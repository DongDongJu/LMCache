# SPDX-License-Identifier: Apache-2.0
"""Happy-path E2E: one exact grow, and one exact move, through the real
coordinator.

Every assertion from PLAN.md section 8 "Required happy-path assertions" is
made from the two services' endpoint-local audits, their admin state, and
the coordinator's durable journal -- correlated by request id, device path,
and confirmed phase, never by cross-process wall clocks.

Grow before move: with the harness's default pool budget (the initially
assigned 64 GiB) the coordinator's first saga is a GROW the allocator
refuses (``NOT_SERVED``, no mutation), and the move follows unchanged; with
a raised budget the GROW is served and no donor is touched.
"""

# Third Party
import pytest
import requests

# Local
from .conftest import (
    DONOR_IP,
    DONOR_RUNTIME,
    DONOR_SPARE,
    GIB,
    GROW_KIND,
    RECEIVER_IP,
    RECEIVER_RUNTIME,
    RECEIVER_SPARE,
    Harness,
    wait_until,
)

pytestmark = pytest.mark.kind


def _usage_reads_before_first_post(records: list[dict]) -> int:
    """Count ``/instances/usage`` reads the coordinator made before mutating."""
    count = 0
    for record in records:
        if record["kind"] != "request":
            continue
        if record["method"] == "POST":
            break
        if record["path"] == "/instances/usage":
            count += 1
    return count


def test_happy_path_moves_exactly_one_device(harness: Harness) -> None:
    initial = harness.allocator.state()
    assert initial["global"]["assigned_runtime_gib"] == 64
    assert harness.outside_status() == {DONOR_IP: [DONOR_RUNTIME], RECEIVER_IP: []}

    harness.memcoord.seed_inventory()
    assert harness.scenario_posts() == [] and harness.allocator_posts() == []
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["state"] == "COMPLETE", move
    assert move["outcome"] == "SUCCEEDED", move
    assert move["kind"] == "move"
    # 0. Grow before move: the exhausted pool refused the receiver's GROW
    #    first (one refused POST, nothing mutated, no cooldown), so the
    #    move is the second archived saga.
    probe = harness.grow_probe()
    journal = harness.memcoord.client.journal()
    assert [(m["kind"], m["outcome"]) for m in journal["history"]] == [
        ("grow", "NOT_SERVED"),
        ("move", "SUCCEEDED"),
    ]
    assert move["created_at"] - probe["updated_at"] < 10.0  # no cooldown between
    assert set(journal["grow_backoffs"]) == {RECEIVER_IP}

    scenario_audit = harness.scenario.audit()
    allocator_audit = harness.allocator.audit()
    move_ids = {r["body"]["request_id"] for r in harness.move_allocator_posts()}

    # 1. Two eligible samples cause zero mutation; the third permits one move.
    assert _usage_reads_before_first_post(scenario_audit) >= 3
    # Preflight: both /status reads precede any mutation.
    first_post = next(i for i, r in enumerate(scenario_audit) if r["method"] == "POST")
    statuses = {
        r["service"]
        for r in scenario_audit[:first_post]
        if r["kind"] == "request" and r["path"] == "/status"
    }
    assert statuses == {"mp-donor", "mp-receiver"}

    # 2. Exact mutation subsequence per service, and the causal order across
    #    services from the journal's effect ledger.
    mp_posts = harness.scenario_posts()
    assert [
        (r["service"], r["path"], (r["body"] or {}).get("mode")) for r in mp_posts
    ] == [
        ("mp-donor", "/reconfigure/dax/remove", "drain"),
        ("mp-donor", "/reconfigure/dax/remove", "evict"),
        ("mp-receiver", "/reconfigure/dax/add", None),
    ]
    assert [r["operation"] for r in harness.move_allocator_posts()] == [
        "deallocate",
        "allocate",
    ]
    effects = move["effects"]
    order = ["donor_drain", "donor_evict", "deallocate", "allocate", "receiver_add"]
    assert list(effects) == order
    intents = [effects[name]["intent_at"] for name in order]
    assert intents == sorted(intents)
    # 3. Deallocation only after the old path is no longer readable: the
    #    evict was confirmed from status before the deallocation intent.
    assert effects["donor_evict"]["confirmed"] is True
    assert effects["donor_evict"]["confirmed_at"] <= effects["deallocate"]["intent_at"]
    donor_state = next(
        i
        for i in harness.scenario.state()["instances"]
        if i["instance_id"] == "mp-donor"
    )
    tombstones = [
        d for d in donor_state["devices"] if d["device_path"] == DONOR_RUNTIME
    ]
    assert [d["state"] for d in tombstones] == ["removed"]

    # 4. Exact frozen bodies, no extra fields; 5. every echo and size validated.
    posts = harness.move_allocator_posts()
    deallocation, allocation = posts[0]["body"], posts[1]["body"]
    assert deallocation == {
        "request_id": move["deallocation_request_id"],
        "target_node": DONOR_IP,
        "device_path": DONOR_RUNTIME,
    }
    assert allocation == {
        "request_id": move["allocation_request_id"],
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    assert move["deallocation_request_id"] and move["allocation_request_id"]
    assert move["deallocation_request_id"] != move["allocation_request_id"]
    responses = [
        r
        for r in allocator_audit
        if r["kind"] == "response"
        and r["operation"] != "status"
        and r["request_id"] in move_ids
    ]
    assert responses[0]["body"]["request_id"] == move["deallocation_request_id"]
    assert responses[1]["body"]["request_id"] == move["allocation_request_id"]
    assert set(responses[0]["body"]) == {
        "status",
        "request_id",
        "target_node",
        "device_path",
        "released_size_gib",
    }
    assert set(responses[1]["body"]) == {
        "status",
        "request_id",
        "target_node",
        "device_path",
        "requested_size_gib",
        "granted_size_gib",
    }
    assert move["released_size_gib"] == move["granted_size_gib"] == 64
    assert effects["deallocate"]["response"]["released_size_gib"] == 64
    assert effects["allocate"]["response"]["requested_size_gib"] == 64

    # 6. Receiver add uses adapter 0, the returned path, and "64GiB".
    assert mp_posts[2]["body"] == {
        "adapter_index": 0,
        "device_path": RECEIVER_RUNTIME,
        "size": "64GiB",
    }
    assert move["new_path"] == RECEIVER_RUNTIME

    # 7. COMPLETE only after the new path is active and capacity converged.
    assert effects["receiver_add"]["confirmed"] is True
    instances = {i["instance_id"]: i for i in harness.scenario.state()["instances"]}
    assert instances["mp-donor"]["capacity_bytes"] == 64 * GIB
    assert instances["mp-receiver"]["capacity_bytes"] == 128 * GIB
    receiver_devices = {
        d["device_path"]: d for d in instances["mp-receiver"]["devices"]
    }
    assert receiver_devices[RECEIVER_RUNTIME]["state"] == "active"

    # 8. Final outside assigned GiB equals the pre-move value; a zero-assigned
    #    gap existed between the two outside mutations.
    final = harness.allocator.state()
    assert final["global"]["assigned_runtime_gib"] == 64
    mutations = [r for r in allocator_audit if r["kind"] == "mutation"]
    assert [(m["operation"], m["device_path"]) for m in mutations] == [
        ("deallocate", DONOR_RUNTIME),
        ("allocate", RECEIVER_RUNTIME),
    ]
    assert harness.outside_status() == {DONOR_IP: [], RECEIVER_IP: [RECEIVER_RUNTIME]}

    # 9. Managed inventory holds the receiver path; the mock still holds both
    #    paths under their original workers.
    journal = harness.memcoord.client.journal()
    assert [a["device_path"] for a in journal["inventory"]] == [RECEIVER_RUNTIME]
    assert journal["inventory"][0]["worker_ip"] == RECEIVER_IP
    nodes = final["nodes"]
    assert {d["path"] for d in nodes[DONOR_IP]["devices"]} >= {
        DONOR_RUNTIME,
        DONOR_SPARE,
    }
    assert {d["path"] for d in nodes[RECEIVER_IP]["devices"]} >= {
        RECEIVER_RUNTIME,
        RECEIVER_SPARE,
    }

    # 10. Cooldown prevents a second move.
    harness.memcoord.client.wait_cycles(4, timeout=30)
    assert harness.memcoord.client.active_move() is None
    assert len(harness.move_allocator_posts()) == 2
    assert len(harness.allocator_posts()) == 3  # plus the refused GROW probe
    rejections = harness.memcoord.client.status()["last_cycle"]["rejections"]
    assert any(r["reason"] == "cooldown" for r in rejections), rejections

    # 11. Exactly one deallocation, one allocation, one successful add.
    assert len(harness.scenario_posts()) == 3
    counters = harness.memcoord.client.status()["counters"]
    assert counters["succeeded"] == 1 and counters["not_served"] == 1
    assert counters["grown"] == 0

    # 12. Conservation and immutable bindings/sizes.
    for node_ip, node in nodes.items():
        assert (
            node["free_runtime_gib"] + node["assigned_runtime_gib"]
            == node["fixed_runtime_inventory_gib"]
        ), node_ip
        for device in node["devices"]:
            assert device["size_gib"] == 64
            assert device["path"] in {
                d["path"] for d in initial["nodes"][node_ip]["devices"]
            }
    assert (
        final["global"]["free_runtime_gib"] + final["global"]["assigned_runtime_gib"]
        == final["global"]["fixed_runtime_inventory_gib"]
    )
    assert harness.memcoord.client.readyz().status_code == 200


def test_dry_run_proposes_but_never_mutates(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.memcoord.start(actuation_enabled=False)
    harness.memcoord.client.wait_cycles(6, timeout=60)
    status = harness.memcoord.client.status()
    proposal = status["last_cycle"]["proposal"]
    assert proposal is not None
    # Grow before move: a dry run proposes the receiver's GROW (nothing is
    # probed, so the move alternative is never reached).
    assert proposal["kind"] == "grow"
    assert proposal["receiver"] == "mp-receiver"
    assert proposal["receiver_worker_ip"] == RECEIVER_IP
    assert proposal["request_size_gib"] == 64
    assert "donor" not in proposal
    assert any(
        r["reason"] == "actuation_disabled" for r in status["last_cycle"]["rejections"]
    )
    assert harness.scenario_posts() == []
    assert harness.allocator_posts() == []
    assert harness.memcoord.client.active_move() is None
    assert status["counters"]["proposed"] >= 1


@pytest.mark.local_only
def test_discovery_moves_a_device_without_any_allowlist(harness: Harness) -> None:
    """The default mode derives ownership from outside status alone.

    No ``seed_inventory``, no ``adoption_file``, no ``--adopt``: the donor's
    runtime path is claimed because the outside service lists it under the
    donor's worker IP, and the same single move follows.
    """
    assert harness.outside_status() == {DONOR_IP: [DONOR_RUNTIME], RECEIVER_IP: []}

    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["state"] == "COMPLETE", move
    assert move["outcome"] == "SUCCEEDED", move

    # Exactly the one device the outside service confirmed, and nothing else
    # (after the refused GROW probe of the exhausted pool).
    harness.grow_probe()
    assert [r["operation"] for r in harness.move_allocator_posts()] == [
        "deallocate",
        "allocate",
    ]
    status = harness.memcoord.client.status()
    origins = {a["origin"] for a in status["inventory"]}
    assert origins <= {"discovered", "allocated"}, status["inventory"]


def test_discovery_declines_a_path_outside_status_does_not_confirm(
    harness: Harness,
) -> None:
    """A live device the outside service does not list is never claimed."""
    response = requests.post(
        f"{harness.endpoints.allocator_public_url}/api/v2/apps/lmcache/deallocations",
        json={
            "request_id": "unown-donor-runtime",
            "target_node": DONOR_IP,
            "device_path": DONOR_RUNTIME,
        },
        timeout=5,
    )
    assert response.status_code < 300, response.text
    assert harness.outside_status() == {DONOR_IP: [], RECEIVER_IP: []}
    # The deallocation freed pool room; keep the pool exhausted so the
    # receiver's GROW probe is refused and only discovery is under test.
    harness.set_pool_budget(0)
    setup_posts = len(harness.allocator_posts())

    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(8, timeout=60)
    assert harness.scenario_posts() == []
    # The only further allocator POST is the refused GROW probe.
    harness.grow_probe()
    assert len(harness.move_allocator_posts()) == setup_posts
    status = harness.memcoord.client.status()
    assert status["inventory"] == []
    skipped = status["last_cycle"]["discovery"]["skipped"]
    assert DONOR_RUNTIME in skipped, skipped


def test_present_assigned_device_is_attached_then_adopted(harness: Harness) -> None:
    """Attach orchestration: present + outside-assigned + unattached -> add.

    The donor's spare path is made present in the fake MP server's watched
    directory and assigned to the donor by the allocator, but not attached.
    The coordinator must issue exactly one ``/reconfigure/dax/add`` for it
    (no outside POST), and discovery must adopt it on a later cycle.
    """
    harness.set_pool_budget(128)  # the setup assignment below needs room
    response = requests.post(
        f"{harness.endpoints.allocator_public_url}/api/v2/apps/lmcache/allocations",
        json={
            "request_id": "assign-donor-spare",
            "target_node": DONOR_IP,
            "request_size_gib": 64,
            "mode": "devdax",
            "purpose": "lmcache-dax",
            "access": "exclusive",
        },
        timeout=5,
    )
    assert response.status_code < 300, response.text
    assert response.json()["device_path"] == DONOR_SPARE
    assert harness.outside_status()[DONOR_IP] == [DONOR_RUNTIME, DONOR_SPARE]
    setup_posts = len(harness.allocator_posts())
    harness.scenario.post(
        "/__test/present_devices",
        {"instance_id": "mp-donor", "device_path": DONOR_SPARE, "size_bytes": 64 * GIB},
    )
    # Keep the fleet ineligible for a move so only attach orchestration acts.
    harness.scenario.post(
        "/__test/usage", {"instance_id": "mp-receiver", "used_bytes": 8 * GIB}
    )

    harness.memcoord.start()

    def _attached() -> bool:
        return any(
            r["service"] == "mp-donor"
            and r["path"] == "/reconfigure/dax/add"
            and r["body"]
            == {
                "adapter_index": 0,
                "device_path": DONOR_SPARE,
                "size": "64GiB",
            }
            for r in harness.scenario_posts()
        )

    wait_until(_attached, 60.0, what="attach of the donor spare")

    def _adopted() -> bool:
        return DONOR_SPARE in [
            a["device_path"] for a in harness.memcoord.client.journal()["inventory"]
        ]

    wait_until(_adopted, 60.0, what="discovery adopting the attached spare")
    harness.memcoord.client.wait_cycles(3, timeout=30)

    # Exactly one add, no outside mutation, and the device is live + owned.
    assert [(r["service"], r["path"]) for r in harness.scenario_posts()] == [
        ("mp-donor", "/reconfigure/dax/add")
    ]
    assert len(harness.allocator_posts()) == setup_posts
    donor = next(
        i
        for i in harness.scenario.state()["instances"]
        if i["instance_id"] == "mp-donor"
    )
    assert {d["device_path"]: d["state"] for d in donor["devices"]}[DONOR_SPARE] == (
        "active"
    )
    assert donor["capacity_bytes"] == 192 * GIB
    status = harness.memcoord.client.status()
    adopted = next(a for a in status["inventory"] if a["device_path"] == DONOR_SPARE)
    assert adopted["origin"] == "discovered"
    assert adopted["worker_ip"] == DONOR_IP
    assert status["counters"]["attached"] == 1
    attachments = status["last_cycle"]["attachments"]
    assert attachments["planned"] == [] and attachments["attached"] == []
    assert attachments["skipped"][DONOR_SPARE] == "already attached"
    assert harness.memcoord.client.active_move() is None
    assert harness.memcoord.client.readyz().status_code == 200


def test_explicit_adoption_command_populates_inventory(harness: Harness) -> None:
    """Adoption requires the exact (worker_ip, path, size) tuple."""
    result = harness.memcoord.adopt(
        {
            "allocations": [
                {
                    "worker_ip": DONOR_IP,
                    "device_path": DONOR_RUNTIME,
                    "allocation_size_gib": 64,
                    "device_map_size_bytes": 64 * GIB,
                },
                {  # bootstrap: rejected (index 0)
                    "worker_ip": DONOR_IP,
                    "device_path": "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0",
                    "allocation_size_gib": 64,
                    "device_map_size_bytes": 64 * GIB,
                },
                {  # free at the outside service: rejected
                    "worker_ip": RECEIVER_IP,
                    "device_path": RECEIVER_RUNTIME,
                    "allocation_size_gib": 64,
                    "device_map_size_bytes": 64 * GIB,
                },
            ]
        }
    )
    assert result.returncode == 0, result.stderr
    assert f"adopted {DONOR_RUNTIME}" in result.stdout
    assert "rejected /dev/dax-cxl/lmcache-e2e--mp-196/dax0.0" in result.stdout
    assert f"rejected {RECEIVER_RUNTIME}" in result.stdout
    assert harness.scenario_posts() == [] and harness.allocator_posts() == []

    harness.memcoord.config_overrides = {}
    harness.memcoord.start()
    journal = harness.memcoord.client.journal()
    assert journal["initialized"] is True
    assert [a["device_path"] for a in journal["inventory"]] == [DONOR_RUNTIME]
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"


def test_grow_adds_memory_without_a_donor(harness: Harness) -> None:
    """Grow before move: a pool with room serves the receiver directly.

    With the budget raised the coordinator's first saga is a GROW: one
    allocation POST for the receiver's own worker and one receiver add; no
    donor device is drained, removed, or deallocated.
    """
    harness.reset(pool_budget_gib=128)
    initial = harness.allocator.state()
    assert initial["global"]["assigned_runtime_gib"] == 64
    assert harness.outside_status() == {DONOR_IP: [DONOR_RUNTIME], RECEIVER_IP: []}
    harness.memcoord.seed_inventory()
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120, kind=GROW_KIND)
    assert move["kind"] == "grow"
    assert move["state"] == "COMPLETE" and move["outcome"] == "SUCCEEDED", move
    assert move["move_id"].startswith("grow-")

    # Exactly one outside POST (the allocation) and one MP POST (the add).
    posts = harness.allocator_posts()
    assert [r["operation"] for r in posts] == ["allocate"]
    assert posts[0]["body"] == {
        "request_id": move["allocation_request_id"],
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    responses = [
        r
        for r in harness.allocator.audit()
        if r["kind"] == "response" and r["operation"] == "allocate"
    ]
    assert responses[0]["body"]["device_path"] == RECEIVER_RUNTIME
    assert responses[0]["body"]["granted_size_gib"] == 64
    mp_posts = harness.scenario_posts()
    assert [(r["service"], r["path"]) for r in mp_posts] == [
        ("mp-receiver", "/reconfigure/dax/add")
    ]
    assert mp_posts[0]["body"] == {
        "adapter_index": 0,
        "device_path": RECEIVER_RUNTIME,
        "size": "64GiB",
    }
    assert list(move["effects"]) == ["allocate", "receiver_add"]
    assert all(e["confirmed"] for e in move["effects"].values())
    assert move["new_path"] == RECEIVER_RUNTIME
    assert move["granted_size_gib"] == 64
    assert move["donor"]["instance_id"] == "" and move["old_path"] == ""

    # The donor is untouched; the pool grew by one device.
    instances = {i["instance_id"]: i for i in harness.scenario.state()["instances"]}
    donor_devices = {
        d["device_path"]: d["state"] for d in instances["mp-donor"]["devices"]
    }
    assert donor_devices[DONOR_RUNTIME] == "active"
    assert instances["mp-donor"]["capacity_bytes"] == 128 * GIB
    assert instances["mp-receiver"]["capacity_bytes"] == 128 * GIB
    receiver_devices = {
        d["device_path"]: d["state"] for d in instances["mp-receiver"]["devices"]
    }
    assert receiver_devices[RECEIVER_RUNTIME] == "active"
    final = harness.allocator.state()
    assert final["global"]["assigned_runtime_gib"] == 128
    assert harness.outside_status() == {
        DONOR_IP: [DONOR_RUNTIME],
        RECEIVER_IP: [RECEIVER_RUNTIME],
    }
    mutations = [r for r in harness.allocator.audit() if r["kind"] == "mutation"]
    assert [(m["operation"], m["device_path"]) for m in mutations] == [
        ("allocate", RECEIVER_RUNTIME)
    ]
    for node_ip, node in final["nodes"].items():
        assert (
            node["free_runtime_gib"] + node["assigned_runtime_gib"]
            == node["fixed_runtime_inventory_gib"]
        ), node_ip

    # Inventory gained the receiver path; the donor entry stays.
    journal = harness.memcoord.client.journal()
    inventory = {a["device_path"]: a for a in journal["inventory"]}
    assert set(inventory) == {DONOR_RUNTIME, RECEIVER_RUNTIME}
    grown = inventory[RECEIVER_RUNTIME]
    assert grown["origin"] == "allocated"
    assert grown["worker_ip"] == RECEIVER_IP
    assert grown["instance_id"] == "mp-receiver"
    assert grown["allocation_size_gib"] == 64
    assert grown["device_map_size_bytes"] == 64 * GIB
    assert inventory[DONOR_RUNTIME]["origin"] == "adopted"
    # Cooldown on the receiver only; no backoff.
    assert len(journal["cooldowns"]) == 1
    assert journal["grow_backoffs"] == {}
    assert journal["history"][-1]["kind"] == "grow"
    counters = harness.memcoord.client.status()["counters"]
    assert counters["succeeded"] == 1 and counters["grown"] == 1
    assert counters["proposed"] == 1 and counters["not_served"] == 0

    # Cooldown prevents a second saga; nothing else is posted.
    harness.memcoord.client.wait_cycles(4, timeout=30)
    assert harness.memcoord.client.active_move() is None
    assert len(harness.allocator_posts()) == 1
    assert len(harness.scenario_posts()) == 1
    rejections = harness.memcoord.client.status()["last_cycle"]["rejections"]
    assert any(r["reason"] == "cooldown" for r in rejections), rejections
    assert harness.memcoord.client.readyz().status_code == 200
