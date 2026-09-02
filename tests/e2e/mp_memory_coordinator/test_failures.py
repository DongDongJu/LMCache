# SPDX-License-Identifier: Apache-2.0
"""The CI failure matrix from PLAN.md section 8.

Each case resets both services, injects one fault, starts the real
coordinator, and asserts the pass condition from the two audits and the
durable journal. Cases that kill/restart the coordinator are ``local_only``.
"""

# Standard
from pathlib import Path
import json

# Third Party
import pytest
import requests

# Local
from .conftest import (
    DONOR_IP,
    DONOR_RUNTIME,
    GIB,
    RECEIVER_IP,
    RECEIVER_RUNTIME,
    Harness,
    wait_until,
)

pytestmark = pytest.mark.kind


def _admin_barrier_hit(admin, name: str) -> bool:
    """Whether the named barrier currently holds a request.

    The scenario server reports a list with ``hit``; the mock allocator a
    mapping with ``status == "waiting"``.
    """
    barriers = admin.state().get("barriers", [])
    if isinstance(barriers, dict):
        entry = barriers.get(name, {})
        return bool(entry.get("hit")) or entry.get("status") == "waiting"
    return any(
        b.get("name") == name and (b.get("hit") or b.get("status") == "waiting")
        for b in barriers
    )


def _barrier_hit(harness: Harness, name: str) -> bool:
    return _admin_barrier_hit(harness.scenario, name)


def _no_posts(harness: Harness) -> None:
    """No POST at all: the coordinator never even reached a proposal."""
    assert harness.scenario_posts() == [], harness.scenario_posts()
    assert harness.allocator_posts() == [], harness.allocator_posts()


def _no_move_posts(harness: Harness) -> None:
    """No MOVE POST: at most the receiver's refused GROW probe was issued."""
    assert harness.scenario_posts() == [], harness.scenario_posts()
    assert harness.move_allocator_posts() == [], harness.allocator_posts()
    assert harness.allocator.state()["global"]["assigned_runtime_gib"] == 64
    for record in harness.memcoord.client.journal()["history"]:
        assert record["kind"] == "grow" and record["outcome"] == "NOT_SERVED"


def _outside_posts(harness: Harness) -> list[str]:
    """Outside operations of the MOVE (the refused GROW probe excluded)."""
    return [r["operation"] for r in harness.move_allocator_posts()]


def _mp_posts(harness: Harness) -> list[tuple[str, str]]:
    return [
        (r["service"], (r["body"] or {}).get("mode") or "add")
        for r in harness.scenario_posts()
    ]


def _conserved(harness: Harness) -> None:
    state = harness.allocator.state()
    for node in state["nodes"].values():
        assert (
            node["free_runtime_gib"] + node["assigned_runtime_gib"]
            == node["fixed_runtime_inventory_gib"]
        )
    assert state["global"]["assigned_runtime_gib"] == 64, state["global"]


def _assert_donor_restored(harness: Harness, move: dict) -> None:
    assert move["state"] == "COMPLETE" and move["outcome"] == "ROLLED_BACK", move
    assert harness.outside_status()[DONOR_IP] == [DONOR_RUNTIME]
    assert harness.outside_status()[RECEIVER_IP] == []
    donor = next(
        i
        for i in harness.scenario.state()["instances"]
        if i["instance_id"] == "mp-donor"
    )
    live = [
        d
        for d in donor["devices"]
        if d["device_path"] == DONOR_RUNTIME and d["state"] == "active"
    ]
    assert len(live) == 1, donor["devices"]
    journal = harness.memcoord.client.journal()
    assert [a["device_path"] for a in journal["inventory"]] == [DONOR_RUNTIME]
    _conserved(harness)


# -- eligibility --------------------------------


@pytest.mark.parametrize(
    "fault",
    [
        {"coordinator": {"null_ratio": ["mp-receiver"]}},
        {"coordinator": {"undeclared_capacity": ["mp-receiver"]}},
        {"coordinator": {"shared_dax": ["mp-donor"]}},
        {"coordinator": {"unregistered": ["mp-donor"]}},
        {"coordinator": {"worker_ip_override": {"mp-donor": None}}},
        {"coordinator": {"worker_ip_override": {"mp-receiver": DONOR_IP}}},
    ],
    ids=[
        "null_ratio",
        "undeclared",
        "shared_dax",
        "unregistered",
        "missing_ip",
        "dup_ip",
    ],
)
def test_eligibility_faults_cause_zero_mutation(harness: Harness, fault: dict) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults(fault)
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(6, timeout=60)
    # A donor-side fault leaves the receiver HIGH: its GROW probe is refused
    # by the exhausted pool and nothing else may follow.
    _no_move_posts(harness)
    assert harness.memcoord.client.active_move() is None


def test_snapshot_race_discards_sample(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults(
        {
            "coordinator": {
                "identity_flip": {
                    "instance_id": "mp-receiver",
                    "field": "registration_time",
                    "every_n_reads": 2,
                }
            }
        }
    )
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(6, timeout=60)
    _no_posts(harness)
    rejections = harness.memcoord.client.status()["last_cycle"]["rejections"]
    assert any(
        r["reason"] in ("identity_changed_between_reads", "history_not_stable")
        for r in rejections
    ), rejections


@pytest.mark.parametrize("adapters", [0, 2])
def test_adapter_gate(harness: Harness, adapters: int) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"mp": {"mp-donor": {"adapters": adapters}}})
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(6, timeout=60)
    _no_move_posts(harness)


def test_ownership_gate_without_approved_runtime_path(harness: Harness) -> None:
    """The donor's runtime path is live in the MP but the outside service no
    longer calls it assigned: the coordinator owns nothing, discovery
    declines it with a reason, and nothing is movable."""
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
    # The deallocation freed pool room; keep the pool exhausted so the
    # receiver's GROW probe is refused and the ownership gate is what is
    # under test.
    harness.set_pool_budget(0)
    setup_posts = len(harness.allocator_posts())

    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(8, timeout=60)
    assert harness.scenario_posts() == []
    harness.grow_probe()
    assert len(harness.move_allocator_posts()) == setup_posts
    status = harness.memcoord.client.status()
    assert status["inventory"] == []
    assert DONOR_RUNTIME in status["last_cycle"]["discovery"]["skipped"]
    rejections = status["last_cycle"]["rejections"]
    assert any(r["reason"] == "no_managed_runtime_device" for r in rejections), (
        rejections
    )


def test_live_ratio_mismatch(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    # Coordinator says 8/128 (LOW); live DAX shows the donor nearly full.
    harness.scenario.post(
        "/__test/devices",
        {
            "instance_id": "mp-donor",
            "device_path": DONOR_RUNTIME,
            "used_bytes": 60 * GIB,
        },
    )
    harness.scenario.post(
        "/__test/devices",
        {
            "instance_id": "mp-donor",
            "device_path": "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0",
            "used_bytes": 60 * GIB,
        },
    )
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(8, timeout=60)
    _no_move_posts(harness)
    rejections = harness.memcoord.client.status()["last_cycle"]["rejections"]
    assert any(r["reason"] == "live_ratio_mismatch" for r in rejections), rejections


# -- drain --------------------------------


def test_busy_drain_waits_for_counters_and_tolerates_409(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.post(
        "/__test/devices",
        {
            "instance_id": "mp-donor",
            "device_path": DONOR_RUNTIME,
            "locked_key_count": 2,
        },
    )
    harness.scenario.faults({"mp": {"mp-donor": {"evict_409_count": 2}}})
    harness.memcoord.start()
    harness.memcoord.client.wait_state({"DONOR_DRAINING"}, timeout=60)
    harness.memcoord.client.wait_cycles(4, timeout=30)
    assert _outside_posts(harness) == []
    donor = next(
        i
        for i in harness.scenario.state()["instances"]
        if i["instance_id"] == "mp-donor"
    )
    assert [
        d["state"] for d in donor["devices"] if d["device_path"] == DONOR_RUNTIME
    ] == ["draining"]
    harness.scenario.post(
        "/__test/devices",
        {
            "instance_id": "mp-donor",
            "device_path": DONOR_RUNTIME,
            "locked_key_count": 0,
        },
    )
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"
    posts = _mp_posts(harness)
    assert posts.count(("mp-donor", "evict")) == 3  # two 409s while busy, then success
    evict_responses = [
        r
        for r in harness.scenario.audit()
        if r["kind"] == "response"
        and r["path"] == "/reconfigure/dax/remove"
        and r["status_code"] == 409
    ]
    assert len(evict_responses) == 2
    assert _outside_posts(harness) == ["deallocate", "allocate"]
    _conserved(harness)


def test_drain_deadline_blocks_without_outside_post(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.post(
        "/__test/devices",
        {
            "instance_id": "mp-donor",
            "device_path": DONOR_RUNTIME,
            "borrowed_slot_count": 1,
        },
    )
    harness.memcoord.start(drain_timeout_seconds=3.0)
    move = harness.memcoord.client.wait_terminal(timeout=90)
    assert move["state"] == "BLOCKED", move
    assert "undrain" in move["block_reason"]
    assert _outside_posts(harness) == []
    assert harness.memcoord.client.readyz().status_code == 503
    assert harness.memcoord.client.status()["counters"]["blocked"] == 1


def test_removed_tombstone_is_never_owned_and_health_stays_valid(
    harness: Harness,
) -> None:
    harness.memcoord.seed_inventory()
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"
    donor = next(
        i
        for i in harness.scenario.state()["instances"]
        if i["instance_id"] == "mp-donor"
    )
    assert "removed" in [d["state"] for d in donor["devices"]]
    # Observation keeps working after the tombstone appears: the coordinator
    # stays ready and never treats the tombstone as a device.
    harness.memcoord.client.wait_cycles(3, timeout=30)
    assert harness.memcoord.client.readyz().status_code == 200
    assert DONOR_RUNTIME not in [
        a["device_path"] for a in harness.memcoord.client.journal()["inventory"]
    ]


def test_delayed_capacity_defers_complete(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"coordinator": {"delayed_capacity_seconds": 6.0}})
    harness.memcoord.start()
    move = harness.memcoord.client.wait_state({"ALLOCATED"}, timeout=120)
    receiver_added = move["effects"].get("receiver_add", {}).get("confirmed", False)
    if not receiver_added:
        wait_until(
            lambda: (
                (harness.memcoord.client.active_move() or {})
                .get("effects", {})
                .get("receiver_add", {})
                .get("confirmed", False)
                or harness.memcoord.client.active_move() is None
            ),
            timeout=60,
            what="receiver add confirmed",
        )
    # Still not complete while capacity is stale.
    assert harness.memcoord.client.active_move() is not None
    assert harness.memcoord.client.last_move() is None
    final = harness.memcoord.client.wait_terminal(timeout=120)
    assert final["outcome"] == "SUCCEEDED"
    assert _outside_posts(harness) == ["deallocate", "allocate"]


# -- outside faults --------------------------------


def test_allocation_failure_restores_donor(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults(
        {"operation": "allocate", "mode": "fail_before_mutation", "status_code": 500}
    )
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    assert _outside_posts(harness) == ["deallocate", "allocate", "allocate"]
    assert harness.move_allocator_posts()[2]["body"]["target_node"] == DONOR_IP
    assert _mp_posts(harness) == [
        ("mp-donor", "drain"),
        ("mp-donor", "evict"),
        ("mp-donor", "add"),
    ]


def test_no_receiver_local_match_fails_without_other_size_or_node(
    harness: Harness,
) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults({"operation": "allocate", "mode": "insufficient_capacity"})
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    bodies = [
        r["body"]
        for r in harness.move_allocator_posts()
        if r["operation"] == "allocate"
    ]
    assert (
        bodies[0]["target_node"] == RECEIVER_IP and bodies[0]["request_size_gib"] == 64
    )
    assert bodies[1]["target_node"] == DONOR_IP and bodies[1]["request_size_gib"] == 64


@pytest.mark.parametrize(
    "fault",
    [
        {
            "operation": "deallocate",
            "mode": "missing_field",
            "missing_field_name": "released_size_gib",
        },
        {"operation": "deallocate", "mode": "wrong_echo", "echo_field": "device_path"},
    ],
    ids=["missing_field", "wrong_echo"],
)
def test_deallocation_contract_violation_blocks_safely(
    harness: Harness, fault: dict
) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults(fault)
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    # The mutation happened but its size/echo is unprovable: BLOCKED, and no
    # dependent allocation was issued.
    assert move["state"] == "BLOCKED", move
    assert _outside_posts(harness) == ["deallocate"]
    assert harness.memcoord.client.readyz().status_code == 503


def test_allocation_wrong_echo_releases_and_restores(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults(
        {"operation": "allocate", "mode": "wrong_echo", "echo_field": "request_id"}
    )
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    assert _outside_posts(harness) == [
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
    ]
    release = harness.move_allocator_posts()[2]["body"]
    assert (
        release["target_node"] == RECEIVER_IP
        and release["device_path"] == RECEIVER_RUNTIME
    )
    assert ("mp-receiver", "add") not in _mp_posts(harness)


def test_wrong_size_releases_receiver_path_and_restores_donor(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults(
        {"operation": "allocate", "mode": "wrong_size", "size_gib_override": 32}
    )
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    assert _outside_posts(harness) == [
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
    ]
    assert ("mp-receiver", "add") not in _mp_posts(harness)
    assert harness.allocator.state()["nodes"][RECEIVER_IP]["assigned_runtime_gib"] == 0


@pytest.mark.parametrize(
    "override",
    ["../etc/dax0.1", "/dev/other/dax0.1", DONOR_RUNTIME, "relative/dax0.1"],
    ids=["traversal", "wrong_prefix", "donor_path", "relative"],
)
def test_invalid_returned_path_never_adds(harness: Harness, override: str) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults(
        {"operation": "allocate", "mode": "invalid_path", "path_override": override}
    )
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    assert ("mp-receiver", "add") not in _mp_posts(harness)
    assert _outside_posts(harness) == [
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
    ]


@pytest.mark.local_only
def test_attach_failure_transient_then_persistent(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"mp": {"mp-receiver": {"add_fail_count": 1}}})
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"
    assert _mp_posts(harness).count(("mp-receiver", "add")) == 2
    assert _outside_posts(harness) == ["deallocate", "allocate"]

    harness.reset()
    harness.memcoord.stop()
    harness.memcoord.state_dir = harness.memcoord.state_dir.parent / "state-persistent"
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"mp": {"mp-receiver": {"add_always_fail": True}}})
    harness.memcoord.start(dax_add_max_attempts=2)
    move = harness.memcoord.client.wait_terminal(timeout=120)
    _assert_donor_restored(harness, move)
    assert _mp_posts(harness).count(("mp-receiver", "add")) == 2
    assert _outside_posts(harness) == [
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
    ]


@pytest.mark.parametrize("operation", ["deallocate", "allocate"])
def test_commit_then_drop_blocks_without_blind_retry(
    harness: Harness, operation: str
) -> None:
    harness.memcoord.seed_inventory()
    harness.allocator.faults({"operation": operation, "mode": "commit_then_drop"})
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["state"] == "BLOCKED", move
    assert _outside_posts(harness).count(operation) == 1
    if operation == "deallocate":
        assert "allocate" not in _outside_posts(harness)
    else:
        assert ("mp-receiver", "add") not in _mp_posts(harness)
    mutations = [r for r in harness.allocator.audit() if r["kind"] == "mutation"]
    assert mutations[-1]["operation"] == operation  # the effect did commit


# -- coordinator outage / restart / re-registration -------------------------------


def test_coordinator_outage_before_and_during_move(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"coordinator": {"unavailable": True}})
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(4, timeout=60)
    _no_posts(harness)
    assert harness.memcoord.client.readyz().status_code == 503
    # Park the move after the donor drain, then take the coordinator down.
    harness.scenario.barrier(
        {"instance_id": "mp-donor", "operation": "drain", "when": "after", "name": "o"}
    )
    harness.scenario.clear_faults()
    wait_until(lambda: _barrier_hit(harness, "o"), 90, what="drain barrier hit")
    harness.scenario.faults({"coordinator": {"unavailable": True}})
    harness.scenario.release("o")
    posts_before = len(harness.scenario_posts()) + len(harness.allocator_posts())
    harness.memcoord.client.wait_cycles(5, timeout=60)
    assert (
        len(harness.scenario_posts()) + len(harness.allocator_posts()) == posts_before
    )
    harness.scenario.clear_faults()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"


def test_coordinator_restart_resets_history(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.scenario.faults({"coordinator": {"undeclared_capacity": ["mp-receiver"]}})
    harness.memcoord.start()
    harness.memcoord.client.wait_cycles(4, timeout=60)
    _no_posts(harness)
    reads_before = len(
        [r for r in harness.scenario.audit() if r["path"] == "/instances/usage"]
    )
    harness.scenario.clear_faults()
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"
    audit = harness.scenario.audit()
    first_post = next(i for i, r in enumerate(audit) if r["method"] == "POST")
    fresh_reads = (
        len([r for r in audit[:first_post] if r["path"] == "/instances/usage"])
        - reads_before
    )
    assert fresh_reads >= 3


def test_mp_reregistration_reconciles_before_policy(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.memcoord.start(actuation_enabled=False)
    harness.memcoord.client.wait_cycles(3, timeout=60)
    harness.scenario.post("/__test/instances/mp-donor/reregister", {"bump": "both"})
    harness.memcoord.client.wait_cycles(4, timeout=60)
    _no_posts(harness)
    journal = harness.memcoord.client.journal()
    assert [a["device_path"] for a in journal["inventory"]] == [DONOR_RUNTIME]
    assert journal["inventory"][0]["last_confirmed_state"] == "active"
    status = harness.memcoord.client.status()
    assert status["last_cycle"]["coordinator_reachable"] is True


# -- journal damage / crash recovery / restart -------------------------------------


@pytest.mark.local_only
@pytest.mark.parametrize("damage", ["truncate", "checksum", "version"])
def test_journal_damage_is_unready_and_inert(harness: Harness, damage: str) -> None:
    harness.memcoord.seed_inventory()
    path = harness.memcoord.journal_path()
    if damage == "truncate":
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])
    else:
        envelope = json.loads(path.read_text())
        if damage == "checksum":
            envelope["payload"]["initialized"] = False
        else:
            envelope["schema_version"] = 42
        path.write_text(json.dumps(envelope))
    harness.memcoord.start()
    assert harness.memcoord.client.healthz().status_code == 503
    assert harness.memcoord.client.readyz().status_code == 503
    harness.memcoord.client.wait_cycles(3, timeout=60)
    _no_posts(harness)


@pytest.mark.local_only
@pytest.mark.parametrize(
    ("service", "barrier"),
    [
        (
            "scenario",
            {"instance_id": "mp-donor", "operation": "drain", "when": "after"},
        ),
        (
            "scenario",
            {"instance_id": "mp-donor", "operation": "evict", "when": "after"},
        ),
        ("allocator", {"operation": "deallocate", "when": "before"}),
        ("allocator", {"operation": "deallocate", "when": "after"}),
        ("allocator", {"operation": "allocate", "when": "after"}),
        (
            "scenario",
            {"instance_id": "mp-receiver", "operation": "add", "when": "after"},
        ),
    ],
    ids=[
        "after_drain",
        "after_evict",
        "before_dealloc",
        "after_dealloc",
        "after_alloc",
        "after_add",
    ],
)
def test_crash_recovery_at_every_effect(
    harness: Harness, service: str, barrier: dict
) -> None:
    """Kill the coordinator while an effect is in flight, restart, and require
    a terminal success/rollback or a safe BLOCKED with no duplicate POST."""
    harness.memcoord.seed_inventory()
    admin = harness.scenario if service == "scenario" else harness.allocator
    spec = dict(barrier)
    spec["name"] = "crash"
    admin.barrier(spec)
    harness.memcoord.start()

    try:
        wait_until(
            lambda: _admin_barrier_hit(admin, "crash"), timeout=90, what="barrier hit"
        )
        harness.memcoord.kill()
    finally:
        # Always let the parked request finish before the next reset.
        admin.release("crash")
    harness.memcoord.start()
    move = harness.memcoord.client.wait_terminal(timeout=150)
    outside = _outside_posts(harness)
    assert outside.count("deallocate") <= 1 or move["outcome"] == "ROLLED_BACK", outside
    assert outside.count("allocate") <= 1 or move["outcome"] == "ROLLED_BACK", outside
    # No blind retry: a duplicate request id never reaches the allocator.
    request_ids = [r["body"]["request_id"] for r in harness.allocator_posts()]
    assert len(request_ids) == len(set(request_ids))
    if move["state"] == "BLOCKED":
        # Only an outside POST whose outcome was lost may block.
        assert service == "allocator", move
        assert harness.memcoord.client.readyz().status_code == 503
    else:
        assert move["state"] == "COMPLETE"
        assert move["outcome"] in ("SUCCEEDED", "ROLLED_BACK")
        _conserved(harness)
    if service == "scenario":
        assert move["outcome"] == "SUCCEEDED", move


@pytest.mark.local_only
def test_cooldown_survives_restart(harness: Harness) -> None:
    harness.memcoord.seed_inventory()
    harness.memcoord.start(cooldown_seconds=600.0)
    move = harness.memcoord.client.wait_terminal(timeout=120)
    assert move["outcome"] == "SUCCEEDED"
    # Make a second move attractive: the receiver now holds the managed
    # device and is LOW-ish; the donor stays LOW. Cooldown must still win.
    harness.memcoord.start(cooldown_seconds=600.0)
    harness.memcoord.client.wait_cycles(5, timeout=60)
    assert harness.memcoord.client.active_move() is None
    assert _outside_posts(harness) == ["deallocate", "allocate"]
    assert harness.memcoord.client.journal()["cooldowns"]


@pytest.mark.local_only
def test_scaling_down_and_up_preserves_journal_and_inventory(
    harness: Harness, tmp_path: Path
) -> None:
    harness.memcoord.seed_inventory()
    harness.memcoord.start(actuation_enabled=False)
    harness.memcoord.client.wait_cycles(2, timeout=60)
    before = harness.memcoord.client.journal()
    harness.memcoord.stop()
    harness.memcoord.start(actuation_enabled=False)
    after = harness.memcoord.client.journal()

    def _stable(inventory: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in a.items() if k != "last_confirmed_at"} for a in inventory
        ]

    assert _stable(after["inventory"]) == _stable(before["inventory"])
    assert after["initialized"] is True
    _no_posts(harness)


# -- grow before move --------------------------------


def test_pool_exhausted_falls_back_to_donor_move(harness: Harness) -> None:
    """The receiver's GROW is refused by the exhausted pool (``NOT_SERVED``,
    zero side effects, no cooldown); the very next saga is the donor move."""
    harness.memcoord.seed_inventory()
    harness.memcoord.start()
    history = harness.memcoord.client.wait_history(2, timeout=150)
    grow, move = history
    assert grow["kind"] == "grow" and grow["state"] == "COMPLETE"
    assert grow["outcome"] == "NOT_SERVED"
    assert list(grow["effects"]) == ["allocate"]
    allocate = grow["effects"]["allocate"]
    assert allocate["dispatched"] is True and allocate["confirmed"] is False
    assert allocate["error"] == "explicit failure 409"
    assert allocate["failure"] == "explicit"
    assert allocate["before_paths"] == []
    assert grow["new_path"] == "" and grow["donor"]["instance_id"] == ""
    assert move["kind"] == "move" and move["outcome"] == "SUCCEEDED"
    assert move["donor"]["instance_id"] == "mp-donor"
    assert move["receiver"]["instance_id"] == "mp-receiver"

    # One refused probe, then the move's two outside POSTs; distinct ids.
    posts = harness.allocator_posts()
    assert [r["operation"] for r in posts] == ["allocate", "deallocate", "allocate"]
    probe = posts[0]["body"]
    assert probe["request_id"] == grow["allocation_request_id"]
    assert probe["target_node"] == RECEIVER_IP and probe["request_size_gib"] == 64
    assert posts[2]["body"]["request_id"] == move["allocation_request_id"]
    assert len({r["body"]["request_id"] for r in posts}) == 3
    responses = [
        r
        for r in harness.allocator.audit()
        if r["kind"] == "response" and r["request_id"] == probe["request_id"]
    ]
    assert [r["status_code"] for r in responses] == [409]
    # The refused probe mutated nothing at the allocator.
    mutations = [r for r in harness.allocator.audit() if r["kind"] == "mutation"]
    assert [(m["operation"], m["device_path"]) for m in mutations] == [
        ("deallocate", DONOR_RUNTIME),
        ("allocate", RECEIVER_RUNTIME),
    ]
    assert _mp_posts(harness) == [
        ("mp-donor", "drain"),
        ("mp-donor", "evict"),
        ("mp-receiver", "add"),
    ]
    journal = harness.memcoord.client.journal()
    # No cooldown after NOT_SERVED (the move started well within the 10 s
    # cooldown), only a per-worker grow backoff.
    assert move["created_at"] - grow["updated_at"] < 10.0
    assert journal["grow_backoffs"][RECEIVER_IP] > grow["updated_at"]
    assert len(journal["cooldowns"]) == 2
    counters = harness.memcoord.client.status()["counters"]
    assert counters["not_served"] == 1 and counters["succeeded"] == 1
    assert counters["grown"] == 0 and counters["proposed"] == 2
    _conserved(harness)
