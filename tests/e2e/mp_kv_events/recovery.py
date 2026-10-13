# SPDX-License-Identifier: Apache-2.0
"""Recovery tests mutate real private caches while preserving router ownership."""

# Standard
from typing import TYPE_CHECKING
import json
import re
import time

# Third Party
from fixtures import fixture
from residency import (
    assert_target_absent,
    assert_target_resident,
    target_evictions,
    target_l1_keys,
)
from run import write_json

if TYPE_CHECKING:
    # Third Party
    from scenarios import Driver


def control(driver: "Driver", replica: str, action: str, **fields: object) -> dict:
    """Invoke the guarded MP test module through the selected real transport."""
    scheme = "tcp" if driver.state["transport"] == "zmq" else "grpc"
    output = driver.exec(
        replica,
        "import subprocess,sys; subprocess.run([sys.executable,"
        "'/opt/mp-kv-events/mp_test_hooks.py','--url',sys.argv[1],sys.argv[2]],"
        "check=True)",
        f"{scheme}://127.0.0.1:5555",
        json.dumps({"action": action, **fields}),
    )
    return json.loads(output.splitlines()[-1])


def verify_recovered_load(driver: "Driver", owner: str, index: int, label: str) -> None:
    """Require fresh complete residency, unbiased routing, and completed H2D."""

    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    baseline = driver.warm(tokens, owner)
    driver.evict_gpu(tokens, owner)
    driver.probe_owner(tokens, owner)
    driver.measured(tokens, owner, True, baseline, label)


def reconnect(driver: "Driver", owner: str, index: int) -> dict:
    """T08: interrupt only the event source, evict real data, and replay."""

    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    driver.warm(tokens, owner)
    before = driver.snapshot(f"T08-{owner}-before")
    gpu = driver.ledger(owner).prefix(tokens, "GPU")
    cpu_hashes = {
        str(block["hash"]) for block in driver.ledger(owner).prefix(tokens, "CPU")
    }
    assert cpu_hashes
    cursor = before[owner]["status"]["kv_events"]["next_seq"] - 1
    world = driver.state["tp"]
    model = driver.model_name(owner)
    try:
        status = control(driver, owner, "disconnect")
        assert status["kv_events_e2e"]["disconnected"]
        driver.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        # The transfer/request path remains alive during an event-only outage.
        driver.active_barrier(owner)
        barrier_tokens = driver.requests[-1]["tokens"]

        def gap_store_committed() -> bool:
            records = driver.records(owner, "mp-events.jsonl")
            try:
                keys = target_l1_keys(
                    records, barrier_tokens, driver.state["chunk_size"], model, world
                )
                assert_target_resident(records, keys)
                return True
            except AssertionError:
                return False

        driver.wait(
            gap_store_committed, "real STORE completed during outage", timeout=30
        )
        # This second setup step gives a store-only fallback a publication
        # opportunity while admission remains closed. Its unique fixture token
        # must not appear in CPU Stored, including native subblock-sized events.
        driver.active_barrier(owner)
        assert driver.ledger(owner).prefix(tokens, "GPU") == gpu
        gap = driver.records(owner, "native-events.jsonl")[
            len(before[owner]["native"]) :
        ]
        assert not [
            event
            for record in gap
            for event in record["payload"][1]
            if event["type"] == "BlockStored"
            and event.get("medium") in ("CPU", "CPU_PINNED")
            and event["token_ids"]
            and event["token_ids"][0] == barrier_tokens[0]
        ], "A local CPU-store source became active during subscription loss"
        replay = control(
            driver, owner, "events", model_name=model, world_size=world, cursor=cursor
        )
        assert not replay["lost"], "Use the retained-log profile for T08"
        removals = [
            event
            for event in replay["events"]
            if event["kind"] == "removed"
            and cpu_hashes.intersection(str(value) for value in event["block_hashes"])
        ]
        assert cpu_hashes == {
            str(value) for event in removals for value in event["block_hashes"]
        }, "The real MP removals do not cover the fixture's native identities"
        removal_cursor = max(event["seq"] for event in removals)
    finally:
        control(driver, owner, "reconnect")

    def received_replay() -> bool:
        log = driver.exec(
            owner,
            "from pathlib import Path; "
            "print(Path('/artifacts/engine.log').read_text())",
        )[len(before[owner]["engine_log"]) :]
        assert "reset=protocol_error" not in log
        return any(
            int(epoch) == replay["incarnation"] and int(next_cursor) >= removal_cursor
            for epoch, next_cursor in re.findall(
                r"MP_KV_SUBSCRIBER received_ns=\d+ incarnation=(\d+) cursor=(\d+)", log
            )
        )

    driver.wait(
        received_replay, "same-source replay reaches the actual removal", timeout=30
    )
    driver.active_barrier(owner)
    driver.wait(
        lambda: not driver.ledger(owner).prefix(tokens, "CPU"),
        "replayed removal after event-only disconnect",
    )
    status = control(driver, owner, "status")
    assert status["kv_events"]["subscribers"] == 1
    after = driver.snapshot(f"T08-{owner}-after")
    observed = after[owner]["native"][len(before[owner]["native"]) :]
    transitions = assert_cpu_replay_order(observed, cpu_hashes)
    assert (
        after[owner]["status"]["kv_events"]["incarnation"]
        == (before[owner]["status"]["kv_events"]["incarnation"])
    )
    verify_recovered_load(driver, owner, index + 1, f"T08-{owner}-recovered")
    write_json(
        driver.root / f"T08-{owner}-ordered-replay.json",
        {
            "source_records": replay,
            "native_transitions": transitions,
            "outage_native_records": gap,
            "gap_store_tokens": barrier_tokens,
        },
    )
    return {
        "status": "PASS",
        "transport": driver.state["transport"],
        "native_transitions": transitions,
    }


def assert_cpu_replay_order(
    records: list[dict], hashes: set[str]
) -> dict[str, list[str]]:
    """Require CPU invalidation, historical replay, then the physical removal."""
    history: dict[str, list[str]] = {block_hash: [] for block_hash in hashes}
    for record in records:
        for event in record["payload"][1]:
            if event.get("medium") not in ("CPU", "CPU_PINNED"):
                continue
            if event["type"] not in ("BlockStored", "BlockRemoved"):
                continue
            for block_hash in event["block_hashes"]:
                identity = str(block_hash)
                if identity in history:
                    history[identity].append(
                        "stored" if event["type"] == "BlockStored" else "removed"
                    )
    assert history and all(
        kinds == ["removed", "stored", "removed"] for kinds in history.values()
    ), ("CPU invalidation/replay order changed", history)
    return history


def overflow(driver: "Driver", owner: str, index: int) -> dict:
    """T09: truncate replay with real stores while subscription admission is shut."""

    initial = control(driver, owner, "status")["kv_events"]
    capacity = initial["log_capacity"]
    if capacity > 64:
        return {
            "status": "BLOCKED",
            "reason": "Run the overflow profile with deploy --event-log-size 8",
        }
    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    driver.warm(tokens, owner)
    before = driver.snapshot(f"T09-{owner}-before-overflow")
    previous_cursor = before[owner]["status"]["kv_events"]["next_seq"] - 1
    try:
        control(driver, owner, "disconnect")
        driver.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        for number in range(capacity + 2):
            driver.request(
                fixture(index + number + 1, chunk_size=driver.state["chunk_size"]),
                owner,
            )
        status = control(driver, owner, "status")["kv_events"]
        assert status["next_seq"] - status["log_depth"] > previous_cursor + 1
        assert status["log_depth"] == capacity
        source = control(
            driver,
            owner,
            "events",
            model_name=driver.model_name(owner),
            world_size=driver.state["tp"],
            cursor=previous_cursor,
        )
        assert source["lost"], "Server did not report actual retained-log loss"
    finally:
        control(driver, owner, "reconnect")
    receipt = {}

    def explicit_loss_received() -> bool:
        nonlocal receipt
        log = driver.exec(
            owner,
            "from pathlib import Path; "
            "print(Path('/artifacts/engine.log').read_text())",
        )[len(before[owner]["engine_log"]) :]
        try:
            receipt = require_loss_receipt(
                log, status["incarnation"], status["next_seq"] - 1
            )
            return True
        except AssertionError:
            return False

    driver.wait(
        explicit_loss_received,
        "subscriber receives explicit retained-log loss",
        timeout=30,
    )
    driver.active_barrier(owner)
    driver.wait(
        lambda: not driver.ledger(owner).prefix(tokens, "CPU"),
        "truncated replay cannot resurrect an evicted placement",
    )
    driver.snapshot(f"T09-{owner}-after-overflow")
    verify_recovered_load(driver, owner, index + capacity + 3, f"T09-{owner}-recovered")
    write_json(
        driver.root / f"T09-{owner}-loss-receipt.json",
        {"source": source, "receipt": receipt},
    )
    return {
        "status": "PASS",
        "scope": (
            "real-operation server log overflow; local queue and bus loss "
            "have separate focused tests"
        ),
        "capacity": capacity,
        "loss_receipt": receipt,
    }


def require_loss_receipt(log: str, incarnation: int, cursor: int) -> dict:
    """Require explicit loss followed by receipt through the truncated source log."""
    marker = "MP_KV_SUBSCRIBER reset=events_lost"
    offset = log.find(marker)
    assert offset >= 0, "Missing explicit event-loss reset"
    for timestamp, epoch, position in re.findall(
        r"MP_KV_SUBSCRIBER received_ns=(\d+) incarnation=(\d+) cursor=(\d+)",
        log[offset:],
    ):
        if int(epoch) == incarnation and int(position) >= cursor:
            return {
                "received_ns": int(timestamp),
                "incarnation": int(epoch),
                "cursor": int(position),
            }
    raise AssertionError("Loss reset did not reach the required replay cursor")


def refresh_worker(driver: "Driver", owner: str, previous: int) -> dict:
    """Verify discovery replaced the stopped engine's serving identity."""
    starts = [
        entry["time_ns"]
        for entry in driver.records(owner, "lifecycle.jsonl")
        if entry["action"] == "start" and entry["process"] == "engine"
    ]
    driver.native_epoch_ns[owner] = starts[-1]
    metrics = "\n".join(
        line
        for line in driver.local_http(owner, 8081, "/metrics").splitlines()
        if line.startswith("dynamo_component_inflight_requests{")
        and 'dynamo_endpoint="generate"' in line
    )
    current = driver.worker_identity(owner)
    assert current != previous
    endpoint = []
    for label in ("dynamo_namespace", "dynamo_component", "dynamo_endpoint"):
        values = set(re.findall(label + r'="([^"]+)"', metrics))
        assert len(values) == 1, (label, values)
        endpoint.append(values.pop())
    script = """
import asyncio, json, sys
from dynamo.runtime import DistributedRuntime
async def inspect():
    runtime = DistributedRuntime(asyncio.get_running_loop(), 'etcd', 'nats')
    try:
        client = await runtime.endpoint(sys.argv[1]).client()
        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            ids = list(client.instance_ids())
            if int(sys.argv[2]) not in ids and int(sys.argv[3]) in ids:
                print(json.dumps(ids))
                return
            await asyncio.sleep(0.2)
        raise RuntimeError('Engine identity did not converge in discovery')
    finally:
        runtime.shutdown()
asyncio.run(inspect())
"""
    output = driver.exec(owner, script, ".".join(endpoint), str(previous), str(current))
    identities = json.loads(output.splitlines()[-1])
    driver.workers[owner] = current
    return {
        "previous": previous,
        "replacement": current,
        "discovered": identities,
        "publisher_epoch_ns": starts[-1],
    }


def warm_restart(driver: "Driver", owner: str, index: int) -> dict:
    """T10: restart only the engine and use retained MP CPU data through Dynamo."""

    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    baseline = driver.warm(tokens, owner)
    before = control(driver, owner, "status")["kv_events"]
    physical_keys = target_l1_keys(
        driver.records(owner, "mp-events.jsonl"),
        tokens,
        driver.state["chunk_size"],
        driver.model_name(owner),
        driver.state["tp"],
    )
    assert_target_resident(driver.records(owner, "mp-events.jsonl"), physical_keys)
    if before["log_capacity"] < 1024:
        return {
            "status": "BLOCKED",
            "reason": "Warm retained-log restart requires the normal large-log profile",
        }
    old_worker = driver.workers[owner]
    driver.local_http(owner, 9900, "/restart/engine", "POST", timeout=720)
    identities = refresh_worker(driver, owner, old_worker)
    after = control(driver, owner, "status")["kv_events"]
    assert before["incarnation"] == after["incarnation"]
    assert_target_resident(driver.records(owner, "mp-events.jsonl"), physical_keys)
    driver.active_barrier(owner)
    driver.wait(
        lambda: bool(driver.ledger(owner).prefix(tokens, "CPU")),
        "warm CPU replay under replacement engine identity",
    )
    assert not driver.ledger(owner).prefix(tokens, "GPU")
    driver.probe_owner(tokens, owner)
    driver.measured(tokens, owner, True, baseline, f"T10-{owner}-warm-restart")
    write_json(driver.root / f"T10-{owner}-identities.json", identities)
    return {"status": "PASS", **identities}


def incarnation(driver: "Driver", owner: str, index: int) -> dict:
    """T10: restart MP separately and withdraw its prior CPU incarnation."""

    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    driver.warm(tokens, owner)
    snapshot = driver.snapshot(f"T10-{owner}-before-mp-restart")
    before = control(driver, owner, "status")["kv_events"]["incarnation"]
    needs_engine_restart = driver.state["engine"] == "sglang"
    driver.local_http(
        owner,
        9900,
        "/restart/mp",
        "POST",
        {
            "wait_for_unhealthy_heartbeats": (
                0 if needs_engine_restart else driver.state["tp"] + 1
            )
        },
        timeout=180,
    )
    after = control(driver, owner, "status")["kv_events"]["incarnation"]
    assert before != after
    registration = None
    if not needs_engine_restart:
        # Wait for the existing unhealthy-to-healthy callback to restore every
        # real GPU context, rather than treating a successful PING as readiness.
        expected = snapshot[owner]["status"]

        def transfers_registered() -> bool:
            nonlocal registration
            registration = json.loads(driver.local_http(owner, 8080, "/status"))
            return (
                set(registration["registered_gpu_ids"])
                == set(expected["registered_gpu_ids"])
                and registration["cache_context_meta"] == expected["cache_context_meta"]
                and registration["kv_events"]["subscribers"] == 1
            )

        driver.wait(
            transfers_registered, "vLLM transfer contexts re-registered", timeout=45
        )
        driver.active_barrier(owner)
    driver.wait(
        lambda: not driver.ledger(owner).prefix(tokens, "CPU"),
        "old-incarnation CPU withdrawal",
    )
    withdrawn = driver.snapshot(f"T10-{owner}-new-incarnation")
    identities = None
    if needs_engine_restart:
        # SGLang's heartbeat only pings. Its normal idle publisher can withdraw
        # CPU state, but a new engine must register transfer contexts before use.
        log = withdrawn[owner]["engine_log"][len(snapshot[owner]["engine_log"]) :]
        assert not re.search(
            r"MP_KV_TRACE (gpu_match|lookup|retrieve_submitted) ", log
        ), "SGLang performed inference before transfer re-registration"
        previous_worker = driver.workers[owner]
        driver.local_http(owner, 9900, "/restart/engine", "POST", timeout=720)
        identities = refresh_worker(driver, owner, previous_worker)
        ready = control(driver, owner, "status")["kv_events"]
        assert ready["incarnation"] == after and ready["subscribers"] == 1
    verify_recovered_load(driver, owner, index + 1, f"T10-{owner}-fresh-load")
    result = {
        "status": "PASS",
        "previous_incarnation": before,
        "incarnation": after,
        "engine_restart_required": needs_engine_restart,
        "engine_identities": identities,
        "restored_transfer_registration": registration,
    }
    write_json(driver.root / f"T10-{owner}-mp-restart-procedure.json", result)
    return result


def truncated_restart(driver: "Driver", owner: str, index: int) -> dict:
    """T10: a cold engine cannot invent warm placements absent from retained replay."""
    capacity = control(driver, owner, "status")["kv_events"]["log_capacity"]
    if capacity > 64:
        return {
            "status": "BLOCKED",
            "reason": "Run truncated warm replay with deploy --event-log-size 8",
        }
    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    driver.warm(tokens, owner)
    physical_keys = target_l1_keys(
        driver.records(owner, "mp-events.jsonl"),
        tokens,
        driver.state["chunk_size"],
        driver.model_name(owner),
        driver.state["tp"],
    )
    cutoff = control(driver, owner, "status")["kv_events"]["next_seq"]
    for number in range(capacity + 2):
        driver.request(
            fixture(index + number + 1, chunk_size=driver.state["chunk_size"]),
            owner,
        )
    driver.active_barrier(owner)
    before = control(driver, owner, "status")["kv_events"]
    assert before["next_seq"] - before["log_depth"] > cutoff
    assert_target_resident(driver.records(owner, "mp-events.jsonl"), physical_keys)
    assert driver.ledger(owner).prefix(tokens, "CPU"), "Warm fixture was evicted"
    previous = driver.workers[owner]
    driver.local_http(owner, 9900, "/restart/engine", "POST", timeout=720)
    identities = refresh_worker(driver, owner, previous)
    driver.active_barrier(owner)
    assert not driver.ledger(owner).prefix(tokens, "CPU"), (
        "Cold subscriber reconstructed a warm entry outside retained history"
    )
    after = control(driver, owner, "status")["kv_events"]
    assert before["incarnation"] == after["incarnation"]
    assert_target_resident(driver.records(owner, "mp-events.jsonl"), physical_keys)
    driver.snapshot(f"T10-{owner}-truncated-replay")
    return {
        "status": "PASS",
        "undiscovered_warm_tokens": tokens,
        "exact_warm_l1_keys": physical_keys,
        "replay_capacity": capacity,
        **identities,
    }


def idle(driver: "Driver", owner: str, index: int, *, paused: bool = False) -> dict:
    """T11: measure physical eviction to reception and publication with no inference."""

    tokens = fixture(index, chunk_size=driver.state["chunk_size"])
    driver.warm(tokens, owner)
    other = "b" if owner == "a" else "a"
    shorter = tokens[: driver.state["chunk_size"]] + [198]
    driver.warm(shorter, other)
    driver.evict_gpu(tokens, owner)
    driver.evict_gpu(shorter, other)
    driver.probe_owner(tokens, owner)
    target_hashes = {
        str(block["hash"]) for block in driver.ledger(owner).prefix(tokens, "CPU")
    }
    assert target_hashes
    request_count = sum(not request["query_only"] for request in driver.requests)
    physical_before = driver.records(owner, "mp-events.jsonl")
    physical_keys = target_l1_keys(
        physical_before,
        tokens,
        driver.state["chunk_size"],
        driver.model_name(owner),
        driver.state["tp"],
    )
    assert_target_resident(physical_before, physical_keys)
    before = len(physical_before)
    engine_log_offset = int(
        driver.exec(
            owner,
            "from pathlib import Path; "
            "print(Path('/artifacts/engine.log').stat().st_size)",
        )
    )
    pause_lifecycle = []
    discovery_before = driver.worker_identity(owner) if paused else None
    pause_log_start = len(driver.records(owner, "lifecycle.jsonl")) if paused else 0
    if paused:
        driver.local_http(owner, 9900, "/pause/engine", "POST")
    try:
        driver.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        removals: list[dict] = []

        def physical_eviction_recorded() -> bool:
            nonlocal removals
            records = driver.records(owner, "mp-events.jsonl")
            removals = target_evictions(records[before:], physical_keys)
            try:
                assert_target_absent(records, physical_keys)
            except AssertionError:
                return False
            return bool(removals)

        driver.wait(physical_eviction_recorded, "physical L1 eviction evidence")
        physical_ns = min(int(event["timestamp"] * 1e9) for event in removals)
        deadline_ns = physical_ns + 5_000_000_000
        publication_ns = None
        router_ns = None
        router_decisions = []
        stability_probe_times = []
        stable = None
        while time.time_ns() < deadline_ns:
            if publication_ns is None:
                ledger = driver.ledger(owner)
                if target_hashes.isdisjoint(ledger.blocks["CPU"]):
                    publication_ns = max(
                        record["received_ns"]
                        for record in ledger.records
                        if record["received_ns"] >= physical_ns
                        and any(
                            event["type"] == "BlockRemoved"
                            and event.get("medium") in ("CPU", "CPU_PINNED")
                            and target_hashes.intersection(
                                str(value) for value in event["block_hashes"]
                            )
                            for event in record["payload"][1]
                        )
                    )
            if publication_ns is not None:
                decision = driver.request(tokens, query=True)
                router_decisions.append(decision["selected"]["decode_worker_id"])
                if router_decisions[-1] == driver.workers[other]:
                    observed = decision["ended_ns"]
                    if observed <= deadline_ns:
                        router_ns = observed
                    break
            time.sleep(0.1)
        if router_ns is not None:
            # The convergence timestamp remains bounded by physical eviction.
            # Check stability separately, without repeatedly fetching the ledger.
            stability_probe_times.append(router_ns)
            for _ in range(19):
                decision = driver.request(tokens, query=True)
                router_decisions.append(decision["selected"]["decode_worker_id"])
                stability_probe_times.append(decision["ended_ns"])
            stable = all(
                worker == driver.workers[other] for worker in router_decisions[-20:]
            )
        status = control(driver, owner, "status")["kv_events"]
        log = driver.exec(
            owner,
            "from pathlib import Path\nimport sys\n"
            "with Path('/artifacts/engine.log').open('rb') as stream:\n"
            "    stream.seek(int(sys.argv[1]))\n"
            "    print(stream.read().decode())\n",
            str(engine_log_offset),
        )
        unexpected_requests = [
            line
            for line in log.splitlines()
            if re.search(r"MP_KV_TRACE (gpu_match|lookup|retrieve_submitted) ", line)
        ]
        arrivals = [
            int(timestamp)
            for timestamp, epoch, cursor in re.findall(
                r"MP_KV_SUBSCRIBER received_ns=(\d+) incarnation=(\d+) cursor=(\d+)",
                log,
            )
            if int(epoch) == status["incarnation"]
            and int(cursor) >= status["next_seq"] - 1
            and int(timestamp) >= physical_ns
        ]
        assert (
            sum(not request["query_only"] for request in driver.requests)
            == request_count
        ), "Idle window included inference"
    finally:
        if paused:
            driver.local_http(owner, 9900, "/resume/engine", "POST")
    if paused:
        discovery_after = driver.worker_identity(owner)
        assert discovery_after == discovery_before, (
            "Pause changed Dynamo discovery identity"
        )
        pause_lifecycle = [
            row
            for row in driver.records(owner, "lifecycle.jsonl")[pause_log_start:]
            if row["action"] in ("pause", "resume")
        ]
        assert [row["action"] for row in pause_lifecycle] == ["pause", "resume"]
        assert all(row["children"] for row in pause_lifecycle)
        assert all(
            child["pid"] != row["discovery_parent_pid"]
            for row in pause_lifecycle
            for child in row["children"]
        )
    label = f"T11-{owner}-{'paused' if paused else 'idle'}"
    result = {
        "status": (
            "FAIL"
            if unexpected_requests or stable is False
            else "PASS"
            if router_ns is not None
            else "LIMITATION"
        ),
        "paused": paused,
        "pause_scope": (
            "engine scheduler/model descendants; "
            "Dynamo parent and discovery remain running"
            if paused
            else None
        ),
        "pause_lifecycle": pause_lifecycle,
        "discovery_identity": discovery_before,
        "physical_eviction_ns": physical_ns,
        "physical_fixture_keys": physical_keys,
        "physical_fixture_removals": removals,
        "subscriber_received_ns": min(arrivals) if arrivals else None,
        "publication_observed_ns": publication_ns,
        "router_removal_observed_ns": router_ns,
        "deadline_ns": deadline_ns,
        "query_only_router_decisions": router_decisions,
        "router_stability_probe_times_ns": stability_probe_times,
        "router_stability_ok": stable,
        "inference_free": not unexpected_requests,
        "unexpected_inference_traces": unexpected_requests,
        "observation_seconds": 5,
    }
    write_json(driver.root / (label + ".json"), result)
    assert not unexpected_requests, "Idle window included engine inference"
    assert stable is not False, "Router choice changed after initial convergence"
    if not paused:
        assert arrivals, "Idle subscriber did not receive actual eviction"
        if driver.state["engine"] == "sglang":
            assert router_ns is not None, "SGLang normal-idle router convergence failed"
    driver.active_barrier(owner)
    driver.wait(
        lambda: not driver.ledger(owner).prefix(tokens, "CPU"),
        "active publication after separate idle measurement",
    )
    driver.probe_owner(
        tokens, other, opportunity_ns=driver.last_publication_opportunity_ns[owner]
    )
    return result
