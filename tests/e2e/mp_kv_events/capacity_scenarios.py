# SPDX-License-Identifier: Apache-2.0
"""T04 real L1 capacity eviction, distinct from the explicit clear scenario."""

# Standard
from typing import TYPE_CHECKING
import logging
import time

# Third Party
from fixtures import fixture
from residency import (
    assert_target_resident,
    identity,
    l1_owners,
    target_evictions,
    target_l1_keys,
)
from run import write_json

if TYPE_CHECKING:
    # Third Party
    from scenarios import Driver


def capacity_removal(driver: "Driver", owner: str, index: int) -> dict:
    """Evict A's exact physical CPU keys through stores, then route/load from B."""
    other = "b" if owner == "a" else "a"
    chunk_size = driver.state["chunk_size"]
    tokens = fixture(index, 2, chunk_size)
    baseline = driver.warm(tokens, owner)
    driver.warm(tokens, other)
    driver.evict_gpu(tokens, owner)
    driver.evict_gpu(tokens, other)
    before = driver.snapshot(f"T04-capacity-{owner}-before")
    keys = {
        replica: target_l1_keys(
            before[replica]["mp"],
            tokens,
            chunk_size,
            driver.model_name(replica),
            driver.state["tp"],
        )
        for replica in (owner, other)
    }
    for replica in (owner, other):
        assert_target_resident(before[replica]["mp"], keys[replica])
        assert not driver.ledger(replica).prefix(tokens, "GPU")
    other_cpu = driver.ledger(other).prefix(tokens, "CPU")
    target_native = {
        str(block["hash"]) for block in driver.ledger(owner).prefix(tokens, "CPU")
    }
    assert target_native and other_cpu
    pressure = []
    for number in range(256):
        current = driver.records(owner, "mp-events.jsonl")
        owners = l1_owners(current)
        if not any(identity(key) in owners for key in keys[owner]):
            break
        # These are ordinary inference STOREs, never an eviction hook or clear.
        pressure.append(
            driver.request(
                fixture(index + 1000 + number, max(1, 1024 // chunk_size), chunk_size),
                owner,
            )
        )
    else:
        raise AssertionError("256 real pressure requests did not evict fixture L1 keys")
    physical = driver.records(owner, "mp-events.jsonl")
    removed = target_evictions(physical[len(before[owner]["mp"]) :], keys[owner])
    assert removed, "No real L1 removal for the pressured fixture"
    owners = l1_owners(physical)
    assert not any(identity(key) in owners for key in keys[owner])
    assert_target_resident(driver.records(other, "mp-events.jsonl"), keys[other])
    assert driver.ledger(other).prefix(tokens, "CPU") == other_cpu
    # Explicit publication barrier is setup traffic; the measured request below
    # is untargeted. Verify the barrier did not rebuild any fixture GPU/CPU KV.
    previous_requests = len(driver.requests)
    driver.wait_received(owner)
    publication = driver.active_barrier(owner)
    barrier = driver.requests[previous_requests:]
    driver.wait(
        lambda: not (target_native & set(driver.ledger(owner).blocks["CPU"])),
        "all native CPU fixture IDs removed after capacity eviction",
    )
    driver.probe_owner(tokens, other, opportunity_ns=publication["ended_ns"])
    after_barrier = driver.snapshot(f"T04-capacity-{owner}-after-barrier")
    source_log = after_barrier[owner]["engine_log"][len(before[owner]["engine_log"]) :]
    assert "MP_KV_SUBSCRIBER reset=" not in source_log, (
        "Subscription loss cannot substitute for capacity eviction"
    )
    native_changes = after_barrier[owner]["native"][len(before[owner]["native"]) :]
    assert not any(
        event["type"] == "AllBlocksCleared"
        for record in native_changes
        for event in record["payload"][1]
    ), "A global reset cannot substitute for CPU capacity eviction"
    native_removed = [
        str(block_hash)
        for record in native_changes
        for event in record["payload"][1]
        if event["type"] == "BlockRemoved"
        and event.get("medium") in ("CPU", "CPU_PINNED")
        for block_hash in event["block_hashes"]
        if str(block_hash) in target_native
    ]
    assert (
        len(native_removed) == len(target_native)
        and set(native_removed) == target_native
    ), ("Capacity eviction must publish the exact fixture IDs once", native_removed)
    assert not any(
        identity(key) in l1_owners(after_barrier[owner]["mp"]) for key in keys[owner]
    )
    assert_target_resident(after_barrier[other]["mp"], keys[other])
    for replica in (owner, other):
        assert not driver.ledger(replica).prefix(tokens, "GPU"), (
            "Setup rebuilt fixture GPU KV"
        )
    measured = driver.measured(
        tokens, other, True, baseline, f"T04-capacity-{owner}-selected-{other}"
    )
    evidence = {
        "status": "PASS",
        "evicted_owner": owner,
        "selected_owner": other,
        "exact_fixture_keys": keys,
        "physical_removals": removed,
        "native_removed": native_removed,
        "pressure_requests": pressure,
        "active_barrier": barrier,
        "measured": measured,
    }
    write_json(driver.root / f"T04-capacity-{owner}-evidence.json", evidence)
    return evidence


def run_capacity(driver: "Driver") -> list[dict]:
    """Run genuine capacity-pressure T04 with A/B ownership reversed."""
    results = []
    for role, owner in enumerate(("a", "b")):
        started = time.monotonic()
        row: dict[str, object] = {"scenario": "T04-capacity", "owner": owner}
        try:
            capacity_removal(driver, owner, 40000 + 2000 * role)
            row["status"] = "PASS"
        except Exception as error:
            logging.getLogger(__name__).exception("Capacity eviction scenario failed")
            row.update(status="FAIL", reason=str(error))
        row["duration_seconds"] = time.monotonic() - started
        results.append(row)
    return results
