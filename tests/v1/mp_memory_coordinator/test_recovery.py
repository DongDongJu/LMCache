# SPDX-License-Identifier: Apache-2.0
"""Tests for the pure recovery/saga decision function.

Every saved state is exercised with evidence variants: the happy path,
already-completed effects, missing paths, ambiguous outside results,
lost identities, and gates (leadership, coordinator outage).
"""

# Standard
from pathlib import Path
import dataclasses
import json

# Third Party
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    GIB,
    DaxHotplugStatus,
    DaxRemoveMode,
    EffectName,
    EffectRecord,
    InstanceIdentity,
    MoveOutcome,
    MoveRecord,
    MoveState,
    RollbackStep,
)
from lmcache.v1.mp_memory_coordinator.recovery import (
    Block,
    DoEffect,
    Evidence,
    Finish,
    Hold,
    Participant,
    Persist,
    decide,
    validate_returned_path,
)

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "e2e"
    / "mp_memory_coordinator"
    / "fixtures"
    / "golden"
)
DONOR_IP = "192.0.2.40"
RECEIVER_IP = "192.0.2.41"
BOOT_D = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
OLD = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
BOOT_R = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.0"
NEW = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"
CONFIG = MPMemoryCoordinatorConfig(drain_timeout_seconds=100.0, dax_add_max_attempts=2)

DONOR = InstanceIdentity(
    instance_id="mp-donor",
    registration_time=1.0,
    endpoint="10.0.0.11:8080",
    worker_ip=DONOR_IP,
)
RECEIVER = InstanceIdentity(
    instance_id="mp-receiver",
    registration_time=1.0,
    endpoint="10.0.0.12:8080",
    worker_ip=RECEIVER_IP,
)


def _device_template() -> dict:
    body = json.loads((GOLDEN / "mp_reconfigure_dax_status.json").read_text())
    return body["adapters"][0]["status"]["devices"][0]


def dax(devices: list[tuple[str, str, int]], **counters: int) -> DaxHotplugStatus:
    """Build a hotplug status from ``(path, state, used_gib)`` entries."""
    entries = []
    for index, (path, state, used) in enumerate(devices):
        device = dict(_device_template())
        device.update(
            {
                "index": index,
                "device_id": index,
                "device_path": path,
                "state": state,
                "is_healthy": state not in ("closed", "removed"),
                "closing": state in ("closed", "removed"),
                "max_dax_size_bytes": 64 * GIB,
                "slot_bytes": 1 << 20,
                "max_slots": 64 * 1024,
                "live_slot_count": used * 1024,
            }
        )
        device.update(counters)
        entries.append(device)
    live = [d for d in entries if d["state"] not in ("closed", "removed")]
    return DaxHotplugStatus.model_validate(
        {
            "hotplug_enabled": True,
            "slot_bytes": 1 << 20,
            "total_capacity_bytes": sum(d["max_dax_size_bytes"] for d in live),
            "total_used_bytes": sum(
                d["live_slot_count"] * d["slot_bytes"] for d in live
            ),
            "devices": entries,
        }
    )


def record(state: MoveState, **overrides: object) -> MoveRecord:
    fields: dict[str, object] = dict(
        move_id="move-1",
        state=state,
        donor=DONOR,
        receiver=RECEIVER,
        donor_capacity_bytes=128 * GIB,
        receiver_capacity_bytes=64 * GIB,
        old_path=OLD,
        old_device_index=1,
        old_map_size_bytes=64 * GIB,
        old_slot_capacity_bytes=64 * GIB,
        allocation_size_gib=64,
        deallocation_request_id="move-1-deallocate",
        allocation_request_id="move-1-allocate",
        release_request_id="move-1-release",
        restore_request_id="move-1-restore",
        created_at=0.0,
        updated_at=0.0,
    )
    fields.update(overrides)
    return MoveRecord.model_validate(fields)


def effect(name: EffectName, **overrides: object) -> EffectRecord:
    fields: dict[str, object] = dict(name=name, intent_at=0.0)
    fields.update(overrides)
    return EffectRecord.model_validate(fields)


def evidence(**overrides: object) -> Evidence:
    base = Evidence(
        now=10.0,
        leader=True,
        coordinator_reachable=True,
        donor_identity_ok=True,
        receiver_identity_ok=True,
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "active", 4)]),
        receiver_dax=dax([(BOOT_R, "active", 56)]),
        outside={DONOR_IP: [OLD], RECEIVER_IP: []},
        donor_capacity_bytes=128 * GIB,
        receiver_capacity_bytes=64 * GIB,
    )
    return dataclasses.replace(base, **overrides)  # type: ignore[arg-type]


# -- SELECTED ---------------------------------------------------------------------


def test_selected_starts_with_donor_drain() -> None:
    decision = decide(record(MoveState.SELECTED), evidence(), CONFIG)
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.DONOR_DRAIN
    assert decision.remove_mode is DaxRemoveMode.DRAIN
    assert decision.participant is Participant.DONOR
    assert decision.device_path == OLD
    assert decision.intent_state is MoveState.DONOR_DRAINING
    assert not decision.is_outside


@pytest.mark.parametrize(
    "gate",
    [
        {"leader": False},
        {"coordinator_reachable": False},
        {"donor_identity_ok": False},
    ],
)
def test_selected_holds_when_gate_fails(gate: dict) -> None:
    decision = decide(record(MoveState.SELECTED), evidence(**gate), CONFIG)
    assert isinstance(decision, Hold)


def test_selected_aborts_when_receiver_vanished_or_path_not_owned() -> None:
    decision = decide(
        record(MoveState.SELECTED), evidence(receiver_identity_ok=False), CONFIG
    )
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK

    decision = decide(
        record(MoveState.SELECTED),
        evidence(outside={DONOR_IP: [], RECEIVER_IP: [OLD]}),
        CONFIG,
    )
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK

    decision = decide(
        record(MoveState.SELECTED),
        evidence(donor_dax=dax([(BOOT_D, "active", 4)])),
        CONFIG,
    )
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK


def test_selected_holds_without_outside_or_dax_status() -> None:
    assert isinstance(
        decide(record(MoveState.SELECTED), evidence(outside=None), CONFIG), Hold
    )
    assert isinstance(
        decide(record(MoveState.SELECTED), evidence(donor_dax=None), CONFIG), Hold
    )


# -- DONOR_DRAINING ---------------------------------------------------------------


def test_draining_waits_for_busy_references_then_evicts() -> None:
    draining = record(MoveState.DONOR_DRAINING, drain_started_at=5.0)
    busy = evidence(
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "draining", 4)], locked_key_count=2)
    )
    assert isinstance(decide(draining, busy, CONFIG), Hold)

    idle = evidence(donor_dax=dax([(BOOT_D, "active", 4), (OLD, "draining", 4)]))
    decision = decide(draining, idle, CONFIG)
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.DONOR_EVICT
    assert decision.remove_mode is DaxRemoveMode.EVICT


def test_draining_blocks_on_drain_timeout_without_outside_call() -> None:
    draining = record(MoveState.DONOR_DRAINING, drain_started_at=5.0)
    busy = evidence(
        now=5.0 + CONFIG.drain_timeout_seconds + 1,
        donor_dax=dax(
            [(BOOT_D, "active", 4), (OLD, "draining", 4)], borrowed_slot_count=1
        ),
    )
    decision = decide(draining, busy, CONFIG)
    assert isinstance(decision, Block)
    assert "undrain" in decision.reason


def test_draining_re_drains_if_device_is_still_active() -> None:
    decision = decide(record(MoveState.DONOR_DRAINING), evidence(), CONFIG)
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.DONOR_DRAIN


def test_draining_confirms_removal_from_tombstone_or_absence() -> None:
    draining = record(MoveState.DONOR_DRAINING)
    draining.effects[EffectName.DONOR_EVICT.value] = effect(
        EffectName.DONOR_EVICT, dispatched=True, attempts=1
    )
    tombstone = evidence(donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0)]))
    decision = decide(draining, tombstone, CONFIG)
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.DONOR_REMOVED
    assert decision.confirm_effect is EffectName.DONOR_EVICT

    absent = evidence(donor_dax=dax([(BOOT_D, "active", 4)]))
    decision = decide(draining, absent, CONFIG)
    assert isinstance(decision, Persist) and decision.state is MoveState.DONOR_REMOVED


def test_draining_rolls_back_when_receiver_vanishes() -> None:
    decision = decide(
        record(MoveState.DONOR_DRAINING), evidence(receiver_identity_ok=False), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ROLLING_BACK
    assert decision.rollback_step is RollbackStep.DONOR_EVICT


# -- DONOR_REMOVED / DEALLOCATING -------------------------------------------------


def _removed_evidence(**overrides) -> Evidence:
    fields = dict(donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0)]))
    fields.update(overrides)
    return evidence(**fields)


def test_donor_removed_issues_exact_deallocation_once() -> None:
    decision = decide(record(MoveState.DONOR_REMOVED), _removed_evidence(), CONFIG)
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.DEALLOCATE
    assert decision.is_outside
    assert decision.intent_state is MoveState.DEALLOCATING
    assert decision.deallocation is not None
    assert decision.deallocation.model_dump() == {
        "request_id": "move-1-deallocate",
        "target_node": DONOR_IP,
        "device_path": OLD,
    }
    assert decision.before_paths == [OLD]


def test_donor_removed_blocks_when_path_not_owned_and_holds_when_readable() -> None:
    decision = decide(
        record(MoveState.DONOR_REMOVED),
        _removed_evidence(outside={DONOR_IP: [], RECEIVER_IP: []}),
        CONFIG,
    )
    assert isinstance(decision, Block)
    decision = decide(record(MoveState.DONOR_REMOVED), evidence(), CONFIG)
    assert isinstance(decision, Hold)


def test_donor_removed_gates_on_leadership_identity_and_coordinator() -> None:
    for gate in (
        {"leader": False},
        {"coordinator_reachable": False},
        {"donor_identity_ok": False},
    ):
        decision = decide(
            record(MoveState.DONOR_REMOVED), _removed_evidence(**gate), CONFIG
        )
        assert isinstance(decision, Hold), gate
    decision = decide(
        record(MoveState.DONOR_REMOVED),
        _removed_evidence(receiver_identity_ok=False),
        CONFIG,
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.DONOR_READD


def test_deallocating_confirms_on_absence_and_blocks_when_still_listed() -> None:
    rec = record(MoveState.DEALLOCATING)
    rec.effects[EffectName.DEALLOCATE.value] = effect(
        EffectName.DEALLOCATE,
        request_id="move-1-deallocate",
        before_paths=[OLD],
        dispatched=True,
        attempts=1,
        response={"status": "DONE", "released_size_gib": 64},
    )
    decision = decide(
        rec, _removed_evidence(outside={DONOR_IP: [], RECEIVER_IP: []}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.DEALLOCATED
    assert decision.fields == {"released_size_gib": 64}
    assert decision.confirm_effect is EffectName.DEALLOCATE

    decision = decide(rec, _removed_evidence(), CONFIG)
    assert isinstance(decision, Block)


def test_deallocating_dispatched_without_outcome_blocks() -> None:
    """Crash after dispatch: never re-issue the POST."""
    rec = record(MoveState.DEALLOCATING)
    rec.effects[EffectName.DEALLOCATE.value] = effect(
        EffectName.DEALLOCATE, dispatched=True, attempts=1
    )
    cases: list[dict[str, list[str]]] = [
        {DONOR_IP: [OLD], RECEIVER_IP: []},
        {DONOR_IP: [], RECEIVER_IP: []},
    ]
    for outside in cases:
        decision = decide(rec, _removed_evidence(outside=outside), CONFIG)
        assert isinstance(decision, Block), outside
        assert "no retry" in decision.reason


def test_deallocating_intent_not_dispatched_may_still_issue_the_single_post() -> None:
    rec = record(MoveState.DEALLOCATING)
    rec.effects[EffectName.DEALLOCATE.value] = effect(EffectName.DEALLOCATE)
    decision = decide(rec, _removed_evidence(), CONFIG)
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.DEALLOCATE


def test_deallocating_explicit_failure_restores_donor_when_path_still_owned() -> None:
    rec = record(MoveState.DEALLOCATING)
    rec.effects[EffectName.DEALLOCATE.value] = effect(
        EffectName.DEALLOCATE, dispatched=True, attempts=1, error="explicit failure 409"
    )
    decision = decide(rec, _removed_evidence(), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ROLLING_BACK
    assert decision.rollback_step is RollbackStep.DONOR_READD

    decision = decide(
        rec, _removed_evidence(outside={DONOR_IP: [], RECEIVER_IP: []}), CONFIG
    )
    assert isinstance(decision, Block)


# -- DEALLOCATED / ALLOCATING -----------------------------------------------------


def _deallocated_evidence(**overrides) -> Evidence:
    fields = dict(
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0)]),
        outside={DONOR_IP: [], RECEIVER_IP: []},
    )
    fields.update(overrides)
    return evidence(**fields)


def test_deallocated_issues_exact_allocation_with_released_size() -> None:
    rec = record(MoveState.DEALLOCATED, released_size_gib=64)
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.ALLOCATE
    assert decision.intent_state is MoveState.ALLOCATING
    assert decision.allocation is not None
    assert decision.allocation.model_dump() == {
        "request_id": "move-1-allocate",
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    assert decision.before_paths == []


def test_deallocated_blocks_without_released_size_and_rolls_back_without_receiver():
    assert isinstance(
        decide(record(MoveState.DEALLOCATED), _deallocated_evidence(), CONFIG), Block
    )
    decision = decide(
        record(MoveState.DEALLOCATED, released_size_gib=64),
        _deallocated_evidence(receiver_identity_ok=False),
        CONFIG,
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ALLOCATE


def _allocating(**effect_fields) -> MoveRecord:
    rec = record(MoveState.ALLOCATING, released_size_gib=64)
    fields = dict(
        request_id="move-1-allocate", before_paths=[], dispatched=True, attempts=1
    )
    fields.update(effect_fields)
    rec.effects[EffectName.ALLOCATE.value] = effect(EffectName.ALLOCATE, **fields)
    return rec


def test_allocating_validates_returned_path_against_before_after_sets() -> None:
    rec = _allocating(
        response={"status": "DONE", "device_path": NEW, "granted_size_gib": 64}
    )
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ALLOCATED
    assert decision.fields == {"new_path": NEW, "granted_size_gib": 64}


@pytest.mark.parametrize(
    ("returned", "outside"),
    [
        ("../etc/dax", {DONOR_IP: [], RECEIVER_IP: [NEW]}),
        ("dax0.1", {DONOR_IP: [], RECEIVER_IP: [NEW]}),
        ("/dev/other/dax0.1", {DONOR_IP: [], RECEIVER_IP: [NEW]}),
        (OLD, {DONOR_IP: [], RECEIVER_IP: [NEW]}),  # wrong echo: donor path
        (NEW, {DONOR_IP: [NEW], RECEIVER_IP: [NEW]}),  # listed under two nodes
    ],
)
def test_allocating_invalid_path_releases_the_proven_allocation(
    returned: str, outside: dict
) -> None:
    rec = _allocating(
        response={"status": "DONE", "device_path": returned, "granted_size_gib": 64}
    )
    decision = decide(rec, _deallocated_evidence(outside=outside), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert decision.fields == {"new_path": NEW}


def test_allocating_invalid_path_without_provable_effect_blocks() -> None:
    rec = _allocating(
        response={
            "status": "DONE",
            "device_path": "/dev/other/x",
            "granted_size_gib": 64,
        }
    )
    # Nothing new appeared: cannot prove which allocation (if any) is ours.
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Block)
    # Two new paths appeared: not uniquely ours.
    decision = decide(
        rec,
        _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW, BOOT_R]}),
        CONFIG,
    )
    assert isinstance(decision, Block)


def test_allocating_explicit_failure_restores_donor() -> None:
    rec = _allocating(error="explicit failure 409")
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ALLOCATE


def test_allocating_contract_violation_with_visible_effect_releases_it() -> None:
    """Wrong requested/granted size: the path was assigned; release it."""
    rec = _allocating(error="contract violation: sizes disagree")
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert decision.fields == {"new_path": NEW}


def test_allocating_dispatched_without_outcome_blocks() -> None:
    rec = _allocating()
    cases: list[dict[str, list[str]]] = [
        {DONOR_IP: [], RECEIVER_IP: []},
        {DONOR_IP: [], RECEIVER_IP: [NEW]},
    ]
    for outside in cases:
        decision = decide(rec, _deallocated_evidence(outside=outside), CONFIG)
        assert isinstance(decision, Block), outside


# -- ALLOCATED ----------------------------------------------------------------------


def _allocated(**overrides) -> MoveRecord:
    fields = dict(released_size_gib=64, new_path=NEW, granted_size_gib=64)
    fields.update(overrides)
    return record(MoveState.ALLOCATED, **fields)


def test_allocated_adds_on_receiver_with_saved_map_size() -> None:
    decision = decide(
        _allocated(),
        _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]}),
        CONFIG,
    )
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.RECEIVER_ADD
    assert decision.participant is Participant.RECEIVER
    assert decision.device_path == NEW
    assert decision.size_bytes == 64 * GIB


def test_allocated_confirms_active_then_waits_for_capacity_then_succeeds() -> None:
    rec = _allocated()
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=1
    )
    active = _deallocated_evidence(
        outside={DONOR_IP: [], RECEIVER_IP: [NEW]},
        receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)]),
    )
    decision = decide(rec, active, CONFIG)
    assert isinstance(decision, Persist)
    assert decision.confirm_effect is EffectName.RECEIVER_ADD
    assert decision.fields == {
        "new_device_index": 1,
        "new_slot_capacity_bytes": 64 * GIB,
    }

    rec.effects[EffectName.RECEIVER_ADD.value].confirmed = True
    stale = decide(rec, active, CONFIG)
    assert isinstance(stale, Hold) and "capacity" in stale.reason

    converged = _deallocated_evidence(
        outside={DONOR_IP: [], RECEIVER_IP: [NEW]},
        receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)]),
        donor_capacity_bytes=64 * GIB,
        receiver_capacity_bytes=128 * GIB,
    )
    decision = decide(rec, converged, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.SUCCEEDED


def test_allocated_persistent_add_failure_releases_receiver() -> None:
    rec = _allocated()
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=2, error="400"
    )
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER


def test_allocated_transient_add_failure_retries_within_bound() -> None:
    rec = _allocated()
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=1, error="400"
    )
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.RECEIVER_ADD


# -- ROLLING_BACK --------------------------------------------------------------------


def test_rollback_donor_evict_then_readd_then_rolled_back() -> None:
    rec = record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.DONOR_EVICT,
        drain_started_at=1.0,
    )
    idle = evidence(donor_dax=dax([(BOOT_D, "active", 4), (OLD, "draining", 4)]))
    decision = decide(rec, idle, CONFIG)
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.DONOR_EVICT
    assert decision.rollback_step is RollbackStep.DONOR_EVICT

    removed = evidence(donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0)]))
    decision = decide(rec, removed, CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.DONOR_READD

    rec.rollback_step = RollbackStep.DONOR_READD
    decision = decide(rec, removed, CONFIG)
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.DONOR_READD
    assert decision.device_path == OLD and decision.size_bytes == 64 * GIB

    re_added = evidence(
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0), (OLD, "active", 0)])
    )
    decision = decide(rec, re_added, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK

    not_owned = evidence(
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0)]),
        outside={DONOR_IP: [], RECEIVER_IP: []},
    )
    assert isinstance(decide(rec, not_owned, CONFIG), Block)


def test_rollback_release_receiver_then_restore_donor() -> None:
    rec = record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RELEASE_RECEIVER,
        released_size_gib=64,
        new_path=NEW,
    )
    listed = _deallocated_evidence(outside={DONOR_IP: [], RECEIVER_IP: [NEW]})
    decision = decide(rec, listed, CONFIG)
    assert (
        isinstance(decision, DoEffect)
        and decision.effect is EffectName.RELEASE_RECEIVER
    )
    assert decision.deallocation is not None
    assert decision.deallocation.model_dump() == {
        "request_id": "move-1-release",
        "target_node": RECEIVER_IP,
        "device_path": NEW,
    }

    # Attached on the receiver: never release underneath a live mapping.
    attached = _deallocated_evidence(
        outside={DONOR_IP: [], RECEIVER_IP: [NEW]},
        receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)]),
    )
    assert isinstance(decide(rec, attached, CONFIG), Block)

    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER,
        dispatched=True,
        attempts=1,
        response={"status": "DONE", "released_size_gib": 64},
    )
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ALLOCATE

    rec.rollback_step = RollbackStep.RESTORE_DONOR_ALLOCATE
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert (
        isinstance(decision, DoEffect)
        and decision.effect is EffectName.RESTORE_ALLOCATE
    )
    assert decision.allocation is not None
    assert decision.allocation.target_node == DONOR_IP
    assert decision.allocation.request_size_gib == 64

    rec.effects[EffectName.RESTORE_ALLOCATE.value] = effect(
        EffectName.RESTORE_ALLOCATE,
        dispatched=True,
        attempts=1,
        before_paths=[],
        response={"status": "DONE", "device_path": OLD, "granted_size_gib": 64},
    )
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: []}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ADD
    assert decision.fields == {"restored_path": OLD}

    rec.rollback_step = RollbackStep.RESTORE_DONOR_ADD
    rec.restored_path = OLD
    decision = decide(
        rec, _deallocated_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: []}), CONFIG
    )
    assert isinstance(decision, DoEffect) and decision.effect is EffectName.RESTORE_ADD

    restored = _deallocated_evidence(
        outside={DONOR_IP: [OLD], RECEIVER_IP: []},
        donor_dax=dax([(BOOT_D, "active", 4), (OLD, "removed", 0), (OLD, "active", 0)]),
    )
    decision = decide(rec, restored, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK


def test_rollback_restore_refused_finishes_rolled_back_without_leak() -> None:
    rec = record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RESTORE_DONOR_ALLOCATE,
        released_size_gib=64,
    )
    rec.effects[EffectName.RESTORE_ALLOCATE.value] = effect(
        EffectName.RESTORE_ALLOCATE,
        dispatched=True,
        attempts=1,
        error="explicit failure 409",
    )
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK


def test_rollback_ambiguous_outside_results_block() -> None:
    rec = record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RELEASE_RECEIVER,
        new_path=NEW,
    )
    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER, dispatched=True, attempts=1
    )
    assert isinstance(decide(rec, _deallocated_evidence(), CONFIG), Block)

    rec = record(
        MoveState.ROLLING_BACK, rollback_step=RollbackStep.RESTORE_DONOR_ALLOCATE
    )
    rec.effects[EffectName.RESTORE_ALLOCATE.value] = effect(
        EffectName.RESTORE_ALLOCATE, dispatched=True, attempts=1
    )
    assert isinstance(decide(rec, _deallocated_evidence(), CONFIG), Block)


# -- terminal states ---------------------------------------------------------------


def test_terminal_states_never_act() -> None:
    assert isinstance(decide(record(MoveState.BLOCKED), evidence(), CONFIG), Hold)
    assert isinstance(decide(record(MoveState.COMPLETE), evidence(), CONFIG), Hold)


# -- path validation helper --------------------------------------------------------


def test_validate_returned_path_rules() -> None:
    outside = {DONOR_IP: [], RECEIVER_IP: [NEW]}
    assert validate_returned_path(NEW, RECEIVER_IP, [], outside, CONFIG) == ""
    assert validate_returned_path("", RECEIVER_IP, [], outside, CONFIG)
    assert validate_returned_path("relative/dax", RECEIVER_IP, [], outside, CONFIG)
    assert validate_returned_path(
        "/dev/dax-cxl/../dax0.1", RECEIVER_IP, [], outside, CONFIG
    )
    assert validate_returned_path("/dev/other/dax0.1", RECEIVER_IP, [], outside, CONFIG)
    assert validate_returned_path(NEW, RECEIVER_IP, [NEW], outside, CONFIG)
    assert validate_returned_path(
        NEW, RECEIVER_IP, [], {DONOR_IP: [NEW], RECEIVER_IP: [NEW]}, CONFIG
    )
