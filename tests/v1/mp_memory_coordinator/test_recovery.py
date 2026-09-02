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
    NO_DONOR,
    DaxHotplugStatus,
    DaxRemoveMode,
    EffectFailure,
    EffectName,
    EffectRecord,
    InstanceIdentity,
    MoveKind,
    MoveOutcome,
    MoveRecord,
    MoveState,
    RollbackStep,
)
from lmcache.v1.mp_memory_coordinator.recovery import (
    GROW_MAX_RECEIVER_REBINDS,
    Block,
    Decision,
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


# -- GROW: allocate for the receiver, add on the receiver, no donor -------------

RECEIVER_2 = InstanceIdentity(
    instance_id="mp-receiver-2",
    registration_time=2.0,
    endpoint="10.0.0.12:8080",
    worker_ip=RECEIVER_IP,
)


def grow_record(state: MoveState, **overrides: object) -> MoveRecord:
    """A GROW record: donor-less, requesting 64 GiB for the receiver."""
    fields: dict[str, object] = dict(
        move_id="grow-1",
        kind=MoveKind.GROW,
        state=state,
        donor=NO_DONOR,
        receiver=RECEIVER,
        donor_capacity_bytes=0,
        receiver_capacity_bytes=64 * GIB,
        old_path="",
        old_device_index=-1,
        old_map_size_bytes=64 * GIB,
        old_slot_capacity_bytes=0,
        allocation_size_gib=64,
        deallocation_request_id="",
        allocation_request_id="grow-1-allocate",
        release_request_id="grow-1-release",
        restore_request_id="",
        created_at=0.0,
        updated_at=0.0,
    )
    fields.update(overrides)
    return MoveRecord.model_validate(fields)


def _grow_evidence(**overrides: object) -> Evidence:
    """Evidence with every donor input absent: a GROW must not need any."""
    fields: dict[str, object] = dict(
        donor_identity_ok=True,
        donor_dax=None,
        donor_capacity_bytes=None,
        receiver_worker_registered=True,
    )
    fields.update(overrides)
    return evidence(**fields)


def _grow_allocating(**effect_fields: object) -> MoveRecord:
    rec = grow_record(MoveState.ALLOCATING)
    fields: dict[str, object] = dict(
        request_id="grow-1-allocate", before_paths=[], dispatched=True, attempts=1
    )
    fields.update(effect_fields)
    rec.effects[EffectName.ALLOCATE.value] = effect(EffectName.ALLOCATE, **fields)
    return rec


def _grow_allocated(**overrides: object) -> MoveRecord:
    fields: dict[str, object] = dict(new_path=NEW, granted_size_gib=64)
    fields.update(overrides)
    return grow_record(MoveState.ALLOCATED, **fields)


def _expect_grow_allocate(decision: Decision) -> DoEffect:
    assert isinstance(decision, DoEffect), decision
    assert decision.effect is EffectName.ALLOCATE
    assert decision.intent_state is MoveState.ALLOCATING
    assert decision.participant is Participant.RECEIVER
    assert decision.is_outside
    assert decision.allocation is not None
    assert decision.allocation.model_dump() == {
        "request_id": "grow-1-allocate",
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    return decision


def test_grow_selected_issues_exact_allocation_once() -> None:
    decision = _expect_grow_allocate(
        decide(grow_record(MoveState.SELECTED), _grow_evidence(), CONFIG)
    )
    assert decision.before_paths == []
    # The receiver's current outside paths are the ledger's before-set.
    decision = _expect_grow_allocate(
        decide(
            grow_record(MoveState.SELECTED),
            _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [BOOT_R]}),
            CONFIG,
        )
    )
    assert decision.before_paths == [BOOT_R]


def test_grow_selected_ignores_donor_evidence_but_gates_on_the_rest() -> None:
    # Donor inputs never matter for a GROW.
    _expect_grow_allocate(
        decide(
            grow_record(MoveState.SELECTED),
            _grow_evidence(donor_identity_ok=False),
            CONFIG,
        )
    )
    for gate in ({"leader": False}, {"coordinator_reachable": False}):
        decision = decide(
            grow_record(MoveState.SELECTED), _grow_evidence(**gate), CONFIG
        )
        assert isinstance(decision, Hold), gate
    # Receiver gone while the coordinator answers: abort with zero effects.
    decision = decide(
        grow_record(MoveState.SELECTED),
        _grow_evidence(receiver_identity_ok=False),
        CONFIG,
    )
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK
    assert "before any effect" in decision.note


def test_grow_selected_holds_without_outside_or_receiver_dax() -> None:
    assert isinstance(
        decide(grow_record(MoveState.SELECTED), _grow_evidence(outside=None), CONFIG),
        Hold,
    )
    assert isinstance(
        decide(
            grow_record(MoveState.SELECTED), _grow_evidence(receiver_dax=None), CONFIG
        ),
        Hold,
    )


def test_grow_selected_blocks_on_an_inconsistent_size_or_a_stray_ledger() -> None:
    bad = grow_record(MoveState.SELECTED, old_map_size_bytes=64 * GIB + 1)
    assert isinstance(decide(bad, _grow_evidence(), CONFIG), Block)
    zero = grow_record(MoveState.SELECTED, allocation_size_gib=0, old_map_size_bytes=0)
    assert isinstance(decide(zero, _grow_evidence(), CONFIG), Block)
    stray = grow_record(MoveState.SELECTED)
    stray.effects[EffectName.ALLOCATE.value] = effect(EffectName.ALLOCATE)
    decision = decide(stray, _grow_evidence(), CONFIG)
    assert isinstance(decision, Block) and "SELECTED" in decision.reason


def test_grow_allocating_unsent_intent_reissues_the_single_post() -> None:
    """Intent persisted, POST provably never sent: the same single POST."""
    rec = _grow_allocating(dispatched=False, attempts=0)
    decision = _expect_grow_allocate(decide(rec, _grow_evidence(), CONFIG))
    assert decision.allocation is not None
    assert decision.allocation.request_id == "grow-1-allocate"
    # Receiver lost in that state: nothing was sent, so nothing to undo.
    lost = decide(rec, _grow_evidence(receiver_identity_ok=False), CONFIG)
    assert isinstance(lost, Finish) and lost.outcome is MoveOutcome.ROLLED_BACK
    assert isinstance(decide(rec, _grow_evidence(outside=None), CONFIG), Hold)
    # A gate failure holds exactly like SELECTED.
    assert isinstance(decide(rec, _grow_evidence(leader=False), CONFIG), Hold)


def test_grow_allocating_validates_returned_path_and_persists_allocated() -> None:
    rec = _grow_allocating(
        response={"status": "DONE", "device_path": NEW, "granted_size_gib": 64}
    )
    decision = decide(
        rec, _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ALLOCATED
    assert decision.confirm_effect is EffectName.ALLOCATE
    assert decision.fields == {"new_path": NEW, "granted_size_gib": 64}


def test_grow_allocating_explicit_refusal_finishes_not_served() -> None:
    rec = _grow_allocating(error="explicit failure 409", failure=EffectFailure.EXPLICIT)
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Finish)
    assert decision.outcome is MoveOutcome.NOT_SERVED
    assert "64 GiB" in decision.note and RECEIVER_IP in decision.note
    # Any explicit status is treated the same (5xx/429 included): the path
    # set diff proves nothing was assigned.
    rec = _grow_allocating(error="explicit failure 503", failure=EffectFailure.EXPLICIT)
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.NOT_SERVED
    # A receiver-side path that vanished is still "no new path".
    rec = _grow_allocating(
        error="explicit failure 409",
        failure=EffectFailure.EXPLICIT,
        before_paths=[BOOT_R],
    )
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.NOT_SERVED


def test_grow_allocating_contract_violation_without_effect_blocks() -> None:
    rec = _grow_allocating(
        error="contract violation: missing device_path", failure=EffectFailure.CONTRACT
    )
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Block)
    assert "contract" in decision.reason
    # An error whose class was never recorded (older ledger) is not proof
    # of a refusal either.
    rec = _grow_allocating(error="explicit failure 409")
    assert isinstance(decide(rec, _grow_evidence(), CONFIG), Block)


@pytest.mark.parametrize("failure", [EffectFailure.EXPLICIT, EffectFailure.CONTRACT])
def test_grow_allocating_failure_with_one_visible_path_releases_it(
    failure: EffectFailure,
) -> None:
    rec = _grow_allocating(error="something failed", failure=failure)
    decision = decide(
        rec, _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ROLLING_BACK
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert decision.fields == {"new_path": NEW}
    # Two new paths: not uniquely ours.
    decision = decide(
        rec,
        _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW, BOOT_R]}),
        CONFIG,
    )
    assert isinstance(decision, Block)


@pytest.mark.parametrize(
    "returned",
    ["../etc/dax", "dax0.1", "/dev/other/dax0.1", OLD],
)
def test_grow_allocating_invalid_path_releases_or_blocks(returned: str) -> None:
    rec = _grow_allocating(
        response={"status": "DONE", "device_path": returned, "granted_size_gib": 64}
    )
    decision = decide(
        rec, _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert decision.fields == {"new_path": NEW}
    assert isinstance(decide(rec, _grow_evidence(), CONFIG), Block)


def test_grow_allocating_dispatched_without_outcome_blocks() -> None:
    rec = _grow_allocating()
    cases: list[dict[str, list[str]]] = [
        {DONOR_IP: [OLD], RECEIVER_IP: []},
        {DONOR_IP: [OLD], RECEIVER_IP: [NEW]},
    ]
    for outside in cases:
        decision = decide(rec, _grow_evidence(outside=outside), CONFIG)
        assert isinstance(decision, Block), outside
        assert "no retry" in decision.reason
    # Even with the receiver gone: never re-sent, never held forever.
    decision = decide(rec, _grow_evidence(receiver_identity_ok=False), CONFIG)
    assert isinstance(decision, Block)
    assert isinstance(decide(rec, _grow_evidence(outside=None), CONFIG), Hold)
    assert isinstance(
        decide(grow_record(MoveState.ALLOCATING), _grow_evidence(), CONFIG), Block
    )


def test_grow_allocated_adds_on_receiver_with_request_map_size() -> None:
    decision = decide(
        _grow_allocated(),
        _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]}),
        CONFIG,
    )
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.RECEIVER_ADD
    assert decision.participant is Participant.RECEIVER
    assert decision.device_path == NEW
    assert decision.size_bytes == 64 * GIB
    assert decision.intent_state is MoveState.ALLOCATED


def test_grow_allocated_confirms_active_then_waits_for_receiver_capacity_only() -> None:
    rec = _grow_allocated()
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=1
    )
    active = _grow_evidence(
        outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]},
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
    converged = dataclasses.replace(active, receiver_capacity_bytes=128 * GIB)
    decision = decide(rec, converged, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.SUCCEEDED
    assert decision.warning == ""
    assert converged.donor_capacity_bytes is None  # never consulted
    gone = dataclasses.replace(
        converged, receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "removed", 0)])
    )
    assert isinstance(decide(rec, gone, CONFIG), Block)
    assert isinstance(
        decide(rec, dataclasses.replace(converged, receiver_dax=None), CONFIG), Hold
    )


def test_grow_allocated_bounded_convergence_wait_finishes_with_warning() -> None:
    """A4: active on the receiver and listed by the allocator, but the usage
    view never converges -> SUCCEEDED after the timeout, with a warning."""
    rec = _grow_allocated(updated_at=100.0)
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=1, confirmed=True
    )
    receiver_dax = dax([(BOOT_R, "active", 56), (NEW, "active", 0)])
    stale = _grow_evidence(
        now=100.0 + CONFIG.capacity_convergence_timeout_seconds,
        outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]},
        receiver_dax=receiver_dax,
    )
    assert isinstance(decide(rec, stale, CONFIG), Hold)
    late = dataclasses.replace(
        stale, now=100.0 + CONFIG.capacity_convergence_timeout_seconds + 1
    )
    decision = decide(rec, late, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.SUCCEEDED
    assert "did not converge" in decision.warning
    # A readable allocator that no longer lists the path under the receiver
    # contradicts the proven allocation: terminal, never a silent hold.
    unlisted = dataclasses.replace(late, outside={DONOR_IP: [OLD], RECEIVER_IP: []})
    decision = decide(rec, unlisted, CONFIG)
    assert isinstance(decision, Block) and "lists it under []" in decision.reason
    elsewhere = dataclasses.replace(
        late, outside={DONOR_IP: [OLD, NEW], RECEIVER_IP: [NEW]}
    )
    decision = decide(rec, elsewhere, CONFIG)
    assert isinstance(decision, Block) and DONOR_IP in decision.reason
    # Before the timeout a lagging usage view is still only a hold.
    early_unlisted = dataclasses.replace(stale, outside={DONOR_IP: [OLD]})
    assert isinstance(decide(rec, early_unlisted, CONFIG), Hold)
    # An unreadable allocator holds for a further drain_timeout_seconds past
    # the convergence timeout (the allocation cannot be re-verified), then
    # blocks: the wait is bounded either way.
    outage_bound = (
        100.0
        + CONFIG.capacity_convergence_timeout_seconds
        + CONFIG.drain_timeout_seconds
    )
    unreadable = dataclasses.replace(late, outside=None, now=outage_bound)
    decision = decide(rec, unreadable, CONFIG)
    assert isinstance(decision, Hold) and "allocator unreadable" in decision.reason
    decision = decide(
        rec, dataclasses.replace(unreadable, now=outage_bound + 1), CONFIG
    )
    assert isinstance(decision, Block) and "allocator unreadable" in decision.reason
    # Convergence still wins at any time, allocator or not.
    converged = dataclasses.replace(
        unreadable, now=outage_bound + 1, receiver_capacity_bytes=128 * GIB
    )
    decision = decide(rec, converged, CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.SUCCEEDED
    # A MOVE keeps waiting regardless of the timeout.
    move = _allocated(updated_at=100.0)
    move.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=1, confirmed=True
    )
    decision = decide(
        move,
        _deallocated_evidence(
            now=late.now,
            outside={DONOR_IP: [], RECEIVER_IP: [NEW]},
            receiver_dax=receiver_dax,
        ),
        CONFIG,
    )
    assert isinstance(decision, Hold)


def test_grow_allocated_persistent_add_failure_releases_receiver() -> None:
    rec = _grow_allocated()
    rec.effects[EffectName.RECEIVER_ADD.value] = effect(
        EffectName.RECEIVER_ADD, dispatched=True, attempts=2, error="400"
    )
    decision = decide(
        rec, _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]}), CONFIG
    )
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER


def _lost_evidence(**overrides: object) -> Evidence:
    fields: dict[str, object] = dict(
        receiver_identity_ok=False,
        receiver_worker_registered=False,
        receiver_dax=None,
        outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]},
    )
    fields.update(overrides)
    return _grow_evidence(**fields)


def test_grow_allocated_receiver_lost_rebinds_holds_then_releases_or_blocks() -> None:
    """A2 in ALLOCATED: rebind, bounded hold, then release or block."""
    rec = _grow_allocated(updated_at=50.0)
    # (a) exactly one accepted instance on the worker: rebind to it.
    decision = decide(
        rec,
        _lost_evidence(
            receiver_replacement=RECEIVER_2, receiver_worker_registered=True
        ),
        CONFIG,
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ALLOCATED and decision.receiver == RECEIVER_2
    # The rebind forgets the add confirmed on the old identity so the add
    # is re-driven from the new identity's status, and it is counted.
    assert decision.unconfirm_effect is EffectName.RECEIVER_ADD
    assert decision.fields == {"receiver_rebinds": 1}
    # (b) within the grace: hold.
    within = _lost_evidence(now=50.0 + CONFIG.drain_timeout_seconds)
    assert isinstance(decide(rec, within, CONFIG), Hold)
    # After the grace, status unreadable and no instance registered on the
    # worker -> provably unattached; the release waits for an instance.
    late = 50.0 + CONFIG.drain_timeout_seconds + 1
    decision = decide(rec, _lost_evidence(now=late), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert "once an instance is back" in decision.note
    # Status readable and the path absent -> provably unattached, whatever
    # raw membership says.
    for registered in (True, False):
        decision = decide(
            rec,
            _lost_evidence(
                now=late,
                receiver_worker_registered=registered,
                receiver_dax=dax([(BOOT_R, "active", 56)]),
            ),
            CONFIG,
        )
        assert isinstance(decision, Persist), registered
        assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    # Something registered on the worker but unreadable: attachment unknown.
    decision = decide(
        rec, _lost_evidence(now=late, receiver_worker_registered=True), CONFIG
    )
    assert isinstance(decision, Block) and "attachment unknown" in decision.reason
    # Readable and attached: never release underneath a mapping, and a
    # readable status beats raw membership (a sandwich-rejected receiver
    # whose path is live is attached, not vanished-and-unattached).
    for registered in (True, False):
        decision = decide(
            rec,
            _lost_evidence(
                now=late,
                receiver_worker_registered=registered,
                receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)]),
            ),
            CONFIG,
        )
        assert isinstance(decision, Block), registered
        assert "is attached" in decision.reason


def test_grow_release_receiver_lost_rebinds_or_blocks_after_grace() -> None:
    """A2 in ROLLING_BACK/RELEASE_RECEIVER."""
    rec = grow_record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RELEASE_RECEIVER,
        new_path=NEW,
        updated_at=50.0,
    )
    decision = decide(
        rec,
        _lost_evidence(
            receiver_replacement=RECEIVER_2, receiver_worker_registered=True
        ),
        CONFIG,
    )
    assert isinstance(decision, Persist)
    assert decision.state is MoveState.ROLLING_BACK
    assert decision.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert decision.receiver == RECEIVER_2
    assert decision.unconfirm_effect is EffectName.RECEIVER_ADD
    assert decision.fields == {"receiver_rebinds": 1}
    assert isinstance(
        decide(rec, _lost_evidence(now=50.0 + CONFIG.drain_timeout_seconds), CONFIG),
        Hold,
    )
    # With nobody back on the worker the release is never POSTed (its gate
    # needs a matching identity): the step blocks after the second grace.
    decision = decide(
        rec, _lost_evidence(now=50.0 + CONFIG.drain_timeout_seconds + 1), CONFIG
    )
    assert isinstance(decision, Block)
    assert "matching identity" in decision.reason


@pytest.mark.parametrize(
    ("state", "step"),
    [
        (MoveState.ALLOCATED, RollbackStep.NONE),
        (MoveState.ROLLING_BACK, RollbackStep.RELEASE_RECEIVER),
    ],
)
def test_grow_receiver_rebinds_are_capped_then_block(
    state: MoveState, step: RollbackStep
) -> None:
    """A2, bounded: a receiver that keeps re-registering is rebound at most
    ``GROW_MAX_RECEIVER_REBINDS`` times per saga; the next loss with a
    replacement blocks (even within the grace, which a rebind would reset),
    naming what is known about the path's attachment."""
    replaced = _lost_evidence(
        now=1.0, receiver_replacement=RECEIVER_2, receiver_worker_registered=True
    )
    rec = grow_record(
        state,
        rollback_step=step,
        new_path=NEW,
        updated_at=0.0,
        receiver_rebinds=GROW_MAX_RECEIVER_REBINDS - 1,
    )
    decision = decide(rec, replaced, CONFIG)
    assert isinstance(decision, Persist) and decision.receiver == RECEIVER_2
    assert decision.fields == {"receiver_rebinds": GROW_MAX_RECEIVER_REBINDS}
    capped = grow_record(
        state,
        rollback_step=step,
        new_path=NEW,
        updated_at=0.0,
        receiver_rebinds=GROW_MAX_RECEIVER_REBINDS,
    )
    decision = decide(capped, replaced, CONFIG)
    assert isinstance(decision, Block)
    assert f"{GROW_MAX_RECEIVER_REBINDS} times" in decision.reason
    assert "attachment unknown" in decision.reason
    attached = dataclasses.replace(
        replaced, receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)])
    )
    decision = decide(capped, attached, CONFIG)
    assert isinstance(decision, Block) and "is attached" in decision.reason
    absent = dataclasses.replace(replaced, receiver_dax=dax([(BOOT_R, "active", 56)]))
    decision = decide(capped, absent, CONFIG)
    assert isinstance(decision, Block) and "provably unattached" in decision.reason
    # The cap only concerns rebinds: with nobody to rebind to, the grace
    # still holds and its expiry still decides as before.
    assert isinstance(decide(capped, _lost_evidence(now=1.0), CONFIG), Hold)
    # An intact receiver is never affected by the count.
    intact = _grow_evidence(
        outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]},
        receiver_dax=dax([(BOOT_R, "active", 56)]),
    )
    assert not isinstance(decide(capped, intact, CONFIG), Block)


def test_grow_rollback_release_then_rolled_back_without_restore() -> None:
    rec = grow_record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RELEASE_RECEIVER,
        new_path=NEW,
        updated_at=50.0,
    )
    listed = _grow_evidence(outside={DONOR_IP: [OLD], RECEIVER_IP: [NEW]})
    decision = decide(rec, listed, CONFIG)
    assert isinstance(decision, DoEffect)
    assert decision.effect is EffectName.RELEASE_RECEIVER
    assert decision.deallocation is not None
    assert decision.deallocation.model_dump() == {
        "request_id": "grow-1-release",
        "target_node": RECEIVER_IP,
        "device_path": NEW,
    }
    # Never release while the receiver's status is unreadable; block after
    # the grace.
    unreadable = dataclasses.replace(listed, receiver_dax=None)
    assert isinstance(decide(rec, unreadable, CONFIG), Hold)
    late = dataclasses.replace(unreadable, now=50.0 + CONFIG.drain_timeout_seconds + 1)
    assert isinstance(decide(rec, late, CONFIG), Block)
    # Attached on the receiver: never release underneath a live mapping.
    attached = dataclasses.replace(
        listed, receiver_dax=dax([(BOOT_R, "active", 56), (NEW, "active", 0)])
    )
    assert isinstance(decide(rec, attached, CONFIG), Block)
    # Not solely under the receiver.
    shared = dataclasses.replace(
        listed, outside={DONOR_IP: [OLD, NEW], RECEIVER_IP: [NEW]}
    )
    assert isinstance(decide(rec, shared, CONFIG), Block)

    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER,
        dispatched=True,
        attempts=1,
        response={"status": "DONE", "released_size_gib": 64},
    )
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK
    # Even with the receiver's status unreadable: the outside proof suffices.
    decision = decide(rec, _grow_evidence(receiver_dax=None), CONFIG)
    assert isinstance(decision, Finish) and decision.outcome is MoveOutcome.ROLLED_BACK
    assert isinstance(decide(rec, listed, CONFIG), Block)  # still listed
    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER, dispatched=True, attempts=1, error="409"
    )
    assert isinstance(decide(rec, listed, CONFIG), Block)
    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER, dispatched=True, attempts=1
    )
    assert isinstance(decide(rec, listed, CONFIG), Block)


@pytest.mark.parametrize(
    ("state", "step"),
    [
        (MoveState.DONOR_DRAINING, RollbackStep.NONE),
        (MoveState.DONOR_REMOVED, RollbackStep.NONE),
        (MoveState.DEALLOCATING, RollbackStep.NONE),
        (MoveState.DEALLOCATED, RollbackStep.NONE),
        (MoveState.ROLLING_BACK, RollbackStep.DONOR_EVICT),
        (MoveState.ROLLING_BACK, RollbackStep.DONOR_READD),
        (MoveState.ROLLING_BACK, RollbackStep.RESTORE_DONOR_ALLOCATE),
        (MoveState.ROLLING_BACK, RollbackStep.RESTORE_DONOR_ADD),
        (MoveState.ROLLING_BACK, RollbackStep.NONE),
    ],
)
def test_grow_never_enters_donor_states(state: MoveState, step: RollbackStep) -> None:
    rec = grow_record(state, rollback_step=step)
    decision = decide(rec, _grow_evidence(), CONFIG)
    assert isinstance(decision, Block)
    assert "GROW" in decision.reason


def test_grow_terminal_states_never_act() -> None:
    assert isinstance(
        decide(grow_record(MoveState.BLOCKED), _grow_evidence(), CONFIG), Hold
    )
    assert isinstance(
        decide(grow_record(MoveState.COMPLETE), _grow_evidence(), CONFIG), Hold
    )


def test_move_record_default_kind_keeps_existing_decisions() -> None:
    assert record(MoveState.SELECTED).kind is MoveKind.MOVE
    assert record(MoveState.SELECTED).has_donor
    rec = _allocating(error="explicit failure 409", failure=EffectFailure.EXPLICIT)
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ALLOCATE
    rec = record(
        MoveState.ROLLING_BACK,
        rollback_step=RollbackStep.RELEASE_RECEIVER,
        new_path=NEW,
    )
    rec.effects[EffectName.RELEASE_RECEIVER.value] = effect(
        EffectName.RELEASE_RECEIVER,
        dispatched=True,
        attempts=1,
        response={"status": "DONE", "released_size_gib": 64},
    )
    decision = decide(rec, _deallocated_evidence(), CONFIG)
    assert isinstance(decision, Persist)
    assert decision.rollback_step is RollbackStep.RESTORE_DONOR_ALLOCATE
    # A MOVE never rebinds and never releases on a vanished receiver.
    allocated = _allocated(updated_at=0.0)
    decision = decide(
        allocated,
        _deallocated_evidence(
            now=1e6,
            receiver_identity_ok=False,
            outside={DONOR_IP: [], RECEIVER_IP: [NEW]},
            receiver_replacement=RECEIVER_2,
        ),
        CONFIG,
    )
    assert isinstance(decision, Hold)
