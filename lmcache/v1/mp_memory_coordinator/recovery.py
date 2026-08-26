# SPDX-License-Identifier: Apache-2.0
"""The pure saga decision function.

:func:`decide` maps a durable :class:`MoveRecord` plus fresh read-only
evidence (membership identities, live DAX status of both participants, the
outside status, and the usage-view capacities) to exactly one next safe
action. The controller executes that action -- at most one side effect per
cycle -- persists the result, and calls :func:`decide` again. Because the
function only ever reads the journal and current status, a restart at any
point simply re-enters the same loop: that *is* the recovery path.

Safety rules encoded here (see the design doc for the table):

* every side effect requires leadership, a reachable MP Coordinator, and
  unchanged sandwich identity of both participants;
* an outside POST is issued at most once: an effect that is ``dispatched``
  without a recorded response or explicit failure has an unknown outcome
  and the move enters ``BLOCKED``;
* DAX effects are re-driven from status (drain/evict/add are verifiable and
  idempotent-safe), never assumed from a response;
* whenever an expected effect, size, or path cannot be proven from status,
  the move enters ``BLOCKED`` and nothing further is mutated.
"""

# Standard
from dataclasses import dataclass, field
from enum import Enum
import posixpath

# First Party
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_ACTIVE_STATE,
    DAX_DRAINING_STATE,
    AllocationRequest,
    DaxDeviceStatus,
    DaxHotplugStatus,
    DaxRemoveMode,
    DeallocationRequest,
    EffectName,
    EffectRecord,
    MoveOutcome,
    MoveRecord,
    MoveState,
    OutsideStatus,
    RollbackStep,
)


class Participant(str, Enum):
    """Which MP server a DAX effect targets."""

    DONOR = "donor"
    RECEIVER = "receiver"


@dataclass(frozen=True)
class Evidence:
    """Everything :func:`decide` may look at, gathered by the controller.

    Attributes:
        now: Wall-clock time.
        leader: Whether this process currently holds leadership.
        coordinator_reachable: Whether the sandwich read succeeded.
        donor_identity_ok: Donor accepted by the sandwich read, unchanged.
        receiver_identity_ok: Receiver accepted, unchanged.
        donor_dax: Donor's live hotplug status (``None`` if unreadable).
        receiver_dax: Receiver's live hotplug status (``None`` if unreadable).
        outside: Outside status (``None`` if unreadable).
        donor_capacity_bytes: Donor ``l2/dax`` capacity from the usage view
            (``None`` when unavailable).
        receiver_capacity_bytes: Receiver capacity from the usage view.
    """

    now: float
    leader: bool
    coordinator_reachable: bool
    donor_identity_ok: bool
    receiver_identity_ok: bool
    donor_dax: DaxHotplugStatus | None
    receiver_dax: DaxHotplugStatus | None
    outside: OutsideStatus | None
    donor_capacity_bytes: int | None
    receiver_capacity_bytes: int | None


@dataclass(frozen=True)
class Hold:
    """Do nothing this cycle; poll again."""

    reason: str


@dataclass(frozen=True)
class Persist:
    """Record a state change or an effect confirmation; no side effect.

    Attributes:
        state: New state.
        rollback_step: New rollback sub-state.
        confirm_effect: Effect to mark ``confirmed``.
        fields: Scalar record fields to set.
        note: Log text.
    """

    state: MoveState
    rollback_step: RollbackStep = RollbackStep.NONE
    confirm_effect: EffectName | None = None
    fields: dict[str, str | int | float | bool] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class DoEffect:
    """Perform one side effect (a single POST) after persisting intent.

    Attributes:
        effect: Ledger name.
        intent_state: State to persist together with the intent.
        rollback_step: Rollback sub-state to persist with the intent.
        participant: Target MP server for DAX effects.
        device_path: DAX device path (DAX effects).
        size_bytes: DAX map size (add effects).
        remove_mode: Drain or evict (remove effects).
        deallocation: Outside deallocation request (if an outside effect).
        allocation: Outside allocation request (if an outside effect).
        before_paths: Outside path set of the target node before the POST.
    """

    effect: EffectName
    intent_state: MoveState
    rollback_step: RollbackStep = RollbackStep.NONE
    participant: Participant = Participant.DONOR
    device_path: str = ""
    size_bytes: int = 0
    remove_mode: DaxRemoveMode | None = None
    deallocation: DeallocationRequest | None = None
    allocation: AllocationRequest | None = None
    before_paths: list[str] = field(default_factory=list)

    @property
    def is_outside(self) -> bool:
        """Whether this effect is an outside POST."""
        return self.deallocation is not None or self.allocation is not None


@dataclass(frozen=True)
class Block:
    """Enter ``BLOCKED``: terminal, no further mutation."""

    reason: str


@dataclass(frozen=True)
class Finish:
    """Enter ``COMPLETE`` with an outcome."""

    outcome: MoveOutcome
    note: str = ""


Decision = Hold | Persist | DoEffect | Block | Finish


def _tombstone(dax: DaxHotplugStatus, device_path: str) -> DaxDeviceStatus | None:
    """Return a terminal entry for ``device_path``, if one exists."""
    for device in dax.devices:
        if device.device_path == device_path and device.is_terminal:
            return device
    return None


def _paths(outside: OutsideStatus, node: str) -> list[str]:
    """Return the outside path list of ``node`` (empty when unlisted)."""
    return list(outside.get(node, []))


def _owners(outside: OutsideStatus, device_path: str) -> list[str]:
    """Return every node listing ``device_path``."""
    return [node for node, paths in outside.items() if device_path in paths]


def validate_returned_path(
    device_path: str,
    node: str,
    before_paths: list[str],
    outside: OutsideStatus,
    config: MPMemoryCoordinatorConfig,
) -> str:
    """Prove that an allocation's returned path is ours and well-formed.

    Args:
        device_path: The path the outside service returned.
        node: The ``target_node`` that was requested.
        before_paths: The node's outside paths captured before the POST.
        outside: The current outside status.
        config: For ``allowed_device_path_prefix``.

    Returns:
        ``""`` when the path is proven; otherwise the first violated rule.
    """
    if not device_path or not device_path.startswith("/"):
        return "returned path is not absolute"
    if posixpath.normpath(device_path) != device_path or ".." in device_path.split("/"):
        return "returned path is not normalized or contains '..'"
    if not device_path.startswith(config.allowed_device_path_prefix):
        return f"returned path is outside {config.allowed_device_path_prefix}"
    if device_path in before_paths:
        return "returned path was already listed before the allocation"
    new_paths = sorted(set(_paths(outside, node)) - set(before_paths))
    if new_paths != [device_path]:
        return f"returned path is not the unique new path of {node}: {new_paths}"
    if _owners(outside, device_path) != [node]:
        return f"returned path is listed under {_owners(outside, device_path)}"
    return ""


def _gate(record: MoveRecord, evidence: Evidence, needs_receiver: bool) -> str:
    """Return why a side effect may not be issued now (``""`` if it may)."""
    if not evidence.leader:
        return "not leader"
    if not evidence.coordinator_reachable:
        return "MP Coordinator unreachable"
    if not evidence.donor_identity_ok:
        return "donor identity changed or missing"
    if needs_receiver and not evidence.receiver_identity_ok:
        return "receiver identity changed or missing"
    return ""


def _gated(decision: DoEffect, record: MoveRecord, evidence: Evidence) -> Decision:
    """Wrap a :class:`DoEffect` in the pre-POST checks."""
    needs_receiver = (
        decision.participant is Participant.RECEIVER
        or (
            decision.allocation is not None
            and decision.allocation.target_node == record.receiver.worker_ip
        )
        or (
            decision.deallocation is not None
            and decision.deallocation.target_node == record.receiver.worker_ip
        )
    )
    # Before the first outside call the receiver must still be there, or the
    # move is pointless; afterwards a vanished receiver is handled per state.
    if record.state in (MoveState.SELECTED, MoveState.DONOR_REMOVED):
        needs_receiver = True
    reason = _gate(record, evidence, needs_receiver)
    if reason:
        return Hold(f"{decision.effect.value} deferred: {reason}")
    return decision


def _outside_unknown(effect: EffectRecord) -> bool:
    """Whether an outside effect was dispatched with no recorded outcome."""
    return effect.dispatched and not effect.response and not effect.error


def _drive_donor_remove(
    record: MoveRecord,
    evidence: Evidence,
    config: MPMemoryCoordinatorConfig,
    *,
    on_removed: Decision,
    intent_state: MoveState,
    rollback_step: RollbackStep,
) -> Decision:
    """Drive drain -> evict of the donor's old path until it is removed.

    Shared by the forward path and the pre-deallocation rollback.
    """
    dax = evidence.donor_dax
    if dax is None:
        return Hold("donor DAX status unavailable")
    live = dax.find_live(record.old_path)
    if live is None:
        if _tombstone(dax, record.old_path) is None:
            # Neither live nor a tombstone: it is not attached, which is the
            # post-remove condition we need; note it for the operator.
            return on_removed
        return on_removed
    if live.state == DAX_ACTIVE_STATE:
        return _gated(
            DoEffect(
                EffectName.DONOR_DRAIN,
                intent_state,
                rollback_step,
                Participant.DONOR,
                record.old_path,
                remove_mode=DaxRemoveMode.DRAIN,
            ),
            record,
            evidence,
        )
    if live.state != DAX_DRAINING_STATE:
        return Hold(f"donor device state {live.state}; waiting")
    if live.busy_references > 0:
        started = record.drain_started_at or record.created_at
        if evidence.now - started > config.drain_timeout_seconds:
            return Block(
                f"drain of {record.old_path} exceeded {config.drain_timeout_seconds}s "
                f"with {live.busy_references} busy references; no undrain API"
            )
        return Hold(f"donor device busy: {live.busy_references} references")
    return _gated(
        DoEffect(
            EffectName.DONOR_EVICT,
            intent_state,
            rollback_step,
            Participant.DONOR,
            record.old_path,
            remove_mode=DaxRemoveMode.EVICT,
        ),
        record,
        evidence,
    )


def _drive_add(
    record: MoveRecord,
    evidence: Evidence,
    config: MPMemoryCoordinatorConfig,
    *,
    effect: EffectName,
    participant: Participant,
    device_path: str,
    dax: DaxHotplugStatus | None,
    intent_state: MoveState,
    rollback_step: RollbackStep,
    on_active: Decision,
    on_persistent_failure: Decision,
) -> Decision:
    """Drive a DAX add until the path is active, bounded by attempts."""
    if dax is None:
        return Hold(f"{participant.value} DAX status unavailable")
    live = dax.find_live(device_path)
    if live is not None and live.state == DAX_ACTIVE_STATE:
        return on_active
    if live is not None:
        return Hold(f"{device_path} on {participant.value} is {live.state}")
    ledger = record.effect(effect)
    if ledger is not None and ledger.attempts >= config.dax_add_max_attempts:
        return on_persistent_failure
    return _gated(
        DoEffect(
            effect,
            intent_state,
            rollback_step,
            participant,
            device_path,
            size_bytes=record.old_map_size_bytes,
        ),
        record,
        evidence,
    )


def _receiver_lost(record: MoveRecord, evidence: Evidence) -> bool:
    """Whether the receiver vanished while the coordinator is reachable."""
    return evidence.coordinator_reachable and not evidence.receiver_identity_ok


def decide(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """Return the next safe action for ``record`` given ``evidence``.

    Args:
        record: The durable move.
        evidence: Fresh read-only observations.
        config: Timeouts and limits.

    Returns:
        Exactly one :data:`Decision`.
    """
    if record.state is MoveState.BLOCKED:
        return Hold("blocked: operator action required")
    if record.state is MoveState.COMPLETE:
        return Hold("complete")
    if record.state is MoveState.SELECTED:
        return _decide_selected(record, evidence, config)
    if record.state is MoveState.DONOR_DRAINING:
        return _decide_donor_draining(record, evidence, config)
    if record.state is MoveState.DONOR_REMOVED:
        return _decide_donor_removed(record, evidence, config)
    if record.state is MoveState.DEALLOCATING:
        return _decide_deallocating(record, evidence, config)
    if record.state is MoveState.DEALLOCATED:
        return _decide_deallocated(record, evidence, config)
    if record.state is MoveState.ALLOCATING:
        return _decide_allocating(record, evidence, config)
    if record.state is MoveState.ALLOCATED:
        return _decide_allocated(record, evidence, config)
    return _decide_rolling_back(record, evidence, config)


def _decide_selected(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """SELECTED: nothing has happened yet; abort freely, else start draining."""
    if record.effect(EffectName.DONOR_DRAIN) is not None:
        # Intent persisted before a crash: status decides what happened.
        return _decide_donor_draining(record, evidence, config)
    if _receiver_lost(record, evidence):
        return Finish(MoveOutcome.ROLLED_BACK, "receiver vanished before any effect")
    if evidence.outside is None:
        return Hold("outside status unavailable")
    if _owners(evidence.outside, record.old_path) != [record.donor.worker_ip]:
        return Finish(
            MoveOutcome.ROLLED_BACK,
            f"{record.old_path} is not listed solely under the donor in outside "
            f"status ({_owners(evidence.outside, record.old_path)})",
        )
    if evidence.donor_dax is None:
        return Hold("donor DAX status unavailable")
    live = evidence.donor_dax.find_live(record.old_path)
    if live is None or live.state != DAX_ACTIVE_STATE or live.index <= 0:
        return Finish(
            MoveOutcome.ROLLED_BACK, f"{record.old_path} is no longer an active device"
        )
    return _drive_donor_remove(
        record,
        evidence,
        config,
        on_removed=Persist(MoveState.DONOR_REMOVED, note="old path already removed"),
        intent_state=MoveState.DONOR_DRAINING,
        rollback_step=RollbackStep.NONE,
    )


def _decide_donor_draining(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """DONOR_DRAINING: wait for busy references, evict, confirm removal."""
    if _receiver_lost(record, evidence):
        return Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.DONOR_EVICT,
            note="receiver vanished while draining; rolling back",
        )
    evict = record.effect(EffectName.DONOR_EVICT)
    confirm = evict.name if evict is not None and not evict.confirmed else None
    return _drive_donor_remove(
        record,
        evidence,
        config,
        on_removed=Persist(
            MoveState.DONOR_REMOVED,
            confirm_effect=confirm,
            note="old path no longer readable on the donor",
        ),
        intent_state=MoveState.DONOR_DRAINING,
        rollback_step=RollbackStep.NONE,
    )


def _decide_donor_removed(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """DONOR_REMOVED: issue the single deallocation POST."""
    if _receiver_lost(record, evidence):
        return Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.DONOR_READD,
            note="receiver vanished before deallocation; re-adding donor path",
        )
    if evidence.outside is None:
        return Hold("outside status unavailable")
    if evidence.donor_dax is not None and evidence.donor_dax.find_live(record.old_path):
        return Hold("old path is readable again on the donor; waiting")
    if _owners(evidence.outside, record.old_path) != [record.donor.worker_ip]:
        return Block(
            f"{record.old_path} not listed solely under donor {record.donor.worker_ip} "
            "before deallocation; ownership unprovable"
        )
    return _gated(
        DoEffect(
            EffectName.DEALLOCATE,
            MoveState.DEALLOCATING,
            deallocation=DeallocationRequest(
                request_id=record.deallocation_request_id,
                target_node=record.donor.worker_ip,
                device_path=record.old_path,
            ),
            before_paths=_paths(evidence.outside, record.donor.worker_ip),
        ),
        record,
        evidence,
    )


def _decide_deallocating(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """DEALLOCATING: intent persisted; confirm, roll back, or block."""
    effect = record.effect(EffectName.DEALLOCATE)
    if effect is None:
        return Block("DEALLOCATING without a deallocation intent record")
    if evidence.outside is None:
        return Hold("outside status unavailable")
    owners = _owners(evidence.outside, record.old_path)
    if effect.response:
        if owners:
            return Block(
                f"deallocation reported DONE but {record.old_path} is still listed "
                f"under {owners}"
            )
        return Persist(
            MoveState.DEALLOCATED,
            confirm_effect=EffectName.DEALLOCATE,
            fields={"released_size_gib": int(effect.response["released_size_gib"])},
            note="old path absent from outside status",
        )
    if effect.error:
        if owners == [record.donor.worker_ip]:
            return Persist(
                MoveState.ROLLING_BACK,
                RollbackStep.DONOR_READD,
                note=f"deallocation refused ({effect.error}); restoring donor",
            )
        return Block(
            f"deallocation failed ({effect.error}) but outside status changed: {owners}"
        )
    if _outside_unknown(effect):
        return Block(
            "deallocation was dispatched but its outcome is unknown; released "
            "size unprovable, no retry"
        )
    # Intent persisted, never dispatched (crash before the POST): it is
    # provably unsent, so the single POST may still be issued.
    return _decide_donor_removed(record, evidence, config)


def _decide_deallocated(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """DEALLOCATED: issue the single allocation POST to the receiver."""
    if evidence.outside is None:
        return Hold("outside status unavailable")
    if _receiver_lost(record, evidence):
        return Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.RESTORE_DONOR_ALLOCATE,
            note="receiver vanished after deallocation; restoring donor",
        )
    if record.released_size_gib <= 0:
        return Block("released_size_gib unknown after deallocation")
    return _gated(
        DoEffect(
            EffectName.ALLOCATE,
            MoveState.ALLOCATING,
            allocation=AllocationRequest(
                request_id=record.allocation_request_id,
                target_node=record.receiver.worker_ip,
                request_size_gib=record.released_size_gib,
            ),
            before_paths=_paths(evidence.outside, record.receiver.worker_ip),
        ),
        record,
        evidence,
    )


def _decide_allocating(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """ALLOCATING: validate the returned path, roll back, or block."""
    effect = record.effect(EffectName.ALLOCATE)
    if effect is None:
        return Block("ALLOCATING without an allocation intent record")
    if evidence.outside is None:
        return Hold("outside status unavailable")
    node = record.receiver.worker_ip
    new_paths = sorted(set(_paths(evidence.outside, node)) - set(effect.before_paths))
    if effect.response:
        path = str(effect.response["device_path"])
        problem = validate_returned_path(
            path, node, effect.before_paths, evidence.outside, config
        )
        if not problem:
            return Persist(
                MoveState.ALLOCATED,
                confirm_effect=EffectName.ALLOCATE,
                fields={
                    "new_path": path,
                    "granted_size_gib": int(effect.response["granted_size_gib"]),
                },
                note="returned path proven under the receiver",
            )
        if len(new_paths) == 1:
            return Persist(
                MoveState.ROLLING_BACK,
                RollbackStep.RELEASE_RECEIVER,
                fields={"new_path": new_paths[0]},
                note=f"allocation returned an invalid path ({problem}); releasing "
                f"the proven allocation {new_paths[0]}",
            )
        return Block(
            f"allocation returned an invalid path ({problem}); effect unprovable"
        )
    if effect.error:
        if not new_paths:
            return Persist(
                MoveState.ROLLING_BACK,
                RollbackStep.RESTORE_DONOR_ALLOCATE,
                note=f"allocation failed ({effect.error}); restoring donor",
            )
        if len(new_paths) == 1:
            return Persist(
                MoveState.ROLLING_BACK,
                RollbackStep.RELEASE_RECEIVER,
                fields={"new_path": new_paths[0]},
                note=f"allocation failed ({effect.error}) but {new_paths[0]} appeared; "
                "releasing it",
            )
        return Block(
            f"allocation failed ({effect.error}); receiver set diff {new_paths}"
        )
    if _outside_unknown(effect):
        return Block(
            "allocation was dispatched but its outcome is unknown; granted size "
            "unprovable, no retry"
        )
    return _decide_deallocated(record, evidence, config)


def _decide_allocated(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """ALLOCATED: attach on the receiver, then wait for capacity convergence."""
    add = record.effect(EffectName.RECEIVER_ADD)
    dax = evidence.receiver_dax
    if add is not None and add.confirmed:
        if dax is None:
            return Hold("receiver DAX status unavailable")
        live = dax.find_live(record.new_path)
        if live is None or live.state != DAX_ACTIVE_STATE:
            return Block(f"{record.new_path} no longer active on the receiver")
        expected_donor = record.donor_capacity_bytes - record.old_slot_capacity_bytes
        expected_receiver = record.receiver_capacity_bytes + live.slot_capacity_bytes
        if (
            evidence.donor_capacity_bytes == expected_donor
            and evidence.receiver_capacity_bytes == expected_receiver
        ):
            return Finish(MoveOutcome.SUCCEEDED, "capacity converged")
        return Hold(
            f"waiting for capacity: donor {evidence.donor_capacity_bytes} -> "
            f"{expected_donor}, receiver {evidence.receiver_capacity_bytes} -> "
            f"{expected_receiver}"
        )
    on_active = Persist(
        MoveState.ALLOCATED,
        confirm_effect=EffectName.RECEIVER_ADD,
        fields=_active_fields(dax, record.new_path),
        note="new path active on the receiver",
    )
    return _drive_add(
        record,
        evidence,
        config,
        effect=EffectName.RECEIVER_ADD,
        participant=Participant.RECEIVER,
        device_path=record.new_path,
        dax=dax,
        intent_state=MoveState.ALLOCATED,
        rollback_step=RollbackStep.NONE,
        on_active=on_active,
        on_persistent_failure=Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.RELEASE_RECEIVER,
            note="receiver add failed persistently; releasing the receiver path",
        ),
    )


def _active_fields(
    dax: DaxHotplugStatus | None, device_path: str
) -> dict[str, str | int | float | bool]:
    """Record index and slot capacity of a live device (empty if unknown)."""
    if dax is None:
        return {}
    live = dax.find_live(device_path)
    if live is None:
        return {}
    return {
        "new_device_index": live.index,
        "new_slot_capacity_bytes": live.slot_capacity_bytes,
    }


def _decide_rolling_back(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """ROLLING_BACK: drive the sub-state machine to ROLLED_BACK or BLOCKED."""
    step = record.rollback_step
    if step is RollbackStep.DONOR_EVICT:
        return _drive_donor_remove(
            record,
            evidence,
            config,
            on_removed=Persist(
                MoveState.ROLLING_BACK,
                RollbackStep.DONOR_READD,
                note="old path removed; re-adding it to the donor",
            ),
            intent_state=MoveState.ROLLING_BACK,
            rollback_step=RollbackStep.DONOR_EVICT,
        )
    if step is RollbackStep.DONOR_READD:
        if evidence.outside is None:
            return Hold("outside status unavailable")
        if _owners(evidence.outside, record.old_path) != [record.donor.worker_ip]:
            return Block(
                f"cannot re-add {record.old_path}: not listed solely under the donor"
            )
        return _drive_add(
            record,
            evidence,
            config,
            effect=EffectName.DONOR_READD,
            participant=Participant.DONOR,
            device_path=record.old_path,
            dax=evidence.donor_dax,
            intent_state=MoveState.ROLLING_BACK,
            rollback_step=RollbackStep.DONOR_READD,
            on_active=Finish(MoveOutcome.ROLLED_BACK, "old path re-added to donor"),
            on_persistent_failure=Block(
                f"re-adding {record.old_path} to the donor failed persistently"
            ),
        )
    if step is RollbackStep.RELEASE_RECEIVER:
        return _decide_release_receiver(record, evidence)
    if step is RollbackStep.RESTORE_DONOR_ALLOCATE:
        return _decide_restore_allocate(record, evidence, config)
    if step is RollbackStep.RESTORE_DONOR_ADD:
        return _drive_add(
            record,
            evidence,
            config,
            effect=EffectName.RESTORE_ADD,
            participant=Participant.DONOR,
            device_path=record.restored_path,
            dax=evidence.donor_dax,
            intent_state=MoveState.ROLLING_BACK,
            rollback_step=RollbackStep.RESTORE_DONOR_ADD,
            on_active=Finish(MoveOutcome.ROLLED_BACK, "donor restored"),
            on_persistent_failure=Block(
                f"attaching restored path {record.restored_path} failed persistently"
            ),
        )
    return Block(f"ROLLING_BACK with unknown step {step.value}")


def _decide_release_receiver(record: MoveRecord, evidence: Evidence) -> Decision:
    """Release a proven receiver allocation, then restore the donor."""
    if evidence.outside is None:
        return Hold("outside status unavailable")
    node = record.receiver.worker_ip
    if evidence.receiver_dax is not None and evidence.receiver_dax.find_live(
        record.new_path
    ):
        return Block(
            f"{record.new_path} is attached on the receiver; detach it manually "
            "before it can be released"
        )
    effect = record.effect(EffectName.RELEASE_RECEIVER)
    owners = _owners(evidence.outside, record.new_path)
    if effect is not None and effect.response:
        if owners:
            return Block(
                f"release reported DONE but {record.new_path} still under {owners}"
            )
        return Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.RESTORE_DONOR_ALLOCATE,
            confirm_effect=EffectName.RELEASE_RECEIVER,
            note="receiver path released",
        )
    if effect is not None and effect.error:
        return Block(f"releasing {record.new_path} failed: {effect.error}")
    if effect is not None and _outside_unknown(effect):
        return Block("release of the receiver path has an unknown outcome")
    if owners != [node]:
        return Block(f"{record.new_path} not listed solely under receiver: {owners}")
    return _gated(
        DoEffect(
            EffectName.RELEASE_RECEIVER,
            MoveState.ROLLING_BACK,
            RollbackStep.RELEASE_RECEIVER,
            deallocation=DeallocationRequest(
                request_id=record.release_request_id,
                target_node=node,
                device_path=record.new_path,
            ),
            before_paths=_paths(evidence.outside, node),
        ),
        record,
        evidence,
    )


def _decide_restore_allocate(
    record: MoveRecord, evidence: Evidence, config: MPMemoryCoordinatorConfig
) -> Decision:
    """Best-effort: allocate the same GiB back to the donor."""
    if evidence.outside is None:
        return Hold("outside status unavailable")
    node = record.donor.worker_ip
    effect = record.effect(EffectName.RESTORE_ALLOCATE)
    if effect is not None and effect.response:
        path = str(effect.response["device_path"])
        problem = validate_returned_path(
            path, node, effect.before_paths, evidence.outside, config
        )
        if problem:
            return Block(f"donor restore returned an invalid path ({problem})")
        return Persist(
            MoveState.ROLLING_BACK,
            RollbackStep.RESTORE_DONOR_ADD,
            confirm_effect=EffectName.RESTORE_ALLOCATE,
            fields={"restored_path": path},
            note="donor restore allocation proven",
        )
    if effect is not None and effect.error:
        new_paths = sorted(
            set(_paths(evidence.outside, node)) - set(effect.before_paths)
        )
        if new_paths:
            return Block(
                f"donor restore failed ({effect.error}) but {new_paths} appeared"
            )
        return Finish(
            MoveOutcome.ROLLED_BACK,
            f"donor restore refused ({effect.error}); the old capacity stays free at "
            "the outside service and leaves the managed inventory",
        )
    if effect is not None and _outside_unknown(effect):
        return Block("donor restore allocation has an unknown outcome")
    return _gated(
        DoEffect(
            EffectName.RESTORE_ALLOCATE,
            MoveState.ROLLING_BACK,
            RollbackStep.RESTORE_DONOR_ALLOCATE,
            allocation=AllocationRequest(
                request_id=record.restore_request_id,
                target_node=node,
                request_size_gib=record.allocation_size_gib,
            ),
            before_paths=_paths(evidence.outside, node),
        ),
        record,
        evidence,
    )
