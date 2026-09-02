# SPDX-License-Identifier: Apache-2.0
"""The control loop: observe, propose, and drive one saga at a time.

Each cycle performs a sandwich read. With no active saga it reads each
relevant MP server's DAX status once, discovers newly owned devices from
outside status (see :mod:`lmcache.v1.mp_memory_coordinator.discovery`),
attaches present devices the outside service assigns to that worker (see
:mod:`lmcache.v1.mp_memory_coordinator.attachment`), reconciles the
inventory, updates the pressure history, ranks candidates, preflights
receivers for a donor-less GROW first and donor/receiver pairs for a MOVE
only when no receiver can grow, and logs a structured dry-run proposal;
with ``actuation_enabled`` and leadership it persists a ``SELECTED``
record. With an active saga it gathers :class:`Evidence`, asks
:func:`decide` for the one next action, executes at most one side effect,
and persists the result -- the persist-before-effect discipline of the
design doc.

I/O is abstracted behind :class:`Remote` so the controller is tested with
injected fakes; :class:`HttpRemote` binds it to the real clients.
"""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Protocol
import asyncio
import json
import time
import uuid

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.adoption import (
    AdoptionEntry,
    AdoptionResult,
    adopt,
)
from lmcache.v1.mp_memory_coordinator.attachment import (
    AttachPlan,
    plan_attachments,
)
from lmcache.v1.mp_memory_coordinator.clients import (
    AmbiguousMutationError,
    ClientConnectionError,
    ClientError,
)
from lmcache.v1.mp_memory_coordinator.clients.memory_allocation_client import (
    MemoryAllocationClient,
    OutsideContractError,
    OutsideExplicitFailure,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_server_client import (
    DaxRemoveResult,
    MPServerClient,
    format_gib,
)
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.discovery import discover
from lmcache.v1.mp_memory_coordinator.leader import LeaderElector
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_ACTIVE_STATE,
    GIB,
    NO_DONOR,
    AllocationOrigin,
    AllocationRequest,
    AllocationResponse,
    DaxAddResponse,
    DaxDeviceNotFound,
    DaxHotplugStatus,
    DaxRemoveBlocked,
    DaxRemoveMode,
    DeallocationRequest,
    DeallocationResponse,
    EffectFailure,
    EffectName,
    EffectRecord,
    InstanceIdentity,
    JournalDocument,
    ManagedAllocation,
    MoveKind,
    MoveOutcome,
    MoveRecord,
    MoveState,
    OutsideStatus,
)
from lmcache.v1.mp_memory_coordinator.persistence.rebalance_journal import (
    JournalError,
    RebalanceJournal,
)
from lmcache.v1.mp_memory_coordinator.policy import (
    Candidates,
    GrowProposal,
    LivePreflight,
    MembershipSnapshot,
    MoveProposal,
    PressureHistory,
    Proposal,
    Rejection,
    RejectionReason,
    evaluate_grow,
    evaluate_pair,
    fetch_preflight,
    rank_candidates,
    read_sandwich,
)
from lmcache.v1.mp_memory_coordinator.recovery import (
    Block,
    Decision,
    DoEffect,
    Evidence,
    Finish,
    Hold,
    Participant,
    Persist,
    decide,
)

logger = init_logger(__name__)

_HISTORY_LIMIT = 20


class Remote(Protocol):
    """Every remote read and side effect the controller performs."""

    async def sandwich(self) -> MembershipSnapshot:
        """One sandwich membership read."""
        ...

    async def preflight(self, identity: InstanceIdentity) -> LivePreflight | None:
        """``/status`` + ``/reconfigure/dax/status`` of one MP server."""
        ...

    async def dax_status(self, identity: InstanceIdentity) -> DaxHotplugStatus | None:
        """The single DAX adapter's hotplug status, or ``None``."""
        ...

    async def outside_status(self) -> OutsideStatus | None:
        """The outside status, or ``None`` when unreadable."""
        ...

    async def remove_device(
        self, identity: InstanceIdentity, device_path: str, mode: DaxRemoveMode
    ) -> DaxRemoveResult:
        """``POST /reconfigure/dax/remove`` once."""
        ...

    async def add_device(
        self, identity: InstanceIdentity, device_path: str, size_bytes: int
    ) -> DaxAddResponse:
        """``POST /reconfigure/dax/add`` once."""
        ...

    async def deallocate(self, request: DeallocationRequest) -> DeallocationResponse:
        """One outside deallocation POST."""
        ...

    async def allocate(self, request: AllocationRequest) -> AllocationResponse:
        """One outside allocation POST."""
        ...


class HttpRemote:
    """:class:`Remote` over the three HTTP clients."""

    def __init__(
        self,
        coordinator: MPCoordinatorClient,
        mp_client: MPServerClient,
        allocator: MemoryAllocationClient,
        *,
        adapter_index: int,
        clock: Callable[[], float],
    ) -> None:
        """Args:
        coordinator: MP Coordinator client.
        mp_client: MP server client.
        allocator: Outside service client.
        adapter_index: Backend-local DAX adapter index for every DAX POST.
        clock: Wall-clock source.
        """
        self._coordinator = coordinator
        self._mp = mp_client
        self._allocator = allocator
        self._adapter_index = adapter_index
        self._clock = clock

    async def sandwich(self) -> MembershipSnapshot:
        """See :class:`Remote`."""
        return await read_sandwich(self._coordinator, self._clock)

    async def preflight(self, identity: InstanceIdentity) -> LivePreflight | None:
        """See :class:`Remote`."""
        return await fetch_preflight(self._mp, identity)

    async def dax_status(self, identity: InstanceIdentity) -> DaxHotplugStatus | None:
        """See :class:`Remote`."""
        try:
            status = await self._mp.get_dax_status(identity.base_url)
        except ClientError as exc:
            logger.warning("dax status of %s failed: %s", identity.instance_id, exc)
            return None
        if len(status.adapters) != 1:
            return None
        return status.adapters[0].status

    async def outside_status(self) -> OutsideStatus | None:
        """See :class:`Remote`."""
        try:
            return await self._allocator.get_status()
        except ClientError as exc:
            logger.warning("outside status failed: %s", exc)
            return None

    async def remove_device(
        self, identity: InstanceIdentity, device_path: str, mode: DaxRemoveMode
    ) -> DaxRemoveResult:
        """See :class:`Remote`."""
        return await self._mp.remove_dax_device(
            identity.base_url,
            adapter_index=self._adapter_index,
            device_path=device_path,
            mode=mode,
        )

    async def add_device(
        self, identity: InstanceIdentity, device_path: str, size_bytes: int
    ) -> DaxAddResponse:
        """See :class:`Remote`."""
        return await self._mp.add_dax_device(
            identity.base_url,
            adapter_index=self._adapter_index,
            device_path=device_path,
            size=format_gib(size_bytes),
        )

    async def deallocate(self, request: DeallocationRequest) -> DeallocationResponse:
        """See :class:`Remote`."""
        return await self._allocator.deallocate(request)

    async def allocate(self, request: AllocationRequest) -> AllocationResponse:
        """See :class:`Remote`."""
        return await self._allocator.allocate(request)


@dataclass
class CycleReport:
    """Structured summary of one control cycle (logged as JSON).

    Attributes:
        at: Wall-clock time of the cycle.
        leader: Leadership at the start of the cycle.
        coordinator_reachable: Whether the sandwich read succeeded.
        accepted: Accepted instance ids.
        rejections: Structured rejections.
        proposal: The dry-run proposal, if any; its ``kind`` is ``"grow"``
            (receiver, size) or ``"move"`` (donor, receiver, device).
        move_id: Active move id (``""`` if none).
        move_state: Active move state (``""`` if none).
        decision: Decision type and reason for an active move.
        discovery: What device discovery adopted and skipped this cycle
            (empty while a move is active, when discovery does not run).
        attachments: What attach orchestration did this cycle: ``planned``
            (eligible adds), ``attached`` (adds confirmed active),
            ``would_attach`` (adds withheld: dry run or stopping), ``failed``
            (``path -> error``), ``skipped`` (``path -> reason`` for devices
            not planned); or ``skipped_pass`` with a reason when the pass
            did not run (outside status unavailable) or its planned adds
            were withheld (leadership lost). Empty while a move is active.
        error: Error text, if the cycle failed.
    """

    at: float
    leader: bool = False
    coordinator_reachable: bool = False
    accepted: list[str] = field(default_factory=list)
    rejections: list[dict[str, str]] = field(default_factory=list)
    proposal: dict[str, str | int | float] | None = None
    move_id: str = ""
    move_state: str = ""
    decision: str = ""
    discovery: dict[str, object] = field(default_factory=dict)
    attachments: dict[str, object] = field(default_factory=dict)
    error: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable form."""
        return {
            "at": self.at,
            "leader": self.leader,
            "coordinator_reachable": self.coordinator_reachable,
            "accepted": self.accepted,
            "rejections": self.rejections,
            "proposal": self.proposal,
            "move_id": self.move_id,
            "move_state": self.move_state,
            "decision": self.decision,
            "discovery": self.discovery,
            "attachments": self.attachments,
            "error": self.error,
        }


def new_move_record(proposal: MoveProposal, now: float) -> MoveRecord:
    """Build a ``SELECTED`` record from a proposal.

    Request ids are deterministic per move and distinct per operation; they
    aid audit only and imply no outside naming convention.

    Args:
        proposal: The checked proposal.
        now: Wall-clock time.

    Returns:
        The new record.
    """
    move_id = f"move-{int(now)}-{uuid.uuid4().hex[:8]}"
    device = proposal.choice.device
    return MoveRecord(
        move_id=move_id,
        state=MoveState.SELECTED,
        donor=proposal.donor.identity,
        receiver=proposal.receiver.identity,
        donor_capacity_bytes=proposal.donor.capacity_bytes,
        receiver_capacity_bytes=proposal.receiver.capacity_bytes,
        old_path=device.device_path,
        old_device_index=device.index,
        old_map_size_bytes=device.max_dax_size_bytes,
        old_slot_capacity_bytes=device.slot_capacity_bytes,
        allocation_size_gib=proposal.choice.allocation.allocation_size_gib,
        deallocation_request_id=f"{move_id}-deallocate",
        allocation_request_id=f"{move_id}-allocate",
        release_request_id=f"{move_id}-release",
        restore_request_id=f"{move_id}-restore",
        created_at=now,
        updated_at=now,
    )


def new_grow_record(proposal: GrowProposal, now: float) -> MoveRecord:
    """Build a ``SELECTED`` GROW record from a proposal.

    The record has no donor: ``donor`` is :data:`NO_DONOR` and every
    donor-side field is empty, while ``old_map_size_bytes`` carries the map
    size to add on the receiver (``request_size_gib * GIB``) so the receiver
    add and the inventory entry read it unchanged. Request ids are
    deterministic per saga and audit-only.

    Args:
        proposal: The checked GROW proposal.
        now: Wall-clock time.

    Returns:
        The new record.
    """
    move_id = f"grow-{int(now)}-{uuid.uuid4().hex[:8]}"
    return MoveRecord(
        move_id=move_id,
        state=MoveState.SELECTED,
        kind=MoveKind.GROW,
        donor=NO_DONOR,
        receiver=proposal.receiver.identity,
        donor_capacity_bytes=0,
        receiver_capacity_bytes=proposal.receiver.capacity_bytes,
        old_path="",
        old_device_index=-1,
        old_map_size_bytes=proposal.request_size_gib * GIB,
        old_slot_capacity_bytes=0,
        allocation_size_gib=proposal.request_size_gib,
        deallocation_request_id="",
        allocation_request_id=f"{move_id}-allocate",
        release_request_id=f"{move_id}-release",
        restore_request_id="",
        created_at=now,
        updated_at=now,
    )


class RebalanceController:
    """Owns the journal document and drives cycles."""

    def __init__(
        self,
        config: MPMemoryCoordinatorConfig,
        journal: RebalanceJournal,
        remote: Remote,
        leader: LeaderElector,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Args:
        config: The configuration.
        journal: The durable journal.
        remote: Remote reads and effects.
        leader: Leadership source.
        clock: Wall-clock source.
        """
        self._config = config
        self._journal = journal
        self._remote = remote
        self._leader = leader
        self._clock = clock
        self._document = JournalDocument()
        self._history = PressureHistory(config.stable_samples)
        self._journal_error = ""
        self._loaded = False
        self._reconciled = False
        self._stopping = False
        self._was_leader = False
        self._last_report = CycleReport(at=0.0)
        self._cycle_lock = asyncio.Lock()
        # device_path -> wall-clock time of the last failed attach; an entry
        # is dropped once it is older than ``cooldown_seconds`` or the path
        # is attached, so the map never outgrows the set of failing paths.
        self._attach_failures: dict[str, float] = {}
        # Successful adds since this process started. Deliberately not
        # persisted: attaching is idempotent, so a restart that starts from
        # zero loses nothing that matters.
        self._attached = 0

    # -- lifecycle ------------------------------------------------------------

    def load(self) -> None:
        """Load the journal; a corrupt journal makes the process unready.

        Never raises: the failure is recorded and every cycle then refuses
        to act until an operator repairs the file.
        """
        try:
            self._document = self._journal.load()
            self._journal_error = ""
            self._loaded = True
        except JournalError as exc:
            logger.error("journal unusable; refusing to mutate: %s", exc)
            self._journal_error = str(exc)
            self._loaded = False

    @property
    def document(self) -> JournalDocument:
        """The in-memory journal document (read-only for callers)."""
        return self._document

    @property
    def journal_error(self) -> str:
        """Why the journal is unusable (``""`` when it is fine)."""
        return self._journal_error

    @property
    def last_report(self) -> CycleReport:
        """The most recent cycle report."""
        return self._last_report

    @property
    def attached_devices(self) -> int:
        """Devices attached by attach orchestration since this process started.

        In-memory only (restarts from zero); see :meth:`status`.
        """
        return self._attached

    def request_stop(self) -> None:
        """Start no new move; the current cycle completes and persists."""
        self._stopping = True

    def readiness(self) -> tuple[bool, str]:
        """Return ``(ready, reason)`` for ``/readyz``.

        Ready means: journal loaded, current leader, MP Coordinator reached
        on the last cycle, that cycle did not fail, inventory reconciled,
        and no BLOCKED move.
        """
        if self._journal_error:
            return False, f"journal: {self._journal_error}"
        if not self._loaded:
            return False, "journal not loaded"
        if not self._leader.is_leader():
            return False, "not leader"
        if self._last_report.error:
            return False, f"last cycle failed: {self._last_report.error}"
        if not self._last_report.coordinator_reachable:
            return False, "MP Coordinator unreachable"
        if not self._reconciled:
            return False, "inventory not reconciled"
        move = self._document.active_move
        if move is not None and move.state is MoveState.BLOCKED:
            return False, f"move {move.move_id} BLOCKED: {move.block_reason}"
        return True, "ok"

    def status(self) -> dict[str, object]:
        """Return a JSON-serializable status for ``/status``.

        ``counters`` carries the persisted journal counters plus
        ``attached``, the in-memory count of :attr:`attached_devices`;
        ``grow_backoffs`` lists only backoffs that are still active.
        """
        move = self._document.active_move
        now = self._clock()
        return {
            "leader": self._leader.is_leader(),
            "leader_identity": self._leader.identity,
            "actuation_enabled": self._config.actuation_enabled,
            "journal_error": self._journal_error,
            "initialized": self._document.initialized,
            "inventory": [a.model_dump(mode="json") for a in self._document.inventory],
            "cooldowns": dict(self._document.cooldowns),
            "grow_backoffs": {
                ip: until
                for ip, until in self._document.grow_backoffs.items()
                if until > now
            },
            "history": self._history.snapshot(),
            "active_move": move.model_dump(mode="json") if move is not None else None,
            "counters": {
                **self._document.counters.model_dump(),
                "attached": self._attached,
            },
            "last_cycle": self._last_report.as_dict(),
        }

    async def adopt_once(
        self,
        entries: list[AdoptionEntry],
        *,
        coordinator: MPCoordinatorClient,
        mp_client: MPServerClient,
        allocator: MemoryAllocationClient,
    ) -> AdoptionResult:
        """Run one adoption pass under the cycle lock and persist it.

        Args:
            entries: The operator-approved allowlist.
            coordinator: MP Coordinator client.
            mp_client: MP server client.
            allocator: Outside service client.

        Returns:
            The adoption result.

        Raises:
            ClientError: If the MP Coordinator or outside status is
                unreachable; the journal is left untouched and uninitialized
                so a later pass can retry.
            JournalError: If the journal cannot be written.
        """
        async with self._cycle_lock:
            result = await adopt(
                entries,
                self._document,
                coordinator=coordinator,
                mp_client=mp_client,
                allocator=allocator,
                config=self._config,
                clock=self._clock,
            )
            self._save()
            return result

    async def run_forever(self, stop: asyncio.Event) -> None:
        """Run cycles until ``stop`` is set.

        Polls faster (``dax_poll_interval_seconds``) while a move is
        active, else every ``poll_interval_seconds``.
        """
        while not stop.is_set() and not self._stopping:
            try:
                await self.run_once()
            except Exception:  # noqa: BLE001 -- the loop must survive
                logger.exception("control cycle failed")
            interval = (
                self._config.dax_poll_interval_seconds
                if self._document.active_move is not None
                else self._config.poll_interval_seconds
            )
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> CycleReport:
        """Run one cycle and return its report.

        Any failure of the cycle body is recorded in ``report.error`` (and
        logged with its traceback) and the report still replaces the last
        one, so readiness never reflects a stale successful cycle while the
        loop keeps failing.

        Returns:
            The report (also logged as one JSON line).
        """
        async with self._cycle_lock:
            report = CycleReport(at=self._clock(), leader=self._leader.is_leader())
            if self._journal_error:
                report.error = f"journal unusable: {self._journal_error}"
            else:
                try:
                    await self._cycle(report)
                except JournalError as exc:
                    self._journal_error = str(exc)
                    report.error = f"journal write failed: {exc}"
                except Exception as exc:  # noqa: BLE001 -- reported, never hidden
                    logger.exception("control cycle failed")
                    report.error = f"cycle failed: {exc}"
            self._last_report = report
            logger.info("memcoord cycle %s", json.dumps(report.as_dict(), default=str))
            return report

    # -- one cycle -------------------------------------------------------------

    async def _cycle(self, report: CycleReport) -> None:
        """The body of one cycle."""
        snapshot = await self._remote.sandwich()
        report.coordinator_reachable = snapshot.coordinator_reachable
        report.accepted = sorted(snapshot.samples)
        report.rejections = [r.as_dict() for r in snapshot.rejections]
        if not report.leader:
            # Standby: observe for readiness only. A non-leader never writes
            # the journal, so shared storage sees a single writer.
            self._was_leader = False
            report.decision = "standby: not leader"
            return
        if not self._was_leader:
            # Newly (re)elected: the previous leader may have written the
            # journal; start from what is durable, not from stale memory.
            self.load()
            self._history = PressureHistory(self._config.stable_samples)
            self._was_leader = True
            if self._journal_error:
                report.error = f"journal unusable: {self._journal_error}"
                return
        move = self._document.active_move
        if move is not None:
            report.move_id = move.move_id
            report.move_state = move.state.value
            if move.state is MoveState.BLOCKED:
                report.decision = f"blocked: {move.block_reason}"
                return
            decision = await self._advance(move, snapshot, report.leader)
            report.decision = _describe(decision)
            move = self._document.active_move
            report.move_state = move.state.value if move is not None else "COMPLETE"
            return

        self._history.observe(snapshot, self._config)
        if not snapshot.coordinator_reachable:
            return
        statuses = await self._collect_dax_statuses(snapshot)
        # One outside read serves both discovery and attach orchestration so
        # the two never disagree about ownership within a cycle.
        outside = await self._remote.outside_status()
        report.discovery = self._discover(snapshot, statuses, outside)
        report.attachments = await self._attach(snapshot, statuses, outside)
        self._reconcile_inventory(snapshot, statuses)
        if report.attachments.get("attached") or report.attachments.get("failed"):
            # The sandwich read predates the add: the adapter's capacity has
            # changed (or may have, after an ambiguous failure), so a move
            # selected now would carry stale capacities in its record and
            # never converge. Re-observe before proposing anything.
            report.decision = "attach issued; re-observing next cycle"
            return
        now = self._clock()
        # Expired grow backoffs leave the document here, in the idle cycle
        # that could propose again; /status filters them out at any time.
        self._document.grow_backoffs = {
            ip: until
            for ip, until in self._document.grow_backoffs.items()
            if until > now
        }
        candidates = rank_candidates(
            snapshot, self._history, self._document.cooldowns, now
        )
        report.rejections.extend(r.as_dict() for r in candidates.rejections)
        proposal, rejections = await self._evaluate(candidates)
        report.rejections.extend(r.as_dict() for r in rejections)
        if proposal is None:
            return
        report.proposal = proposal.as_dict()
        self._document.counters.proposed += 1
        if not self._config.actuation_enabled:
            report.rejections.append(
                Rejection("", RejectionReason.ACTUATION_DISABLED, "dry run").as_dict()
            )
            self._save()
            return
        if self._stopping:
            report.rejections.append(
                Rejection("", RejectionReason.NOT_LEADER, "stopping").as_dict()
            )
            self._save()
            return
        if not await self._leader.ensure_leader():
            report.rejections.append(
                Rejection("", RejectionReason.NOT_LEADER).as_dict()
            )
            self._save()
            return
        record = (
            new_grow_record(proposal, self._clock())
            if isinstance(proposal, GrowProposal)
            else new_move_record(proposal, self._clock())
        )
        self._document.active_move = record
        self._save()
        report.move_id = record.move_id
        logger.info(
            "%s %s SELECTED: %s",
            record.kind.value,
            record.move_id,
            json.dumps(proposal.as_dict()),
        )
        decision = await self._advance(record, snapshot, True)
        report.decision = _describe(decision)
        move = self._document.active_move
        report.move_state = move.state.value if move is not None else "COMPLETE"

    async def _evaluate(
        self, candidates: Candidates
    ) -> tuple[Proposal | None, list[Rejection]]:
        """Preflight candidates; return the first proposal or the rejections.

        Grow before move: every stable-HIGH receiver, best first, is first
        evaluated for a donor-less GROW; the first eligible one is proposed.
        Only when no receiver can grow (each is in grow backoff or fails a
        receiver-only check) are donor/receiver pairs evaluated for a MOVE.
        Each instance's live preflight is read at most once per cycle.

        Args:
            candidates: The ranked, cooldown-free candidates of this cycle.

        Returns:
            ``(proposal, rejections)``; the proposal is ``None`` when nothing
            is eligible, and the rejections explain every skipped candidate
            (each distinct rejection once, even when both passes hit it).
        """
        rejections: list[Rejection] = []
        preflights: dict[str, LivePreflight | None] = {}

        async def _get(identity: InstanceIdentity) -> LivePreflight | None:
            if identity.instance_id not in preflights:
                preflights[identity.instance_id] = await self._remote.preflight(
                    identity
                )
            return preflights[identity.instance_id]

        now = self._clock()
        for receiver in candidates.receivers:
            receiver_pf = await _get(receiver.identity)
            if receiver_pf is None:
                rejections.append(
                    Rejection(
                        receiver.identity.instance_id,
                        RejectionReason.PREFLIGHT_UNAVAILABLE,
                    )
                )
                continue
            grow = evaluate_grow(
                receiver,
                receiver_pf,
                self._document.inventory,
                self._document.grow_backoffs,
                self._config,
                now,
            )
            if isinstance(grow, GrowProposal):
                return grow, _unique_rejections(rejections)
            rejections.extend(grow)

        for donor in candidates.donors:
            donor_pf = await _get(donor.identity)
            if donor_pf is None:
                rejections.append(
                    Rejection(
                        donor.identity.instance_id,
                        RejectionReason.PREFLIGHT_UNAVAILABLE,
                    )
                )
                continue
            for receiver in candidates.receivers:
                if receiver.identity.worker_ip == donor.identity.worker_ip:
                    continue
                receiver_pf = await _get(receiver.identity)
                if receiver_pf is None:
                    rejections.append(
                        Rejection(
                            receiver.identity.instance_id,
                            RejectionReason.PREFLIGHT_UNAVAILABLE,
                        )
                    )
                    continue
                result = evaluate_pair(
                    donor,
                    receiver,
                    donor_pf,
                    receiver_pf,
                    self._document.inventory,
                    self._config,
                )
                if isinstance(result, MoveProposal):
                    return result, _unique_rejections(rejections)
                rejections.extend(result)
        return None, _unique_rejections(rejections)

    async def _collect_dax_statuses(
        self, snapshot: MembershipSnapshot
    ) -> dict[str, DaxHotplugStatus]:
        """Read every accepted instance's DAX adapter status once per cycle.

        Any accepted instance may be holding an undiscovered device, and
        reconciliation needs the same documents, so each is read exactly
        once. Unreadable instances are simply absent from the result.

        Args:
            snapshot: The current sandwich read.

        Returns:
            ``instance_id -> DaxHotplugStatus`` for every instance read
            successfully.
        """
        statuses: dict[str, DaxHotplugStatus] = {}
        for sample in snapshot.samples.values():
            identity = sample.identity
            status = await self._remote.dax_status(identity)
            if status is not None:
                statuses[identity.instance_id] = status
        return statuses

    def _discover(
        self,
        snapshot: MembershipSnapshot,
        statuses: Mapping[str, DaxHotplugStatus],
        outside: OutsideStatus | None,
    ) -> dict[str, object]:
        """Run one discovery pass and persist anything it adopted.

        Discovery needs the outside status to prove ownership; when that
        read failed the pass is skipped and the existing inventory is left
        untouched, so a transient outside outage never shrinks what the
        coordinator manages.

        Args:
            snapshot: The current sandwich read.
            statuses: Per-instance DAX status from
                :meth:`_collect_dax_statuses`.
            outside: This cycle's outside status, or ``None`` when the read
                failed.

        Returns:
            The JSON-friendly discovery summary for the cycle report.
        """
        if outside is None:
            return {"skipped_pass": "outside status unavailable"}
        result = discover(
            snapshot.samples,
            statuses,
            outside,
            self._document,
            self._config,
            self._clock(),
        )
        if result.discovered:
            self._save()
        return result.as_dict()

    async def _attach(
        self,
        snapshot: MembershipSnapshot,
        statuses: Mapping[str, DaxHotplugStatus],
        outside: OutsideStatus | None,
    ) -> dict[str, object]:
        """Attach present devices the outside service assigns to their worker.

        Runs only with no active move (the caller guarantees it): a saga's
        receiver add and a donor's post-evict window must never be raced by
        a second writer of the same adapter. With ``actuation_enabled`` off
        or after :meth:`request_stop` the plan is only reported
        (``would_attach``); when leadership cannot be renewed the planned
        adds are withheld and reported as ``skipped_pass``. Each planned add
        is issued at most once per cycle; a failure is remembered so the
        path is not retried before ``cooldown_seconds``. Nothing is
        journaled and nothing is persisted: a same-size re-add is idempotent
        on the MP server, the next cycle's discovery adopts the device
        through the unchanged ownership rule, and the success count is
        in-memory only. Never raises: client errors from the add are caught
        and reported.

        Args:
            snapshot: The current sandwich read.
            statuses: Per-instance DAX status from
                :meth:`_collect_dax_statuses`.
            outside: This cycle's outside status, or ``None`` when the read
                failed (the pass is then skipped).

        Returns:
            The JSON-friendly attachment summary for the cycle report (see
            :class:`CycleReport`).
        """
        if outside is None:
            return {"skipped_pass": "outside status unavailable"}
        now = self._clock()
        self._attach_failures = {
            path: at
            for path, at in self._attach_failures.items()
            if at + self._config.cooldown_seconds > now
        }
        report = plan_attachments(
            snapshot.samples,
            statuses,
            outside,
            self._config,
            failures=self._attach_failures,
            now=now,
        )
        summary: dict[str, object] = {
            "planned": [plan.as_dict() for plan in report.planned],
            "attached": [],
            "would_attach": [],
            "failed": {},
            "skipped": dict(report.skipped),
        }
        if not report.planned:
            return summary
        if not self._config.actuation_enabled or self._stopping:
            summary["would_attach"] = [plan.device_path for plan in report.planned]
            return summary
        if not await self._leader.ensure_leader():
            summary["skipped_pass"] = "not leader"
            return summary
        attached: list[str] = []
        failed: dict[str, str] = {}
        for plan in report.planned:
            error = await self._attach_one(plan)
            if error:
                failed[plan.device_path] = error
                self._attach_failures[plan.device_path] = self._clock()
            else:
                attached.append(plan.device_path)
                self._attach_failures.pop(plan.device_path, None)
                self._attached += 1
        summary["attached"] = attached
        summary["failed"] = failed
        return summary

    async def _attach_one(self, plan: AttachPlan) -> str:
        """Issue one add for ``plan`` and classify the outcome.

        Args:
            plan: The planned add.

        Returns:
            ``""`` when the server answered with an ``active`` entry, else
            the error text (also logged at WARNING).
        """
        try:
            added = await self._remote.add_device(
                plan.identity, plan.device_path, plan.size_bytes
            )
        except ClientError as exc:
            logger.warning(
                "attach %s on %s failed: %s",
                plan.device_path,
                plan.identity.instance_id,
                exc,
            )
            return str(exc)[:200]
        if added.device.state != DAX_ACTIVE_STATE:
            error = f"add returned state {added.device.state}"
            logger.warning(
                "attach %s on %s: %s",
                plan.device_path,
                plan.identity.instance_id,
                error,
            )
            return error
        logger.info(
            "attached %s on %s (%d GiB, index %d; outside status confirms ownership)",
            plan.device_path,
            plan.identity.instance_id,
            plan.size_bytes // GIB,
            added.device.index,
        )
        return ""

    def _reconcile_inventory(
        self,
        snapshot: MembershipSnapshot,
        statuses: Mapping[str, DaxHotplugStatus],
    ) -> None:
        """Re-bind managed allocations to current instances and confirm them.

        An MP re-registration (new instance id or epoch for the same worker)
        updates the allocation's ``instance_id``; a device that is no
        longer live is logged but never re-attached from here -- no outside
        POST ever results from reconciliation.

        Args:
            snapshot: The current sandwich read.
            statuses: Per-instance DAX status from
                :meth:`_collect_dax_statuses`.
        """
        by_worker = {
            s.identity.worker_ip: s.identity for s in snapshot.samples.values()
        }
        changed = False
        for allocation in self._document.inventory:
            identity = by_worker.get(allocation.worker_ip)
            if identity is None:
                logger.warning(
                    "managed %s: no accepted instance for worker %s",
                    allocation.device_path,
                    allocation.worker_ip,
                )
                continue
            if identity.instance_id != allocation.instance_id:
                logger.info(
                    "managed %s: instance %s -> %s",
                    allocation.device_path,
                    allocation.instance_id,
                    identity.instance_id,
                )
                allocation.instance_id = identity.instance_id
                changed = True
            dax = statuses.get(identity.instance_id)
            if dax is None:
                continue
            live = dax.find_live(allocation.device_path)
            state = live.state if live is not None else "absent"
            if state != allocation.last_confirmed_state:
                logger.warning(
                    "managed %s on %s: state %s -> %s",
                    allocation.device_path,
                    allocation.instance_id,
                    allocation.last_confirmed_state,
                    state,
                )
                allocation.last_confirmed_state = state
                changed = True
            allocation.last_confirmed_at = self._clock()
            if live is not None:
                allocation.slot_capacity_bytes = live.slot_capacity_bytes
        if changed:
            self._save()
        self._reconciled = True

    # -- active move ---------------------------------------------------------------

    async def _advance(
        self, record: MoveRecord, snapshot: MembershipSnapshot, leader: bool
    ) -> Decision:
        """Gather evidence, decide, and apply exactly one decision."""
        evidence = await self._evidence(record, snapshot, leader)
        decision = decide(record, evidence, self._config)
        await self._apply(record, decision, evidence)
        return decision

    async def _evidence(
        self, record: MoveRecord, snapshot: MembershipSnapshot, leader: bool
    ) -> Evidence:
        """Collect the read-only evidence :func:`decide` needs.

        A GROW has no donor: its donor identity is reported as intact, no
        donor status is read, and its donor capacity is ``None``.
        """
        donor_sample = snapshot.samples.get(record.donor.instance_id)
        receiver_sample = snapshot.samples.get(record.receiver.instance_id)
        on_worker = [
            s.identity
            for s in snapshot.samples.values()
            if s.identity.worker_ip == record.receiver.worker_ip
        ]
        replacement = (
            on_worker[0]
            if len(on_worker) == 1 and on_worker[0] != record.receiver
            else None
        )
        return Evidence(
            now=self._clock(),
            leader=leader,
            coordinator_reachable=snapshot.coordinator_reachable,
            donor_identity_ok=(
                not record.has_donor or snapshot.still_matches(record.donor)
            ),
            receiver_identity_ok=snapshot.still_matches(record.receiver),
            donor_dax=(
                await self._remote.dax_status(record.donor)
                if record.has_donor
                else None
            ),
            receiver_dax=await self._remote.dax_status(record.receiver),
            outside=await self._remote.outside_status(),
            donor_capacity_bytes=(
                donor_sample.capacity_bytes
                if record.has_donor and donor_sample is not None
                else None
            ),
            receiver_capacity_bytes=(
                receiver_sample.capacity_bytes if receiver_sample is not None else None
            ),
            receiver_replacement=replacement,
            receiver_worker_registered=(
                record.receiver.worker_ip in snapshot.registered_worker_ips
            ),
        )

    async def _apply(
        self, record: MoveRecord, decision: Decision, evidence: Evidence
    ) -> None:
        """Apply one decision to the record and persist."""
        if isinstance(decision, Hold):
            logger.debug("move %s hold: %s", record.move_id, decision.reason)
            return
        if isinstance(decision, Persist):
            self._persist_transition(record, decision)
            return
        if isinstance(decision, Block):
            self._block(record, decision.reason)
            return
        if isinstance(decision, Finish):
            self._finish(record, decision)
            return
        await self._perform(record, decision)

    def _persist_transition(self, record: MoveRecord, decision: Persist) -> None:
        """Apply a :class:`Persist` decision."""
        record.state = decision.state
        record.rollback_step = decision.rollback_step
        if decision.confirm_effect is not None:
            ledger = record.effects.get(decision.confirm_effect.value)
            if ledger is not None:
                ledger.confirmed = True
                ledger.confirmed_at = self._clock()
        if decision.unconfirm_effect is not None:
            ledger = record.effects.get(decision.unconfirm_effect.value)
            if ledger is not None:
                ledger.confirmed = False
                ledger.confirmed_at = 0.0
        for name, value in decision.fields.items():
            setattr(record, name, value)
        if decision.receiver is not None:
            record.receiver = decision.receiver
        record.updated_at = self._clock()
        logger.info(
            "move %s -> %s/%s: %s",
            record.move_id,
            record.state.value,
            record.rollback_step.value,
            decision.note,
        )
        self._save()

    def _block(self, record: MoveRecord, reason: str) -> None:
        """Enter BLOCKED (terminal)."""
        record.state = MoveState.BLOCKED
        record.block_reason = reason
        record.updated_at = self._clock()
        self._document.counters.blocked += 1
        logger.error("move %s BLOCKED: %s", record.move_id, reason)
        self._save()

    def _finish(self, record: MoveRecord, decision: Finish) -> None:
        """Enter COMPLETE, update inventory, cooldowns and backoffs, archive.

        One save covers everything: a ``SUCCEEDED`` MOVE swaps the old path
        for the new one in the inventory and a ``SUCCEEDED`` GROW only adds
        the new one; ``NOT_SERVED`` (GROW) changes no inventory and writes no
        cooldown, only the receiver worker's grow backoff (see
        :func:`_grow_backoff_seconds`); every other outcome cools the
        participants (the receiver alone for a GROW).
        """
        now = self._clock()
        record.state = MoveState.COMPLETE
        record.outcome = decision.outcome
        record.updated_at = now
        record.last_error = decision.note
        if decision.warning:
            logger.warning("move %s: %s", record.move_id, decision.warning)
        deallocated = record.effect(EffectName.DEALLOCATE)
        old_gone = deallocated is not None and deallocated.confirmed
        if decision.outcome is MoveOutcome.SUCCEEDED:
            if record.has_donor:
                self._document.inventory = [
                    a
                    for a in self._document.inventory
                    if a.device_path != record.old_path
                ]
            self._document.inventory.append(
                ManagedAllocation(
                    worker_ip=record.receiver.worker_ip,
                    instance_id=record.receiver.instance_id,
                    device_path=record.new_path,
                    allocation_size_gib=record.granted_size_gib,
                    device_map_size_bytes=record.old_map_size_bytes,
                    slot_capacity_bytes=record.new_slot_capacity_bytes,
                    adapter_index=self._config.adapter_index,
                    origin=AllocationOrigin.ALLOCATED,
                    last_confirmed_state="active",
                    last_confirmed_at=now,
                )
            )
            self._document.counters.succeeded += 1
            if not record.has_donor:
                self._document.counters.grown += 1
        elif decision.outcome is MoveOutcome.NOT_SERVED:
            self._document.counters.not_served += 1
            self._document.grow_backoffs[record.receiver.worker_ip] = (
                now + _grow_backoff_seconds(self._config)
            )
        else:
            if old_gone:
                self._document.inventory = [
                    a
                    for a in self._document.inventory
                    if a.device_path != record.old_path
                ]
            if record.restored_path:
                self._document.inventory.append(
                    ManagedAllocation(
                        worker_ip=record.donor.worker_ip,
                        instance_id=record.donor.instance_id,
                        device_path=record.restored_path,
                        allocation_size_gib=record.allocation_size_gib,
                        device_map_size_bytes=record.old_map_size_bytes,
                        slot_capacity_bytes=record.old_slot_capacity_bytes,
                        adapter_index=self._config.adapter_index,
                        origin=AllocationOrigin.RESTORED,
                        last_confirmed_state="active",
                        last_confirmed_at=now,
                    )
                )
            self._document.counters.rolled_back += 1
        if decision.outcome is not MoveOutcome.NOT_SERVED:
            until = now + self._config.cooldown_seconds
            if record.has_donor:
                self._document.cooldowns[record.donor.key] = until
            self._document.cooldowns[record.receiver.key] = until
        self._document.history.append(record)
        del self._document.history[:-_HISTORY_LIMIT]
        self._document.active_move = None
        logger.info(
            "move %s COMPLETE outcome=%s: %s",
            record.move_id,
            record.outcome.value,
            decision.note,
        )
        self._save()

    async def _perform(self, record: MoveRecord, effect: DoEffect) -> None:
        """Persist intent, re-check the gate, issue one POST, persist result.

        An existing ledger entry (an intent that was never dispatched) is
        reused as is -- same request id, same ``before_paths`` -- so a
        re-issued POST is still the single POST of that effect.
        """
        now = self._clock()
        ledger = record.effects.get(effect.effect.value)
        if ledger is None:
            request_id = ""
            if effect.deallocation is not None:
                request_id = effect.deallocation.request_id
            elif effect.allocation is not None:
                request_id = effect.allocation.request_id
            ledger = EffectRecord(
                name=effect.effect,
                request_id=request_id,
                intent_at=now,
                before_paths=list(effect.before_paths),
            )
            record.effects[effect.effect.value] = ledger
            if effect.effect is EffectName.DONOR_DRAIN and not record.drain_started_at:
                record.drain_started_at = now
        record.state = effect.intent_state
        record.rollback_step = effect.rollback_step
        record.updated_at = now
        self._save()  # intent durable before any POST

        # Before the POST: renew, prove the participant identities are still
        # current, then renew once more so a slow sandwich cannot leave us
        # dispatching under an expired or lost Lease.
        if not await self._leader.ensure_leader():
            record.last_error = f"{effect.effect.value}: lost leadership before POST"
            self._save()
            return
        fresh = await self._remote.sandwich()
        needs_receiver = (
            effect.participant is Participant.RECEIVER
            or (
                effect.allocation is not None
                and effect.allocation.target_node == record.receiver.worker_ip
            )
            or (
                effect.deallocation is not None
                and effect.deallocation.target_node == record.receiver.worker_ip
            )
        )
        if not fresh.coordinator_reachable:
            record.last_error = (
                f"{effect.effect.value}: MP Coordinator unreachable before POST"
            )
            self._save()
            return
        if record.has_donor and not fresh.still_matches(record.donor):
            record.last_error = f"{effect.effect.value}: donor identity check failed"
            self._save()
            return
        if needs_receiver and not fresh.still_matches(record.receiver):
            record.last_error = f"{effect.effect.value}: receiver identity check failed"
            self._save()
            return
        if not await self._leader.ensure_leader():
            record.last_error = (
                f"{effect.effect.value}: lost leadership after identity check "
                "before POST"
            )
            self._save()
            return

        if effect.is_outside:
            await self._perform_outside(record, effect, ledger)
        else:
            await self._perform_dax(record, effect, ledger)

    async def _perform_outside(
        self, record: MoveRecord, effect: DoEffect, ledger: EffectRecord
    ) -> None:
        """Issue one outside POST; record response, failure, or ambiguity.

        A connect failure delivered nothing: ``dispatched`` is reset so the
        same ledger (request id, ``before_paths``) is re-issued by a later
        cycle, up to ``get_retry_attempts`` attempts, after which the saga
        blocks. Any other outcome leaves ``dispatched`` set and is never
        followed by another POST for that ledger.
        """
        ledger.attempts += 1
        ledger.dispatched = True
        ledger.error = ""
        ledger.failure = EffectFailure.NONE
        self._save()  # dispatched durable before the POST
        try:
            if effect.deallocation is not None:
                response = await self._remote.deallocate(effect.deallocation)
                ledger.response = _documented(response.model_dump(mode="json"))
            elif effect.allocation is not None:
                allocation = await self._remote.allocate(effect.allocation)
                ledger.response = _documented(allocation.model_dump(mode="json"))
                if effect.effect is EffectName.ALLOCATE:
                    record.new_path = allocation.device_path  # raw, before validation
                else:
                    record.restored_path = allocation.device_path
        except OutsideExplicitFailure as exc:
            ledger.error = f"explicit failure {exc.status_code}"
            ledger.failure = EffectFailure.EXPLICIT
            record.last_error = str(exc)
        except OutsideContractError as exc:
            ledger.error = f"contract violation: {exc}"
            ledger.failure = EffectFailure.CONTRACT
            record.last_error = str(exc)
            if effect.effect is EffectName.ALLOCATE and "device_path" in exc.fields:
                record.new_path = str(exc.fields["device_path"])
        except ClientConnectionError as exc:
            # Provably never sent: not dispatched after all.
            ledger.dispatched = False
            record.last_error = str(exc)
            if ledger.attempts >= self._config.get_retry_attempts:
                self._save()
                self._block(
                    record,
                    f"{effect.effect.value}: outside service unreachable after "
                    f"{ledger.attempts} attempts",
                )
                return
        except AmbiguousMutationError as exc:
            record.last_error = f"{effect.effect.value}: {exc}"
        record.updated_at = self._clock()
        self._save()

    async def _perform_dax(
        self, record: MoveRecord, effect: DoEffect, ledger: EffectRecord
    ) -> None:
        """Issue one DAX POST; the next cycle confirms it from status."""
        identity = (
            record.donor
            if effect.participant is Participant.DONOR
            else (record.receiver)
        )
        ledger.attempts += 1
        ledger.dispatched = True
        ledger.error = ""
        self._save()
        try:
            if effect.remove_mode is not None:
                result = await self._remote.remove_device(
                    identity, effect.device_path, effect.remove_mode
                )
                if isinstance(result, DaxRemoveBlocked):
                    ledger.error = (
                        f"409 busy: locked={result.locked_key_count} "
                        f"borrowed={result.borrowed_slot_count}"
                    )
                elif isinstance(result, DaxDeviceNotFound):
                    ledger.error = "404 device not found"
                else:
                    ledger.response = _documented(result.model_dump(mode="json"))
            else:
                added = await self._remote.add_device(
                    identity, effect.device_path, effect.size_bytes
                )
                ledger.response = {
                    "status": added.status,
                    "state": added.device.state,
                    "index": added.device.index,
                }
        except ClientError as exc:
            ledger.error = str(exc)[:200]
            record.last_error = str(exc)
        record.updated_at = self._clock()
        self._save()

    def _save(self) -> None:
        """Persist the document (raises :class:`JournalError` on failure)."""
        try:
            self._journal.save(self._document)
        except OSError as exc:
            raise JournalError(f"cannot write journal: {exc}") from exc


def _grow_backoff_seconds(config: MPMemoryCoordinatorConfig) -> float:
    """Duration of the grow backoff written by a ``NOT_SERVED`` finish.

    ``cooldown_seconds``, but never less than two idle polls: the backoff
    must still be active in the idle cycle that follows the finish (one
    ``poll_interval_seconds`` of sleep plus that cycle's own reads), or that
    cycle would propose the same refused GROW again instead of running the
    donor search, forever.

    Args:
        config: For ``cooldown_seconds`` and ``poll_interval_seconds``.

    Returns:
        Seconds from the finish until the worker may be proposed a GROW.
    """
    return max(config.cooldown_seconds, 2 * config.poll_interval_seconds)


def _unique_rejections(rejections: list[Rejection]) -> list[Rejection]:
    """Return ``rejections`` without repeats, first occurrence order kept.

    The GROW pass and the MOVE pass can reject the same receiver for the
    same reason (once per donor in the MOVE pass); the report lists it once.
    """
    seen: set[Rejection] = set()
    unique: list[Rejection] = []
    for rejection in rejections:
        if rejection not in seen:
            seen.add(rejection)
            unique.append(rejection)
    return unique


def _documented(dump: dict[str, object]) -> dict[str, str | int | float | bool]:
    """Keep only scalar members of a response dump for the ledger."""
    return {k: v for k, v in dump.items() if isinstance(v, (str, int, float, bool))}


def _describe(decision: Decision) -> str:
    """One-line description of a decision for the cycle report."""
    if isinstance(decision, Hold):
        return f"hold: {decision.reason}"
    if isinstance(decision, Persist):
        return (
            f"persist {decision.state.value}/{decision.rollback_step.value}: "
            f"{decision.note}"
        )
    if isinstance(decision, DoEffect):
        return f"effect {decision.effect.value}"
    if isinstance(decision, Block):
        return f"block: {decision.reason}"
    return f"finish {decision.outcome.value}: {decision.note}"
