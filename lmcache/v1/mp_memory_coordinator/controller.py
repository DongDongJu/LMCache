# SPDX-License-Identifier: Apache-2.0
"""The control loop: observe, propose, and drive one move at a time.

Each cycle performs a sandwich read. With no active move it reads each
relevant MP server's DAX status once, discovers newly owned devices from
outside status (see :mod:`lmcache.v1.mp_memory_coordinator.discovery`),
reconciles the inventory, updates the pressure history, ranks candidates,
preflights the best pair, and logs a structured dry-run proposal; with
``actuation_enabled`` and leadership it persists a ``SELECTED`` record. With
an active move it gathers :class:`Evidence`, asks :func:`decide` for the one
next action, executes at most one side effect, and persists the result --
the persist-before-effect discipline of the design doc.

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
    EffectName,
    EffectRecord,
    InstanceIdentity,
    JournalDocument,
    ManagedAllocation,
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
    LivePreflight,
    MembershipSnapshot,
    MoveProposal,
    PressureHistory,
    Rejection,
    RejectionReason,
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
        proposal: The dry-run proposal, if any.
        move_id: Active move id (``""`` if none).
        move_state: Active move state (``""`` if none).
        decision: Decision type and reason for an active move.
        discovery: What device discovery adopted and skipped this cycle
            (empty while a move is active, when discovery does not run).
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

    def request_stop(self) -> None:
        """Start no new move; the current cycle completes and persists."""
        self._stopping = True

    def readiness(self) -> tuple[bool, str]:
        """Return ``(ready, reason)`` for ``/readyz``.

        Ready means: journal loaded, current leader, MP Coordinator reached
        on the last cycle, inventory reconciled, and no BLOCKED move.
        """
        if self._journal_error:
            return False, f"journal: {self._journal_error}"
        if not self._loaded:
            return False, "journal not loaded"
        if not self._leader.is_leader():
            return False, "not leader"
        if not self._last_report.coordinator_reachable:
            return False, "MP Coordinator unreachable"
        if not self._reconciled:
            return False, "inventory not reconciled"
        move = self._document.active_move
        if move is not None and move.state is MoveState.BLOCKED:
            return False, f"move {move.move_id} BLOCKED: {move.block_reason}"
        return True, "ok"

    def status(self) -> dict[str, object]:
        """Return a JSON-serializable status for ``/status``."""
        move = self._document.active_move
        return {
            "leader": self._leader.is_leader(),
            "leader_identity": self._leader.identity,
            "actuation_enabled": self._config.actuation_enabled,
            "journal_error": self._journal_error,
            "initialized": self._document.initialized,
            "inventory": [a.model_dump(mode="json") for a in self._document.inventory],
            "cooldowns": dict(self._document.cooldowns),
            "history": self._history.snapshot(),
            "active_move": move.model_dump(mode="json") if move is not None else None,
            "counters": self._document.counters.model_dump(),
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
        outside = await self._remote.outside_status()
        report.discovery = self._discover(snapshot, statuses, outside)
        self._reconcile_inventory(snapshot, statuses)
        candidates = rank_candidates(
            snapshot, self._history, self._document.cooldowns, self._clock()
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
        record = new_move_record(proposal, self._clock())
        self._document.active_move = record
        self._save()
        report.move_id = record.move_id
        logger.info(
            "move %s SELECTED: %s", record.move_id, json.dumps(proposal.as_dict())
        )
        decision = await self._advance(record, snapshot, True)
        report.decision = _describe(decision)
        move = self._document.active_move
        report.move_state = move.state.value if move is not None else "COMPLETE"

    async def _evaluate(
        self, candidates
    ) -> tuple[MoveProposal | None, list[Rejection]]:
        """Preflight ranked pairs; return the first proposal or the rejections."""
        rejections: list[Rejection] = []
        preflights: dict[str, LivePreflight | None] = {}

        async def _get(identity: InstanceIdentity) -> LivePreflight | None:
            if identity.instance_id not in preflights:
                preflights[identity.instance_id] = await self._remote.preflight(
                    identity
                )
            return preflights[identity.instance_id]

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
                    return result, rejections
                rejections.extend(result)
        return None, rejections

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
        """Collect the read-only evidence :func:`decide` needs."""
        donor_sample = snapshot.samples.get(record.donor.instance_id)
        receiver_sample = snapshot.samples.get(record.receiver.instance_id)
        return Evidence(
            now=self._clock(),
            leader=leader,
            coordinator_reachable=snapshot.coordinator_reachable,
            donor_identity_ok=snapshot.still_matches(record.donor),
            receiver_identity_ok=snapshot.still_matches(record.receiver),
            donor_dax=await self._remote.dax_status(record.donor),
            receiver_dax=await self._remote.dax_status(record.receiver),
            outside=await self._remote.outside_status(),
            donor_capacity_bytes=(
                donor_sample.capacity_bytes if donor_sample is not None else None
            ),
            receiver_capacity_bytes=(
                receiver_sample.capacity_bytes if receiver_sample is not None else None
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
        for name, value in decision.fields.items():
            setattr(record, name, value)
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
        """Enter COMPLETE, update inventory and cooldowns, archive."""
        now = self._clock()
        record.state = MoveState.COMPLETE
        record.outcome = decision.outcome
        record.updated_at = now
        record.last_error = decision.note
        deallocated = record.effect(EffectName.DEALLOCATE)
        old_gone = deallocated is not None and deallocated.confirmed
        if decision.outcome is MoveOutcome.SUCCEEDED:
            self._document.inventory = [
                a for a in self._document.inventory if a.device_path != record.old_path
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
        until = now + self._config.cooldown_seconds
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
        """Persist intent, re-check the gate, issue one POST, persist result."""
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

        # Immediately before the POST: leadership and identity again.
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
        if not fresh.coordinator_reachable or not fresh.still_matches(record.donor):
            record.last_error = f"{effect.effect.value}: donor identity check failed"
            self._save()
            return
        if needs_receiver and not fresh.still_matches(record.receiver):
            record.last_error = f"{effect.effect.value}: receiver identity check failed"
            self._save()
            return

        if effect.is_outside:
            await self._perform_outside(record, effect, ledger)
        else:
            await self._perform_dax(record, effect, ledger)

    async def _perform_outside(
        self, record: MoveRecord, effect: DoEffect, ledger: EffectRecord
    ) -> None:
        """Issue one outside POST; record response, failure, or ambiguity."""
        ledger.attempts += 1
        ledger.dispatched = True
        ledger.error = ""
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
            record.last_error = str(exc)
        except OutsideContractError as exc:
            ledger.error = f"contract violation: {exc}"
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
