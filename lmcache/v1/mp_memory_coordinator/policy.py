# SPDX-License-Identifier: Apache-2.0
"""Observation and the dry-run policy.

Observation: the MP Coordinator's ``/instances`` and ``/instances/usage``
are separate snapshots, so every accepted sample is built from a *sandwich*
read (``instances A -> instances/usage -> instances B``). An instance is
accepted only when its ``registration_time``, ``ip``, ``http_port`` and
``metadata.worker_ip`` match in A and B, and its usage row is registered,
has declared capacity, and carries a non-null private ``l2/dax`` ratio.
Everything else is a typed :class:`Rejection` that the dry-run log reports.

Policy: pressure history, candidate ranking, live preflight rules, GROW
size derivation, and donor device selection. Everything after the I/O
helpers is pure: it takes samples, live status documents, the managed
inventory, cooldowns and grow backoffs, and returns either a proposal or
typed rejections. Selection is deterministic: receivers rank by descending
ratio then ``instance_id``, donors by ascending ratio then ``instance_id``.
The controller first asks :func:`evaluate_grow` for each receiver, best
first (grow before move); only when no receiver can grow does it run
:func:`evaluate_pair` over donor/receiver pairs, and the first pair that
passes every check is proposed.
"""

# Standard
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.clients import ClientError
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_server_client import MPServerClient
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_ACTIVE_STATE,
    DAX_BACKEND,
    GIB,
    CoordinatorInstance,
    DaxDeviceStatus,
    DaxHotplugStatus,
    DaxReconfigureStatus,
    InstanceIdentity,
    InstanceSample,
    InstanceUsage,
    ManagedAllocation,
    MoveKind,
    MPStatus,
)

logger = init_logger(__name__)


class RejectionReason(str, Enum):
    """Why an instance (or a candidate pair) was not eligible."""

    COORDINATOR_UNREACHABLE = "coordinator_unreachable"
    NOT_IN_BOTH_READS = "not_in_both_reads"
    IDENTITY_CHANGED = "identity_changed_between_reads"
    MISSING_WORKER_IP = "missing_worker_ip"
    DUPLICATE_WORKER_IP = "duplicate_worker_ip"
    MISSING_USAGE = "missing_usage"
    UNREGISTERED = "unregistered"
    UNDECLARED_CAPACITY = "undeclared_capacity"
    NO_PRIVATE_DAX = "no_private_l2_dax"
    NULL_RATIO = "null_usage_ratio"
    HISTORY_NOT_STABLE = "history_not_stable"
    COOLDOWN = "cooldown"
    MOVE_IN_PROGRESS = "move_in_progress"
    PREFLIGHT_UNAVAILABLE = "preflight_unavailable"
    PREFLIGHT_FAILED = "preflight_failed"
    LIVE_RATIO_MISMATCH = "live_ratio_mismatch"
    NO_MANAGED_DEVICE = "no_managed_runtime_device"
    MIN_DEVICES = "min_devices_per_instance"
    PROJECTED_DONOR_RATIO = "projected_donor_ratio"
    INSUFFICIENT_GAP = "insufficient_ratio_gap"
    GROW_BACKOFF = "grow_backoff"
    GROW_SIZE_UNDETERMINABLE = "grow_size_undeterminable"
    ACTUATION_DISABLED = "actuation_disabled"
    NOT_LEADER = "not_leader"


@dataclass(frozen=True)
class Rejection:
    """One structured rejection for the dry-run log.

    Attributes:
        instance_id: The instance (or ``"<donor>/<receiver>"`` pair).
        reason: The reason code.
        detail: Free-form detail.
    """

    instance_id: str
    reason: RejectionReason
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-friendly form."""
        return {
            "instance_id": self.instance_id,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class MembershipSnapshot:
    """The result of one sandwich read.

    Attributes:
        coordinator_reachable: ``False`` when any of the three reads
            failed; then ``samples`` is empty and no mutation may follow.
        samples: Accepted samples keyed by ``instance_id``.
        rejections: Every instance that was seen but not accepted.
        sampled_at: Wall-clock time of the read.
        registered_worker_ips: ``worker_ip`` of every instance seen in
            either ``/instances`` read, accepted or rejected (instances
            without a ``worker_ip`` cannot be attributed and are absent).
            Raw membership: "is anything registered on that worker?"
    """

    coordinator_reachable: bool
    samples: dict[str, InstanceSample] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)
    sampled_at: float = 0.0
    registered_worker_ips: frozenset[str] = frozenset()

    def identity_of(self, instance_id: str) -> InstanceIdentity | None:
        """Return the accepted identity of ``instance_id``, if any."""
        sample = self.samples.get(instance_id)
        return sample.identity if sample is not None else None

    def still_matches(self, identity: InstanceIdentity) -> bool:
        """Whether ``identity`` is accepted in this snapshot unchanged."""
        return self.identity_of(identity.instance_id) == identity


@dataclass(frozen=True)
class LivePreflight:
    """One MP server's live ``/status`` and ``/reconfigure/dax/status``."""

    status: MPStatus
    dax: DaxReconfigureStatus


async def read_sandwich(
    coordinator: MPCoordinatorClient, clock: Callable[[], float]
) -> MembershipSnapshot:
    """Perform one sandwich read and join membership with usage.

    Args:
        coordinator: The MP Coordinator client.
        clock: Wall-clock source for ``sampled_at``.

    Returns:
        The snapshot. A transport or schema failure on any of the three
        reads yields ``coordinator_reachable=False`` with no samples.
    """
    try:
        first = await coordinator.get_instances()
        usage = await coordinator.get_fleet_usage()
        second = await coordinator.get_instances()
    except ClientError as exc:
        logger.warning("sandwich read failed: %s", exc)
        return MembershipSnapshot(coordinator_reachable=False, sampled_at=clock())
    return join_sandwich(first.instances, usage.instances, second.instances, clock())


def join_sandwich(
    first: list[CoordinatorInstance],
    usage_rows: list[InstanceUsage],
    second: list[CoordinatorInstance],
    sampled_at: float,
) -> MembershipSnapshot:
    """Join the three reads of a sandwich into accepted samples.

    Pure; see the module docstring for the acceptance rules.

    Args:
        first: ``/instances`` read before the usage read.
        usage_rows: ``/instances/usage`` rows.
        second: ``/instances`` read after the usage read.
        sampled_at: Wall-clock time to stamp on the samples.

    Returns:
        The snapshot with ``coordinator_reachable=True``.
    """
    rejections: list[Rejection] = []
    after = {instance.instance_id: instance for instance in second}
    usage = {row.instance_id: row for row in usage_rows}
    identities: dict[str, InstanceIdentity] = {}
    for instance in first:
        other = after.get(instance.instance_id)
        if other is None:
            rejections.append(
                Rejection(instance.instance_id, RejectionReason.NOT_IN_BOTH_READS)
            )
            continue
        if (
            instance.registration_time != other.registration_time
            or instance.ip != other.ip
            or instance.http_port != other.http_port
            or instance.worker_ip != other.worker_ip
        ):
            rejections.append(
                Rejection(instance.instance_id, RejectionReason.IDENTITY_CHANGED)
            )
            continue
        if not instance.worker_ip:
            rejections.append(
                Rejection(instance.instance_id, RejectionReason.MISSING_WORKER_IP)
            )
            continue
        identities[instance.instance_id] = InstanceIdentity(
            instance_id=instance.instance_id,
            registration_time=instance.registration_time,
            endpoint=instance.endpoint,
            worker_ip=instance.worker_ip,
        )

    by_worker: dict[str, list[str]] = {}
    for instance_id, identity in identities.items():
        by_worker.setdefault(identity.worker_ip, []).append(instance_id)
    for worker_ip, owners in by_worker.items():
        if len(owners) > 1:
            for instance_id in owners:
                rejections.append(
                    Rejection(
                        instance_id,
                        RejectionReason.DUPLICATE_WORKER_IP,
                        f"worker_ip {worker_ip} claimed by {sorted(owners)}",
                    )
                )
                identities.pop(instance_id, None)

    samples: dict[str, InstanceSample] = {}
    for instance_id, identity in identities.items():
        row = usage.get(instance_id)
        if row is None:
            rejections.append(Rejection(instance_id, RejectionReason.MISSING_USAGE))
            continue
        if not row.registered:
            rejections.append(Rejection(instance_id, RejectionReason.UNREGISTERED))
            continue
        if not row.declared_capacity:
            rejections.append(
                Rejection(instance_id, RejectionReason.UNDECLARED_CAPACITY)
            )
            continue
        module = row.private_dax()
        if module is None:
            rejections.append(Rejection(instance_id, RejectionReason.NO_PRIVATE_DAX))
            continue
        if module.usage_ratio is None:
            rejections.append(Rejection(instance_id, RejectionReason.NULL_RATIO))
            continue
        samples[instance_id] = InstanceSample(
            identity=identity,
            registered=row.registered,
            declared_capacity=row.declared_capacity,
            used_bytes=module.used_bytes,
            capacity_bytes=module.capacity_bytes,
            usage_ratio=module.usage_ratio,
            sampled_at=sampled_at,
        )
    return MembershipSnapshot(
        coordinator_reachable=True,
        samples=samples,
        rejections=rejections,
        sampled_at=sampled_at,
        registered_worker_ips=frozenset(
            instance.worker_ip for instance in (*first, *second) if instance.worker_ip
        ),
    )


async def fetch_preflight(
    mp_client: MPServerClient, identity: InstanceIdentity
) -> LivePreflight | None:
    """Read one MP server's ``/status`` and ``/reconfigure/dax/status``.

    Args:
        mp_client: The shared MP server client.
        identity: Which server (its ``endpoint``).

    Returns:
        Both documents, or ``None`` if either read failed (the caller
        rejects the instance with ``PREFLIGHT_UNAVAILABLE``).
    """
    try:
        status = await mp_client.get_status(identity.base_url)
        dax = await mp_client.get_dax_status(identity.base_url)
    except ClientError as exc:
        logger.warning("preflight of %s failed: %s", identity.instance_id, exc)
        return None
    return LivePreflight(status=status, dax=dax)


_DAX_REQUIRED_OPERATIONS = ("status", "add", "remove")


class PressureLevel(str, Enum):
    """Classification of one ``used/capacity`` ratio."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"


def classify(ratio: float, config: MPMemoryCoordinatorConfig) -> PressureLevel:
    """Classify a ratio against the configured LOW/HIGH thresholds.

    Args:
        ratio: ``used_bytes / capacity_bytes``.
        config: Thresholds.

    Returns:
        ``HIGH`` when ``ratio >= high_ratio``, ``LOW`` when
        ``ratio <= low_ratio``, otherwise ``NORMAL``.
    """
    if ratio >= config.high_ratio:
        return PressureLevel.HIGH
    if ratio <= config.low_ratio:
        return PressureLevel.LOW
    return PressureLevel.NORMAL


@dataclass
class _HistoryEntry:
    """Consecutive same-level sample count for one identity."""

    identity: InstanceIdentity
    level: PressureLevel
    count: int


class PressureHistory:
    """Consecutive-sample history keyed by full instance identity.

    Any identity change (instance id, registration epoch, endpoint, or
    worker IP) or a gap in accepted samples resets the count, so a move is
    proposed only after ``stable_samples`` consecutive accepted samples of
    the same instance with the same classification.
    """

    def __init__(self, stable_samples: int) -> None:
        """Args:
        stable_samples: Consecutive samples required for stability.
        """
        self._stable_samples = stable_samples
        self._entries: dict[str, _HistoryEntry] = {}

    def observe(
        self, snapshot: MembershipSnapshot, config: MPMemoryCoordinatorConfig
    ) -> dict[str, PressureLevel]:
        """Record one snapshot.

        Instances missing from the snapshot (rejected or gone) lose their
        history; an unreachable coordinator resets everything.

        Args:
            snapshot: The accepted samples of this cycle.
            config: Thresholds.

        Returns:
            ``instance_id`` -> level recorded this cycle.
        """
        if not snapshot.coordinator_reachable:
            self._entries.clear()
            return {}
        levels: dict[str, PressureLevel] = {}
        for instance_id, sample in snapshot.samples.items():
            if sample.usage_ratio is None:
                continue
            level = classify(sample.usage_ratio, config)
            entry = self._entries.get(instance_id)
            if (
                entry is not None
                and entry.identity == sample.identity
                and entry.level == level
            ):
                entry.count += 1
            else:
                self._entries[instance_id] = _HistoryEntry(sample.identity, level, 1)
            levels[instance_id] = level
        for instance_id in list(self._entries):
            if instance_id not in snapshot.samples:
                del self._entries[instance_id]
        return levels

    def stable_level(self, identity: InstanceIdentity) -> PressureLevel | None:
        """Return the level ``identity`` has held for ``stable_samples``.

        Args:
            identity: The full identity to look up.

        Returns:
            The stable level, or ``None`` when the history is shorter,
            belongs to a different identity, or is absent.
        """
        entry = self._entries.get(identity.instance_id)
        if entry is None or entry.identity != identity:
            return None
        if entry.count < self._stable_samples:
            return None
        return entry.level

    def count(self, identity: InstanceIdentity) -> int:
        """Return the consecutive count for ``identity`` (``0`` if none)."""
        entry = self._entries.get(identity.instance_id)
        if entry is None or entry.identity != identity:
            return 0
        return entry.count

    def snapshot(self) -> dict[str, dict[str, str | int]]:
        """Return a JSON-friendly view for logs and the status endpoint."""
        return {
            instance_id: {"level": entry.level.value, "count": entry.count}
            for instance_id, entry in self._entries.items()
        }


@dataclass(frozen=True)
class Candidates:
    """Ranked, history-stable, cooldown-free candidates of one cycle.

    Attributes:
        receivers: Stable-HIGH samples, best first.
        donors: Stable-LOW samples, best first.
        rejections: Why other accepted samples were not candidates.
    """

    receivers: list[InstanceSample] = field(default_factory=list)
    donors: list[InstanceSample] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)


def rank_candidates(
    snapshot: MembershipSnapshot,
    history: PressureHistory,
    cooldowns: dict[str, float],
    now: float,
) -> Candidates:
    """Rank stable receivers and donors.

    Args:
        snapshot: The accepted samples of this cycle.
        history: History already updated with ``snapshot``.
        cooldowns: Identity key -> time until which the instance is cooling.
        now: Current wall-clock time.

    Returns:
        The ranked candidates; ties break on ``instance_id``.
    """
    receivers: list[InstanceSample] = []
    donors: list[InstanceSample] = []
    rejections: list[Rejection] = []
    for instance_id, sample in snapshot.samples.items():
        level = history.stable_level(sample.identity)
        if level is None:
            rejections.append(
                Rejection(
                    instance_id,
                    RejectionReason.HISTORY_NOT_STABLE,
                    f"count={history.count(sample.identity)}",
                )
            )
            continue
        until = cooldowns.get(sample.identity.key, 0.0)
        if until > now:
            rejections.append(
                Rejection(instance_id, RejectionReason.COOLDOWN, f"until={until:.0f}")
            )
            continue
        if level is PressureLevel.HIGH:
            receivers.append(sample)
        elif level is PressureLevel.LOW:
            donors.append(sample)
    receivers.sort(key=lambda s: (-(s.usage_ratio or 0.0), s.identity.instance_id))
    donors.sort(key=lambda s: ((s.usage_ratio or 0.0), s.identity.instance_id))
    return Candidates(receivers=receivers, donors=donors, rejections=rejections)


def preflight_problems(
    preflight: LivePreflight, config: MPMemoryCoordinatorConfig
) -> list[str]:
    """Check one MP server's live status against the move preconditions.

    Requires a healthy engine and storage manager, exactly one DAX L2
    adapter that is healthy, not closing, and hot-pluggable, and a DAX
    reconfiguration status with exactly one adapter at backend-local
    index 0 supporting status/add/remove whose every non-tombstone device
    is healthy, not closing, and active.

    Args:
        preflight: The live documents.
        config: For the adapter index.

    Returns:
        Human-readable problems; empty when the server passes.
    """
    problems: list[str] = []
    status = preflight.status
    if not status.is_healthy:
        problems.append("engine is_healthy=false")
    if not status.storage_manager.is_healthy:
        problems.append("storage_manager is_healthy=false")
    dax_adapters = [
        a for a in status.storage_manager.l2_adapters if a.type == DAX_BACKEND
    ]
    if len(dax_adapters) != 1:
        problems.append(
            f"expected exactly one DAX L2 adapter, found {len(dax_adapters)}"
        )
    else:
        adapter = dax_adapters[0]
        if not adapter.is_healthy:
            problems.append("DAX adapter is_healthy=false")
        if adapter.closing:
            problems.append("DAX adapter closing=true")
        if not adapter.hotplug_enabled:
            problems.append("DAX adapter hotplug_enabled=false")

    dax = preflight.dax
    if not dax.enabled:
        problems.append("dax reconfigure enabled=false")
    if dax.num_adapters != 1 or len(dax.adapters) != 1:
        problems.append(f"dax num_adapters={dax.num_adapters}, expected 1")
        return problems
    reconf = dax.adapters[0]
    if reconf.adapter_index != config.adapter_index:
        problems.append(f"dax adapter_index={reconf.adapter_index}")
    if reconf.backend != DAX_BACKEND:
        problems.append(f"dax backend={reconf.backend!r}")
    missing = [
        op for op in _DAX_REQUIRED_OPERATIONS if op not in reconf.supported_operations
    ]
    if missing:
        problems.append(f"dax adapter lacks operations {missing}")
    if not reconf.status.hotplug_enabled:
        problems.append("dax status hotplug_enabled=false")
    for device in reconf.status.live_devices():
        if not device.is_healthy or device.closing or device.state != DAX_ACTIVE_STATE:
            problems.append(
                f"device {device.device_path} state={device.state} "
                f"healthy={device.is_healthy} closing={device.closing}"
            )
    return problems


def live_ratio(dax: DaxHotplugStatus) -> float | None:
    """Return ``total_used_bytes / total_capacity_bytes`` of a live status.

    Args:
        dax: The adapter's hotplug status.

    Returns:
        The ratio, or ``None`` when the capacity is zero.
    """
    if dax.total_capacity_bytes <= 0:
        return None
    return dax.total_used_bytes / dax.total_capacity_bytes


@dataclass(frozen=True)
class DeviceChoice:
    """The donor device chosen for a move."""

    device: DaxDeviceStatus
    allocation: ManagedAllocation


def choose_donor_device(
    donor: InstanceIdentity,
    dax: DaxHotplugStatus,
    inventory: list[ManagedAllocation],
    config: MPMemoryCoordinatorConfig,
) -> DeviceChoice | Rejection:
    """Pick the least-used movable managed device of a donor.

    A device is movable when its DAX index is ``> 0``, it is active and
    healthy, its path is in the managed inventory under the donor's worker
    IP with a whole-GiB map that equals the allocation size, and it starts
    with the allowed prefix. Removing it must leave at least
    ``min_devices_per_instance`` active devices.

    Args:
        donor: The donor identity.
        dax: The donor's live hotplug status.
        inventory: The full managed inventory.
        config: Limits.

    Returns:
        The choice, or a rejection naming the first failed rule.
    """
    managed = {a.device_path: a for a in inventory if a.worker_ip == donor.worker_ip}
    active = [d for d in dax.live_devices() if d.state == DAX_ACTIVE_STATE]
    movable: list[DeviceChoice] = []
    for device in active:
        allocation = managed.get(device.device_path)
        if device.index <= 0 or allocation is None:
            continue
        if allocation.adapter_index != config.adapter_index:
            continue
        if not device.device_path.startswith(config.allowed_device_path_prefix):
            continue
        if not device.is_healthy or device.closing:
            continue
        if device.max_dax_size_bytes != allocation.device_map_size_bytes:
            continue
        if not allocation.is_size_consistent():
            continue
        movable.append(DeviceChoice(device=device, allocation=allocation))
    if not movable:
        return Rejection(
            donor.instance_id,
            RejectionReason.NO_MANAGED_DEVICE,
            f"active={[d.device_path for d in active]} managed={sorted(managed)}",
        )
    if len(active) - 1 < config.min_devices_per_instance:
        return Rejection(
            donor.instance_id,
            RejectionReason.MIN_DEVICES,
            f"active={len(active)} min={config.min_devices_per_instance}",
        )
    movable.sort(key=lambda c: (c.device.live_slot_count, c.device.device_path))
    return movable[0]


@dataclass(frozen=True)
class MoveProposal:
    """A fully checked move the controller may start.

    Attributes:
        donor: The donor sample.
        receiver: The receiver sample.
        choice: The donor device and its managed allocation.
        donor_live_capacity_bytes: Donor DAX ``total_capacity_bytes``.
        receiver_live_capacity_bytes: Receiver DAX ``total_capacity_bytes``.
        projected_donor_ratio: Donor ratio after removal.
    """

    donor: InstanceSample
    receiver: InstanceSample
    choice: DeviceChoice
    donor_live_capacity_bytes: int
    receiver_live_capacity_bytes: int
    projected_donor_ratio: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-friendly form for the dry-run log (``kind: move``)."""
        return {
            "kind": MoveKind.MOVE.value,
            "donor": self.donor.identity.instance_id,
            "donor_worker_ip": self.donor.identity.worker_ip,
            "receiver": self.receiver.identity.instance_id,
            "receiver_worker_ip": self.receiver.identity.worker_ip,
            "device_path": self.choice.device.device_path,
            "device_index": self.choice.device.index,
            "allocation_size_gib": self.choice.allocation.allocation_size_gib,
            "projected_donor_ratio": round(self.projected_donor_ratio, 4),
        }


def evaluate_pair(
    donor: InstanceSample,
    receiver: InstanceSample,
    donor_preflight: LivePreflight,
    receiver_preflight: LivePreflight,
    inventory: list[ManagedAllocation],
    config: MPMemoryCoordinatorConfig,
) -> MoveProposal | list[Rejection]:
    """Apply every live check to one donor/receiver pair.

    Args:
        donor: Stable-LOW sample.
        receiver: Stable-HIGH sample.
        donor_preflight: Donor's live documents.
        receiver_preflight: Receiver's live documents.
        inventory: The managed inventory.
        config: Thresholds and limits.

    Returns:
        A proposal, or the rejections that stopped it (never empty).
    """
    rejections: list[Rejection] = []
    for sample, preflight in ((donor, donor_preflight), (receiver, receiver_preflight)):
        problems = preflight_problems(preflight, config)
        if problems:
            rejections.append(
                Rejection(
                    sample.identity.instance_id,
                    RejectionReason.PREFLIGHT_FAILED,
                    "; ".join(problems),
                )
            )
    if rejections:
        return rejections

    donor_dax = donor_preflight.dax.adapters[0].status
    receiver_dax = receiver_preflight.dax.adapters[0].status
    donor_live = live_ratio(donor_dax)
    receiver_live = live_ratio(receiver_dax)
    if donor_live is None or classify(donor_live, config) is not PressureLevel.LOW:
        rejections.append(
            Rejection(
                donor.identity.instance_id,
                RejectionReason.LIVE_RATIO_MISMATCH,
                f"live={donor_live}",
            )
        )
    if (
        receiver_live is None
        or classify(receiver_live, config) is not PressureLevel.HIGH
    ):
        rejections.append(
            Rejection(
                receiver.identity.instance_id,
                RejectionReason.LIVE_RATIO_MISMATCH,
                f"live={receiver_live}",
            )
        )
    if rejections:
        return rejections
    if (receiver.usage_ratio or 0.0) - (donor.usage_ratio or 0.0) < (
        config.minimum_ratio_gap
    ):
        return [
            Rejection(
                f"{donor.identity.instance_id}/{receiver.identity.instance_id}",
                RejectionReason.INSUFFICIENT_GAP,
                f"gap={(receiver.usage_ratio or 0.0) - (donor.usage_ratio or 0.0):.3f}",
            )
        ]

    choice = choose_donor_device(donor.identity, donor_dax, inventory, config)
    if isinstance(choice, Rejection):
        return [choice]
    remaining = donor_dax.total_capacity_bytes - choice.device.slot_capacity_bytes
    if remaining <= 0:
        return [
            Rejection(
                donor.identity.instance_id,
                RejectionReason.PROJECTED_DONOR_RATIO,
                f"remaining capacity {remaining} <= 0",
            )
        ]
    projected = donor_dax.total_used_bytes / remaining
    if projected > config.projected_donor_max_ratio:
        return [
            Rejection(
                donor.identity.instance_id,
                RejectionReason.PROJECTED_DONOR_RATIO,
                f"projected={projected:.3f} max={config.projected_donor_max_ratio}",
            )
        ]
    return MoveProposal(
        donor=donor,
        receiver=receiver,
        choice=choice,
        donor_live_capacity_bytes=donor_dax.total_capacity_bytes,
        receiver_live_capacity_bytes=receiver_dax.total_capacity_bytes,
        projected_donor_ratio=projected,
    )


# -- GROW: receiver-only proposal ----------------------------------------------------


class GrowSizeSource(str, Enum):
    """Where a GROW request size came from (logged with the proposal)."""

    RECEIVER_INVENTORY = "receiver_inventory"
    FLEET_INVENTORY = "fleet_inventory"
    BOOTSTRAP_DEVICE = "bootstrap_device"


@dataclass(frozen=True)
class GrowSize:
    """The derived GROW request size.

    Attributes:
        size_gib: Whole GiB to request from the outside service.
        source: Which derivation tier produced it.
    """

    size_gib: int
    source: GrowSizeSource


def _most_common_size(allocations: list[ManagedAllocation]) -> int:
    """Return the most common consistent allocation size (ties: largest, 0 if none)."""
    sizes = Counter(
        a.allocation_size_gib for a in allocations if a.is_size_consistent()
    )
    if not sizes:
        return 0
    return max(sizes, key=lambda size: (sizes[size], size))


def derive_grow_size(
    receiver: InstanceIdentity,
    dax: DaxHotplugStatus,
    inventory: list[ManagedAllocation],
) -> GrowSize | Rejection:
    """Derive the size a GROW should request for ``receiver``.

    The frozen outside API exposes neither device sizes nor free capacity
    and serves exact-size matches only, so the size is taken from what the
    allocator has demonstrably served before, in order: (a) the most common
    size among size-consistent managed allocations on the receiver's worker,
    (b) the most common size across the whole managed inventory, (c) the
    receiver's live index-0 (bootstrap) device map size rounded down to whole
    GiB. Ties prefer the larger size. No configuration key is involved.

    Args:
        receiver: The receiver identity (its ``worker_ip`` and id).
        dax: The receiver's live hotplug status.
        inventory: The full managed inventory.

    Returns:
        The size and its source, or a ``GROW_SIZE_UNDETERMINABLE`` rejection
        when no tier yields a positive whole GiB.
    """
    local = _most_common_size(
        [a for a in inventory if a.worker_ip == receiver.worker_ip]
    )
    if local > 0:
        return GrowSize(local, GrowSizeSource.RECEIVER_INVENTORY)
    fleet = _most_common_size(inventory)
    if fleet > 0:
        return GrowSize(fleet, GrowSizeSource.FLEET_INVENTORY)
    bootstrap = [d for d in dax.live_devices() if d.index == 0]
    if bootstrap and bootstrap[0].max_dax_size_bytes // GIB > 0:
        return GrowSize(
            bootstrap[0].max_dax_size_bytes // GIB, GrowSizeSource.BOOTSTRAP_DEVICE
        )
    return Rejection(
        receiver.instance_id,
        RejectionReason.GROW_SIZE_UNDETERMINABLE,
        "no managed allocation size and no whole-GiB bootstrap device",
    )


@dataclass(frozen=True)
class GrowProposal:
    """A fully checked GROW the controller may start: allocate, then add.

    Attributes:
        receiver: The receiver sample.
        request_size_gib: Outside allocation size to request.
        size_source: Which tier derived ``request_size_gib``.
        receiver_live_capacity_bytes: Receiver DAX ``total_capacity_bytes``.
        receiver_live_ratio: Receiver live ``used/capacity``.
    """

    receiver: InstanceSample
    request_size_gib: int
    size_source: GrowSizeSource
    receiver_live_capacity_bytes: int
    receiver_live_ratio: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return a JSON-friendly form for the dry-run log (``kind: grow``)."""
        return {
            "kind": MoveKind.GROW.value,
            "receiver": self.receiver.identity.instance_id,
            "receiver_worker_ip": self.receiver.identity.worker_ip,
            "request_size_gib": self.request_size_gib,
            "size_source": self.size_source.value,
            "receiver_live_capacity_bytes": self.receiver_live_capacity_bytes,
            "receiver_live_ratio": round(self.receiver_live_ratio, 4),
        }


Proposal = MoveProposal | GrowProposal
"""What one cycle may start: a donor-less GROW or a donor/receiver MOVE."""


def evaluate_grow(
    receiver: InstanceSample,
    receiver_preflight: LivePreflight,
    inventory: list[ManagedAllocation],
    grow_backoffs: Mapping[str, float],
    config: MPMemoryCoordinatorConfig,
    now: float,
) -> GrowProposal | list[Rejection]:
    """Apply every receiver-only check for a GROW to one stable-HIGH sample.

    In order: no active grow backoff for the receiver's worker, a clean live
    preflight, a live ratio that still classifies HIGH, and a derivable
    request size. No donor, gap, or projection rule applies: a GROW touches
    nothing but the receiver and the outside pool.

    Args:
        receiver: Stable-HIGH sample (already cooldown-free).
        receiver_preflight: Receiver's live documents.
        inventory: The managed inventory (size derivation).
        grow_backoffs: ``worker_ip`` -> time until which GROW is not proposed.
        config: Thresholds and the adapter index.
        now: Current wall-clock time.

    Returns:
        A proposal, or the rejections that stopped it (never empty).
    """
    instance_id = receiver.identity.instance_id
    until = grow_backoffs.get(receiver.identity.worker_ip, 0.0)
    if until > now:
        return [
            Rejection(instance_id, RejectionReason.GROW_BACKOFF, f"until={until:.0f}")
        ]
    problems = preflight_problems(receiver_preflight, config)
    if problems:
        return [
            Rejection(
                instance_id, RejectionReason.PREFLIGHT_FAILED, "; ".join(problems)
            )
        ]
    receiver_dax = receiver_preflight.dax.adapters[0].status
    live = live_ratio(receiver_dax)
    if live is None or classify(live, config) is not PressureLevel.HIGH:
        return [
            Rejection(instance_id, RejectionReason.LIVE_RATIO_MISMATCH, f"live={live}")
        ]
    size = derive_grow_size(receiver.identity, receiver_dax, inventory)
    if isinstance(size, Rejection):
        return [size]
    return GrowProposal(
        receiver=receiver,
        request_size_gib=size.size_gib,
        size_source=size.source,
        receiver_live_capacity_bytes=receiver_dax.total_capacity_bytes,
        receiver_live_ratio=live,
    )
