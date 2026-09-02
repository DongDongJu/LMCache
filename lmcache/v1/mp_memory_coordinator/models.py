# SPDX-License-Identifier: Apache-2.0
"""Local wire DTOs and durable domain records of the MP Memory Coordinator.

Wire models mirror the *documented* responses of three remotes -- the MP
Coordinator (``/instances``, ``/instances/usage``), an MP server (``/status``,
``/reconfigure/dax/*``), and the frozen outside Memory Allocation API. They
are deliberately local: this package never imports MP Coordinator models.
Responses allow unknown fields (forward compatibility) but reject missing
required ones; outside *requests* forbid extra fields so an internal value
can never leak into an outside body.

Domain records (:class:`ManagedAllocation`, :class:`MoveRecord`,
:class:`JournalDocument`) are what the journal persists.
"""

# Standard
from enum import Enum
from typing import Final, Literal

# Third Party
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

GIB = 1024**3
"""Bytes per GiB; DAX map sizes must be whole GiB to be movable."""

OUTSIDE_STATUS_PATH = "/api/v2/apps/lmcache"
OUTSIDE_DEALLOCATIONS_PATH = "/api/v2/apps/lmcache/deallocations"
OUTSIDE_ALLOCATIONS_PATH = "/api/v2/apps/lmcache/allocations"
OUTSIDE_STATUS_DONE: Final = "DONE"
OUTSIDE_MODE: Final = "devdax"
OUTSIDE_PURPOSE: Final = "lmcache-dax"
OUTSIDE_ACCESS: Final = "exclusive"

MP_STATUS_PATH = "/status"
MP_HEALTHCHECK_PATH = "/healthcheck"
MP_DAX_STATUS_PATH = "/reconfigure/dax/status"
MP_DAX_REMOVE_PATH = "/reconfigure/dax/remove"
MP_DAX_ADD_PATH = "/reconfigure/dax/add"

COORDINATOR_INSTANCES_PATH = "/instances"
COORDINATOR_USAGE_PATH = "/instances/usage"

WORKER_IP_METADATA_KEY = "worker_ip"
"""Registration ``metadata`` key carrying the worker's host IP."""

DAX_BACKEND = "dax"
DAX_TERMINAL_STATES = frozenset({"closed", "removed"})
"""Tombstone states: never selected, owned, or attached."""
DAX_ACTIVE_STATE = "active"
DAX_DRAINING_STATE = "draining"


# -- MP Coordinator wire models ----------------------------------------------


class CoordinatorInstance(BaseModel):
    """One entry of ``GET /instances``."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    ip: str
    http_port: int
    registration_time: float
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        """Direct MP HTTP endpoint, ``ip:http_port``."""
        return f"{self.ip}:{self.http_port}"

    @property
    def worker_ip(self) -> str:
        """The ``metadata.worker_ip`` value, or ``""`` when absent."""
        return self.metadata.get(WORKER_IP_METADATA_KEY, "")


class CoordinatorInstances(BaseModel):
    """Body of ``GET /instances``."""

    model_config = ConfigDict(extra="allow")

    instances: list[CoordinatorInstance]


class ModuleUsage(BaseModel):
    """One compartment of ``GET /instances/usage``."""

    model_config = ConfigDict(extra="allow")

    tier: str
    backend: str
    shared: bool
    used_bytes: int
    capacity_bytes: int
    usage_ratio: float | None


class InstanceUsage(BaseModel):
    """One instance of ``GET /instances/usage``."""

    model_config = ConfigDict(extra="allow")

    instance_id: str
    registered: bool
    declared_capacity: bool
    modules: list[ModuleUsage]

    def private_dax(self) -> ModuleUsage | None:
        """Return the private ``l2/dax`` compartment, if declared.

        Returns:
            The non-shared ``l2``/``dax`` module, or ``None`` when the
            instance reports none (or only a shared one).
        """
        for module in self.modules:
            if (
                module.tier == "l2"
                and module.backend == DAX_BACKEND
                and not module.shared
            ):
                return module
        return None


class FleetUsage(BaseModel):
    """Body of ``GET /instances/usage``."""

    model_config = ConfigDict(extra="allow")

    instances: list[InstanceUsage]
    shared_modules: list[ModuleUsage]


# -- MP server wire models -----------------------------------------------------


class L2AdapterStatus(BaseModel):
    """One ``storage_manager.l2_adapters[]`` entry of ``GET /status``."""

    model_config = ConfigDict(extra="allow")

    is_healthy: bool
    type: str
    closing: bool = False
    hotplug_enabled: bool = False


class StorageManagerStatus(BaseModel):
    """The ``storage_manager`` section of ``GET /status``."""

    model_config = ConfigDict(extra="allow")

    is_healthy: bool
    l2_adapters: list[L2AdapterStatus]
    num_l2_adapters: int


class MPStatus(BaseModel):
    """Body of an MP server's ``GET /status``."""

    model_config = ConfigDict(extra="allow")

    is_healthy: bool
    storage_manager: StorageManagerStatus


class DaxDeviceStatus(BaseModel):
    """One device of ``GET /reconfigure/dax/status``."""

    model_config = ConfigDict(extra="allow")

    index: int
    device_id: int
    device_path: str
    state: str
    is_healthy: bool
    closing: bool
    max_dax_size_bytes: int
    slot_bytes: int
    max_slots: int
    live_slot_count: int
    locked_key_count: int
    borrowed_slot_count: int
    active_read_count: int
    active_write_count: int
    inflight_store_tasks: int
    inflight_lookup_tasks: int
    inflight_load_tasks: int

    @property
    def slot_capacity_bytes(self) -> int:
        """Capacity the device contributes: ``max_slots * slot_bytes``."""
        return self.max_slots * self.slot_bytes

    @property
    def is_terminal(self) -> bool:
        """Whether the entry is a ``closed``/``removed`` tombstone."""
        return self.state in DAX_TERMINAL_STATES

    @property
    def busy_references(self) -> int:
        """Locks, borrows, active I/O and adapter in-flight tasks."""
        return (
            self.locked_key_count
            + self.borrowed_slot_count
            + self.active_read_count
            + self.active_write_count
            + self.inflight_store_tasks
            + self.inflight_lookup_tasks
            + self.inflight_load_tasks
        )


class DaxHotplugStatus(BaseModel):
    """The ``status`` section of one DAX adapter."""

    model_config = ConfigDict(extra="allow")

    hotplug_enabled: bool
    slot_bytes: int
    total_capacity_bytes: int
    total_used_bytes: int
    devices: list[DaxDeviceStatus]

    def live_devices(self) -> list[DaxDeviceStatus]:
        """Return every non-tombstone device entry."""
        return [d for d in self.devices if not d.is_terminal]

    def find_live(self, device_path: str) -> DaxDeviceStatus | None:
        """Return the non-tombstone entry for ``device_path``, if any."""
        for device in self.devices:
            if device.device_path == device_path and not device.is_terminal:
                return device
        return None


class DaxAdapterStatus(BaseModel):
    """One adapter of ``GET /reconfigure/dax/status``."""

    model_config = ConfigDict(extra="allow")

    backend: str
    supported_operations: list[str]
    status: DaxHotplugStatus
    adapter_index: int


class DaxReconfigureStatus(BaseModel):
    """Body of ``GET /reconfigure/dax/status``."""

    model_config = ConfigDict(extra="allow")

    enabled: bool
    backend: str
    num_adapters: int
    adapters: list[DaxAdapterStatus]


class DaxRemoveMode(str, Enum):
    """``mode`` of ``POST /reconfigure/dax/remove`` this coordinator uses."""

    DRAIN = "drain"
    EVICT = "evict"


class DaxRemoveResponse(BaseModel):
    """Successful body of ``POST /reconfigure/dax/remove``."""

    model_config = ConfigDict(extra="allow")

    status: str
    operation: str
    adapter_index: int
    device_path: str
    index: int
    state: str


class DaxRemoveBlocked(BaseModel):
    """A ``409`` from ``POST /reconfigure/dax/remove``: the device is busy."""

    model_config = ConfigDict(extra="allow")

    status: str = "blocked"
    reason: str = ""
    locked_key_count: int = 0
    borrowed_slot_count: int = 0


class DaxDeviceNotFound(BaseModel):
    """A ``404`` from ``POST /reconfigure/dax/remove``."""

    model_config = ConfigDict(extra="allow")

    error: str = ""


class DaxAddResponse(BaseModel):
    """Successful body of ``POST /reconfigure/dax/add``."""

    model_config = ConfigDict(extra="allow")

    status: str
    operation: str
    adapter_index: int
    device: DaxDeviceStatus


# -- Outside Memory Allocation wire models (frozen) -----------------------------


class DeallocationRequest(BaseModel):
    """Exact body of ``POST /api/v2/apps/lmcache/deallocations``."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    target_node: str
    device_path: str


class DeallocationResponse(BaseModel):
    """Required fields of a deallocation response."""

    model_config = ConfigDict(extra="allow")

    status: StrictStr
    request_id: StrictStr
    target_node: StrictStr
    device_path: StrictStr
    released_size_gib: StrictInt


class AllocationRequest(BaseModel):
    """Exact body of ``POST /api/v2/apps/lmcache/allocations``.

    The request field is ``request_size_gib``; the response echoes it as
    ``requested_size_gib``. The names differ on purpose.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    target_node: str
    request_size_gib: int
    mode: Literal["devdax"] = OUTSIDE_MODE
    purpose: Literal["lmcache-dax"] = OUTSIDE_PURPOSE
    access: Literal["exclusive"] = OUTSIDE_ACCESS


class AllocationResponse(BaseModel):
    """Required fields of an allocation response."""

    model_config = ConfigDict(extra="allow")

    status: StrictStr
    request_id: StrictStr
    target_node: StrictStr
    device_path: StrictStr
    requested_size_gib: StrictInt
    granted_size_gib: StrictInt


OutsideStatus = dict[str, list[str]]
"""``GET /api/v2/apps/lmcache``: bare ``target_node -> device_path[]``."""


def parse_outside_status(raw: object) -> OutsideStatus:
    """Validate the bare ``target_node -> device_path[]`` status object.

    Args:
        raw: The decoded JSON body.

    Returns:
        A new ``dict`` with each path list copied.

    Raises:
        ValueError: If the body is not a mapping of strings to lists of
            strings (a wrapper object, a list, or nested structures are all
            rejected).
    """
    if not isinstance(raw, dict):
        raise ValueError("outside status must be a JSON object")
    result: OutsideStatus = {}
    for node, paths in raw.items():
        if not isinstance(node, str):
            raise ValueError("outside status keys must be strings")
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError(f"outside status entry {node!r} must be a string list")
        result[node] = list(paths)
    return result


# -- Domain records ---------------------------------------------------------------


class InstanceIdentity(BaseModel):
    """What makes an MP instance *the same* instance across samples.

    Attributes:
        instance_id: The registered id.
        registration_time: Registration epoch reported by the coordinator;
            any change invalidates a sample.
        endpoint: Direct MP HTTP endpoint ``ip:port``.
        worker_ip: ``metadata.worker_ip`` -- the outside ``target_node``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance_id: str
    registration_time: float
    endpoint: str
    worker_ip: str

    @property
    def key(self) -> str:
        """A stable string key for history and cooldown maps."""
        return "|".join(
            (
                self.instance_id,
                repr(self.registration_time),
                self.endpoint,
                self.worker_ip,
            )
        )

    @property
    def base_url(self) -> str:
        """``http://ip:port`` of the direct MP HTTP API."""
        return f"http://{self.endpoint}"


class InstanceSample(BaseModel):
    """A joined membership + occupancy sample of one instance.

    Attributes:
        identity: The instance identity the sample belongs to.
        registered: ``registered`` from the usage view.
        declared_capacity: ``declared_capacity`` from the usage view.
        used_bytes: Private ``l2/dax`` bytes used.
        capacity_bytes: Private ``l2/dax`` declared capacity.
        usage_ratio: ``used/capacity`` or ``None`` when undeclared.
        sampled_at: Wall-clock time the sample was built.
    """

    model_config = ConfigDict(extra="forbid")

    identity: InstanceIdentity
    registered: bool
    declared_capacity: bool
    used_bytes: int
    capacity_bytes: int
    usage_ratio: float | None
    sampled_at: float


class AllocationOrigin(str, Enum):
    """How a managed allocation entered the inventory."""

    ADOPTED = "adopted"
    ALLOCATED = "allocated"
    DISCOVERED = "discovered"
    RESTORED = "restored"


class ManagedAllocation(BaseModel):
    """One Memory-Coordinator-managed Device-DAX allocation.

    Attributes:
        worker_ip: Owning worker (outside ``target_node``).
        instance_id: MP instance currently exposing the device.
        device_path: Exact device path.
        allocation_size_gib: Outside allocation size in GiB.
        device_map_size_bytes: DAX ``max_dax_size_bytes`` of the mapping.
        slot_capacity_bytes: ``max_slots * slot_bytes`` at last confirmation.
        adapter_index: Backend-local DAX adapter index (always ``0``).
        origin: See :class:`AllocationOrigin`.
        last_confirmed_state: DAX ``state`` at last confirmation.
        last_confirmed_at: Wall-clock time of the last confirmation.
    """

    model_config = ConfigDict(extra="forbid")

    worker_ip: str
    instance_id: str
    device_path: str
    allocation_size_gib: int
    device_map_size_bytes: int
    slot_capacity_bytes: int
    adapter_index: int
    origin: AllocationOrigin
    last_confirmed_state: str
    last_confirmed_at: float

    def is_size_consistent(self) -> bool:
        """Whether the map is whole GiB and equals the allocation size."""
        return (
            self.device_map_size_bytes % GIB == 0
            and self.device_map_size_bytes // GIB == self.allocation_size_gib
        )


class MoveState(str, Enum):
    """Saga states. ``COMPLETE`` and ``BLOCKED`` are terminal."""

    SELECTED = "SELECTED"
    DONOR_DRAINING = "DONOR_DRAINING"
    DONOR_REMOVED = "DONOR_REMOVED"
    DEALLOCATING = "DEALLOCATING"
    DEALLOCATED = "DEALLOCATED"
    ALLOCATING = "ALLOCATING"
    ALLOCATED = "ALLOCATED"
    COMPLETE = "COMPLETE"
    ROLLING_BACK = "ROLLING_BACK"
    BLOCKED = "BLOCKED"


TERMINAL_MOVE_STATES = frozenset({MoveState.COMPLETE, MoveState.BLOCKED})


class MoveOutcome(str, Enum):
    """Terminal outcome; ``PENDING`` until ``state == COMPLETE``."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    ROLLED_BACK = "ROLLED_BACK"


class RollbackStep(str, Enum):
    """Sub-state of ``ROLLING_BACK``."""

    NONE = "NONE"
    DONOR_EVICT = "DONOR_EVICT"
    DONOR_READD = "DONOR_READD"
    RELEASE_RECEIVER = "RELEASE_RECEIVER"
    RESTORE_DONOR_ALLOCATE = "RESTORE_DONOR_ALLOCATE"
    RESTORE_DONOR_ADD = "RESTORE_DONOR_ADD"


class EffectName(str, Enum):
    """Every side effect a move can perform, each recorded once."""

    DONOR_DRAIN = "donor_drain"
    DONOR_EVICT = "donor_evict"
    DEALLOCATE = "deallocate"
    ALLOCATE = "allocate"
    RECEIVER_ADD = "receiver_add"
    RELEASE_RECEIVER = "release_receiver"
    RESTORE_ALLOCATE = "restore_allocate"
    RESTORE_ADD = "restore_add"
    DONOR_READD = "donor_readd"


ScalarJSON = str | int | float | bool
"""Scalar JSON values a confirmed response is reduced to."""


class EffectRecord(BaseModel):
    """Durable intent/confirmation of one side effect.

    Attributes:
        name: Which effect.
        request_id: Outside request id (empty for DAX effects).
        intent_at: When the intent was persisted (before the POST).
        before_paths: Outside path set of the target node captured before
            the POST (outside effects only).
        dispatched: Persisted immediately before the POST is sent. An
            outside effect that is ``dispatched`` but has neither a
            ``response`` nor an ``error`` after a restart has an unknown
            outcome and blocks the move; one that is not dispatched was
            provably never sent.
        attempts: POSTs issued for DAX effects (outside effects: at most 1).
        confirmed: Whether the effect was confirmed against status.
        confirmed_at: When confirmation was persisted.
        response: Documented response fields, once received.
        error: Last error text.
    """

    model_config = ConfigDict(extra="forbid")

    name: EffectName
    request_id: str = ""
    intent_at: float
    before_paths: list[str] = Field(default_factory=list)
    dispatched: bool = False
    attempts: int = 0
    confirmed: bool = False
    confirmed_at: float = 0.0
    response: dict[str, ScalarJSON] = Field(default_factory=dict)
    error: str = ""


class MoveRecord(BaseModel):
    """The durable record of one move (see the design doc for the saga).

    Attributes:
        move_id: Unique id.
        state: Current :class:`MoveState`.
        outcome: :class:`MoveOutcome`; set when ``state == COMPLETE``.
        rollback_step: Sub-state while ``ROLLING_BACK``.
        donor: Donor identity snapshot at selection.
        receiver: Receiver identity snapshot at selection.
        donor_capacity_bytes: Donor ``l2/dax`` capacity at selection.
        receiver_capacity_bytes: Receiver ``l2/dax`` capacity at selection.
        old_path: Donor device path being moved.
        old_device_index: Donor DAX index of ``old_path`` at selection.
        old_map_size_bytes: DAX map size of ``old_path``.
        old_slot_capacity_bytes: Slot capacity of ``old_path``.
        allocation_size_gib: Outside allocation size of ``old_path``.
        deallocation_request_id: Outside id of the donor deallocation.
        allocation_request_id: Outside id of the receiver allocation.
        release_request_id: Outside id of a rollback receiver release.
        restore_request_id: Outside id of a rollback donor allocation.
        released_size_gib: From the deallocation response.
        new_path: Receiver path returned by the allocation (raw, before
            validation).
        granted_size_gib: From the allocation response.
        new_device_index: Receiver DAX index after add.
        new_slot_capacity_bytes: Slot capacity of ``new_path`` after add.
        restored_path: Donor path returned by a rollback allocation.
        effects: Effect ledger keyed by :class:`EffectName` value.
        drain_started_at: When the donor drain intent was persisted.
        capacity_converged: Whether the usage view reflected the move.
        created_at: Creation time.
        updated_at: Last persisted change.
        last_error: Last error text.
        block_reason: Why the move is BLOCKED (terminal).
    """

    model_config = ConfigDict(extra="forbid")

    move_id: str
    state: MoveState
    outcome: MoveOutcome = MoveOutcome.PENDING
    rollback_step: RollbackStep = RollbackStep.NONE
    donor: InstanceIdentity
    receiver: InstanceIdentity
    donor_capacity_bytes: int
    receiver_capacity_bytes: int
    old_path: str
    old_device_index: int
    old_map_size_bytes: int
    old_slot_capacity_bytes: int
    allocation_size_gib: int
    deallocation_request_id: str
    allocation_request_id: str
    release_request_id: str
    restore_request_id: str
    released_size_gib: int = 0
    new_path: str = ""
    granted_size_gib: int = 0
    new_device_index: int = -1
    new_slot_capacity_bytes: int = 0
    restored_path: str = ""
    effects: dict[str, EffectRecord] = Field(default_factory=dict)
    drain_started_at: float = 0.0
    capacity_converged: bool = False
    created_at: float
    updated_at: float
    last_error: str = ""
    block_reason: str = ""

    def effect(self, name: EffectName) -> EffectRecord | None:
        """Return the ledger entry for ``name`` if an intent was persisted."""
        return self.effects.get(name.value)

    @property
    def is_terminal(self) -> bool:
        """Whether no further action will ever be taken."""
        return self.state in TERMINAL_MOVE_STATES


class MoveCounters(BaseModel):
    """Minimal persisted counters."""

    model_config = ConfigDict(extra="forbid")

    proposed: int = 0
    succeeded: int = 0
    rolled_back: int = 0
    blocked: int = 0


JOURNAL_SCHEMA_VERSION = 1


class JournalDocument(BaseModel):
    """Everything the journal persists, as one atomically replaced document.

    Attributes:
        schema_version: Format version; a reader rejects unknown versions.
        initialized: Set once adoption ran (or was explicitly skipped);
            a normal restart never repeats adoption.
        inventory: Managed allocations.
        cooldowns: Identity key -> wall-clock time until which the instance
            may not participate in a move.
        active_move: The single in-progress move, if any.
        history: Recent terminal moves, newest last (bounded).
        counters: See :class:`MoveCounters`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = JOURNAL_SCHEMA_VERSION
    initialized: bool = False
    inventory: list[ManagedAllocation] = Field(default_factory=list)
    cooldowns: dict[str, float] = Field(default_factory=dict)
    active_move: MoveRecord | None = None
    history: list[MoveRecord] = Field(default_factory=list)
    counters: MoveCounters = Field(default_factory=MoveCounters)

    def find_allocation(self, device_path: str) -> ManagedAllocation | None:
        """Return the managed allocation for ``device_path``, if owned."""
        for allocation in self.inventory:
            if allocation.device_path == device_path:
                return allocation
        return None

    def allocations_for(self, worker_ip: str) -> list[ManagedAllocation]:
        """Return managed allocations owned by ``worker_ip``."""
        return [a for a in self.inventory if a.worker_ip == worker_ip]
