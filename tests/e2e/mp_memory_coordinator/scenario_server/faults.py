# SPDX-License-Identifier: Apache-2.0
"""Fault-injection models and one-shot barriers for the scenario server.

Faults are plain pydantic models so the admin port can validate them with
``extra="forbid"``. :class:`FaultSpec` is a *patch*: only the keys present in
the request body change the active :class:`ActiveFaults`.

Barriers pause one named MP mutation (``drain``, ``evict`` or ``add``) either
before or after it touches state, until the admin port releases the barrier.
They are one-shot: the first matching mutation consumes the barrier.
"""

# Standard
from dataclasses import dataclass, field
from typing import Literal
import asyncio
import threading

# Third Party
from pydantic import BaseModel, ConfigDict, Field

IdentityField = Literal["registration_time", "endpoint"]
BarrierOperation = Literal["drain", "evict", "add"]
BarrierPhase = Literal["before", "after"]


class IdentityFlip(BaseModel):
    """Report a different identity field on every ``every_n_reads``-th read.

    Attributes:
        instance_id: Instance whose ``/instances`` entry flips.
        field: ``registration_time`` bumps the epoch by 1.0 on flipped reads;
            ``endpoint`` reports ``http_port + 1``.
        every_n_reads: Period of the flip; the Nth, 2Nth, ... reads flip.
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    field: IdentityField
    every_n_reads: int = Field(default=2, ge=1)


class CoordinatorFaults(BaseModel):
    """Faults applied to the fake MP Coordinator port.

    Attributes:
        unavailable: Every coordinator route returns 503.
        undeclared_capacity: Instances reporting ``declared_capacity=false``
            (module capacities 0, ratios null).
        null_ratio: Instances whose l2/dax ``usage_ratio`` is null.
        shared_dax: Instances whose l2/dax module reports ``shared=true``.
        unregistered: Instances omitted from ``/instances`` and reported with
            ``registered=false`` in ``/instances/usage``.
        worker_ip_override: ``metadata.worker_ip`` replacement per instance;
            ``null`` omits the key entirely.
        identity_flip: Periodic identity mismatch, or ``null`` for none.
        delayed_capacity_seconds: After a DAX add/remove keep publishing the
            previous capacity for this long (0 disables the delay).
    """

    model_config = ConfigDict(extra="forbid")

    unavailable: bool = False
    undeclared_capacity: list[str] = Field(default_factory=list)
    null_ratio: list[str] = Field(default_factory=list)
    shared_dax: list[str] = Field(default_factory=list)
    unregistered: list[str] = Field(default_factory=list)
    worker_ip_override: dict[str, str | None] = Field(default_factory=dict)
    identity_flip: IdentityFlip | None = None
    delayed_capacity_seconds: float = Field(default=0.0, ge=0.0)


class MpFaults(BaseModel):
    """Faults applied to one fake MP server port.

    Attributes:
        status_unavailable: ``/healthcheck``, ``/status`` and
            ``/reconfigure/dax/status`` return 503.
        adapters: Number of DAX adapters reported (0, 1 or 2).
        unhealthy: Adapter, storage manager and engine report unhealthy.
        closing: Adapter reports ``closing=true``.
        hotplug_disabled: DAX status reports ``hotplug_enabled=false`` and
            add/remove return 403.
        evict_409_count: Number of upcoming evicts that return 409 even when
            the device is idle (decremented per blocked evict).
        add_fail_count: Number of upcoming new-device adds that return 400
            ``failed to map DAX device`` (decremented per failure).
        add_always_fail: Every new-device add returns that 400.
        remove_route_failure: ``POST /reconfigure/dax/remove`` returns 500.
    """

    model_config = ConfigDict(extra="forbid")

    status_unavailable: bool = False
    adapters: int = Field(default=1, ge=0, le=2)
    unhealthy: bool = False
    closing: bool = False
    hotplug_disabled: bool = False
    evict_409_count: int = Field(default=0, ge=0)
    add_fail_count: int = Field(default=0, ge=0)
    add_always_fail: bool = False
    remove_route_failure: bool = False


class FaultSpec(BaseModel):
    """Partial fault update accepted by ``POST /__test/faults``.

    Only keys present in the request body are applied; see
    :meth:`ActiveFaults.merge`.

    Attributes:
        coordinator: Coordinator fault patch.
        mp: Per-instance MP fault patches keyed by instance id.
    """

    model_config = ConfigDict(extra="forbid")

    coordinator: CoordinatorFaults = Field(default_factory=CoordinatorFaults)
    mp: dict[str, MpFaults] = Field(default_factory=dict)


class ActiveFaults(BaseModel):
    """The complete set of faults currently in force.

    Attributes:
        coordinator: Active coordinator faults.
        mp: Active MP faults for every instance (never missing an instance).
    """

    coordinator: CoordinatorFaults
    mp: dict[str, MpFaults]

    def merge(self, patch: FaultSpec) -> None:
        """Apply a patch, changing only the keys the patch explicitly set.

        Args:
            patch: Validated patch; ``patch.mp`` may only name instances that
                already exist in :attr:`mp`.

        Raises:
            KeyError: If ``patch.mp`` names an unknown instance id.
        """
        for instance_id in patch.mp:
            if instance_id not in self.mp:
                raise KeyError(instance_id)
        for name in patch.coordinator.model_fields_set:
            setattr(self.coordinator, name, getattr(patch.coordinator, name))
        for instance_id, mp_patch in patch.mp.items():
            target = self.mp[instance_id]
            for name in mp_patch.model_fields_set:
                setattr(target, name, getattr(mp_patch, name))


class BarrierRequest(BaseModel):
    """Body of ``POST /__test/barriers``.

    Attributes:
        instance_id: MP instance whose mutation is paused.
        operation: Mutation kind; ``migrate`` removes count as ``evict``.
        when: ``before`` pauses before state changes, ``after`` pauses after
            the state change but before the response is sent.
        name: Unique barrier name used by the release route.
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    operation: BarrierOperation
    when: BarrierPhase
    name: str = Field(min_length=1)


@dataclass
class Barrier:
    """One armed barrier and its release event.

    Attributes:
        request: The arming request.
        event: Set when the barrier is released.
        hit: True once a mutation has consumed the barrier.
    """

    request: BarrierRequest
    event: threading.Event = field(default_factory=threading.Event)
    hit: bool = False

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable view of this barrier.

        Returns:
            ``{"name", "instance_id", "operation", "when", "hit", "released"}``.
        """
        return {
            "name": self.request.name,
            "instance_id": self.request.instance_id,
            "operation": self.request.operation,
            "when": self.request.when,
            "hit": self.hit,
            "released": self.event.is_set(),
        }


class BarrierRegistry:
    """Thread-safe registry of one-shot barriers.

    The release event is a :class:`threading.Event` so that a request blocked
    in one event loop can be released from another loop or thread (the
    in-process ``TestClient`` runs each request on its own portal).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # name -> barrier; insertion order decides which barrier a mutation
        # consumes first when several match.
        self._barriers: dict[str, Barrier] = {}

    def arm(self, request: BarrierRequest) -> Barrier:
        """Register a new barrier.

        Args:
            request: Validated barrier description.

        Returns:
            The armed barrier.

        Raises:
            ValueError: If a barrier with that name already exists.
        """
        with self._lock:
            if request.name in self._barriers:
                raise ValueError(f"barrier {request.name!r} already exists")
            barrier = Barrier(request=request)
            self._barriers[request.name] = barrier
            return barrier

    def release(self, name: str) -> Barrier:
        """Release a barrier, unblocking the mutation waiting on it.

        Releasing a barrier nobody has hit yet is allowed: the next matching
        mutation then passes through without pausing.

        Args:
            name: Barrier name.

        Returns:
            The released barrier.

        Raises:
            KeyError: If no barrier with that name exists.
        """
        with self._lock:
            barrier = self._barriers[name]
            barrier.event.set()
            return barrier

    def clear(self) -> None:
        """Drop every barrier, releasing any mutation still waiting."""
        with self._lock:
            for barrier in self._barriers.values():
                barrier.event.set()
            self._barriers.clear()

    def snapshot(self) -> list[dict[str, object]]:
        """Return every barrier as JSON-serializable dicts.

        Returns:
            Barrier snapshots in arming order.
        """
        with self._lock:
            return [barrier.snapshot() for barrier in self._barriers.values()]

    async def wait(
        self, instance_id: str, operation: BarrierOperation, when: BarrierPhase
    ) -> None:
        """Pause if an unconsumed barrier matches, until it is released.

        The first matching barrier (in arming order) is consumed even if it
        was already released, so each barrier pauses at most one mutation.

        Args:
            instance_id: Instance performing the mutation.
            operation: Mutation kind.
            when: Phase reached by the mutation.
        """
        with self._lock:
            matching = [
                barrier
                for barrier in self._barriers.values()
                if not barrier.hit
                and barrier.request.instance_id == instance_id
                and barrier.request.operation == operation
                and barrier.request.when == when
            ]
            if not matching:
                return
            barrier = matching[0]
            barrier.hit = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, barrier.event.wait)
