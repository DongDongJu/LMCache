# SPDX-License-Identifier: Apache-2.0
"""In-memory fleet, DAX device, fault and audit state of the scenario server.

One :class:`ScenarioState` backs all four listeners. Every public method takes
the state lock, so handlers running on different event loops (or the
in-process ``TestClient``) see a consistent snapshot. Handlers must await
barriers *outside* these methods; nothing here blocks.

The topology is derived from the shared ``two_server_local_dax.yaml`` fixture
(see ``PLAN.md`` Phase 1A): the lexicographically first node hosts
``mp-donor``, the second hosts ``mp-receiver``. Each instance starts with its
bootstrap device plus every runtime device the fixture marks ``assigned``.
"""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import posixpath
import threading
import time

# Third Party
from pydantic import BaseModel, ConfigDict, Field
import yaml

# Local
from .faults import (
    ActiveFaults,
    BarrierRegistry,
    CoordinatorFaults,
    FaultSpec,
    IdentityFlip,
    MpFaults,
)

GIB = 1 << 30
SLOT_BYTES = 1 << 20
"""Slot geometry: 1 MiB slots, so whole-GiB devices map exactly."""
DAX_ALIGN_BYTES = 2 << 20
"""sysfs ``align`` every fake device reports (2 MiB, the Device-DAX default)."""
DAX_CHAR_MAJOR = 249
"""Character-device major every fake device reports."""
WATCHER_INTERVAL_SECONDS = 1.0
"""``watcher.interval_seconds`` every fake MP server reports."""
L1_CAPACITY_BYTES = 4 * GIB
DONOR_ID = "mp-donor"
RECEIVER_ID = "mp-receiver"
INSTANCE_IDS = (DONOR_ID, RECEIVER_ID)
_DEFAULT_USED_BYTES: dict[str, int] = {DONOR_ID: 8 * GIB, RECEIVER_ID: 56 * GIB}
# Per-device live bytes by device index; devices beyond the tuple start empty.
_DEFAULT_DEVICE_USED_BYTES: dict[str, tuple[int, ...]] = {
    DONOR_ID: (4 * GIB, 4 * GIB),
    RECEIVER_ID: (56 * GIB,),
}
_DAX_RECONFIGURE_OPERATIONS = ("status", "add", "remove", "resize")
_BLOCKED_REASON = "device has externally locked or borrowed slots"

DeviceState = Literal[
    "active",
    "draining",
    "migrating",
    "resizing",
    "removing",
    "closed",
    "failed",
    "removed",
]
_CAPACITY_STATES: frozenset[str] = frozenset(
    {"active", "draining", "migrating", "resizing", "removing"}
)
# Tombstones left by a controlled remove: hold no capacity, serve no I/O,
# report is_healthy=false, but do not poison the adapter health aggregate.
_TERMINAL_STATES: frozenset[str] = frozenset({"closed", "removed"})
# States ignored when resolving a device_path for add/remove.
_LOOKUP_EXCLUDED_STATES: frozenset[str] = frozenset({"closed", "removed", "failed"})

RemoveMode = Literal["migrate", "evict", "drain"]
PhysicalMode = Literal[
    "devdax", "system-ram", "unbound", "not-a-device", "absent", "unknown"
]
IdentityBump = Literal["registration_time", "endpoint", "both"]
AuditKind = Literal["request", "response", "mutation"]
SizeRequest = int | str


def _wall_clock_after(previous: float) -> float:
    """Return the current wall-clock time, strictly greater than ``previous``.

    Returns:
        ``time.time()`` or ``previous`` nudged upward when the clock has not
        advanced, so a re-registration always yields a new epoch.
    """
    now = time.time()
    if now <= previous:
        return previous + 1e-6
    return now


def _usage_ratio(used_bytes: int, capacity_bytes: int) -> float | None:
    """Return ``used/capacity`` or ``None`` when capacity is undeclared (0).

    Returns:
        The ratio, or ``None``.
    """
    if capacity_bytes <= 0:
        return None
    return used_bytes / capacity_bytes


def physical_status(
    device_path: str,
    minor: int,
    mode: PhysicalMode,
    size_bytes: int,
    align_bytes: int = DAX_ALIGN_BYTES,
) -> dict[str, object]:
    """Build one physical-inspection entry with the production key set.

    Mirrors ``DaxPhysicalState.as_dict()`` of the real DAX adapter: what a
    read-only ``stat`` + sysfs probe reports for one path.

    Args:
        device_path: The inspected path.
        minor: Character-device minor number (also the ``daxX.Y`` suffix).
        mode: Physical mode; ``devdax`` is the only attachable one.
        size_bytes: sysfs ``size``.
        align_bytes: sysfs ``align``.

    Returns:
        ``{"device_path", "mode", "present", "major", "minor",
        "kernel_name", "driver", "size_bytes", "align_bytes", "probed_at",
        "detail"}``.
    """
    driver = {"devdax": "device_dax", "system-ram": "kmem"}.get(mode, "")
    present = mode != "absent"
    return {
        "device_path": device_path,
        "mode": mode,
        "present": present,
        "major": DAX_CHAR_MAJOR if present else 0,
        "minor": minor if present else 0,
        "kernel_name": f"dax2.{minor}" if present else "",
        "driver": driver,
        "size_bytes": size_bytes,
        "align_bytes": align_bytes,
        "probed_at": time.time(),
        "detail": "" if mode == "devdax" else f"scenario server reports {mode}",
    }


def _module_status(
    tier: str,
    backend: str,
    shared: bool,
    used_bytes: int,
    capacity_bytes: int,
    usage_ratio: float | None,
) -> dict[str, object]:
    """Build one ``/instances/usage`` module entry with the golden key set.

    Returns:
        ``{"tier", "backend", "shared", "used_bytes", "capacity_bytes",
        "usage_ratio"}``.
    """
    return {
        "tier": tier,
        "backend": backend,
        "shared": shared,
        "used_bytes": used_bytes,
        "capacity_bytes": capacity_bytes,
        "usage_ratio": usage_ratio,
    }


class FixtureDevice(BaseModel):
    """One device entry of ``two_server_local_dax.yaml``."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_gib: int = Field(ge=1)
    role: Literal["bootstrap", "runtime"]
    state: Literal["free", "assigned"]


class FixtureNode(BaseModel):
    """One worker node of ``two_server_local_dax.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    devices: list[FixtureDevice]


class Fixture(BaseModel):
    """Root of ``two_server_local_dax.yaml``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    nodes: dict[str, FixtureNode]


class DaxAddRequest(BaseModel):
    """Body of ``POST /reconfigure/dax/add`` (mirrors production)."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    size: SizeRequest


class DaxRemoveRequest(BaseModel):
    """Body of ``POST /reconfigure/dax/remove`` (mirrors production)."""

    model_config = ConfigDict(extra="forbid")

    adapter_index: int = 0
    device_path: str
    mode: RemoveMode = "migrate"
    force: bool = False


class DeviceUpdate(BaseModel):
    """Body of ``POST /__test/devices``: set any subset of device counters.

    Only fields present in the request are applied. ``used_bytes`` sets
    ``live_slot_count = used_bytes // slot_bytes``.
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    device_path: str
    locked_key_count: int = Field(default=0, ge=0)
    borrowed_slot_count: int = Field(default=0, ge=0)
    active_read_count: int = Field(default=0, ge=0)
    active_write_count: int = Field(default=0, ge=0)
    inflight_store_tasks: int = Field(default=0, ge=0)
    inflight_lookup_tasks: int = Field(default=0, ge=0)
    inflight_load_tasks: int = Field(default=0, ge=0)
    used_bytes: int = Field(default=0, ge=0)


class PresentDevice(BaseModel):
    """Body of ``POST /__test/present_devices``: a path the watcher sees.

    Declares a device that is physically present in the instance's watched
    directory without attaching it. Attached devices are always present;
    this adds the unattached ones (or overrides the physical state of an
    attached path, e.g. to report ``system-ram``).
    """

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    device_path: str
    mode: PhysicalMode = "devdax"
    size_bytes: int = Field(default=64 * GIB, ge=0)
    align_bytes: int = Field(default=DAX_ALIGN_BYTES, ge=0)


_DEVICE_COUNTER_FIELDS = (
    "locked_key_count",
    "borrowed_slot_count",
    "active_read_count",
    "active_write_count",
    "inflight_store_tasks",
    "inflight_lookup_tasks",
    "inflight_load_tasks",
)


@dataclass
class HttpResult:
    """Status code and JSON body a state operation wants the route to send."""

    status_code: int
    body: dict[str, object]


@dataclass
class InstanceEndpoint:
    """Advertised identity of one fake MP instance.

    Attributes:
        ip: ``ip`` reported by ``/instances`` (pod IP in Kubernetes).
        http_port: Primary advertised port; a listener serves the MP app there.
        alt_http_port: Spare advertised port served by the same MP app; an
            ``endpoint`` re-registration toggles between the two.
    """

    ip: str
    http_port: int
    alt_http_port: int


@dataclass
class DaxDevice:
    """Runtime state of one mapped DAX device (mirrors ``DaxDeviceEntry``)."""

    device_id: int
    device_path: str
    max_dax_size_bytes: int
    slot_bytes: int = SLOT_BYTES
    state: DeviceState = "active"
    live_slot_count: int = 0
    locked_key_count: int = 0
    borrowed_slot_count: int = 0
    active_read_count: int = 0
    active_write_count: int = 0
    inflight_store_tasks: int = 0
    inflight_lookup_tasks: int = 0
    inflight_load_tasks: int = 0

    @property
    def max_slots(self) -> int:
        """Slot capacity of the mapping."""
        return self.max_dax_size_bytes // self.slot_bytes

    @property
    def is_tombstone(self) -> bool:
        """True for ``closed``/``removed`` entries left by a remove."""
        return self.state in _TERMINAL_STATES

    @property
    def is_healthy(self) -> bool:
        """False once the device is closed, removed or failed."""
        return self.state not in _LOOKUP_EXCLUDED_STATES

    @property
    def counts_capacity(self) -> bool:
        """True while the device contributes to adapter capacity."""
        return self.state in _CAPACITY_STATES

    def physical(self) -> dict[str, object]:
        """Return the physical state of an attached device (always ``devdax``).

        Returns:
            See :func:`physical_status`.
        """
        return physical_status(
            self.device_path, self.device_id, "devdax", self.max_dax_size_bytes
        )

    def status(self, index: int) -> dict[str, object]:
        """Return the golden device status entry.

        Args:
            index: Position of the device in the adapter device list.

        Returns:
            Dict with exactly the golden ``devices[]`` keys, including the
            device's own ``physical`` state (the hotplug status overrides it
            from the watcher snapshot, as the real adapter does).
        """
        return {
            "is_healthy": self.is_healthy,
            "device_path": self.device_path,
            "max_dax_size_bytes": self.max_dax_size_bytes,
            "slot_bytes": self.slot_bytes,
            "max_slots": self.max_slots,
            "live_slot_count": self.live_slot_count,
            "locked_key_count": self.locked_key_count,
            "borrowed_slot_count": self.borrowed_slot_count,
            "active_read_count": self.active_read_count,
            "active_write_count": self.active_write_count,
            "closing": self.is_tombstone,
            "supports_restart_recovery": False,
            "index": index,
            "device_id": self.device_id,
            "state": self.state,
            "inflight_store_tasks": self.inflight_store_tasks,
            "inflight_lookup_tasks": self.inflight_lookup_tasks,
            "inflight_load_tasks": self.inflight_load_tasks,
            "physical": self.physical(),
        }


@dataclass
class StaleCapacity:
    """Capacity the coordinator keeps publishing during a delayed update.

    Attributes:
        capacity_bytes: The pre-change capacity still being published.
        publish_at: Clock reading after which the live capacity is published.
        pending: True while the stale value is in force.
    """

    capacity_bytes: int = 0
    publish_at: float = 0.0
    pending: bool = False


@dataclass
class ScenarioInstance:
    """One fake MP instance: identity, usage and DAX devices.

    Attributes:
        instance_id: ``mp-donor`` or ``mp-receiver``.
        worker_ip: Node IP from the fixture (``metadata.worker_ip``).
        endpoint: Advertised ip and the two listener ports.
        advertised_port: Port currently reported as ``http_port``.
        registration_time: Registration epoch reported by ``/instances``.
        used_bytes: l2/dax bytes reported by ``/instances/usage``.
        devices: Adapter device list including tombstones, in index order.
        next_device_id: ``device_id`` assigned to the next added device.
        stale_capacity: Delayed-capacity bookkeeping.
        present_devices: Physical state per path declared through the admin
            port; the watcher reports these in addition to every attached
            (or tombstoned) device path.
        watch_directory: The directory the fake presence watcher scans (the
            bootstrap device's directory, fixed at construction).
    """

    instance_id: str
    worker_ip: str
    endpoint: InstanceEndpoint
    advertised_port: int
    registration_time: float
    used_bytes: int
    devices: list[DaxDevice]
    next_device_id: int
    stale_capacity: StaleCapacity = field(default_factory=StaleCapacity)
    present_devices: dict[str, dict[str, object]] = field(default_factory=dict)
    watch_directory: str = ""

    def present(self) -> dict[str, dict[str, object]]:
        """Physical state of every path the watcher currently sees.

        Every adapter entry's path is present (a removed device's node stays
        on the host); admin-declared entries are added and take precedence.

        Returns:
            ``device_path -> physical entry`` (see :func:`physical_status`).
        """
        present: dict[str, dict[str, object]] = {}
        for device in self.devices:
            present.setdefault(device.device_path, device.physical())
        present.update(self.present_devices)
        return present

    def present_sorted(self) -> list[dict[str, object]]:
        """Return :meth:`present` entries ordered by path (copies).

        Returns:
            The ``present_devices`` list a watcher snapshot reports.
        """
        present = self.present()
        return [dict(present[path]) for path in sorted(present)]

    @property
    def capacity_bytes(self) -> int:
        """Live l2/dax capacity: slot capacity of every capacity-holding device."""
        return sum(
            device.max_slots * device.slot_bytes
            for device in self.devices
            if device.counts_capacity
        )


@dataclass
class AuditRecord:
    """One entry of the endpoint-local audit log."""

    seq: int
    kind: AuditKind
    service: str
    method: str
    path: str
    body: object
    status_code: int | None
    response: object
    mutation: dict[str, object] | None
    timestamp: float

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON-serializable dict.

        Returns:
            Dict with keys ``seq, kind, service, method, path, body,
            status_code, response, mutation, timestamp``.
        """
        return {
            "seq": self.seq,
            "kind": self.kind,
            "service": self.service,
            "method": self.method,
            "path": self.path,
            "body": self.body,
            "status_code": self.status_code,
            "response": self.response,
            "mutation": self.mutation,
            "timestamp": self.timestamp,
        }


class AuditLog:
    """Ordered, thread-safe audit log with a strictly increasing ``seq``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[AuditRecord] = []
        self._seq = 0

    @property
    def seq(self) -> int:
        """Sequence number of the newest record (0 when empty)."""
        with self._lock:
            return self._seq

    def record_request(
        self, service: str, method: str, path: str, body: object
    ) -> AuditRecord:
        """Append a ``request`` record.

        Args:
            service: ``coordinator``, ``mp-donor`` or ``mp-receiver``.
            method: HTTP method.
            path: Request path including any query string.
            body: Decoded JSON body, raw text, or ``None`` when empty.

        Returns:
            The appended record.
        """
        return self._append("request", service, method, path, body, None, None, None)

    def record_response(
        self, service: str, method: str, path: str, status_code: int, response: object
    ) -> AuditRecord:
        """Append a ``response`` record.

        Args:
            service: Service that answered.
            method: HTTP method of the request.
            path: Request path including any query string.
            status_code: HTTP status sent.
            response: Decoded JSON response body (or raw text).

        Returns:
            The appended record.
        """
        return self._append(
            "response", service, method, path, None, status_code, response, None
        )

    def record_mutation(
        self, service: str, method: str, path: str, mutation: dict[str, object]
    ) -> AuditRecord:
        """Append a ``mutation`` record.

        Args:
            service: Instance (or ``coordinator``) whose state changed.
            method: HTTP method of the request that caused it.
            path: Path of the request that caused it.
            mutation: Description, e.g. ``{"device_path", "from_state",
                "to_state", ...}`` or a usage/identity change.

        Returns:
            The appended record.
        """
        return self._append(
            "mutation", service, method, path, None, None, None, mutation
        )

    def after(self, seq: int) -> list[dict[str, object]]:
        """Return every record with ``seq`` greater than the given value.

        Args:
            seq: Exclusive lower bound; 0 returns everything.

        Returns:
            Records in sequence order.
        """
        with self._lock:
            return [r.to_dict() for r in self._records if r.seq > seq]

    def clear(self) -> None:
        """Drop all records and restart ``seq`` from 1."""
        with self._lock:
            self._records.clear()
            self._seq = 0

    def _append(
        self,
        kind: AuditKind,
        service: str,
        method: str,
        path: str,
        body: object,
        status_code: int | None,
        response: object,
        mutation: dict[str, object] | None,
    ) -> AuditRecord:
        with self._lock:
            self._seq += 1
            record = AuditRecord(
                seq=self._seq,
                kind=kind,
                service=service,
                method=method,
                path=path,
                body=body,
                status_code=status_code,
                response=response,
                mutation=mutation,
                timestamp=time.time(),
            )
            self._records.append(record)
            return record


class ScenarioState:
    """Fleet, device, fault and audit state shared by the four listeners.

    Args:
        fixture_path: ``two_server_local_dax.yaml``; re-read on every reset.
        endpoints: Advertised endpoint per instance id; must contain exactly
            ``mp-donor`` and ``mp-receiver``.
        clock: Monotonic clock used for delayed capacity publication
            (injectable so tests can advance time deterministically).

    Raises:
        ValueError: If ``endpoints`` does not cover exactly both instances or
            the fixture does not describe exactly two nodes.
    """

    def __init__(
        self,
        fixture_path: Path,
        endpoints: dict[str, InstanceEndpoint],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if set(endpoints) != set(INSTANCE_IDS):
            raise ValueError(f"endpoints must be exactly {INSTANCE_IDS}")
        self._fixture_path = fixture_path
        self._endpoints = endpoints
        self._clock = clock
        self._lock = threading.RLock()
        self.audit = AuditLog()
        self.barriers = BarrierRegistry()
        self._instances: dict[str, ScenarioInstance] = {}
        self._faults = ActiveFaults(coordinator=CoordinatorFaults(), mp={})
        # Number of ``/instances`` reads since the identity flip was armed.
        self._instances_reads = 0
        self.reset()

    # ----------------------------------------------------------------- admin

    def reset(self) -> dict[str, object]:
        """Reload the fixture and defaults; clear faults, barriers and audit.

        Returns:
            The state snapshot after the reset (see :meth:`snapshot`).
        """
        with self._lock:
            fixture = Fixture.model_validate(
                yaml.safe_load(self._fixture_path.read_text())
            )
            if len(fixture.nodes) != 2:
                raise ValueError("fixture must describe exactly two nodes")
            node_ips = sorted(fixture.nodes)
            self._instances = {}
            for instance_id, node_ip in zip(INSTANCE_IDS, node_ips, strict=True):
                self._instances[instance_id] = self._build_instance(
                    instance_id, node_ip, fixture.nodes[node_ip]
                )
            self._faults = ActiveFaults(
                coordinator=CoordinatorFaults(),
                mp={instance_id: MpFaults() for instance_id in INSTANCE_IDS},
            )
            self._instances_reads = 0
            self.barriers.clear()
            self.audit.clear()
            return self._snapshot_locked()

    def snapshot(self) -> dict[str, object]:
        """Return the full state as JSON.

        Returns:
            ``{"seq", "fixture_path", "instances": [...], "faults",
            "barriers": [...], "audit_records"}`` where each instance carries
            identity, ports, usage, capacities and device statuses.
        """
        with self._lock:
            return self._snapshot_locked()

    def apply_faults(self, patch: FaultSpec) -> dict[str, object]:
        """Merge a fault patch into the active faults.

        Arming (or clearing) ``identity_flip`` restarts the read counter.

        Args:
            patch: Validated patch.

        Returns:
            The active faults as JSON.

        Raises:
            KeyError: If ``patch.mp`` names an unknown instance.
        """
        with self._lock:
            self._faults.merge(patch)
            if "identity_flip" in patch.coordinator.model_fields_set:
                self._instances_reads = 0
            return self._faults.model_dump()

    def clear_faults(self) -> dict[str, object]:
        """Reset every fault to its default.

        Returns:
            The active (default) faults as JSON.
        """
        with self._lock:
            self._faults = ActiveFaults(
                coordinator=CoordinatorFaults(),
                mp={instance_id: MpFaults() for instance_id in INSTANCE_IDS},
            )
            self._instances_reads = 0
            return self._faults.model_dump()

    def set_usage(self, instance_id: str, used_bytes: int) -> dict[str, object]:
        """Set the l2/dax ``used_bytes`` the coordinator reports.

        Args:
            instance_id: Target instance.
            used_bytes: New used bytes (the pressure numerator).

        Returns:
            ``{"instance_id", "used_bytes", "capacity_bytes"}``.

        Raises:
            KeyError: If the instance is unknown.
        """
        with self._lock:
            instance = self._instances[instance_id]
            before = instance.used_bytes
            instance.used_bytes = used_bytes
            self.audit.record_mutation(
                instance_id,
                "POST",
                "/__test/usage",
                {
                    "instance_id": instance_id,
                    "used_bytes_before": before,
                    "used_bytes_after": used_bytes,
                },
            )
            return {
                "instance_id": instance_id,
                "used_bytes": used_bytes,
                "capacity_bytes": instance.capacity_bytes,
            }

    def update_device(self, update: DeviceUpdate) -> dict[str, object]:
        """Set a subset of counters on one non-tombstone device.

        Args:
            update: Fields present in ``update.model_fields_set`` are applied.

        Returns:
            The device status entry after the update.

        Raises:
            KeyError: If the instance is unknown.
            LookupError: If no non-tombstone device has that path.
        """
        with self._lock:
            instance = self._instances[update.instance_id]
            found = self._find_device_locked(instance, update.device_path)
            if found is None:
                raise LookupError(update.device_path)
            index, device = found
            changes: dict[str, object] = {}
            for name in _DEVICE_COUNTER_FIELDS:
                if name in update.model_fields_set:
                    setattr(device, name, getattr(update, name))
                    changes[name] = getattr(update, name)
            if "used_bytes" in update.model_fields_set:
                device.live_slot_count = update.used_bytes // device.slot_bytes
                changes["live_slot_count"] = device.live_slot_count
            self.audit.record_mutation(
                update.instance_id,
                "POST",
                "/__test/devices",
                {
                    "instance_id": update.instance_id,
                    "device_path": device.device_path,
                    "index": index,
                    "changes": changes,
                },
            )
            return device.status(index)

    def declare_present_device(self, declared: PresentDevice) -> dict[str, object]:
        """Make the instance's watcher report a path as physically present.

        Nothing is attached: the coordinator decides whether the device is
        also assigned by the outside service and issues the add itself.

        Args:
            declared: The path and its physical state.

        Returns:
            The physical entry now reported under ``watcher.present_devices``.

        Raises:
            KeyError: If the instance is unknown.
        """
        with self._lock:
            instance = self._instances[declared.instance_id]
            entry = physical_status(
                declared.device_path,
                len(instance.present()),
                declared.mode,
                declared.size_bytes,
                declared.align_bytes,
            )
            instance.present_devices[declared.device_path] = entry
            self.audit.record_mutation(
                declared.instance_id,
                "POST",
                "/__test/present_devices",
                {
                    "instance_id": declared.instance_id,
                    "device_path": declared.device_path,
                    "mode": declared.mode,
                    "size_bytes": declared.size_bytes,
                },
            )
            return dict(entry)

    def reregister(self, instance_id: str, bump: IdentityBump) -> dict[str, object]:
        """Permanently change an instance's registration identity.

        ``registration_time`` assigns a fresh, strictly greater epoch.
        ``endpoint`` toggles the advertised ``http_port`` between the primary
        and the alternate listener (both serve the same MP app, so the
        instance stays reachable). ``both`` does both.

        Args:
            instance_id: Target instance.
            bump: Which identity field(s) change.

        Returns:
            ``{"instance_id", "ip", "http_port", "registration_time"}``.

        Raises:
            KeyError: If the instance is unknown.
        """
        with self._lock:
            instance = self._instances[instance_id]
            before = self._identity_locked(instance)
            if bump in ("registration_time", "both"):
                instance.registration_time = _wall_clock_after(
                    instance.registration_time
                )
            if bump in ("endpoint", "both"):
                endpoint = instance.endpoint
                instance.advertised_port = (
                    endpoint.alt_http_port
                    if instance.advertised_port == endpoint.http_port
                    else endpoint.http_port
                )
            after = self._identity_locked(instance)
            self.audit.record_mutation(
                instance_id,
                "POST",
                f"/__test/instances/{instance_id}/reregister",
                {"bump": bump, "before": before, "after": after},
            )
            return after

    # ----------------------------------------------------------- coordinator

    def list_instances(self) -> HttpResult:
        """Serve ``GET /instances``.

        Counts one read for the identity-flip fault. Unregistered instances
        are omitted; ``worker_ip_override`` replaces or omits
        ``metadata.worker_ip``.

        Returns:
            200 ``{"instances": [...]}`` or 503 when unavailable.
        """
        with self._lock:
            unavailable = self._coordinator_unavailable_locked()
            if unavailable is not None:
                return unavailable
            faults = self._faults.coordinator
            self._instances_reads += 1
            flip = faults.identity_flip
            flip_now = (
                flip is not None and self._instances_reads % flip.every_n_reads == 0
            )
            instances: list[dict[str, object]] = []
            for instance in self._instances.values():
                if instance.instance_id in faults.unregistered:
                    continue
                identity = self._identity_locked(instance)
                if flip_now and flip is not None:
                    self._apply_flip(identity, instance, flip)
                metadata: dict[str, object] = {"worker_ip": instance.worker_ip}
                if instance.instance_id in faults.worker_ip_override:
                    override = faults.worker_ip_override[instance.instance_id]
                    metadata = {} if override is None else {"worker_ip": override}
                instances.append(
                    {
                        **identity,
                        "metadata": metadata,
                        "p2p_advertised_url": "",
                        "mq_port": 0,
                    }
                )
            return HttpResult(200, {"instances": instances})

    def fleet_usage(self) -> HttpResult:
        """Serve ``GET /instances/usage``.

        Returns:
            200 ``{"instances": [...], "shared_modules": []}`` or 503.
        """
        with self._lock:
            unavailable = self._coordinator_unavailable_locked()
            if unavailable is not None:
                return unavailable
            return HttpResult(
                200,
                {
                    "instances": [
                        self._instance_usage_locked(instance)
                        for instance in self._instances.values()
                    ],
                    "shared_modules": [],
                },
            )

    def instance_usage(self, instance_id: str) -> HttpResult:
        """Serve ``GET /instances/{instance_id}/usage``.

        Args:
            instance_id: Instance to report on.

        Returns:
            200 with the instance status, 404 for an unknown id, or 503.
        """
        with self._lock:
            unavailable = self._coordinator_unavailable_locked()
            if unavailable is not None:
                return unavailable
            instance = self._instances.get(instance_id)
            if instance is None:
                return HttpResult(404, {"detail": f"unknown instance {instance_id!r}"})
            return HttpResult(200, self._instance_usage_locked(instance))

    def coordinator_health(self) -> HttpResult:
        """Serve ``GET /healthz``.

        Returns:
            200 ``{"status": "ok"}`` or 503 when unavailable.
        """
        with self._lock:
            unavailable = self._coordinator_unavailable_locked()
            if unavailable is not None:
                return unavailable
            return HttpResult(200, {"status": "ok"})

    # -------------------------------------------------------------------- MP

    def healthcheck(self, instance_id: str) -> HttpResult:
        """Serve ``GET /healthcheck`` of one MP instance.

        Returns:
            200 ``{"status": "healthy"}`` or 503 under ``status_unavailable``.
        """
        with self._lock:
            if self._faults.mp[instance_id].status_unavailable:
                return HttpResult(
                    503, {"status": "unhealthy", "reason": "engine not initialized"}
                )
            return HttpResult(200, {"status": "healthy"})

    def mp_status(self, instance_id: str) -> HttpResult:
        """Serve ``GET /status`` of one MP instance with the golden shape.

        Returns:
            200 with the engine status or 503 under ``status_unavailable``.
        """
        with self._lock:
            faults = self._faults.mp[instance_id]
            if faults.status_unavailable:
                return HttpResult(503, {"error": "engine not initialized"})
            instance = self._instances[instance_id]
            adapter = self._adapter_report_locked(instance, faults)
            adapters = [dict(adapter) for _ in range(faults.adapters)]
            healthy = all(bool(a["is_healthy"]) for a in adapters)
            controller_counts = {
                "num_l2_adapters": len(adapters),
                "num_active_adapters": len(adapters),
                "num_draining_adapters": 0,
            }
            return HttpResult(
                200,
                {
                    "is_healthy": healthy,
                    "engine_type": "MPCacheServer",
                    "chunk_size": 256,
                    "hash_algorithm": "blake3",
                    "active_sessions": 0,
                    "storage_manager": {
                        "is_healthy": healthy,
                        "l1_manager": {
                            "is_healthy": True,
                            "total_object_count": 0,
                            "write_locked_count": 0,
                            "read_locked_count": 0,
                            "temporary_count": 0,
                            "memory_used_bytes": 0,
                            "memory_total_bytes": L1_CAPACITY_BYTES,
                            "memory_configured_bytes": L1_CAPACITY_BYTES,
                            "memory_usage_ratio": 0.0,
                            "write_ttl_seconds": 60,
                            "read_ttl_seconds": 60,
                        },
                        "store_controller": {
                            "is_healthy": True,
                            "thread_alive": True,
                            "pending_keys_count": 0,
                            "in_flight_task_count": 0,
                            **controller_counts,
                        },
                        "prefetch_controller": {
                            "is_healthy": True,
                            "thread_alive": True,
                            "max_in_flight": 8,
                            "submission_queue_size": 0,
                            "pending_queue_size": 0,
                            "in_flight_request_count": 0,
                            "lookup_phase_count": 0,
                            "load_phase_count": 0,
                            "completed_results_count": 0,
                            **controller_counts,
                        },
                        "l1_eviction_controller": {
                            "is_healthy": True,
                            "thread_alive": True,
                            "eviction_policy": "noop",
                            "trigger_watermark": 0.8,
                            "eviction_ratio": 0.2,
                        },
                        "l2_eviction_controller": {
                            "is_healthy": True,
                            "thread_alive": True,
                            "adapters": [],
                        },
                        "l2_adapters": adapters,
                        "num_l2_adapters": len(adapters),
                    },
                    "active_prefetch_jobs": 0,
                    "registered_gpu_ids": [],
                    "cache_context_meta": {},
                },
            )

    def dax_status(self, instance_id: str) -> HttpResult:
        """Serve ``GET /reconfigure/dax/status`` with the golden shape.

        Returns:
            200 with the reconfigure status (``enabled=false`` and no
            adapters under the ``adapters=0`` fault) or 503.
        """
        with self._lock:
            faults = self._faults.mp[instance_id]
            if faults.status_unavailable:
                return HttpResult(503, {"error": "engine not initialized"})
            instance = self._instances[instance_id]
            hotplug = self._hotplug_status_locked(instance, faults)
            adapters: list[dict[str, object]] = [
                {
                    "backend": "dax",
                    "supported_operations": list(_DAX_RECONFIGURE_OPERATIONS),
                    "status": dict(hotplug),
                    "adapter_index": adapter_index,
                    "l2_adapter_index": adapter_index,
                }
                for adapter_index in range(faults.adapters)
            ]
            return HttpResult(
                200,
                {
                    "enabled": bool(adapters),
                    "backend": "dax",
                    "num_adapters": len(adapters),
                    "adapters": adapters,
                },
            )

    def remove_device(self, instance_id: str, request: DaxRemoveRequest) -> HttpResult:
        """Apply ``POST /reconfigure/dax/remove`` (drain, evict or migrate).

        Semantics mirror ``DaxL2Adapter.hotplug_remove_device`` with these
        documented simplifications: ``migrate`` behaves exactly like
        ``evict``; a blocked evict leaves the device ``draining``; ``force``
        skips the locked/borrowed check but not the ``evict_409_count``
        fault.

        Args:
            instance_id: Target instance.
            request: Validated request body.

        Returns:
            200 golden drain/remove response; 404 unknown adapter or device
            (tombstones are not matched); 403 hotplug disabled; 409 blocked
            (``{"status": "blocked", "reason", "locked_key_count",
            "borrowed_slot_count"}``); 500 under ``remove_route_failure``.
        """
        with self._lock:
            faults = self._faults.mp[instance_id]
            if faults.remove_route_failure:
                return HttpResult(500, {"error": "internal error"})
            gate = self._hotplug_gate_locked(faults, request.adapter_index)
            if gate is not None:
                return gate
            instance = self._instances[instance_id]
            device_path = request.device_path.strip()
            if not device_path:
                return HttpResult(400, {"error": "device_path must be non-empty"})
            found = self._find_device_locked(instance, device_path)
            if found is None:
                return HttpResult(404, {"error": "DAX device not found"})
            index, device = found
            path = "/reconfigure/dax/remove"
            self._transition_locked(instance, index, device, "draining", path)
            if request.mode == "drain":
                return HttpResult(
                    200,
                    {
                        "status": "ok",
                        "operation": "drain",
                        "adapter_index": 0,
                        "device_path": device.device_path,
                        "index": index,
                        "state": device.state,
                    },
                )
            busy = (
                device.locked_key_count + device.borrowed_slot_count > 0
                and not request.force
            )
            if faults.evict_409_count > 0:
                faults.evict_409_count -= 1
                busy = True
            if busy:
                return HttpResult(
                    409,
                    {
                        "status": "blocked",
                        "reason": _BLOCKED_REASON,
                        "locked_key_count": device.locked_key_count,
                        "borrowed_slot_count": device.borrowed_slot_count,
                    },
                )
            freed = device.live_slot_count
            device.live_slot_count = 0
            self._transition_locked(instance, index, device, "removed", path)
            return HttpResult(
                200,
                {
                    "status": "ok",
                    "operation": "remove",
                    "adapter_index": 0,
                    "device_path": device.device_path,
                    "index": index,
                    "moved_keys": 0,
                    "moved_bytes": 0,
                    "deleted_keys": freed,
                    "source_slots_freed": freed,
                    "state": "removed",
                },
            )

    def add_device(
        self, instance_id: str, adapter_index: int, device_path: str, size_bytes: int
    ) -> HttpResult:
        """Apply ``POST /reconfigure/dax/add``.

        Mirrors ``DaxL2Adapter.hotplug_add_device``: an existing non-tombstone
        entry with the same path and size is returned unchanged (a draining
        device stays draining); a different size is a 409; otherwise a new
        ``active`` entry is appended with the next index and device id. The
        ``add_fail_count`` / ``add_always_fail`` faults only affect the
        new-entry path, like a real mmap failure would.

        Args:
            instance_id: Target instance.
            adapter_index: Requested adapter index.
            device_path: Path to map.
            size_bytes: Resolved mapping size in bytes.

        Returns:
            200 golden add response, 404 unknown adapter, 403 hotplug
            disabled, 400 invalid path/size or mapping failure, 409 size
            conflict.
        """
        with self._lock:
            faults = self._faults.mp[instance_id]
            gate = self._hotplug_gate_locked(faults, adapter_index)
            if gate is not None:
                return gate
            instance = self._instances[instance_id]
            path = device_path.strip()
            if not path:
                return HttpResult(400, {"error": "device_path must be non-empty"})
            if size_bytes // SLOT_BYTES <= 0:
                return HttpResult(400, {"error": "size_bytes does not fit one slot"})
            found = self._find_device_locked(instance, path)
            if found is not None:
                index, device = found
                if device.max_dax_size_bytes != size_bytes:
                    return HttpResult(
                        409,
                        {"error": "device_path already active with a different size"},
                    )
                return HttpResult(200, self._add_response(device.status(index)))
            if faults.add_always_fail:
                return HttpResult(400, {"error": "failed to map DAX device"})
            if faults.add_fail_count > 0:
                faults.add_fail_count -= 1
                return HttpResult(400, {"error": "failed to map DAX device"})
            capacity_before = instance.capacity_bytes
            device = DaxDevice(
                device_id=instance.next_device_id,
                device_path=path,
                max_dax_size_bytes=size_bytes,
            )
            instance.next_device_id += 1
            instance.devices.append(device)
            index = len(instance.devices) - 1
            self._record_transition_locked(
                instance, index, device, None, "/reconfigure/dax/add", capacity_before
            )
            return HttpResult(200, self._add_response(device.status(index)))

    # ---------------------------------------------------------------- private

    def _build_instance(
        self, instance_id: str, node_ip: str, node: FixtureNode
    ) -> ScenarioInstance:
        """Create an instance from its fixture node (bootstrap + assigned)."""
        endpoint = self._endpoints[instance_id]
        used_per_device = _DEFAULT_DEVICE_USED_BYTES.get(instance_id, ())
        devices: list[DaxDevice] = []
        for entry in node.devices:
            if entry.role != "bootstrap" and entry.state != "assigned":
                continue
            index = len(devices)
            used = used_per_device[index] if index < len(used_per_device) else 0
            devices.append(
                DaxDevice(
                    device_id=index,
                    device_path=entry.path,
                    max_dax_size_bytes=entry.size_gib * GIB,
                    live_slot_count=used // SLOT_BYTES,
                )
            )
        if not devices:
            raise ValueError(f"fixture node {node_ip} has no bootstrap device")
        return ScenarioInstance(
            instance_id=instance_id,
            worker_ip=node_ip,
            endpoint=endpoint,
            advertised_port=endpoint.http_port,
            registration_time=time.time(),
            used_bytes=_DEFAULT_USED_BYTES.get(instance_id, 0),
            devices=devices,
            next_device_id=len(devices),
            watch_directory=posixpath.dirname(devices[0].device_path),
        )

    def _snapshot_locked(self) -> dict[str, object]:
        return {
            "seq": self.audit.seq,
            "fixture_path": str(self._fixture_path),
            "instances": [
                {
                    **self._identity_locked(instance),
                    "worker_ip": instance.worker_ip,
                    "primary_http_port": instance.endpoint.http_port,
                    "alt_http_port": instance.endpoint.alt_http_port,
                    "registered": instance.instance_id
                    not in self._faults.coordinator.unregistered,
                    "used_bytes": instance.used_bytes,
                    "capacity_bytes": instance.capacity_bytes,
                    "published_capacity_bytes": self._published_capacity_locked(
                        instance
                    ),
                    "l1_capacity_bytes": L1_CAPACITY_BYTES,
                    "devices": [
                        device.status(index)
                        for index, device in enumerate(instance.devices)
                    ],
                    "present_devices": instance.present_sorted(),
                }
                for instance in self._instances.values()
            ],
            "faults": self._faults.model_dump(),
            "barriers": self.barriers.snapshot(),
            "audit_records": self.audit.seq,
        }

    def _identity_locked(self, instance: ScenarioInstance) -> dict[str, object]:
        return {
            "instance_id": instance.instance_id,
            "ip": instance.endpoint.ip,
            "http_port": instance.advertised_port,
            "registration_time": instance.registration_time,
        }

    @staticmethod
    def _apply_flip(
        identity: dict[str, object], instance: ScenarioInstance, flip: IdentityFlip
    ) -> None:
        if flip.instance_id != instance.instance_id:
            return
        if flip.field == "registration_time":
            identity["registration_time"] = instance.registration_time + 1.0
        else:
            identity["http_port"] = instance.advertised_port + 1

    def _coordinator_unavailable_locked(self) -> HttpResult | None:
        if self._faults.coordinator.unavailable:
            return HttpResult(503, {"error": "unavailable"})
        return None

    def _published_capacity_locked(self, instance: ScenarioInstance) -> int:
        stale = instance.stale_capacity
        if stale.pending and self._clock() < stale.publish_at:
            return stale.capacity_bytes
        stale.pending = False
        return instance.capacity_bytes

    def _note_capacity_change_locked(
        self, instance: ScenarioInstance, capacity_before: int
    ) -> None:
        delay = self._faults.coordinator.delayed_capacity_seconds
        if delay <= 0 or capacity_before == instance.capacity_bytes:
            return
        stale = instance.stale_capacity
        if not stale.pending:
            stale.capacity_bytes = capacity_before
            stale.pending = True
        stale.publish_at = self._clock() + delay

    def _instance_usage_locked(self, instance: ScenarioInstance) -> dict[str, object]:
        faults = self._faults.coordinator
        instance_id = instance.instance_id
        declared = instance_id not in faults.undeclared_capacity
        dax_capacity = self._published_capacity_locked(instance) if declared else 0
        l1_capacity = L1_CAPACITY_BYTES if declared else 0
        dax_ratio = _usage_ratio(instance.used_bytes, dax_capacity)
        if instance_id in faults.null_ratio:
            dax_ratio = None
        return {
            "instance_id": instance_id,
            "registered": instance_id not in faults.unregistered,
            "declared_capacity": declared,
            "modules": [
                _module_status(
                    "l1", "dram", False, 0, l1_capacity, _usage_ratio(0, l1_capacity)
                ),
                _module_status(
                    "l2",
                    "dax",
                    instance_id in faults.shared_dax,
                    instance.used_bytes,
                    dax_capacity,
                    dax_ratio,
                ),
            ],
        }

    def _hotplug_status_locked(
        self, instance: ScenarioInstance, faults: MpFaults
    ) -> dict[str, object]:
        present = instance.present()
        devices = [d.status(i) for i, d in enumerate(instance.devices)]
        for status in devices:
            path = str(status["device_path"])
            status["physical"] = dict(present[path])
        return {
            "hotplug_enabled": not faults.hotplug_disabled,
            "slot_bytes": SLOT_BYTES,
            "total_capacity_bytes": instance.capacity_bytes,
            "total_used_bytes": sum(
                d.live_slot_count * d.slot_bytes
                for d in instance.devices
                if d.counts_capacity
            ),
            "devices": devices,
            "watcher": {
                "enabled": True,
                "directory": instance.watch_directory,
                "interval_seconds": WATCHER_INTERVAL_SECONDS,
                "last_scan_at": time.time(),
                "present_devices": instance.present_sorted(),
            },
        }

    def _adapter_report_locked(
        self, instance: ScenarioInstance, faults: MpFaults
    ) -> dict[str, object]:
        hotplug = self._hotplug_status_locked(instance, faults)
        serving = [d for d in instance.devices if not d.is_tombstone]
        healthy = all(d.is_healthy for d in serving) and not faults.unhealthy
        return {
            "is_healthy": healthy,
            "type": "dax",
            "device_path": instance.devices[0].device_path,
            "max_dax_size_bytes": instance.capacity_bytes,
            "slot_bytes": SLOT_BYTES,
            "live_slot_count": sum(d.live_slot_count for d in instance.devices),
            "locked_key_count": sum(d.locked_key_count for d in instance.devices),
            "borrowed_slot_count": sum(d.borrowed_slot_count for d in instance.devices),
            "inflight_store_tasks": sum(
                d.inflight_store_tasks for d in instance.devices
            ),
            "inflight_lookup_tasks": sum(
                d.inflight_lookup_tasks for d in instance.devices
            ),
            "inflight_load_tasks": sum(d.inflight_load_tasks for d in instance.devices),
            "closing": faults.closing,
            "supports_restart_recovery": False,
            "hotplug_enabled": not faults.hotplug_disabled,
            "devices": hotplug["devices"],
            "num_devices": len(instance.devices),
        }

    @staticmethod
    def _hotplug_gate_locked(faults: MpFaults, adapter_index: int) -> HttpResult | None:
        if adapter_index < 0 or adapter_index >= faults.adapters:
            return HttpResult(404, {"error": "dax adapter not found"})
        if faults.hotplug_disabled:
            return HttpResult(403, {"error": "DAX hotplug is disabled"})
        return None

    @staticmethod
    def _find_device_locked(
        instance: ScenarioInstance, device_path: str
    ) -> tuple[int, DaxDevice] | None:
        for index, device in enumerate(instance.devices):
            if device.device_path != device_path:
                continue
            if device.state in _LOOKUP_EXCLUDED_STATES:
                continue
            return index, device
        return None

    @staticmethod
    def _add_response(device_status: dict[str, object]) -> dict[str, object]:
        return {
            "status": "ok",
            "operation": "add",
            "adapter_index": 0,
            "device": device_status,
        }

    def _transition_locked(
        self,
        instance: ScenarioInstance,
        index: int,
        device: DaxDevice,
        to_state: DeviceState,
        path: str,
    ) -> None:
        """Move a device to ``to_state`` and audit it (no-op if unchanged)."""
        if device.state == to_state:
            return
        capacity_before = instance.capacity_bytes
        from_state = device.state
        device.state = to_state
        self._record_transition_locked(
            instance, index, device, from_state, path, capacity_before
        )

    def _record_transition_locked(
        self,
        instance: ScenarioInstance,
        index: int,
        device: DaxDevice,
        from_state: DeviceState | None,
        path: str,
        capacity_before: int,
    ) -> None:
        self._note_capacity_change_locked(instance, capacity_before)
        self.audit.record_mutation(
            instance.instance_id,
            "POST",
            path,
            {
                "instance_id": instance.instance_id,
                "device_path": device.device_path,
                "index": index,
                "device_id": device.device_id,
                "from_state": from_state,
                "to_state": device.state,
                "capacity_bytes_before": capacity_before,
                "capacity_bytes_after": instance.capacity_bytes,
            },
        )
