# SPDX-License-Identifier: Apache-2.0
"""Inventory, audit log, and optional persistence of the mock allocation service.

The inventory is a fixed set of Device-DAX paths per worker IP.  Worker IP,
path, size and role (``bootstrap`` or ``runtime``) never change; only the
``free``/``assigned`` state of runtime devices does.  Bootstrap devices are part
of the topology fixture so that other test services can share it, but the
allocator never lists or moves them.

One ``asyncio.Lock`` guards every mutation and every audit append.  After each
mutation the state re-checks the conservation invariant (per node and globally,
``free + assigned == fixed runtime inventory``) and raises if it is violated:
that is a self-check of the mock, surfaced as a 500 on the public port.
"""

# Standard
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal
import asyncio
import json
import os
import time

# Third Party
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
import yaml

AUDIT_CAPACITY: int = 10_000
"""Maximum number of audit records kept in memory (oldest are dropped)."""

STATE_SCHEMA_VERSION: int = 1
"""``schema_version`` accepted in fixtures and persisted state files."""


class DeviceRole(str, Enum):
    """Fixed role of a device: bootstrap devices are never managed."""

    BOOTSTRAP = "bootstrap"
    RUNTIME = "runtime"


class DeviceState(str, Enum):
    """Mutable state of a runtime device."""

    FREE = "free"
    ASSIGNED = "assigned"


class Operation(str, Enum):
    """Public operation an audit record belongs to."""

    STATUS = "status"
    DEALLOCATE = "deallocate"
    ALLOCATE = "allocate"


class AuditKind(str, Enum):
    """Kind of an audit record."""

    REQUEST = "request"
    RESPONSE = "response"
    MUTATION = "mutation"


class MockServiceError(Exception):
    """Rejection of a public request, carrying the HTTP status to respond with.

    Attributes:
        status_code: HTTP status code for the public error response.
        message: Human-readable reason placed in the ``error`` body field.
    """

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class FixtureDevice(BaseModel):
    """One device entry of a fixture or persisted state document."""

    model_config = ConfigDict(extra="forbid")

    path: str
    size_gib: int = Field(gt=0, strict=True)
    role: DeviceRole
    state: DeviceState

    @field_validator("path")
    @classmethod
    def _path_must_be_absolute_and_normalized(cls, value: str) -> str:
        if not value.startswith("/") or os.path.normpath(value) != value:
            raise ValueError(f"device path must be absolute and normalized: {value!r}")
        return value

    @model_validator(mode="after")
    def _bootstrap_must_be_assigned(self) -> "FixtureDevice":
        if self.role is DeviceRole.BOOTSTRAP and self.state is not DeviceState.ASSIGNED:
            raise ValueError(f"bootstrap device {self.path!r} must be assigned")
        return self


class FixtureNode(BaseModel):
    """One worker entry of a fixture or persisted state document."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    devices: list[FixtureDevice]


class InventoryDocument(BaseModel):
    """Schema shared by the YAML fixture and the persisted JSON state file.

    ``seen_request_ids`` is only present in persisted state files.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    nodes: dict[str, FixtureNode]
    seen_request_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _paths_must_be_globally_unique(self) -> "InventoryDocument":
        seen: set[str] = set()
        for ip, node in self.nodes.items():
            if not ip:
                raise ValueError("node IP must not be empty")
            for device in node.devices:
                if device.path in seen:
                    raise ValueError(f"duplicate device path {device.path!r}")
                seen.add(device.path)
        return self


@dataclass
class Device:
    """A device of the fixed inventory; only ``state`` may change."""

    path: str
    size_gib: int
    role: DeviceRole
    state: DeviceState


@dataclass
class Node:
    """A worker and its devices keyed by path in lexicographic order."""

    ip: str
    name: str
    devices: dict[str, Device]

    def runtime_gib(self, state: DeviceState) -> int:
        """Return the total size in GiB of runtime devices in ``state``."""
        return sum(
            device.size_gib
            for device in self.devices.values()
            if device.role is DeviceRole.RUNTIME and device.state is state
        )


@dataclass(frozen=True)
class DeallocationResult:
    """Outcome of a committed deallocation."""

    request_id: str
    target_node: str
    device_path: str
    released_size_gib: int


@dataclass(frozen=True)
class AllocationResult:
    """Outcome of a committed allocation."""

    request_id: str
    target_node: str
    device_path: str
    requested_size_gib: int
    granted_size_gib: int


@dataclass(frozen=True)
class AuditRecord:
    """One entry of the ordered request/response/mutation audit log.

    Attributes:
        seq: Strictly increasing sequence number starting at 1.
        kind: ``request``, ``response`` or ``mutation``.
        operation: Public operation the record belongs to.
        request_id: ``request_id`` of the request, or ``""`` for status.
        target_node: ``target_node`` of the request, or ``""``.
        device_path: ``device_path`` of the request or mutation, or ``""``.
        status_code: HTTP status for responses; ``0`` for requests, mutations
            and responses whose connection was dropped before a body was sent.
        body: Exact request or response JSON, or for mutations
            ``{"path", "from_state", "to_state", "node"}``.
        timestamp: ``time.time()`` when the record was appended.
    """

    seq: int
    kind: AuditKind
    operation: Operation
    request_id: str
    target_node: str
    device_path: str
    status_code: int
    body: dict[str, object]
    timestamp: float

    def to_dict(self) -> dict[str, object]:
        """Return the record as a JSON object."""
        return {
            "seq": self.seq,
            "kind": self.kind.value,
            "operation": self.operation.value,
            "request_id": self.request_id,
            "target_node": self.target_node,
            "device_path": self.device_path,
            "status_code": self.status_code,
            "body": self.body,
            "timestamp": self.timestamp,
        }


def load_inventory_document(path: Path) -> InventoryDocument:
    """Parse and validate a fixture (YAML) or persisted state (JSON) file.

    Args:
        path: File to load.  ``.json`` files are parsed as JSON, anything else
            as YAML.

    Returns:
        The validated document.

    Raises:
        ValueError: If the file is not a mapping or fails schema validation.
        OSError: If the file cannot be read.
    """
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    try:
        return InventoryDocument.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid inventory document: {exc}") from exc


class MockAllocatorState:
    """Lock-protected inventory, request-ID ledger, audit log and persistence.

    Construction loads the persisted state file when one is configured and
    exists, otherwise the fixture.  When a state file is configured it is
    rewritten atomically after every committed mutation and on reset.
    """

    def __init__(self, fixture_path: Path, state_file: Path | None) -> None:
        """Load the initial inventory.

        Args:
            fixture_path: YAML fixture describing the topology.
            state_file: JSON file to persist state to, or ``None`` to keep the
                state purely in memory.

        Raises:
            ValueError: If the fixture or state file fails validation.
            OSError: If a file cannot be read or written.
        """
        self._fixture_path = fixture_path
        self._state_file = state_file
        self._lock = asyncio.Lock()
        self._nodes: dict[str, Node] = {}
        self._fixed_runtime_gib: dict[str, int] = {}
        self._seen_request_ids: list[str] = []
        self._audit: deque[AuditRecord] = deque(maxlen=AUDIT_CAPACITY)
        self._next_seq = 1
        if state_file is not None and state_file.exists():
            self._load_document(load_inventory_document(state_file))
        else:
            self._load_document(load_inventory_document(fixture_path))
        self._persist()

    @property
    def fixture_path(self) -> Path:
        """Fixture the state was, or will be on reset, loaded from."""
        return self._fixture_path

    async def current_seq(self) -> int:
        """Return the sequence number of the last appended audit record."""
        async with self._lock:
            return self._next_seq - 1

    async def status_view(self) -> dict[str, list[str]]:
        """Return the public status body.

        Returns:
            ``{node_ip: [assigned runtime paths, sorted]}`` for every configured
            node; nodes without assigned runtime devices map to ``[]``.
            Bootstrap devices never appear.
        """
        async with self._lock:
            return {
                ip: sorted(
                    device.path
                    for device in node.devices.values()
                    if device.role is DeviceRole.RUNTIME
                    and device.state is DeviceState.ASSIGNED
                )
                for ip, node in self._nodes.items()
            }

    async def deallocate(
        self, request_id: str, target_node: str, device_path: str
    ) -> DeallocationResult:
        """Release an assigned runtime device.

        The request ID is recorded before any other check, so a repeated ID is
        rejected even if the first attempt failed; POSTs are intentionally not
        idempotent.

        Args:
            request_id: Caller-chosen request identifier.
            target_node: Worker IP that must own ``device_path``.
            device_path: Assigned runtime device to release.

        Returns:
            The released device and its recorded size.

        Raises:
            MockServiceError: 409 for a repeated request ID, an already free
                device, or a device owned by another node; 404 for an unknown
                node or path; 403 for a bootstrap device.
        """
        async with self._lock:
            self._record_request_id(request_id)
            node = self._node_or_raise(target_node)
            device = node.devices.get(device_path)
            if device is None:
                owner = self._owner_of(device_path)
                if owner:
                    raise MockServiceError(
                        409,
                        f"wrong owner: device_path {device_path!r} belongs to "
                        f"node {owner!r}, not {target_node!r}",
                    )
                raise MockServiceError(
                    404, f"unknown device_path {device_path!r} on node {target_node!r}"
                )
            if device.role is DeviceRole.BOOTSTRAP:
                raise MockServiceError(
                    403, f"device_path {device_path!r} is a bootstrap device"
                )
            if device.state is DeviceState.FREE:
                raise MockServiceError(
                    409, f"device_path {device_path!r} is already free"
                )
            self._transition(node, device, DeviceState.FREE)
            return DeallocationResult(
                request_id=request_id,
                target_node=target_node,
                device_path=device_path,
                released_size_gib=device.size_gib,
            )

    async def allocate(
        self, request_id: str, target_node: str, request_size_gib: int
    ) -> AllocationResult:
        """Assign the first free runtime device of exactly ``request_size_gib``.

        Selection is the lexicographically first free runtime device on
        ``target_node`` whose recorded size equals ``request_size_gib``.  The
        request ID is recorded before any other check; see :meth:`deallocate`.

        Args:
            request_id: Caller-chosen request identifier.
            target_node: Worker IP that must receive the device.
            request_size_gib: Exact device size to select, in GiB.

        Returns:
            The selected device's pre-existing path and sizes.

        Raises:
            MockServiceError: 409 for a repeated request ID or when no free
                runtime device of that size exists on the node; 404 for an
                unknown node.
        """
        async with self._lock:
            self._record_request_id(request_id)
            node = self._node_or_raise(target_node)
            for device in node.devices.values():
                if (
                    device.role is DeviceRole.RUNTIME
                    and device.state is DeviceState.FREE
                    and device.size_gib == request_size_gib
                ):
                    self._transition(node, device, DeviceState.ASSIGNED)
                    return AllocationResult(
                        request_id=request_id,
                        target_node=target_node,
                        device_path=device.path,
                        requested_size_gib=request_size_gib,
                        granted_size_gib=device.size_gib,
                    )
            raise MockServiceError(
                409,
                f"insufficient capacity: no free device of exactly "
                f"{request_size_gib} GiB on node {target_node!r}",
            )

    async def record_request(
        self,
        operation: Operation,
        request_id: str,
        target_node: str,
        device_path: str,
        body: dict[str, object],
    ) -> int:
        """Append a ``request`` audit record.

        Args:
            operation: Public operation observed.
            request_id: ``request_id`` from the request body, or ``""``.
            target_node: ``target_node`` from the request body, or ``""``.
            device_path: ``device_path`` from the request body, or ``""``.
            body: Exact request JSON.

        Returns:
            The sequence number assigned to the record.
        """
        async with self._lock:
            return self._append_audit(
                AuditKind.REQUEST,
                operation,
                request_id,
                target_node,
                device_path,
                0,
                body,
            )

    async def record_response(
        self,
        operation: Operation,
        request_id: str,
        target_node: str,
        device_path: str,
        status_code: int,
        body: dict[str, object],
    ) -> int:
        """Append a ``response`` audit record.

        Args:
            operation: Public operation observed.
            request_id: ``request_id`` from the request body, or ``""``.
            target_node: ``target_node`` from the request body, or ``""``.
            device_path: ``device_path`` from the request body, or ``""``.
            status_code: HTTP status sent, or ``0`` if the connection was
                dropped before a body was sent.
            body: Exact response JSON (empty when dropped).

        Returns:
            The sequence number assigned to the record.
        """
        async with self._lock:
            return self._append_audit(
                AuditKind.RESPONSE,
                operation,
                request_id,
                target_node,
                device_path,
                status_code,
                body,
            )

    async def audit_after(self, after_seq: int) -> list[dict[str, object]]:
        """Return audit records with ``seq > after_seq`` in sequence order."""
        async with self._lock:
            return [
                record.to_dict() for record in self._audit if record.seq > after_seq
            ]

    async def reset(self) -> None:
        """Reload the fixture, clear the request-ID ledger and audit, and persist.

        Raises:
            ValueError: If the fixture fails validation.
            OSError: If the fixture or state file cannot be accessed.
        """
        async with self._lock:
            self._load_document(load_inventory_document(self._fixture_path))
            self._audit.clear()
            self._next_seq = 1
            self._persist()

    async def snapshot(self) -> dict[str, object]:
        """Return the admin view of inventory, accounting and request IDs.

        Returns:
            ``{"nodes": {ip: {"name", "devices": [{path, size_gib, role,
            state}], "free_runtime_gib", "assigned_runtime_gib",
            "fixed_runtime_inventory_gib"}}, "global": {"free_runtime_gib",
            "assigned_runtime_gib", "fixed_runtime_inventory_gib"},
            "seen_request_ids": [...]}``.
        """
        async with self._lock:
            nodes: dict[str, object] = {}
            for ip, node in self._nodes.items():
                nodes[ip] = {
                    "name": node.name,
                    "devices": [
                        {
                            "path": device.path,
                            "size_gib": device.size_gib,
                            "role": device.role.value,
                            "state": device.state.value,
                        }
                        for device in node.devices.values()
                    ],
                    "free_runtime_gib": node.runtime_gib(DeviceState.FREE),
                    "assigned_runtime_gib": node.runtime_gib(DeviceState.ASSIGNED),
                    "fixed_runtime_inventory_gib": self._fixed_runtime_gib[ip],
                }
            return {
                "nodes": nodes,
                "global": {
                    "free_runtime_gib": self._global_runtime_gib(DeviceState.FREE),
                    "assigned_runtime_gib": self._global_runtime_gib(
                        DeviceState.ASSIGNED
                    ),
                    "fixed_runtime_inventory_gib": sum(
                        self._fixed_runtime_gib.values()
                    ),
                },
                "seen_request_ids": list(self._seen_request_ids),
            }

    def _load_document(self, document: InventoryDocument) -> None:
        """Replace the inventory and request-ID ledger from a validated document."""
        nodes: dict[str, Node] = {}
        fixed: dict[str, int] = {}
        for ip, fixture_node in document.nodes.items():
            devices = {
                entry.path: Device(
                    path=entry.path,
                    size_gib=entry.size_gib,
                    role=entry.role,
                    state=entry.state,
                )
                for entry in sorted(fixture_node.devices, key=lambda d: d.path)
            }
            node = Node(ip=ip, name=fixture_node.name, devices=devices)
            nodes[ip] = node
            fixed[ip] = node.runtime_gib(DeviceState.FREE) + node.runtime_gib(
                DeviceState.ASSIGNED
            )
        self._nodes = nodes
        self._fixed_runtime_gib = fixed
        self._seen_request_ids = list(document.seen_request_ids)
        self._verify_invariants()

    def _record_request_id(self, request_id: str) -> None:
        """Reject a repeated request ID, else remember it. Caller holds the lock."""
        if request_id in self._seen_request_ids:
            raise MockServiceError(409, f"duplicate request_id {request_id!r}")
        self._seen_request_ids.append(request_id)

    def _node_or_raise(self, target_node: str) -> Node:
        """Look up a configured node. Caller holds the lock."""
        node = self._nodes.get(target_node)
        if node is None:
            raise MockServiceError(404, f"unknown target_node {target_node!r}")
        return node

    def _owner_of(self, device_path: str) -> str:
        """Return the IP owning ``device_path`` or ``""`` if no node has it."""
        for ip, node in self._nodes.items():
            if device_path in node.devices:
                return ip
        return ""

    def _transition(self, node: Node, device: Device, to_state: DeviceState) -> None:
        """Commit a state change, audit it, verify invariants and persist.

        Caller holds the lock.
        """
        from_state = device.state
        device.state = to_state
        self._verify_invariants()
        self._append_audit(
            AuditKind.MUTATION,
            Operation.DEALLOCATE
            if to_state is DeviceState.FREE
            else Operation.ALLOCATE,
            "",
            node.ip,
            device.path,
            0,
            {
                "path": device.path,
                "from_state": from_state.value,
                "to_state": to_state.value,
                "node": node.ip,
            },
        )
        self._persist()

    def _append_audit(
        self,
        kind: AuditKind,
        operation: Operation,
        request_id: str,
        target_node: str,
        device_path: str,
        status_code: int,
        body: dict[str, object],
    ) -> int:
        """Append one audit record and return its seq. Caller holds the lock."""
        seq = self._next_seq
        self._next_seq += 1
        self._audit.append(
            AuditRecord(
                seq=seq,
                kind=kind,
                operation=operation,
                request_id=request_id,
                target_node=target_node,
                device_path=device_path,
                status_code=status_code,
                body=body,
                timestamp=time.time(),
            )
        )
        return seq

    def _global_runtime_gib(self, state: DeviceState) -> int:
        """Sum of runtime GiB in ``state`` over all nodes. Caller holds the lock."""
        return sum(node.runtime_gib(state) for node in self._nodes.values())

    def _verify_invariants(self) -> None:
        """Raise ``RuntimeError`` if conservation or path identity is violated."""
        seen_paths: set[str] = set()
        for ip, node in self._nodes.items():
            for path, device in node.devices.items():
                if path != device.path or path in seen_paths:
                    raise RuntimeError(f"path identity violated for {path!r}")
                seen_paths.add(path)
            total = node.runtime_gib(DeviceState.FREE) + node.runtime_gib(
                DeviceState.ASSIGNED
            )
            if total != self._fixed_runtime_gib[ip]:
                raise RuntimeError(
                    f"conservation violated on {ip}: {total} != "
                    f"{self._fixed_runtime_gib[ip]}"
                )
        global_total = self._global_runtime_gib(
            DeviceState.FREE
        ) + self._global_runtime_gib(DeviceState.ASSIGNED)
        if global_total != sum(self._fixed_runtime_gib.values()):
            raise RuntimeError("global conservation violated")

    def _persist(self) -> None:
        """Atomically write the full state to the state file, if configured."""
        if self._state_file is None:
            return
        document = {
            "schema_version": STATE_SCHEMA_VERSION,
            "nodes": {
                ip: {
                    "name": node.name,
                    "devices": [
                        {
                            "path": device.path,
                            "size_gib": device.size_gib,
                            "role": device.role.value,
                            "state": device.state.value,
                        }
                        for device in node.devices.values()
                    ],
                }
                for ip, node in self._nodes.items()
            },
            "seen_request_ids": list(self._seen_request_ids),
        }
        tmp_path = self._state_file.with_name(self._state_file.name + ".tmp")
        tmp_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._state_file)
