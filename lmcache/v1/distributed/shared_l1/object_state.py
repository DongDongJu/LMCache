# SPDX-License-Identifier: Apache-2.0
"""Orchestrator-owned object state of a shared Device-DAX L1."""

# Standard
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from mmap import PAGESIZE
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig, SharedDevDaxConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    DeleteStatus,
    OrchestratorError,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionFencedError,
    RequestRejectedError,
    TokenStatus,
    WireLayout,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient

logger = init_logger(__name__)

_DELETE_RESULTS = {
    DeleteStatus.DELETED: L1Error.SUCCESS,
    DeleteStatus.MISS: L1Error.KEY_NOT_EXIST,
    DeleteStatus.BUSY_WRITING: L1Error.KEY_IS_LOCKED,
}


def _layout_to_wire(layout: MemoryLayoutDesc) -> WireLayout:
    """Return ``layout`` as plain integer dims and dtype names."""
    return WireLayout(
        shapes=tuple(tuple(int(dim) for dim in shape) for shape in layout.shapes),
        dtypes=tuple(str(dtype).removeprefix("torch.") for dtype in layout.dtypes),
    )


def _layout_from_wire(layout: WireLayout) -> MemoryLayoutDesc:
    """Return the memory layout of a wire layout another server wrote.

    Raises:
        ValueError: A dtype name is unknown to this torch build.
    """
    dtypes = [getattr(torch, name, None) for name in layout.dtypes]
    if not all(isinstance(dtype, torch.dtype) for dtype in dtypes):
        raise ValueError(f"unknown torch dtype in a shared-L1 layout: {layout.dtypes}")
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in layout.shapes], dtypes=dtypes
    )


def _check_contract(contract: RegionContract, memory: L1MemoryManagerConfig) -> None:
    """Reject a region this L1 cannot map or serve."""
    if contract.reset_required:
        raise RuntimeError(
            f"Memory orchestrator for region {contract.region_id!r} needs an "
            "offline reset: stop every server using the region, stop the "
            "orchestrator, delete its startup marker, restart"
        )
    if contract.capacity_bytes > memory.size_in_bytes:
        raise ValueError(
            f"Region {contract.region_id!r} holds {contract.capacity_bytes} bytes, "
            f"more than this L1's size_gb allows ({memory.size_in_bytes} bytes)"
        )
    if contract.visibility_mode == "software_fenced" and (
        contract.alignment_bytes < PAGESIZE
    ):
        # Different writers must never write back the same cache line.
        raise ValueError("software_fenced extents require page alignment")


@dataclass
class _WriteGrant:
    """A write extent this server reserved and has not committed or aborted."""

    token: bytes
    memory_obj: TensorMemoryObj


@dataclass
class _ReadHold:
    """The read leases this server holds on one object, and its view."""

    memory_obj: TensorMemoryObj
    leases: list[bytes] = field(default_factory=list)


class OrchestratedObjectState:
    """This server's grants and leases on a region a memory orchestrator owns.

    The orchestrator decides which objects exist and are readable; this class
    only remembers the write grants (until commit or abort) and read leases it
    took. Construct it with :meth:`connect`.

    Args:
        client: A registered client of the region's orchestrator.
        contract: The region's contract, checked at registration.
        visibility: Publish/acquire fences for the region's visibility mode.
        memory: The memory manager that maps the region.
        owner_id: The owning ``L1Manager``'s ID, stamped on every view.
    """

    def __init__(
        self,
        client: MemoryOrchestratorClient,
        contract: RegionContract,
        visibility: Visibility,
        memory: DevDaxL1MemoryManager,
        owner_id: int,
    ) -> None:
        self._client = client
        self._contract = contract
        self._visibility = visibility
        self._memory = memory
        self._owner_id = owner_id
        self._lock = threading.Lock()
        self._writes: dict[tuple[ObjectKey, str], _WriteGrant] = {}
        self._reads: dict[ObjectKey, _ReadHold] = {}
        self._staging_bytes = 0
        self._reachable = True
        self._disabled_reason: str | None = None
        self._closed = False

    @classmethod
    def connect(
        cls,
        shared: SharedDevDaxConfig,
        memory_config: L1MemoryManagerConfig,
        owner_id: int,
    ) -> tuple["OrchestratedObjectState", DevDaxL1MemoryManager]:
        """Join the region: describe, validate, map and register, check epoch.

        Every resource acquired is released when a later step fails.

        Args:
            shared: The L1's ``shared`` object.
            memory_config: The L1's Device-DAX path, offset and size.
            owner_id: The owning ``L1Manager``'s ID.

        Returns:
            The object state and the memory manager that maps the region.

        Raises:
            ValueError: The region does not fit this L1, or its visibility
                mode is unknown.
            RuntimeError: The region needs an offline reset, its visibility
                mode is unsupported on this CPU, or GPU registration fails.
            RegionFencedError: The region changed while it was being mapped.
            OrchestratorError: The orchestrator refused or was unreachable.
            OSError: The region cannot be mapped.
        """
        client = MemoryOrchestratorClient(
            shared.orchestrator,
            shared.region_id,
            shared.client_id,
            rpc_timeout_s=shared.rpc_timeout_seconds,
        )
        with ExitStack() as cleanup:
            cleanup.callback(client.close)
            contract = client.describe_region()
            _check_contract(contract, memory_config)
            visibility = Visibility(contract.visibility_mode)
            memory = DevDaxL1MemoryManager(memory_config, shared_region=contract)
            cleanup.callback(memory.close)
            client.register(
                layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
                mapped_bytes=contract.capacity_bytes,
                visibility_mode=visibility.mode,
            )
            if client.region_epoch != contract.region_epoch:
                raise RegionFencedError("The region changed while it was mapped")
            cleanup.pop_all()
        logger.info(
            "Shared Device-DAX region %r (%d bytes, epoch %d, %s) mapped from %s "
            "at offset %d as client %r",
            shared.region_id,
            contract.capacity_bytes,
            contract.region_epoch,
            visibility.mode,
            memory_config.devdax_path,
            memory_config.devdax_offset_in_bytes,
            shared.client_id,
        )
        return cls(client, contract, visibility, memory, owner_id), memory

    @property
    def capacity_bytes(self) -> int:
        """Return the capacity of the whole region."""
        return self._contract.capacity_bytes

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str,
    ) -> dict[ObjectKey, L1OperationResult]:
        """Ask the orchestrator for one extent per key.

        Args:
            keys: Keys to write.
            is_temporary: One flag per key; shared objects are never temporary.
            layout_desc: Layout of every object in the batch.
            tag: The writer's identity; ``commit`` and ``abort`` take the same.

        Returns:
            Per key ``(SUCCESS, view)``; ``(KEY_NOT_WRITABLE, None)`` when any
            server is writing or committed the key, or the grant is malformed;
            ``(OUT_OF_MEMORY, None)`` when the region is full or the
            orchestrator cannot be used, so ordered overflow tries the next L1.

        Raises:
            ValueError: The lengths differ, or a temporary object is requested.
            RequestRejectedError: A key's layout conflicts with the stored
                object; this is not capacity pressure.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )
        if any(is_temporary):
            raise ValueError("A shared Device-DAX L1 holds no temporary objects")
        ret: dict[ObjectKey, L1OperationResult] = {}
        with self._lock:
            for key in keys:
                if (key, tag) in self._writes:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
        pending = [key for key in keys if key not in ret]
        if not pending or self._skip_request_rpc():
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret
        payload_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        wire = _layout_to_wire(layout_desc)
        try:
            grants = self._client.reserve_write(
                [WriteRequest(key, payload_bytes, wire) for key in pending]
            )
        except RequestRejectedError:
            # A conflicting layout is not capacity pressure: overflowing into
            # a private L1 would hide it.
            raise
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveWrite", exc)
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret
        self._on_rpc_ok()
        bad_tokens: list[bytes] = []
        with self._lock:
            for key, grant in zip(pending, grants, strict=True):
                if grant.status is WriteStatus.OUT_OF_SPACE:
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue
                if grant.status is not WriteStatus.WRITE_GRANTED:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                assert grant.handle is not None and grant.token is not None
                try:
                    memory_obj = self._memory.view(
                        grant.handle.offset, grant.handle.length, layout_desc
                    )
                except ValueError:
                    logger.exception("Shared L1: orchestrator granted a bad extent")
                    bad_tokens.append(grant.token)
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                memory_obj.set_l1_manager(self._owner_id)
                self._writes[(key, tag)] = _WriteGrant(grant.token, memory_obj)
                self._staging_bytes += payload_bytes
                ret[key] = (L1Error.SUCCESS, memory_obj)
        if bad_tokens:
            self._send_tokens("AbortWrite", self._client.abort_write, bad_tokens)
        return ret

    def commit(
        self, keys: list[ObjectKey], tag: str
    ) -> tuple[dict[ObjectKey, L1Error], list[ObjectKey]]:
        """Publish ``tag``'s written extents, then commit them.

        The caller must have waited for every device-to-host copy into them.

        Args:
            keys: Keys whose extents were written.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS`` once committed, ``KEY_NOT_EXIST`` when ``tag``
            reserved nothing for it, or ``KEY_IN_WRONG_STATE`` when the
            orchestrator refused or could not be asked: the write is reported
            failed and its extent is never reused here. Also the committed
            keys, in order.

        Raises:
            Exception: Publishing the bytes failed; the grants are aborted.
        """
        ret, taken = self._take_writes(keys, tag)
        tokens = [grant.token for _, grant in taken]
        statuses = [TokenStatus.STALE_TOKEN] * len(taken)
        if taken and not self._disabled():
            try:
                self._visibility.publish(
                    [(g.memory_obj.data_ptr, g.memory_obj.get_size()) for _, g in taken]
                )
            except Exception:
                self._send_tokens("AbortWrite", self._client.abort_write, tokens)
                raise
            try:
                statuses = self._client.finish_write(tokens)
                self._on_rpc_ok()
            except OrchestratorError as exc:
                self._on_rpc_error("FinishWrite", exc)
        done: list[ObjectKey] = []
        for (key, _), status in zip(taken, statuses, strict=True):
            if status is TokenStatus.OK:
                ret[key] = L1Error.SUCCESS
                done.append(key)
            else:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        return ret, done

    def abort(self, keys: list[ObjectKey], tag: str) -> dict[ObjectKey, L1Error]:
        """Abort ``tag``'s write grants; the keys become writable again.

        The caller must have waited for every copy into the extents.

        Args:
            keys: Keys whose grants to abort.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when ``tag`` reserved
            nothing for it, or ``KEY_IN_WRONG_STATE`` when the orchestrator
            refused or could not be asked.
        """
        ret, taken = self._take_writes(keys, tag)
        tokens = [grant.token for _, grant in taken]
        statuses = self._send_tokens("AbortWrite", self._client.abort_write, tokens)
        for (key, _), status in zip(taken, statuses, strict=True):
            ok = status is TokenStatus.OK
            ret[key] = L1Error.SUCCESS if ok else L1Error.KEY_IN_WRONG_STATE
        return ret

    def reserve_read(
        self, keys: list[ObjectKey], count: int
    ) -> dict[ObjectKey, L1OperationResult]:
        """Lease committed objects and drop this host's stale cached lines.

        Args:
            keys: Keys to read.
            count: Validated leases to take per key, one per reader.

        Returns:
            Per key ``(SUCCESS, view)`` with the leases held, else
            ``(KEY_NOT_EXIST, None)``: missing, still being written, a
            malformed grant (logged), or the orchestrator cannot be used.

        Raises:
            Exception: Dropping cached lines failed; only the leases this
                call took are released.
        """
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_EXIST, None) for key in keys
        }
        if not keys or self._skip_request_rpc():
            return ret
        try:
            grants = self._client.reserve_read(
                [ReadRequest(key, count) for key in keys]
            )
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveRead", exc)
            return ret
        self._on_rpc_ok()
        bad_leases: list[bytes] = []
        taken: list[ObjectKey] = []
        ranges: list[tuple[int, int]] = []
        with self._lock:
            for key, grant in zip(keys, grants, strict=True):
                if grant.status is not ReadStatus.READ_GRANTED:
                    continue
                assert grant.handle is not None and grant.layout is not None
                hold = self._reads.get(key)
                try:
                    if len(grant.leases) != count:
                        raise ValueError("Read grant has the wrong number of leases")
                    layout = _layout_from_wire(grant.layout)
                    memory_obj = self._memory.view(
                        grant.handle.offset, grant.handle.length, layout
                    )
                    if memory_obj.get_size() != grant.payload_bytes:
                        raise ValueError("Read grant payload does not match its layout")
                    if hold is not None and (
                        hold.memory_obj.data_ptr != memory_obj.data_ptr
                        or hold.memory_obj.get_shapes() != layout.shapes
                        or hold.memory_obj.get_dtypes() != layout.dtypes
                    ):
                        raise ValueError("A leased object changed its extent or layout")
                except ValueError:
                    logger.exception("Shared L1: orchestrator granted a bad read")
                    bad_leases.extend(grant.leases)
                    continue
                if hold is None:
                    memory_obj.set_l1_manager(self._owner_id)
                    hold = _ReadHold(memory_obj)
                    self._reads[key] = hold
                hold.leases.extend(grant.leases)
                ret[key] = (L1Error.SUCCESS, hold.memory_obj)
                taken.append(key)
                ranges.append((memory_obj.data_ptr, grant.payload_bytes))
        if bad_leases:
            self._send_tokens("FinishRead", self._client.finish_read, bad_leases)
        try:
            self._visibility.acquire(ranges)
        except Exception:
            # Only this call's leases go back; overlapping readers keep theirs.
            self.finish_read(taken, count)
            raise
        return ret

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return the views of keys this server holds read leases on.

        Args:
            keys: Keys to read.

        Returns:
            Per key ``(SUCCESS, view)`` while a lease is held, else
            ``(KEY_NOT_EXIST, None)``.
        """
        with self._lock:
            ret: dict[ObjectKey, L1OperationResult] = {}
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                else:
                    ret[key] = (L1Error.SUCCESS, hold.memory_obj)
            return ret

    def finish_read(
        self, keys: list[ObjectKey], count: int
    ) -> dict[ObjectKey, L1Error]:
        """Release ``count`` leases per key.

        Every lease of a key is on the same committed object, so releasing
        any ``count`` of them is exact.

        Args:
            keys: Keys to release.
            count: Validated leases to release per key.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when no lease is held, or
            ``KEY_IN_WRONG_STATE`` when the orchestrator refused a lease or
            could not be asked (the lease is dropped here either way).
        """
        ret: dict[ObjectKey, L1Error] = {}
        released: list[tuple[ObjectKey, list[bytes]]] = []
        with self._lock:
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                released.append((key, hold.leases[-count:]))
                del hold.leases[-count:]
                if not hold.leases:
                    del self._reads[key]
        tokens = [token for _, key_tokens in released for token in key_tokens]
        statuses = self._send_tokens("FinishRead", self._client.finish_read, tokens)
        position = 0
        for key, key_tokens in released:
            key_statuses = statuses[position : position + len(key_tokens)]
            position += len(key_tokens)
            ok = all(status is TokenStatus.OK for status in key_statuses)
            ret[key] = L1Error.SUCCESS if ok else L1Error.KEY_IN_WRONG_STATE
        return ret

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Make committed objects unreadable for every server.

        Their bytes stay consumed until a region reset, and leases any server
        already holds on them stay valid until released.

        Args:
            keys: Keys to delete.

        Returns:
            Per key ``SUCCESS`` once deleted, ``KEY_NOT_EXIST`` when no
            committed object exists, or ``KEY_IS_LOCKED`` while the object is
            still being written or the orchestrator cannot be used.
        """
        ret = {key: L1Error.KEY_IS_LOCKED for key in keys}
        if not keys or self._skip_request_rpc():
            return ret
        try:
            statuses = self._client.delete_objects(keys)
        except OrchestratorError as exc:
            self._on_rpc_error("DeleteObjects", exc)
            return ret
        self._on_rpc_ok()
        for key, status in zip(keys, statuses, strict=True):
            ret[key] = _DELETE_RESULTS[status]
        return ret

    def delete_published(self) -> int | None:
        """Delete every committed object written under this server's client_id.

        Objects written before this server restarted count too. Deletion
        follows :meth:`delete`.

        Returns:
            The number of objects deleted, or None when the orchestrator
            cannot be used.
        """
        if self._skip_request_rpc():
            return None
        try:
            deleted = self._client.delete_published()
        except OrchestratorError as exc:
            self._on_rpc_error("DeletePublished", exc)
            return None
        self._on_rpc_ok()
        return deleted

    def get_staging_memory_usage(self) -> int:
        """Return the payload bytes this server reserved and has not committed."""
        with self._lock:
            return self._staging_bytes

    def is_healthy(self) -> bool:
        """Return whether the last RPC succeeded and the region is not disabled."""
        with self._lock:
            return self._reachable and self._disabled_reason is None

    def report_status(self) -> dict[str, object]:
        """Return the L1 status fields for this server's grants and leases.

        Returns:
            The ``L1Manager.report_status`` fields plus ``shared.state``:
            ``available``, ``unavailable`` (the last RPC failed) or ``disabled``.
        """
        with self._lock:
            grants = len(self._writes)
            holds = len(self._reads)
            staging = self._staging_bytes
            disabled = self._disabled_reason
            reachable = self._reachable
        capacity = self._contract.capacity_bytes
        if disabled is not None:
            state = "disabled"
        elif reachable:
            state = "available"
        else:
            state = "unavailable"
        return {
            "is_healthy": state == "available",
            "total_object_count": grants + holds,
            "write_locked_count": grants,
            "read_locked_count": holds,
            "temporary_count": 0,
            "staging_object_count": grants,
            "staging_bytes": staging,
            "memory_used_bytes": staging,
            "memory_total_bytes": capacity,
            "memory_configured_bytes": capacity,
            "capacity_bytes_by_backend": {L1BackendType.DEVDAX.value: capacity},
            "memory_usage_ratio": staging / capacity if capacity > 0 else 0.0,
            # The orchestrator expires writes; read leases do not expire.
            "write_ttl_seconds": None,
            "read_ttl_seconds": None,
            "shared": {"state": state, "disabled_reason": disabled},
        }

    def close(self) -> None:
        """Stop admission and retire this server's grants and leases.

        Call only after the copies into and out of the views have drained.
        """
        with self._lock:
            self._closed = True
            while self._writes:
                self._writes.popitem()[1].memory_obj.invalidate()
            self._reads.clear()
            self._staging_bytes = 0
        self._client.close()

    def _take_writes(
        self, keys: list[ObjectKey], tag: str
    ) -> tuple[dict[ObjectKey, L1Error], list[tuple[ObjectKey, _WriteGrant]]]:
        """Pop ``tag``'s grants for ``keys``; ``KEY_NOT_EXIST`` for keys without."""
        ret: dict[ObjectKey, L1Error] = {}
        taken: list[tuple[ObjectKey, _WriteGrant]] = []
        with self._lock:
            for key in keys:
                grant = self._writes.pop((key, tag), None)
                if grant is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                self._staging_bytes -= grant.memory_obj.get_size()
                taken.append((key, grant))
        return ret, taken

    def _send_tokens(
        self,
        rpc: str,
        call: Callable[[Sequence[bytes]], list[TokenStatus]],
        tokens: list[bytes],
    ) -> list[TokenStatus]:
        """Send FinishRead or AbortWrite; every token is stale if it cannot."""
        if not tokens or self._disabled():
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        try:
            statuses = call(tokens)
        except OrchestratorError as exc:
            self._on_rpc_error(rpc, exc)
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        self._on_rpc_ok()
        return statuses

    def _on_rpc_ok(self) -> None:
        with self._lock:
            was_reachable = self._reachable
            self._reachable = True
        if not was_reachable:
            logger.info("Shared L1: memory orchestrator reachable again")

    def _on_rpc_error(self, rpc: str, exc: OrchestratorError) -> None:
        """Record a failed RPC; a fencing error disables the region for good."""
        with self._lock:
            if isinstance(exc, RegionFencedError):
                if self._disabled_reason is not None:
                    return
                self._disabled_reason = f"{rpc}: {exc}"
                logger.error(
                    "Shared L1 is disabled (%s). It serves nothing until this "
                    "server restarts after the region's offline reset.",
                    self._disabled_reason,
                )
                return
            was_reachable = self._reachable
            self._reachable = False
        if was_reachable:
            logger.warning(
                "Shared L1: %s failed (%s); reads miss and writes overflow",
                rpc,
                exc,
            )

    def _disabled(self) -> bool:
        with self._lock:
            return self._disabled_reason is not None

    def _skip_request_rpc(self) -> bool:
        """Whether a lookup, reservation or delete skips its RPC: closed or disabled."""
        with self._lock:
            return self._closed or self._disabled_reason is not None
