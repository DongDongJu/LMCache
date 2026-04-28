# SPDX-License-Identifier: Apache-2.0
"""Blkio-backed MP L2 adapter with Python-side key tracking.

The native libblkio connector is offset-based and has no durable key index.
This adapter provides the MP L2 contract by maintaining a volatile in-memory
map from ObjectKey to device offsets and sizes.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import select
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

_BLOCK_ALIGN = 4096


def _round_up(value: int, align: int = _BLOCK_ALIGN) -> int:
    """Round a byte count up to the next alignment boundary."""
    return ((value + align - 1) // align) * align


def _obj_to_memoryview(obj: MemoryObj) -> memoryview:
    """Return a byte-oriented memoryview for a memory object."""
    return obj.byte_array  # type: ignore[return-value]


def _offset_key(offset: int) -> str:
    """Encode a blkio native key whose final field is a byte offset."""
    return f"blkio@00000000@{offset:x}"


@dataclass
class _BlkioEntry:
    offset: int
    size: int
    object_size: int
    alloc_size: int


@dataclass
class _PendingStore:
    task_id: L2TaskId
    keys: list[ObjectKey]
    entries: list[_BlkioEntry]


@dataclass
class _PendingLoad:
    task_id: L2TaskId
    keys: list[ObjectKey]
    indices: list[int]
    bitmap_size: int


class BlkioL2AdapterConfig(L2AdapterConfigBase):
    """Configuration for the built-in blkio MP L2 adapter."""

    def __init__(
        self,
        *,
        device_path: str,
        num_workers: int = 4,
        direct_io: bool = True,
    ) -> None:
        """Initialize blkio adapter configuration.

        Args:
            device_path: Block device or dedicated file path.
            num_workers: Number of libblkio worker threads.
            direct_io: Whether libblkio should use direct I/O.
        """
        super().__init__()
        self.device_path = device_path
        self.num_workers = int(num_workers)
        self.direct_io = bool(direct_io)

    @classmethod
    def from_dict(cls, d: dict) -> "BlkioL2AdapterConfig":
        """Build blkio adapter config from ``--l2-adapter`` JSON."""
        device_path = d.get("device_path")
        if not isinstance(device_path, str) or not device_path:
            raise ValueError("device_path must be a non-empty string")
        num_workers = int(d.get("num_workers", 4))
        if num_workers <= 0:
            raise ValueError("num_workers must be > 0")
        direct_io = d.get("direct_io", True)
        if not isinstance(direct_io, bool):
            raise ValueError("direct_io must be a boolean")
        return cls(
            device_path=device_path,
            num_workers=num_workers,
            direct_io=direct_io,
        )

    @classmethod
    def help(cls) -> str:
        """Return human-readable blkio adapter configuration help."""
        return (
            "blkio L2 adapter config fields:\n"
            "- device_path (str): block device or dedicated file path (required)\n"
            "- num_workers (int): libblkio worker threads (default 4)\n"
            "- direct_io (bool): enable direct I/O (default true)\n"
        )


class BlkioL2Adapter(L2AdapterInterface):
    """MP L2 adapter backed by libblkio offset reads and writes."""

    _OP_STORE = "store"
    _OP_LOAD = "load"

    def __init__(
        self,
        config: BlkioL2AdapterConfig,
        l1_memory_desc: L1MemoryDesc | None = None,  # noqa: ARG002
    ) -> None:
        """Create a blkio adapter and start its completion demux thread."""
        super().__init__()
        try:
            # First Party
            from lmcache.lmcache_blkio import LMCacheBlkioClient
        except ImportError as e:
            raise RuntimeError(
                "BlkioL2Adapter requires the lmcache_blkio extension. "
                "Install libblkio development headers and rebuild LMCache."
            ) from e

        self._client = LMCacheBlkioClient(
            config.device_path,
            config.num_workers,
            config.direct_io,
        )
        self._client_fd = int(self._client.event_fd())
        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._lock = threading.Lock()
        self._closed = False
        self._next_task_id: L2TaskId = 0
        self._next_offset = 0
        self._free_ranges: list[tuple[int, int]] = []
        self._index: dict[ObjectKey, _BlkioEntry] = {}
        self._lock_refcnt: dict[ObjectKey, int] = defaultdict(int)
        self._pending_ops: dict[int, tuple[str, _PendingStore | _PendingLoad]] = {}
        self._completed_stores: dict[L2TaskId, bool] = {}
        self._completed_lookups: dict[L2TaskId, Bitmap] = {}
        self._completed_loads: dict[L2TaskId, Bitmap] = {}

        self._stop = threading.Event()
        self._demux_thread = threading.Thread(
            target=self._demux_loop,
            daemon=True,
            name="blkio-l2-demux",
        )
        self._demux_thread.start()

    def get_store_event_fd(self) -> int:
        """Return the eventfd signaled when store tasks complete."""
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the eventfd signaled when lookup tasks complete."""
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        """Return the eventfd signaled when load tasks complete."""
        return self._load_efd.fileno()

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking blkio store task."""
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        memviews = [_obj_to_memoryview(obj) for obj in objects]
        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._next_task_id_locked()
            entries = [
                self._reserve_entry_locked(key, obj, len(memview))
                for key, obj, memview in zip(keys, objects, memviews, strict=True)
            ]
            native_keys = [_offset_key(entry.offset) for entry in entries]
            future_id = int(self._client.submit_batch_set(native_keys, memviews))
            self._pending_ops[future_id] = (
                self._OP_STORE,
                _PendingStore(task_id, list(keys), entries),
            )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        """Drain and return completed store task results."""
        with self._lock:
            completed = self._completed_stores
            self._completed_stores = {}
        return completed

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Complete lookup from the volatile Python key map."""
        if not keys:
            raise ValueError("keys must be non-empty")
        bitmap = Bitmap(len(keys))
        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._next_task_id_locked()
            for i, key in enumerate(keys):
                if key in self._index:
                    bitmap.set(i)
                    self._lock_refcnt[key] += 1
            self._completed_lookups[task_id] = bitmap
        self._lookup_efd.notify()
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed lookup bitmap if available."""
        with self._lock:
            return self._completed_lookups.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Release locks acquired by lookup-and-lock."""
        with self._lock:
            for key in keys:
                refcnt = self._lock_refcnt.get(key, 0)
                if refcnt <= 1:
                    self._lock_refcnt.pop(key, None)
                else:
                    self._lock_refcnt[key] = refcnt - 1

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        """Submit a non-blocking blkio load task for mapped keys."""
        if not keys or not objects:
            raise ValueError("keys and objects must be non-empty")
        if len(keys) != len(objects):
            raise ValueError("keys and objects must have the same length")

        with self._lock:
            self._raise_if_closed_locked()
            task_id = self._next_task_id_locked()
            found: list[tuple[int, ObjectKey, _BlkioEntry]] = [
                (i, key, entry)
                for i, key in enumerate(keys)
                if (entry := self._index.get(key)) is not None
            ]
            if not found:
                self._completed_loads[task_id] = Bitmap(len(keys))
                self._load_efd.notify()
                return task_id

            native_keys = [_offset_key(entry.offset) for _, _, entry in found]
            memviews = [_obj_to_memoryview(objects[i]) for i, _, _ in found]
            future_id = int(self._client.submit_batch_get(native_keys, memviews))
            self._pending_ops[future_id] = (
                self._OP_LOAD,
                _PendingLoad(
                    task_id=task_id,
                    keys=[key for _, key, _ in found],
                    indices=[i for i, _, _ in found],
                    bitmap_size=len(keys),
                ),
            )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Return and remove a completed load bitmap if available."""
        with self._lock:
            return self._completed_loads.pop(task_id, None)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete unlocked keys from the volatile map and recycle ranges."""
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        with self._lock:
            for key in keys:
                if self._lock_refcnt.get(key, 0) > 0:
                    continue
                entry = self._index.pop(key, None)
                if entry is None:
                    continue
                self._free_ranges.append((entry.offset, entry.alloc_size))
                deleted_keys.append(key)
                deleted_sizes.append(entry.object_size)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    def close(self) -> None:
        """Stop the demux thread and close native resources."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._demux_thread.join(timeout=5)
        self._client.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    def report_status(self) -> dict:
        """Return adapter health and volatile index counters."""
        with self._lock:
            return {
                "is_healthy": not self._closed,
                "type": "BlkioL2Adapter",
                "indexed_key_count": len(self._index),
                "locked_key_count": len(self._lock_refcnt),
                "pending_task_count": len(self._pending_ops),
                "restart_recovery": False,
            }

    def _raise_if_closed_locked(self) -> None:
        """Raise if adapter has been closed while holding ``_lock``."""
        if self._closed:
            raise RuntimeError("BlkioL2Adapter is closed")

    def _next_task_id_locked(self) -> L2TaskId:
        """Return and increment the task id while holding ``_lock``."""
        task_id = self._next_task_id
        self._next_task_id += 1
        return task_id

    def _reserve_entry_locked(
        self,
        key: ObjectKey,
        obj: MemoryObj,
        byte_len: int,
    ) -> _BlkioEntry:
        """Reserve or reuse a device offset for a store while holding ``_lock``."""
        object_size = int(obj.get_size())
        alloc_size = _round_up(max(1, byte_len))
        old = self._index.get(key)
        if old is not None and old.alloc_size >= alloc_size:
            return _BlkioEntry(old.offset, byte_len, object_size, old.alloc_size)
        if old is not None:
            self._free_ranges.append((old.offset, old.alloc_size))

        for idx, (offset, size) in enumerate(self._free_ranges):
            if size < alloc_size:
                continue
            self._free_ranges.pop(idx)
            remainder = size - alloc_size
            if remainder > 0:
                self._free_ranges.append((offset + alloc_size, remainder))
            return _BlkioEntry(offset, byte_len, object_size, alloc_size)

        offset = self._next_offset
        self._next_offset += alloc_size
        return _BlkioEntry(offset, byte_len, object_size, alloc_size)

    def _demux_loop(self) -> None:
        """Poll native completions and publish MP adapter task results."""
        poller = select.poll()
        poller.register(self._client_fd, select.POLLIN)
        while not self._stop.is_set():
            events = poller.poll(500)
            if not events:
                continue
            try:
                completions = self._client.drain_completions()
            except Exception:
                logger.exception("blkio drain_completions failed")
                continue
            for future_id, ok, error, result_bools in completions:
                self._handle_completion(int(future_id), bool(ok), error, result_bools)

    def _handle_completion(
        self,
        future_id: int,
        ok: bool,
        error: str,
        result_bools: list[bool] | None,
    ) -> None:
        """Handle one native blkio completion."""
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        accessed_keys: list[ObjectKey] = []
        with self._lock:
            pending = self._pending_ops.pop(future_id, None)
            if pending is None:
                logger.warning("blkio completion for unknown future_id=%d", future_id)
                return
            op_type, payload = pending
            if op_type == self._OP_STORE:
                assert isinstance(payload, _PendingStore)
                per_key = self._result_bools(ok, result_bools, len(payload.keys))
                for key, entry, stored in zip(
                    payload.keys, payload.entries, per_key, strict=True
                ):
                    if stored:
                        stored_size = 0 if key in self._index else entry.object_size
                        self._index[key] = entry
                        stored_keys.append(key)
                        stored_sizes.append(stored_size)
                    else:
                        self._free_ranges.append((entry.offset, entry.alloc_size))
                self._completed_stores[payload.task_id] = all(per_key)
                self._store_efd.notify()
            elif op_type == self._OP_LOAD:
                assert isinstance(payload, _PendingLoad)
                per_key = self._result_bools(ok, result_bools, len(payload.keys))
                bitmap = Bitmap(payload.bitmap_size)
                for key, index, loaded in zip(
                    payload.keys, payload.indices, per_key, strict=True
                ):
                    if loaded:
                        bitmap.set(index)
                        accessed_keys.append(key)
                self._completed_loads[payload.task_id] = bitmap
                self._load_efd.notify()
            elif error:
                logger.warning("blkio native task failed: %s", error)

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)

    def _result_bools(
        self,
        ok: bool,
        result_bools: list[bool] | None,
        expected: int,
    ) -> list[bool]:
        """Normalize native completion results to one bool per submitted key."""
        if result_bools is not None:
            return [bool(v) for v in result_bools[:expected]]
        return [ok] * expected


register_l2_adapter_type("blkio", BlkioL2AdapterConfig)


def _create_blkio_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create a blkio L2 adapter from a registered config."""
    assert isinstance(config, BlkioL2AdapterConfig)
    return BlkioL2Adapter(config, l1_memory_desc)


register_l2_adapter_factory("blkio", _create_blkio_l2_adapter)
