# SPDX-License-Identifier: Apache-2.0
"""L2 adapter for RustRawBlockBackend in MP mode."""

# Standard
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import asyncio
import os
import queue
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import RustRawBlockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.object_key_codec import object_key_to_cache_key
from lmcache.v1.memory_management import AdHocMemoryAllocator, MemoryObj
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.plugins.rust_raw_block_backend import (
    RustRawBlockBackend,
)

logger = init_logger(__name__)

_MP_SLOT_PAYLOAD_BYTES = 64 * 1024 * 1024


def _build_slot_bytes(header_bytes: int, block_align: int) -> int:
    """Round up header + payload to block alignment for fixed-size slots."""
    requested = int(header_bytes) + _MP_SLOT_PAYLOAD_BYTES
    align = int(block_align)
    return ((requested + align - 1) // align) * align


class RustRawBlockL2Adapter(L2AdapterInterface):
    """Thread-safe async adapter over RustRawBlockBackend."""

    def __init__(self, config: RustRawBlockL2AdapterConfig):
        self._config = config
        self._closed = False

        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        self._lock = threading.Lock()
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}

        self._num_queues = max(1, int(torch.cuda.device_count()))
        self._queue_depth = [0 for _ in range(self._num_queues)]
        self._rr_counter = 0

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name="rust-raw-block-l2-loop",
            daemon=True,
        )
        self._loop_thread.start()

        self._queue_executors = self._create_io_queue_executors(self._num_queues)

        self._unlock_stop = threading.Event()
        self._unlock_queue: "queue.SimpleQueue[list]" = queue.SimpleQueue()
        self._unlock_thread = threading.Thread(
            target=self._unlock_loop,
            name="rust-raw-block-l2-unlock",
            daemon=True,
        )

        self._backend = self._build_backend(config)
        self._unlock_thread.start()

        logger.info(
            "RustRawBlockL2Adapter initialized: device=%s, queues=%d",
            config.device_path,
            self._num_queues,
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        if len(keys) != len(objects):
            raise ValueError("submit_store_task requires equal key/object lengths")

        cache_keys = [object_key_to_cache_key(key) for key in keys]
        task_id = self._allocate_task_id()
        queue_idx = self._choose_queue(keys)

        future = self._queue_executors[queue_idx].submit(
            self._backend.store_batch_blocking,
            cache_keys,
            objects,
        )
        future.add_done_callback(
            lambda fut: self._on_store_done(task_id, queue_idx, fut)
        )
        return task_id

    def pop_completed_store_tasks(self) -> dict[L2TaskId, bool]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self,
        keys: list[ObjectKey],
    ) -> L2TaskId:
        cache_keys = [object_key_to_cache_key(key) for key in keys]
        task_id = self._allocate_task_id()

        try:
            bitmap = self._backend.lookup_and_lock(cache_keys)
        except Exception:
            logger.exception("lookup_and_lock failed")
            bitmap = Bitmap(len(keys))

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap

        self._signal_eventfd(self._lookup_efd)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(
        self,
        keys: list[ObjectKey],
    ) -> None:
        if not keys:
            return
        cache_keys = [object_key_to_cache_key(key) for key in keys]
        self._unlock_queue.put(cache_keys)

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        if len(keys) != len(objects):
            raise ValueError("submit_load_task requires equal key/object lengths")

        cache_keys = [object_key_to_cache_key(key) for key in keys]
        task_id = self._allocate_task_id()
        queue_idx = self._choose_queue(keys)

        future = self._queue_executors[queue_idx].submit(
            self._backend.load_into_blocking,
            cache_keys,
            objects,
        )
        future.add_done_callback(
            lambda fut, num_keys=len(keys): self._on_load_done(
                task_id, queue_idx, num_keys, fut
            )
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        self._unlock_stop.set()
        self._unlock_thread.join(timeout=5)

        for executor in self._queue_executors:
            executor.shutdown(wait=True)

        try:
            self._backend.close()
        except Exception:
            logger.exception("Failed closing RustRawBlockBackend")

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        self._loop.close()

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_backend(self, config: RustRawBlockL2AdapterConfig) -> RustRawBlockBackend:
        engine_cfg = LMCacheEngineConfig.from_defaults(
            chunk_size=256,
            local_cpu=True,
            max_local_cpu_size=1.0,
            lmcache_instance_id="mp-rust-raw-block-l2",
        )
        engine_cfg.extra_config = {
            "rust_raw_block.device_path": config.device_path,
            "rust_raw_block.capacity_bytes": int(config.capacity_bytes),
            "rust_raw_block.block_align": int(config.block_align),
            "rust_raw_block.header_bytes": int(config.header_bytes),
            "rust_raw_block.slot_bytes": _build_slot_bytes(
                header_bytes=int(config.header_bytes),
                block_align=int(config.block_align),
            ),
            "rust_raw_block.use_odirect": bool(config.use_odirect),
            "rust_raw_block.meta_total_bytes": int(config.meta_total_bytes),
            "rust_raw_block.meta_enable_periodic": bool(config.meta_enable_periodic),
            "rust_raw_block.enable_zero_copy": True,
        }

        local_cpu = LocalCPUBackend(
            config=engine_cfg,
            metadata=None,
            dst_device="cpu",
            memory_allocator=AdHocMemoryAllocator(device="cpu"),
        )

        return RustRawBlockBackend(
            config=engine_cfg,
            metadata=None,
            local_cpu_backend=local_cpu,
            loop=self._loop,
            dst_device="cpu",
        )

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _create_io_queue_executors(self, num_queues: int) -> list[ThreadPoolExecutor]:
        cpu_sets = self._partition_cpu_sets(num_queues)
        executors: list[ThreadPoolExecutor] = []

        for queue_idx in range(num_queues):
            cpu_set = cpu_sets[queue_idx]
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"l2-rust-io-{queue_idx}",
                initializer=self._io_thread_init,
                initargs=(cpu_set,),
            )
            executors.append(executor)

        return executors

    def _partition_cpu_sets(self, num_queues: int) -> list[set[int] | None]:
        if not hasattr(os, "sched_getaffinity"):
            return [None for _ in range(num_queues)]

        try:
            cpus = sorted(os.sched_getaffinity(0))
        except Exception:
            return [None for _ in range(num_queues)]

        if not cpus:
            return [None for _ in range(num_queues)]

        groups: list[set[int]] = [set() for _ in range(num_queues)]
        for idx, cpu in enumerate(cpus):
            groups[idx % num_queues].add(cpu)

        for idx in range(num_queues):
            if not groups[idx]:
                groups[idx].add(cpus[idx % len(cpus)])

        return groups

    @staticmethod
    def _io_thread_init(cpu_set: set[int] | None) -> None:
        if cpu_set is None or not hasattr(os, "sched_setaffinity"):
            return
        try:
            os.sched_setaffinity(0, cpu_set)
        except Exception:
            pass

    def _allocate_task_id(self) -> L2TaskId:
        with self._lock:
            task_id = self._next_task_id
            self._next_task_id += 1
        return task_id

    def _choose_queue(self, keys: list[ObjectKey]) -> int:
        preferred: int | None = None
        if keys:
            preferred = abs(int(keys[0].kv_rank)) % self._num_queues

        with self._lock:
            rr = self._rr_counter % self._num_queues
            self._rr_counter += 1

            if preferred is None:
                chosen = rr
            else:
                rr_depth = self._queue_depth[rr]
                preferred_depth = self._queue_depth[preferred]
                chosen = preferred if preferred_depth <= rr_depth else rr

            self._queue_depth[chosen] += 1
            return chosen

    def _mark_queue_done(self, queue_idx: int) -> None:
        with self._lock:
            self._queue_depth[queue_idx] = max(0, self._queue_depth[queue_idx] - 1)

    def _on_store_done(
        self,
        task_id: L2TaskId,
        queue_idx: int,
        future: Future,
    ) -> None:
        ok = False
        try:
            ok = bool(future.result())
        except Exception:
            logger.exception("store task failed: task_id=%d", task_id)

        with self._lock:
            self._completed_store_tasks[task_id] = ok

        self._mark_queue_done(queue_idx)
        self._signal_eventfd(self._store_efd)

    def _on_load_done(
        self,
        task_id: L2TaskId,
        queue_idx: int,
        num_keys: int,
        future: Future,
    ) -> None:
        try:
            bitmap = future.result()
        except Exception:
            logger.exception("load task failed: task_id=%d", task_id)
            bitmap = Bitmap(num_keys)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap

        self._mark_queue_done(queue_idx)
        self._signal_eventfd(self._load_efd)

    def _unlock_loop(self) -> None:
        while not self._unlock_stop.is_set() or not self._unlock_queue.empty():
            try:
                keys = self._unlock_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            while True:
                try:
                    self._backend.unlock(keys)
                    break
                except Exception:
                    logger.exception("unlock failed, retrying")
                    time.sleep(0.05)

    @staticmethod
    def _signal_eventfd(fd: int) -> None:
        try:
            os.eventfd_write(fd, 1)
        except OSError:
            pass
