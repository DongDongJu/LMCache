# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Sequence
from concurrent.futures import Future
from typing import Any, Callable, List, Optional, Union
import threading
import time

# Third Party
import torch

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

try:
    # First Party
    from lmcache.lmcache_sysram import LMCacheSysRAMCore

    _SYSRAM_IMPORT_ERROR: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - exercised in build/install flow
    LMCacheSysRAMCore = None  # type: ignore[assignment]
    _SYSRAM_IMPORT_ERROR = exc


logger = init_logger(__name__)


class _SysRAMMemoryAllocator(MemoryAllocatorInterface):
    def __init__(self, core: "LMCacheSysRAMCore", slot_bytes: int):
        self.core = core
        self.slot_bytes = slot_bytes

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        allocated = self.core.allocate_slot()
        if allocated is None:
            return None

        slot_id, raw_tensor = allocated
        return TensorMemoryObj(
            raw_data=raw_tensor,
            metadata=MemoryObjMetadata(
                shape=shapes[0],
                dtype=dtypes[0],
                address=int(slot_id),
                phy_size=self.slot_bytes,
                ref_count=1,
                pin_count=0,
                fmt=fmt,
                shapes=shapes,
                dtypes=dtypes,
            ),
            parent_allocator=self,
        )

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[list[MemoryObj]]:
        memory_objs: list[MemoryObj] = []
        for _ in range(batch_size):
            memory_obj = self.allocate(shapes, dtypes, fmt, allocator_type)
            if memory_obj is None:
                for allocated_obj in memory_objs:
                    allocated_obj.ref_count_down()
                return None
            memory_objs.append(memory_obj)
        return memory_objs

    def free(
        self,
        memory_obj: MemoryObj,
        allocator_type: Optional[str] = None,
    ):
        if not memory_obj.is_valid():
            return
        self.core.release_slot(int(memory_obj.meta.address))
        memory_obj.invalidate()

    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ):
        for memory_obj in memory_objs:
            self.free(memory_obj, allocator_type=allocator_type)


class SysRAMBackend(AllocatorBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
    ):
        super().__init__("cpu")
        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.sysram_lock = threading.Lock()
        self.keys_in_request: List[CacheEngineKey] = []
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        sysram_config = config.get_sysram_backend_config()
        self.promote_on_get = sysram_config["promote_on_get"]
        self.slot_bytes = self.get_full_chunk_size_bytes(metadata)
        self.memory_allocator = self.initialize_allocator(config, metadata)
        self.usage = 0

    def __str__(self):
        return self.__class__.__name__

    def initialize_allocator(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> MemoryAllocatorInterface:
        if LMCacheSysRAMCore is None:
            raise RuntimeError(
                "lmcache.lmcache_sysram is not available. "
                "Reinstall LMCache so the native SysRAM extension is built."
            ) from _SYSRAM_IMPORT_ERROR

        sysram_config = config.get_sysram_backend_config()
        pools = sysram_config["pools"]
        pool_nodes = [pool["numa_node"] for pool in pools]
        pool_sizes_bytes = [int(pool["size_gb"] * 1024**3) for pool in pools]
        self.core = LMCacheSysRAMCore(pool_nodes, pool_sizes_bytes, self.slot_bytes)
        return _SysRAMMemoryAllocator(self.core, self.slot_bytes)

    def get_memory_allocator(self) -> MemoryAllocatorInterface:
        return self.memory_allocator

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.sysram_lock:
            memory_obj = self.dict.get(key)
            if memory_obj is None:
                return False
            if pin:
                memory_obj.pin()
                self.keys_in_request.append(key)
            return True

    def batched_contains(
        self,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        return super().batched_contains(keys, pin)

    def touch_cache(self):
        with self.sysram_lock:
            for key in reversed(self.keys_in_request):
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Optional[Future]:
        self.batched_submit_put_task(
            [key], [memory_obj], on_complete_callback=on_complete_callback
        )
        return None

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        completed_keys: list[CacheEngineKey] = []
        with self.sysram_lock:
            for key, memory_obj in zip(keys, memory_objs, strict=False):
                if key in self.dict:
                    self.cache_policy.update_on_hit(key, self.dict)
                    continue
                if not self.core.bind_key(key.to_string(), int(memory_obj.meta.address)):
                    continue

                memory_obj.ref_count_up()
                self.dict[key] = memory_obj
                self.cache_policy.update_on_put(key)
                self.usage += memory_obj.get_physical_size()
                completed_keys.append(key)

        if completed_keys:
            self.stats_monitor.update_local_storage_usage(self.usage)

        if on_complete_callback is not None:
            for key in completed_keys:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        with self.sysram_lock:
            memory_obj = self.dict.get(key)
            if memory_obj is None:
                return None
            self.cache_policy.update_on_hit(key, self.dict)
            shapes = memory_obj.get_shapes()
            dtypes = memory_obj.get_dtypes()
            fmt = memory_obj.get_memory_format()
            cached_positions = memory_obj.meta.cached_positions

        staging_obj = self.local_cpu_backend.allocate(shapes, dtypes, fmt)
        if staging_obj is None:
            return None

        staging_tensor = staging_obj.raw_tensor
        if staging_tensor is None:
            staging_obj.ref_count_down()
            return None

        if not self.core.copy_out(key.to_string(), staging_tensor[: memory_obj.get_size()]):
            staging_obj.ref_count_down()
            return None

        staging_obj.meta.cached_positions = cached_positions

        if self.promote_on_get and self.local_cpu_backend.use_hot:
            self.local_cpu_backend.submit_put_task(key, staging_obj)

        return staging_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        return None

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        memory_objs = []
        for key in keys:
            memory_obj = self.get_blocking(key)
            if memory_obj is None:
                break
            memory_objs.append(memory_obj)
        return memory_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_chunks = 0
        with self.sysram_lock:
            for key in keys:
                memory_obj = self.dict.get(key)
                if memory_obj is None:
                    return num_hit_chunks
                if pin:
                    memory_obj.pin()
                    self.keys_in_request.append(key)
                num_hit_chunks += 1
        return num_hit_chunks

    def pin(self, key: CacheEngineKey) -> bool:
        with self.sysram_lock:
            memory_obj = self.dict.get(key)
            if memory_obj is None:
                return False
            memory_obj.pin()
            return True

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.sysram_lock:
            memory_obj = self.dict.get(key)
            if memory_obj is None:
                return False
            memory_obj.unpin()
            return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.sysram_lock.acquire()
        memory_obj = self.dict.pop(key, None)
        if memory_obj is None:
            if force:
                self.sysram_lock.release()
            return False

        self.core.erase_key(key.to_string())
        self.usage -= memory_obj.get_physical_size()
        memory_obj.ref_count_down()

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.sysram_lock.release()

        self.stats_monitor.update_local_storage_usage(self.usage)
        return True

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
        if memory_obj is not None or not eviction:
            return memory_obj

        while True:
            wait_other_requests = True
            with self.sysram_lock:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if evict_keys:
                    wait_other_requests = False
                    self.batched_remove(evict_keys, force=False)

            if wait_other_requests:
                if not busy_loop:
                    return None
                time.sleep(0.1)

            memory_obj = self.memory_allocator.allocate(shapes, dtypes, fmt)
            if memory_obj is not None:
                return memory_obj

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        memory_objs = self.memory_allocator.batched_allocate(
            shapes, dtypes, batch_size, fmt
        )
        if memory_objs is not None or not eviction:
            return memory_objs

        while True:
            wait_other_requests = True
            with self.sysram_lock:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.dict, num_candidates=1
                )
                if evict_keys:
                    wait_other_requests = False
                    self.batched_remove(evict_keys, force=False)

            if wait_other_requests:
                if not busy_loop:
                    return None
                time.sleep(0.1)

            memory_objs = self.memory_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
            if memory_objs is not None:
                return memory_objs

    def get_full_chunk_size_bytes(self, metadata: LMCacheMetadata) -> int:
        return get_size_bytes(
            metadata.get_shapes(self.config.chunk_size),
            metadata.get_dtypes(),
        )

    def calculate_chunk_budget(self) -> int:
        return int(self.core.capacity_slots())

    def clear(self) -> int:
        clear_keys = []
        num_cleared_tokens = 0
        with self.sysram_lock:
            for key, memory_obj in self.dict.items():
                if not memory_obj.can_evict:
                    continue
                clear_keys.append(key)
                num_cleared_tokens += memory_obj.get_num_tokens()
        self.batched_remove(clear_keys)
        return num_cleared_tokens

    def get_allocator_backend(self):
        return self

    def get_keys(self) -> List[CacheEngineKey]:
        with self.sysram_lock:
            return list(self.dict.keys())

    def close(self) -> None:
        with self.sysram_lock:
            all_keys = list(self.dict.keys())
        self.batched_remove(all_keys)
