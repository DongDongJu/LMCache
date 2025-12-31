# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import Future
from typing import Any, List, Optional, Sequence
import asyncio
import threading

# Third Party
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObj,
    TensorMemoryAllocator,
    _allocate_unpinned_numa_memory,
    _free_unpinned_numa_memory,
)
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.naive_serde import CreateSerde
from lmcache.v1.system_detection import NUMADetector, get_numa_distance

logger = init_logger(__name__)


class LocalCXLBackend(StorageBackendInterface):
    """Local capacity backend backed by pageable NUMA-bound host memory.

    Intended use:
    - DRAM pinned pool (LocalCPUBackend) remains the allocator/staging tier.
    - This backend stores KV chunks into a larger capacity pool (e.g., CXL NUMA node),
      and re-stages them back into DRAM on get.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        metadata: Optional[LMCacheEngineMetadata] = None,
    ):
        if torch.cuda.is_available():
            super().__init__(dst_device)
        else:
            super().__init__("cpu")

        self.cache_policy = get_cache_policy(config.cache_policy)
        self.dict = self.cache_policy.init_mutable_mapping()
        self.lock = threading.Lock()

        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        hybrid = config.get_hybrid_config()
        self.cxl_numa_node = self._select_cxl_numa_node(config)
        self.max_cache_size = int(float(hybrid["cxl_capacity_pool_gb"]) * 1024**3)
        self.promote_on_get = bool(hybrid["promote_on_get"])
        self.cxl_serde = (
            str(hybrid.get("cxl_serde", "naive") or "naive").strip().lower()
        )
        self.enable_compression = self.cxl_serde == "cachegen"
        self._warned_promote_ignored = False
        self._cached_positions: dict[CacheEngineKey, Optional[torch.Tensor]] = {}

        self.serializer = None
        self.deserializer = None
        if self.enable_compression:
            if metadata is None:
                raise ValueError(
                    "LocalCXLBackend compression requires LMCacheEngineMetadata"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "LocalCXLBackend compression (cachegen) requires CUDA"
                )
            self.serializer, self.deserializer = CreateSerde(
                self.cxl_serde, metadata, config
            )

        if self.max_cache_size <= 0:
            raise ValueError(
                "LocalCXLBackend requires extra_config['cxl_capacity_pool_gb'] > 0"
            )

        self.buffer = _allocate_unpinned_numa_memory(
            size=self.max_cache_size,
            numa_node=self.cxl_numa_node,
        )
        self.allocator = TensorMemoryAllocator(self.buffer)

        self.usage = 0

        # Track pinned keys during a lookup to preserve prefix order.
        self.keys_in_request: List[CacheEngineKey] = []

        logger.info(
            "LocalCXLBackend: capacity pool %.2f GiB on numa_node=%s",
            self.max_cache_size / 1024**3,
            self.cxl_numa_node,
        )
        if self.enable_compression:
            logger.info(
                "LocalCXLBackend: compression enabled (cxl_serde=%s)", self.cxl_serde
            )

    @staticmethod
    def _select_cxl_numa_node(config: LMCacheEngineConfig) -> int:
        extra = config.extra_config or {}

        # Explicit override.
        if "cxl_numa_node" in extra:
            return int(extra.get("cxl_numa_node", -1))

        device_id = 0
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        numa_mapping = NUMADetector.get_numa_mapping(config)
        dram_numa = None
        if numa_mapping is not None:
            dram_numa = numa_mapping.gpu_to_numa_mapping.get(device_id)

        # GPU-specific candidates.
        candidates = extra.get("gpu_to_cxl_numa_candidates")
        if isinstance(candidates, dict) and device_id in candidates:
            cand_list = [int(x) for x in candidates[device_id]]
            if not cand_list:
                return -1
            policy = str(extra.get("cxl_select_policy", "first")).lower()
            if policy == "nearest" and dram_numa is not None:
                return min(
                    cand_list,
                    key=lambda c: get_numa_distance(int(dram_numa), int(c)),
                )
            return int(cand_list[0])

        # Socket/DRAM-NUMA keyed mapping.
        sock_map = extra.get("socket_to_cxl_numa") or extra.get("dram_numa_to_cxl_numa")
        if isinstance(sock_map, dict) and dram_numa is not None:
            if int(dram_numa) in sock_map:
                return int(sock_map[int(dram_numa)])
            if str(int(dram_numa)) in sock_map:
                return int(sock_map[str(int(dram_numa))])

        return -1

    def __str__(self) -> str:
        return "LocalCXLBackend"

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.lock:
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
                self.keys_in_request.append(key)
            return True

    def touch_cache(self) -> None:
        with self.lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.dict)
            self.keys_in_request = []

    def pin(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if key not in self.dict:
                return False
            return self.dict[key].pin()

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.lock:
            if key not in self.dict:
                return False
            return self.dict[key].unpin()

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if force:
            self.lock.acquire()

        memory_obj = self.dict.pop(key, None)
        self._cached_positions.pop(key, None)
        if memory_obj is None:
            if force:
                self.lock.release()
            return False

        self.usage -= memory_obj.get_physical_size()
        self.stats_monitor.update_local_storage_usage(self.usage)

        memory_obj.ref_count_down()

        if force:
            self.cache_policy.update_on_force_evict(key)
            self.lock.release()

        return True

    def _copy_memory_obj(self, src: MemoryObj, dst: MemoryObj) -> None:
        src_raw = src.raw_tensor
        dst_raw = dst.raw_tensor
        assert src_raw is not None and dst_raw is not None
        assert dst_raw.numel() >= src_raw.numel()
        dst_raw[: src_raw.numel()].copy_(src_raw, non_blocking=False)

        # Preserve cached positions if present.
        dst.metadata.cached_positions = src.metadata.cached_positions

    def _allocate_in_capacity(self, src: MemoryObj) -> Optional[MemoryObj]:
        shapes = src.get_shapes()
        dtypes = src.get_dtypes()
        fmt = src.get_memory_format()

        # TensorMemoryAllocator expects shapes/dtypes as list.
        cap_obj = self.allocator.allocate(shapes, dtypes, fmt)
        return cap_obj

    def _allocate_bytes_in_capacity(self, nbytes: int) -> Optional[MemoryObj]:
        # Store compressed payload as a flat uint8 tensor backed by the CXL pool.
        return self.allocator.allocate(
            torch.Size([int(nbytes)]),
            torch.uint8,
            MemoryFormat.BINARY,
        )

    def _evict_until_allocated_bytes(self, nbytes: int) -> Optional[MemoryObj]:
        cap_obj = self._allocate_bytes_in_capacity(nbytes)
        if cap_obj is not None:
            return cap_obj

        max_evict = 64
        while max_evict > 0:
            evict_keys = self.cache_policy.get_evict_candidates(
                self.dict,
                num_candidates=min(8, max_evict),
            )
            if not evict_keys:
                break
            self.batched_remove(evict_keys, force=False)
            max_evict -= len(evict_keys)
            cap_obj = self._allocate_bytes_in_capacity(nbytes)
            if cap_obj is not None:
                return cap_obj
        return None

    def _evict_until_allocated(self, src_obj: MemoryObj) -> Optional[MemoryObj]:
        """Try to make room in the capacity tier and allocate.

        This makes the capacity tier behave like a swap: when the pool is full,
        evict cold chunks (best-effort) to admit new chunks.
        """
        cap_obj = self._allocate_in_capacity(src_obj)
        if cap_obj is not None:
            return cap_obj

        # Best-effort: evict a bounded number of chunks and retry.
        # Keep it small to avoid long stalls under heavy pinning.
        max_evict = 64
        while max_evict > 0:
            evict_keys = self.cache_policy.get_evict_candidates(
                self.dict,
                num_candidates=min(8, max_evict),
            )
            if not evict_keys:
                break
            self.batched_remove(evict_keys, force=False)
            max_evict -= len(evict_keys)
            cap_obj = self._allocate_in_capacity(src_obj)
            if cap_obj is not None:
                return cap_obj

        return None

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: List[MemoryObj],
        transfer_spec: Any = None,
    ) -> Optional[List[Future]]:
        # Synchronous put for now.
        for key, src_obj in zip(keys, objs, strict=False):
            with self.lock:
                if key in self.dict:
                    continue

            compressed: Optional[BytesBufferMemoryObj] = None
            if self.enable_compression:
                assert self.serializer is not None
                compressed = self.serializer.serialize(src_obj)  # type: ignore[assignment]

            with self.lock:
                if self.enable_compression:
                    assert compressed is not None
                    # Cache cached_positions separately (not part of cachegen payload).
                    self._cached_positions[key] = src_obj.metadata.cached_positions
                    cap_obj = self._evict_until_allocated_bytes(
                        len(compressed.byte_array)
                    )
                else:
                    cap_obj = self._evict_until_allocated(src_obj)
                if cap_obj is None:
                    logger.warning(
                        "LocalCXLBackend: allocation failed; skipping put "
                        "(usage=%.2f/%.2f GiB). Consider increasing "
                        "extra_config['cxl_capacity_pool_gb'] and/or reducing "
                        "pinning/working-set.",
                        self.usage / 1024**3,
                        self.max_cache_size / 1024**3,
                    )
                    continue

                copied_bytes = 0
                if self.enable_compression:
                    assert compressed is not None
                    copied_bytes = int(len(compressed.byte_array))
                    dst = cap_obj.raw_tensor
                    assert dst is not None
                    mv = memoryview(compressed.byte_array)
                    if hasattr(torch, "frombuffer"):
                        src_u8 = torch.frombuffer(mv, dtype=torch.uint8)
                    else:
                        # Third Party
                        import numpy as np  # type: ignore

                        src_u8 = torch.from_numpy(np.frombuffer(mv, dtype=np.uint8))
                    dst[: src_u8.numel()].copy_(src_u8, non_blocking=False)
                else:
                    if src_obj.raw_tensor is not None:
                        copied_bytes = int(
                            src_obj.raw_tensor.numel()
                            * src_obj.raw_tensor.element_size()
                        )
                    self._copy_memory_obj(src_obj, cap_obj)
                self.stats_monitor.update_interval_local_cxl_put(copied_bytes)

                cap_obj.ref_count_up()
                self.dict[key] = cap_obj
                self.cache_policy.update_on_put(key)

                self.usage += cap_obj.get_physical_size()
                self.stats_monitor.update_local_storage_usage(self.usage)

        return None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        with self.lock:
            if key not in self.dict:
                return None

            cap_obj = self.dict[key]
            cap_obj.ref_count_up()
            self.cache_policy.update_on_hit(key, self.dict)

        copied_bytes = 0
        if cap_obj.raw_tensor is not None:
            copied_bytes = int(
                cap_obj.raw_tensor.numel() * cap_obj.raw_tensor.element_size()
            )

        if self.enable_compression:
            assert self.deserializer is not None

            class _ByteView:
                def __init__(self, mv):
                    self.byte_array = mv

            decoded = self.deserializer.deserialize(_ByteView(cap_obj.byte_array))  # type: ignore[arg-type]
            cached_positions = self._cached_positions.get(key)
            if cached_positions is not None:
                decoded.metadata.cached_positions = cached_positions
            self.stats_monitor.update_interval_local_cxl_get(copied_bytes)

            if self.promote_on_get and not self._warned_promote_ignored:
                logger.info(
                    "LocalCXLBackend: promote_on_get is ignored when "
                    "compression is enabled (would require extra staging copies)."
                )
                self._warned_promote_ignored = True

            cap_obj.ref_count_down()
            return decoded

        # Uncompressed path: restage into DRAM staging pool.
        fmt = cap_obj.get_memory_format()
        shapes = cap_obj.get_shapes()
        dtypes = cap_obj.get_dtypes()
        staging_obj = self.local_cpu_backend.allocate(shapes, dtypes, fmt)
        assert staging_obj is not None, "DRAM staging allocation failed"

        self._copy_memory_obj(cap_obj, staging_obj)
        self.stats_monitor.update_interval_local_cxl_get(copied_bytes)

        # Optional promotion into DRAM hot cache.
        if self.promote_on_get and self.local_cpu_backend.use_hot:
            if not self.local_cpu_backend.contains(key):
                self.local_cpu_backend.submit_put_task(key, staging_obj)
                self.stats_monitor.update_interval_local_hybrid_promote_to_dram(1)

        cap_obj.ref_count_down()
        return staging_obj

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        transfer_spec: Any = None,
    ) -> list[MemoryObj]:
        # Synchronous restage in the event loop thread for now.
        mem_objs: list[MemoryObj] = []
        for key in keys:
            mem_obj = self.get_blocking(key)
            assert mem_obj is not None, f"Key {key} not found in LocalCXLBackend"
            mem_objs.append(mem_obj)
        return mem_objs

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit = 0
        with self.lock:
            for key in keys:
                if key not in self.dict:
                    return num_hit
                if pin:
                    self.dict[key].pin()
                    self.keys_in_request.append(key)
                num_hit += 1
        return num_hit

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        with self.lock:
            keys = list(self.dict.keys())
        self.batched_remove(keys, force=True)
        _free_unpinned_numa_memory(self.buffer, size=self.max_cache_size)
