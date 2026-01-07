# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 LMCache Authors
"""
LMCache external backend for Modular MAX integration.

This module provides LMCacheExternalBackend which implements MAX's external
KV cache interface, enabling multi-tier KV cache persistence (CPU + Disk).

Configuration via environment variables:
  - LMCACHE_LOCAL_CPU=true          Enable CPU cache
  - LMCACHE_MAX_LOCAL_CPU_SIZE=4.0  CPU cache size in GB
  - LMCACHE_LOCAL_DISK="file:///path/"  Disk cache path
  - LMCACHE_MAX_LOCAL_DISK_SIZE=10.0    Disk cache size in GB
  - LMCACHE_CHUNK_SIZE=256          Tokens per chunk

Key features:
  - Direct DLPack transfer (torch.from_dlpack) for bfloat16 support
  - Async background storage (non-blocking inference)
  - Multi-tier cache (CPU -> Disk)
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    from max.driver import Tensor as MAXTensor

logger = init_logger(__name__)


class AsyncStoreWorker:
    """Background worker for non-blocking KV cache storage."""

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._pending_count = 0
        self._lock = threading.Lock()
        self._start_worker()

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                task = self._queue.get(timeout=0.1)
                if task is None:
                    continue
                callback, args = task
                try:
                    callback(*args)
                except Exception as e:
                    logger.warning(f"Async store task failed: {e}")
                finally:
                    with self._lock:
                        self._pending_count -= 1
            except queue.Empty:
                continue

    def submit(self, callback, *args):
        with self._lock:
            self._pending_count += 1
        self._queue.put((callback, args))

    def shutdown(self, wait: bool = True):
        self._stop_event.set()
        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)


@dataclass
class ExternalCacheLookupResult:
    matched_prefix_len: int
    cache_tier: Optional[str] = None
    metadata: Optional[Any] = None


@dataclass
class ExternalCacheLoadResult:
    success: bool
    loaded_tokens: int
    load_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class ExternalCacheStoreResult:
    event_id: str
    stored_tokens: int


@dataclass
class ExternalCacheStats:
    hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    bytes_loaded: int = 0
    bytes_stored: int = 0
    tier_hits: Dict[str, int] = field(default_factory=dict)


class LMCacheExternalBackend:
    """LMCache external backend for MAX KV cache offloading.
    
    Implements MAX's external KV cache interface to enable multi-tier
    caching with CPU and disk storage backends.
    """

    def __init__(
        self,
        model_name: str = "max-model",
        chunk_size: int = 64,
        page_size: int = 128,
        num_layers: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        kv_dtype: str = "bfloat16",
        **kwargs,
    ):
        self._model_name = model_name
        self._page_size = page_size
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim

        # Load config from environment
        self._config = LMCacheEngineConfig.from_env()
        self._chunk_size = self._config.chunk_size

        logger.info(f"LMCache config: chunk_size={self._chunk_size}, "
                    f"local_cpu={self._config.local_cpu}, "
                    f"local_disk={self._config.local_disk}")

        self._torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(kv_dtype, torch.bfloat16)

        # Asyncio loop for backends
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()

        # Storage backends
        self._cpu_backend: Optional[LocalCPUBackend] = None
        self._disk_backend = None
        self._storage_lock = threading.RLock()
        self._use_simple = False
        self._simple_cache: Dict[int, torch.Tensor] = {}

        # Metadata for backends
        kv_shape = (num_layers, 2, self._chunk_size, num_kv_heads, head_dim)
        self._metadata = LMCacheEngineMetadata(
            model_name=model_name,
            kv_shape=kv_shape,
            kv_dtype=self._torch_dtype,
            world_size=1,
            worker_id=0,
            fmt="vllm",
            role="kv_both",
            use_mla=False,
        )

        self._init_backends()

        # Stats and async tracking
        self._stats = ExternalCacheStats()
        self._pending: Dict[str, threading.Event] = {}
        self._store_lock = threading.Lock()
        self._counter = 0

        # Async worker
        self._async_worker = AsyncStoreWorker()
        logger.info(f"LMCacheExternalBackend initialized: model={model_name}")

    def _init_backends(self):
        try:
            if self._config.local_cpu:
                self._cpu_backend = LocalCPUBackend(
                    config=self._config,
                    metadata=self._metadata,
                    dst_device="cpu",
                )
                logger.info(f"CPU backend: {self._config.max_local_cpu_size}GB")

            if self._config.local_disk:
                from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend
                self._disk_backend = LocalDiskBackend(
                    config=self._config,
                    loop=self._loop,
                    local_cpu_backend=self._cpu_backend,
                    dst_device="cpu",
                    metadata=self._metadata,
                )
                logger.info(f"Disk backend: {self._config.max_local_disk_size}GB")

            self._use_simple = not (self._cpu_backend or self._disk_backend)
        except Exception as e:
            logger.warning(f"Backend init failed: {e}, using simple cache")
            self._use_simple = True

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _make_key(self, tokens: Sequence[int], chunk_idx: int) -> CacheEngineKey:
        end_pos = (chunk_idx + 1) * self._chunk_size
        prefix = list(tokens[:end_pos])
        chunk_hash = int(hashlib.sha256(str(prefix).encode()).hexdigest()[:16], 16)
        return CacheEngineKey(
            fmt="vllm",
            model_name=self._model_name,
            world_size=1,
            worker_id=0,
            chunk_hash=chunk_hash,
            dtype=self._torch_dtype,
        )

    def lookup(self, tokens: Sequence[int]) -> ExternalCacheLookupResult:
        """Check for cached KV data."""
        matched = 0
        tier = None
        num_chunks = len(tokens) // self._chunk_size

        with self._storage_lock:
            for i in range(num_chunks):
                key = self._make_key(tokens, i)
                found = False

                if self._cpu_backend and self._cpu_backend.contains(key):
                    matched = (i + 1) * self._chunk_size
                    tier = "cpu"
                    found = True
                elif self._disk_backend and self._disk_backend.contains(key):
                    matched = (i + 1) * self._chunk_size
                    tier = "disk"
                    found = True
                elif self._use_simple and key.chunk_hash in self._simple_cache:
                    matched = (i + 1) * self._chunk_size
                    tier = "memory"
                    found = True

                if not found:
                    break

        matched = (matched // self._page_size) * self._page_size

        if matched > 0:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        return ExternalCacheLookupResult(matched_prefix_len=matched, cache_tier=tier)

    def load(
        self,
        tokens: Sequence[int],
        prefix_len: int,
        dst_blocks: Sequence[int],
        dst_tensors: Sequence[Any],
    ) -> ExternalCacheLoadResult:
        """Load KV data from cache to device tensors."""
        start_time = time.perf_counter()

        if prefix_len == 0 or not dst_tensors or not dst_blocks:
            return ExternalCacheLoadResult(success=True, loaded_tokens=0)

        try:
            num_chunks = prefix_len // self._chunk_size
            loaded_tokens = 0

            for chunk_idx in range(num_chunks):
                key = self._make_key(tokens, chunk_idx)
                mem_obj = None

                with self._storage_lock:
                    if self._cpu_backend:
                        mem_obj = self._cpu_backend.get_blocking(key)
                    if mem_obj is None and self._disk_backend:
                        mem_obj = self._disk_backend.get_blocking(key)
                    if mem_obj is None and self._use_simple:
                        if key.chunk_hash in self._simple_cache:
                            cached = self._simple_cache[key.chunk_hash]
                            mem_obj = type('MemObj', (), {'tensor': cached})()

                if mem_obj is None or mem_obj.tensor is None:
                    break

                chunk_start = chunk_idx * self._chunk_size
                if self._write_chunk_to_device(
                    mem_obj.tensor, dst_tensors[0], dst_blocks, chunk_start
                ):
                    loaded_tokens += self._chunk_size

                if hasattr(mem_obj, 'ref_count_down'):
                    mem_obj.ref_count_down()

            load_time_ms = (time.perf_counter() - start_time) * 1000
            logger.info(f"Loaded {loaded_tokens} tokens in {load_time_ms:.1f}ms")

            return ExternalCacheLoadResult(
                success=True, loaded_tokens=loaded_tokens, load_time_ms=load_time_ms
            )

        except Exception as e:
            logger.error(f"Load error: {e}")
            return ExternalCacheLoadResult(success=False, loaded_tokens=0, error=str(e))

    def _write_chunk_to_device(
        self,
        kv_tensor: torch.Tensor,
        dst_tensor: Any,
        dst_blocks: Sequence[int],
        start_pos: int,
    ) -> bool:
        """Write KV chunk to MAX device tensor using DLPack."""
        try:
            from max.driver import Tensor as MAXTensor
            from max.dtype import DType

            if kv_tensor.device.type != 'cpu':
                kv_tensor = kv_tensor.cpu()
            if not kv_tensor.is_contiguous():
                kv_tensor = kv_tensor.contiguous()

            kv_dim, num_layers, chunk_size, num_heads, head_dim = kv_tensor.shape
            page_size = dst_tensor.shape[3]
            is_bfloat16 = dst_tensor.dtype == DType.bfloat16

            pos = start_pos
            remaining = chunk_size
            kv_offset = 0

            while remaining > 0:
                block_idx = pos // page_size
                if block_idx >= len(dst_blocks):
                    break

                block_id = dst_blocks[block_idx]
                page_offset = pos % page_size
                take = min(page_size - page_offset, remaining)

                src_slice = kv_tensor[:, :, kv_offset:kv_offset + take, :, :].contiguous()

                # Try DLPack first (supports bfloat16)
                try:
                    src_max = MAXTensor.from_dlpack(src_slice)
                except Exception:
                    # Fallback: numpy with uint16 view for bfloat16
                    if is_bfloat16 and src_slice.dtype == torch.bfloat16:
                        src_np = src_slice.view(torch.uint16).numpy()
                        src_max = MAXTensor.from_numpy(src_np).view(DType.bfloat16)
                    else:
                        src_max = MAXTensor.from_numpy(src_slice.numpy())

                dst_slice = dst_tensor[block_id, :, :, page_offset:page_offset + take, :, :]

                if not dst_tensor.device.is_host:
                    src_max = src_max.to(dst_tensor.device)

                dst_slice.inplace_copy_from(src_max)

                remaining -= take
                pos += take
                kv_offset += take

            return True

        except Exception as e:
            logger.error(f"Write chunk error: {e}")
            return False

    def store(
        self,
        tokens: Sequence[int],
        src_blocks: Sequence[int],
        src_tensors: Sequence[Any],
        start_pos: int = 0,
    ) -> ExternalCacheStoreResult:
        """Store KV data to cache (async)."""
        with self._store_lock:
            self._counter += 1
            event_id = f"store_{self._counter}"
            event = threading.Event()
            self._pending[event_id] = event

        num_tokens = len(tokens) - start_pos
        num_chunks = num_tokens // self._chunk_size

        if not src_tensors or num_chunks == 0:
            event.set()
            return ExternalCacheStoreResult(event_id=event_id, stored_tokens=0)

        # Extract chunks
        chunks_to_store: List[Tuple[CacheEngineKey, torch.Tensor]] = []

        for i in range(num_chunks):
            chunk_start = start_pos + i * self._chunk_size
            chunk_idx = chunk_start // self._chunk_size
            key = self._make_key(tokens, chunk_idx)

            if self._cpu_backend and self._cpu_backend.contains(key):
                continue
            if self._use_simple and key.chunk_hash in self._simple_cache:
                continue

            try:
                kv_cpu = self._extract_chunk(src_tensors[0], src_blocks, chunk_start)
                if kv_cpu is not None:
                    chunks_to_store.append((key, kv_cpu.cpu() if kv_cpu.device.type != 'cpu' else kv_cpu))
            except Exception as e:
                logger.warning(f"Extract chunk {i} failed: {e}")

        if not chunks_to_store:
            event.set()
            return ExternalCacheStoreResult(event_id=event_id, stored_tokens=0)

        def _do_store():
            stored = 0
            for key, kv_cpu in chunks_to_store:
                with self._storage_lock:
                    if self._use_simple:
                        self._simple_cache[key.chunk_hash] = kv_cpu
                        stored += self._chunk_size
                    elif self._cpu_backend:
                        try:
                            mem_obj = self._cpu_backend.allocate(
                                torch.Size(kv_cpu.shape), kv_cpu.dtype,
                                MemoryFormat.KV_2LTD, eviction=True, busy_loop=False
                            )
                            if mem_obj and mem_obj.tensor is not None:
                                mem_obj.tensor.copy_(kv_cpu)
                                self._cpu_backend.submit_put_task(key, mem_obj)
                                if self._disk_backend:
                                    self._disk_backend.submit_put_task(key, mem_obj)
                                stored += self._chunk_size
                        except Exception as e:
                            logger.warning(f"Backend store failed: {e}")
                            self._simple_cache[key.chunk_hash] = kv_cpu
                            stored += self._chunk_size

            if stored > 0:
                logger.info(f"Stored {stored} tokens")
            event.set()

        self._async_worker.submit(_do_store)
        return ExternalCacheStoreResult(
            event_id=event_id, stored_tokens=len(chunks_to_store) * self._chunk_size
        )

    def _extract_chunk(
        self,
        src_tensor: Any,
        src_blocks: Sequence[int],
        start_pos: int,
    ) -> Optional[torch.Tensor]:
        """Extract KV chunk from MAX tensor using torch.from_dlpack."""
        try:
            total_pages, kv_dim, num_layers, page_size, num_heads, head_dim = src_tensor.shape

            collected = []
            remaining = self._chunk_size
            pos = start_pos

            while remaining > 0:
                block_idx = pos // page_size
                if block_idx >= len(src_blocks):
                    break
                block_id = src_blocks[block_idx]
                if block_id >= total_pages:
                    break

                offset = pos % page_size
                take = min(page_size - offset, remaining)

                slice_tensor = src_tensor[block_id, :, :, offset:offset + take, :, :]

                # Use torch.from_dlpack (supports bfloat16)
                try:
                    torch_slice = torch.from_dlpack(slice_tensor)
                    if torch_slice.is_cuda:
                        torch_slice = torch_slice.cpu()
                    collected.append(torch_slice.clone())
                except Exception:
                    # Fallback for bfloat16 via uint16
                    from max.dtype import DType
                    if slice_tensor.dtype == DType.bfloat16:
                        slice_np = slice_tensor.view(DType.uint16).to_numpy()
                        torch_slice = torch.from_numpy(slice_np.copy()).view(torch.bfloat16)
                    else:
                        torch_slice = torch.from_numpy(slice_tensor.to_numpy().copy())
                    collected.append(torch_slice)

                remaining -= take
                pos += take

            return torch.cat(collected, dim=2) if collected else None

        except Exception as e:
            logger.error(f"Extract chunk error: {e}")
            return None

    def is_store_complete(self, event_id: str) -> bool:
        with self._store_lock:
            if event_id not in self._pending:
                return True
            if self._pending[event_id].is_set():
                del self._pending[event_id]
                return True
            return False

    def get_stats(self) -> ExternalCacheStats:
        stats = ExternalCacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            tier_hits=self._stats.tier_hits.copy(),
        )
        total = stats.hits + stats.misses
        stats.hit_rate = stats.hits / total if total > 0 else 0.0
        return stats

    def clear(self) -> int:
        with self._storage_lock:
            count = len(self._simple_cache)
            self._simple_cache.clear()
        return count

    def shutdown(self):
        logger.info("Shutting down LMCacheExternalBackend")
        for e in self._pending.values():
            e.wait(timeout=5.0)
        self._async_worker.shutdown(wait=True)
        if self._disk_backend:
            self._disk_backend.close()
        if self._cpu_backend:
            self._cpu_backend.close()
        self._loop.call_soon_threadsafe(self._loop.stop)


def create_lmcache_backend(config: Dict[str, Any]) -> LMCacheExternalBackend:
    """Factory function to create LMCache backend."""
    return LMCacheExternalBackend(**config)
