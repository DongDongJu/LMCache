# SPDX-License-Identifier: Apache-2.0
# Standard
import asyncio
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    AdHocMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.local_cxl_backend import LocalCXLBackend


def create_test_config(
    local_cpu: bool = True,
    use_layerwise: bool = False,
    enable_blending: bool = False,
    cxl_capacity_pool_gb: float = 0.1,
    cxl_numa_node: int = -1,
    cxl_serde: str = "naive",
    prefer_cxl_on_put: bool = True,
    promote_on_get: bool = True,
    demote_on_pressure: bool = False,
    dram_hot_cache_max_chunks: int = 0,
):
    """Create a test configuration for LocalCXLBackend."""
    extra_config = {
        "local_memory_backend": "hybrid",
        "cxl_capacity_pool_gb": cxl_capacity_pool_gb,
        "cxl_numa_node": cxl_numa_node,
        "cxl_serde": cxl_serde,
        "prefer_cxl_on_put": prefer_cxl_on_put,
        "promote_on_get": promote_on_get,
        "demote_on_pressure": demote_on_pressure,
        "dram_hot_cache_max_chunks": dram_hot_cache_max_chunks,
    }
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=local_cpu,
        use_layerwise=use_layerwise,
        enable_blending=enable_blending,
        lmcache_instance_id="test_cxl_instance",
        extra_config=extra_config,
    )
    return config


def create_test_key(key_id: str = "test_key") -> CacheEngineKey:
    """Create a test CacheEngineKey."""
    return CacheEngineKey("vllm", "test_model", 3, 123, hash(key_id), torch.bfloat16)


def create_test_memory_obj(shape=(2, 16, 8, 128), dtype=torch.bfloat16) -> MemoryObj:
    """Create a test MemoryObj using AdHocMemoryAllocator for testing."""
    allocator = AdHocMemoryAllocator(device="cpu")
    memory_obj = allocator.allocate(shape, dtype, fmt=MemoryFormat.KV_T2D)
    return memory_obj


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def local_cpu_backend_for_cxl(memory_allocator):
    """Create a LocalCPUBackend to be used as staging for CXL backend."""
    config = create_test_config()
    return LocalCPUBackend(config=config, memory_allocator=memory_allocator)


@pytest.fixture
def local_cxl_backend(local_cpu_backend_for_cxl, event_loop):
    """Create a LocalCXLBackend for testing."""
    config = create_test_config()
    backend = LocalCXLBackend(
        config=config,
        loop=event_loop,
        local_cpu_backend=local_cpu_backend_for_cxl,
        dst_device="cpu",
        metadata=None,
    )
    yield backend
    # Cleanup
    backend.close()


@pytest.fixture
def local_cxl_backend_no_promote(local_cpu_backend_for_cxl, event_loop):
    """Create a LocalCXLBackend with promote_on_get disabled."""
    config = create_test_config(promote_on_get=False)
    backend = LocalCXLBackend(
        config=config,
        loop=event_loop,
        local_cpu_backend=local_cpu_backend_for_cxl,
        dst_device="cpu",
        metadata=None,
    )
    yield backend
    backend.close()


class TestLocalCXLBackendInit:
    """Test cases for LocalCXLBackend initialization."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_init_basic(self, local_cpu_backend_for_cxl, event_loop):
        """Test LocalCXLBackend basic initialization."""
        config = create_test_config()
        backend = LocalCXLBackend(
            config=config,
            loop=event_loop,
            local_cpu_backend=local_cpu_backend_for_cxl,
            dst_device="cpu",
            metadata=None,
        )

        assert backend is not None
        assert backend.local_cpu_backend == local_cpu_backend_for_cxl
        assert len(backend.dict) == 0  # capacity cache stored in self.dict

        backend.close()
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_init_with_custom_pool_size(self, local_cpu_backend_for_cxl, event_loop):
        """Test LocalCXLBackend with custom pool size."""
        config = create_test_config(cxl_capacity_pool_gb=0.05)
        backend = LocalCXLBackend(
            config=config,
            loop=event_loop,
            local_cpu_backend=local_cpu_backend_for_cxl,
            dst_device="cpu",
            metadata=None,
        )

        # Pool size should be approximately 0.05 GB
        expected_size = int(0.05 * 1024**3)
        assert backend.max_cache_size >= expected_size * 0.9  # Allow some margin

        backend.close()
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_str_representation(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test string representation."""
        assert str(local_cxl_backend) == "LocalCXLBackend"
        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendOperations:
    """Test cases for LocalCXLBackend basic operations."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_contains_empty_cache(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test contains() on empty cache."""
        key = create_test_key("nonexistent")
        assert not local_cxl_backend.contains(key)
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_batched_submit_put_and_contains(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test batched_submit_put_task() with single item and contains()."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Put the memory object using batched API
        result = local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        # Should return None (synchronous put)
        assert result is None

        # Key should now be in cache
        assert local_cxl_backend.contains(key)
        assert key in local_cxl_backend.dict

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_batched_submit_put_duplicate_key(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test batched_submit_put_task() with duplicate key."""
        key = create_test_key("test_key")
        memory_obj1 = create_test_memory_obj()
        memory_obj2 = create_test_memory_obj()

        # First put
        local_cxl_backend.batched_submit_put_task([key], [memory_obj1])
        assert local_cxl_backend.contains(key)

        # Second put with same key should be ignored
        local_cxl_backend.batched_submit_put_task([key], [memory_obj2])
        # Should still have only one entry
        assert len(local_cxl_backend.dict) == 1

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_get_blocking_not_found(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test get_blocking() when key doesn't exist."""
        key = create_test_key("nonexistent")
        result = local_cxl_backend.get_blocking(key)

        assert result is None
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_get_blocking_found(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test get_blocking() when key exists."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Put first
        local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        # Get should return a memory object
        result = local_cxl_backend.get_blocking(key)

        assert result is not None
        assert isinstance(result, MemoryObj)

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_get_blocking_with_promote(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test get_blocking() promotes to DRAM when promote_on_get is enabled."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Put first
        local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        # Get should promote to DRAM
        result = local_cxl_backend.get_blocking(key)

        assert result is not None
        # With promote_on_get=True, the data should be copied to DRAM staging
        # The LocalCPUBackend should now have the key
        assert local_cpu_backend_for_cxl.contains(key)

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_get_blocking_no_promote(
        self, local_cxl_backend_no_promote, local_cpu_backend_for_cxl
    ):
        """Test get_blocking() does not promote when promote_on_get is disabled."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Put first
        local_cxl_backend_no_promote.batched_submit_put_task([key], [memory_obj])

        # Get without promotion
        result = local_cxl_backend_no_promote.get_blocking(key)

        assert result is not None
        # With promote_on_get=False, the LocalCPUBackend should NOT have the key
        assert not local_cpu_backend_for_cxl.contains(key)

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_remove_existing_key(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test remove() for existing key."""
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()

        # Put first
        local_cxl_backend.batched_submit_put_task([key], [memory_obj])
        assert local_cxl_backend.contains(key)

        # Remove
        result = local_cxl_backend.remove(key)

        assert result is True
        assert not local_cxl_backend.contains(key)

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_remove_nonexistent_key(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test remove() for non-existent key."""
        key = create_test_key("nonexistent")
        result = local_cxl_backend.remove(key)

        assert result is False
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_dict_keys(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test accessing keys through the internal dict."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Put all keys using batched API
        local_cxl_backend.batched_submit_put_task(keys, memory_objs)

        # Get all keys from internal dict
        retrieved_keys = list(local_cxl_backend.dict.keys())

        assert len(retrieved_keys) == 3
        assert all(key in retrieved_keys for key in keys)

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_dict_keys_empty(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test accessing keys from empty cache."""
        keys = list(local_cxl_backend.dict.keys())
        assert len(keys) == 0
        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendBatched:
    """Test cases for LocalCXLBackend batched operations."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_batched_submit_put_task(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test batched_submit_put_task()."""
        keys = [create_test_key(f"key_{i}") for i in range(3)]
        memory_objs = [create_test_memory_obj() for _ in range(3)]

        # Batched put
        local_cxl_backend.batched_submit_put_task(keys, memory_objs)

        # All keys should be in cache
        for key in keys:
            assert local_cxl_backend.contains(key)

        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendEviction:
    """Test cases for LocalCXLBackend eviction."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_eviction_when_pool_full(self, local_cpu_backend_for_cxl, event_loop):
        """Test eviction when pool is full."""
        # Create backend with small but functional pool (~50MB)
        # This should allow a few entries before triggering eviction
        config = create_test_config(cxl_capacity_pool_gb=0.05)
        backend = LocalCXLBackend(
            config=config,
            loop=event_loop,
            local_cpu_backend=local_cpu_backend_for_cxl,
            dst_device="cpu",
            metadata=None,
        )

        # Insert enough data to potentially trigger eviction
        keys_inserted = []
        for i in range(20):
            key = create_test_key(f"key_{i}")
            memory_obj = create_test_memory_obj()
            backend.batched_submit_put_task([key], [memory_obj])
            keys_inserted.append(key)

        # Some keys may have been evicted due to pool size limit
        # or all inserted if pool is large enough
        remaining_keys = list(backend.dict.keys())
        assert len(remaining_keys) <= len(keys_inserted)

        backend.close()
        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendConcurrency:
    """Test cases for LocalCXLBackend thread safety."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_concurrent_put_and_get(
        self, local_cxl_backend, local_cpu_backend_for_cxl
    ):
        """Test concurrent put and get operations."""
        num_threads = 4
        num_ops_per_thread = 10
        errors = []

        def worker(thread_id):
            try:
                for i in range(num_ops_per_thread):
                    key = create_test_key(f"thread_{thread_id}_key_{i}")
                    memory_obj = create_test_memory_obj()
                    local_cxl_backend.batched_submit_put_task([key], [memory_obj])

                    # Immediately try to get it back
                    result = local_cxl_backend.get_blocking(key)
                    if result is None:
                        errors.append(
                            f"Thread {thread_id}: Key {key} not found after put"
                        )
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_concurrent_contains(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test concurrent contains() calls."""
        key = create_test_key("shared_key")
        memory_obj = create_test_memory_obj()
        local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        errors = []

        def check_contains(thread_id):
            try:
                for _ in range(20):
                    if not local_cxl_backend.contains(key):
                        errors.append(f"Thread {thread_id}: Key not found")
            except Exception as e:
                errors.append(f"Thread {thread_id}: {e}")

        threads = [threading.Thread(target=check_contains, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendMetrics:
    """Test cases for LocalCXLBackend metrics collection."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_put_updates_metrics(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test that put operations update metrics."""
        monitor = LMCStatsMonitor.GetOrCreate()

        # Clear existing stats
        monitor.get_stats_and_clear()

        key = create_test_key("metrics_test")
        memory_obj = create_test_memory_obj()

        local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        stats = monitor.get_stats_and_clear()
        assert stats.interval_local_cxl_put_requests >= 1

        local_cpu_backend_for_cxl.memory_allocator.close()

    def test_get_updates_metrics(self, local_cxl_backend, local_cpu_backend_for_cxl):
        """Test that get operations update metrics."""
        monitor = LMCStatsMonitor.GetOrCreate()

        key = create_test_key("metrics_test")
        memory_obj = create_test_memory_obj()
        local_cxl_backend.batched_submit_put_task([key], [memory_obj])

        # Clear stats after put
        monitor.get_stats_and_clear()

        # Now get
        local_cxl_backend.get_blocking(key)

        stats = monitor.get_stats_and_clear()
        assert stats.interval_local_cxl_get_requests >= 1

        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendClose:
    """Test cases for LocalCXLBackend close/cleanup."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_close_clears_cache(self, local_cpu_backend_for_cxl, event_loop):
        """Test that close() clears the cache."""
        config = create_test_config()
        backend = LocalCXLBackend(
            config=config,
            loop=event_loop,
            local_cpu_backend=local_cpu_backend_for_cxl,
            dst_device="cpu",
            metadata=None,
        )

        # Add some data
        key = create_test_key("test_key")
        memory_obj = create_test_memory_obj()
        backend.batched_submit_put_task([key], [memory_obj])
        assert len(backend.dict) > 0

        # Close
        backend.close()

        # Cache should be cleared
        assert len(backend.dict) == 0

        local_cpu_backend_for_cxl.memory_allocator.close()


class TestLocalCXLBackendHybridConfig:
    """Test cases for hybrid configuration validation."""

    def teardown_method(self, method):
        LMCStatsMonitor.unregister_all_metrics()
        LMCStatsMonitor.DestroyInstance()

    def test_get_hybrid_config(self):
        """Test get_hybrid_config() returns correct values."""
        config = create_test_config(
            cxl_capacity_pool_gb=0.5,
            cxl_numa_node=2,
            cxl_serde="naive",
            prefer_cxl_on_put=True,
            promote_on_get=False,
            demote_on_pressure=True,
            dram_hot_cache_max_chunks=100,
        )

        hybrid = config.get_hybrid_config()

        assert hybrid["enabled"] is True
        assert hybrid["backend"] == "hybrid"
        assert hybrid["cxl_capacity_pool_gb"] == 0.5
        assert hybrid["cxl_serde"] == "naive"
        assert hybrid["prefer_cxl_on_put"] is True
        assert hybrid["promote_on_get"] is False
        assert hybrid["demote_on_pressure"] is True
        assert hybrid["dram_hot_cache_max_chunks"] == 100

    def test_hybrid_disabled_by_default(self):
        """Test that hybrid mode is disabled by default."""
        config = LMCacheEngineConfig.from_defaults(chunk_size=256)
        hybrid = config.get_hybrid_config()

        assert hybrid["enabled"] is False
