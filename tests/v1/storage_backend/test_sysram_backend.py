# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest
import torch

pytest.importorskip("lmcache.lmcache_sysram")

# First Party
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.storage_backend.sysram_backend import SysRAMBackend


def create_test_metadata(chunk_size: int = 16) -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(2, 2, chunk_size, 2, 4),
        chunk_size=chunk_size,
        role="worker",
    )


def create_test_config(
    *,
    chunk_size: int = 16,
    max_local_cpu_size: float = 0.01,
    promote_on_get: bool = False,
    sysram_pool_size_gb: float = 0.01,
) -> LMCacheEngineConfig:
    return LMCacheEngineConfig.from_defaults(
        chunk_size=chunk_size,
        local_cpu=True,
        max_local_cpu_size=max_local_cpu_size,
        extra_config={
            "sysram_backend": {
                "enabled": True,
                "pools": [
                    {
                        "numa_node": 0,
                        "size_gb": sysram_pool_size_gb,
                    }
                ],
                "promote_on_get": promote_on_get,
            }
        },
        lmcache_instance_id="test_instance",
    )


def create_test_key(key_id: str) -> CacheEngineKey:
    return CacheEngineKey(
        model_name="test_model",
        world_size=1,
        worker_id=0,
        chunk_hash=hash(key_id),
        dtype=torch.bfloat16,
    )


@pytest.fixture(autouse=True)
def cleanup_singletons():
    yield
    PinMonitor.DestroyInstance()
    LMCStatsMonitor.unregister_all_metrics()
    LMCStatsMonitor.DestroyInstance()


class TestSysRAMBackendConfig:
    def test_get_sysram_backend_config(self):
        config = create_test_config(promote_on_get=True)

        sysram_config = config.get_sysram_backend_config()

        assert sysram_config["enabled"] is True
        assert sysram_config["promote_on_get"] is True
        assert sysram_config["pools"] == [{"numa_node": 0, "size_gb": 0.01}]

    def test_validate_requires_local_cpu(self):
        config = LMCacheEngineConfig.from_defaults(
            local_cpu=False,
            max_local_cpu_size=0.01,
            extra_config={
                "sysram_backend": {
                    "enabled": True,
                    "pools": [{"numa_node": 0, "size_gb": 0.01}],
                }
            },
        )

        with pytest.raises(ValueError, match="local_cpu=True"):
            config.validate()

    def test_validate_requires_pools(self):
        config = LMCacheEngineConfig.from_defaults(
            local_cpu=True,
            max_local_cpu_size=0.01,
            extra_config={"sysram_backend": {"enabled": True, "pools": []}},
        )

        with pytest.raises(ValueError, match="at least one pool"):
            config.validate()


class TestSysRAMBackend:
    def _create_backend(
        self,
        promote_on_get: bool = False,
    ) -> tuple[SysRAMBackend, LocalCPUBackend, LMCacheMetadata]:
        config = create_test_config(promote_on_get=promote_on_get)
        metadata = create_test_metadata(chunk_size=config.chunk_size)
        PinMonitor.GetOrCreate(config)
        local_cpu_backend = LocalCPUBackend(config=config, metadata=metadata)
        backend = SysRAMBackend(
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
        )
        return backend, local_cpu_backend, metadata

    def test_put_get_round_trip(self):
        backend, local_cpu_backend, metadata = self._create_backend()
        key = create_test_key("sysram-roundtrip")
        memory_obj = backend.allocate(
            metadata.get_shapes(metadata.chunk_size),
            metadata.get_dtypes(),
        )
        assert memory_obj is not None
        raw_tensor = memory_obj.raw_tensor
        assert raw_tensor is not None
        raw_tensor[: memory_obj.get_size()].copy_(
            torch.arange(memory_obj.get_size(), dtype=torch.uint8)
        )

        backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()

        assert backend.contains(key) is True
        assert local_cpu_backend.contains(key) is False

        retrieved = backend.get_blocking(key)
        assert retrieved is not None
        retrieved_raw = retrieved.raw_tensor
        assert retrieved_raw is not None
        torch.testing.assert_close(
            retrieved_raw[: retrieved.get_size()],
            torch.arange(retrieved.get_size(), dtype=torch.uint8),
        )
        assert local_cpu_backend.contains(key) is False

        retrieved.ref_count_down()
        backend.close()
        local_cpu_backend.close()

    def test_promote_on_get_warms_local_cpu(self):
        backend, local_cpu_backend, metadata = self._create_backend(promote_on_get=True)
        key = create_test_key("sysram-promote")
        memory_obj = backend.allocate(
            metadata.get_shapes(metadata.chunk_size),
            metadata.get_dtypes(),
        )
        assert memory_obj is not None
        raw_tensor = memory_obj.raw_tensor
        assert raw_tensor is not None
        raw_tensor[: memory_obj.get_size()].fill_(7)

        backend.submit_put_task(key, memory_obj)
        memory_obj.ref_count_down()

        retrieved = backend.get_blocking(key)
        assert retrieved is not None
        assert local_cpu_backend.contains(key) is True

        retrieved.ref_count_down()
        backend.close()
        local_cpu_backend.close()


class TestStorageManagerWithSysRAM:
    def test_overflow_spills_to_sysram_and_promotes_on_get(self):
        chunk_size = 16
        metadata = create_test_metadata(chunk_size=chunk_size)
        # MixedMemoryAllocator aligns to 4 KiB blocks, so reserve space for one chunk.
        one_chunk_gb = 4096 / 1024**3
        config = create_test_config(
            chunk_size=chunk_size,
            max_local_cpu_size=one_chunk_gb,
            promote_on_get=True,
            sysram_pool_size_gb=0.01,
        )

        PinMonitor.GetOrCreate(config)
        manager = StorageManager(
            config=config,
            metadata=metadata,
            event_manager=EventManager(),
        )
        try:
            shapes = metadata.get_shapes(chunk_size)
            dtypes = metadata.get_dtypes()
            key1 = create_test_key("hot")
            key2 = create_test_key("overflow")

            obj1 = manager.allocate(shapes, dtypes, busy_loop=False)
            obj2 = manager.allocate(shapes, dtypes, busy_loop=False)
            assert obj1 is not None
            assert obj2 is not None

            sysram_allocator = manager.storage_backends[
                "SysRAMBackend"
            ].get_memory_allocator()
            assert obj1.parent() is not sysram_allocator
            assert obj2.parent() is sysram_allocator

            raw1 = obj1.raw_tensor
            raw2 = obj2.raw_tensor
            assert raw1 is not None
            assert raw2 is not None
            raw1[: obj1.get_size()].fill_(1)
            raw2[: obj2.get_size()].fill_(2)

            manager.batched_put([key1, key2], [obj1, obj2])

            local_cpu_backend = manager.storage_backends["LocalCPUBackend"]
            sysram_backend = manager.storage_backends["SysRAMBackend"]

            assert local_cpu_backend.contains(key1) is True
            assert sysram_backend.contains(key1) is False
            assert local_cpu_backend.contains(key2) is False
            assert sysram_backend.contains(key2) is True

            retrieved = manager.get(key2)
            assert retrieved is not None
            assert local_cpu_backend.contains(key2) is True

            retrieved.ref_count_down()
        finally:
            manager.close()
