# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import patch
import os
import select
import tempfile
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.raw_block_l2_adapter import (
    RawBlockL2Adapter,
    RawBlockL2AdapterConfig,
)
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L2AdapterEvictionState,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


def _has_ext() -> bool:
    try:
        # Third Party
        import lmcache_rust_raw_block_io  # noqa: F401

        return True
    except Exception:
        return False


requires_raw_block_ext = pytest.mark.skipif(
    not _has_ext(), reason="lmcache_rust_raw_block_io extension not installed"
)


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey]):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self.deleted.append(list(keys))


class _FailingListener(L2AdapterListener):
    def on_l2_keys_stored(self, keys: list[ObjectKey]):
        del keys
        raise RuntimeError("store listener failed")

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        raise RuntimeError("access listener failed")

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        del keys
        raise RuntimeError("delete listener failed")


def _create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _create_memory_obj(size: int = 1024, fill_value: float = 0.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _create_complex_memory_obj(
    size: int = 1024,
    fill_value: complex = 0j,
) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.complex64)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.complex64,
        address=0,
        phy_size=raw_data.numel() * raw_data.element_size(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def _wait_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


def _wait_for_condition(
    predicate,
    timeout: float = 5.0,
    poll_interval: float = 0.05,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return False


def _make_config(
    device_path: str,
    *,
    slot_bytes: int = 64 * 1024,
    capacity_bytes: int = 0,
    meta_total_bytes: int = 1 * 1024 * 1024,
) -> RawBlockL2AdapterConfig:
    return RawBlockL2AdapterConfig(
        device_path=device_path,
        slot_bytes=slot_bytes,
        capacity_bytes=capacity_bytes,
        use_odirect=False,
        block_align=4096,
        header_bytes=4096,
        meta_total_bytes=meta_total_bytes,
        meta_enable_periodic=False,
        num_store_workers=2,
        num_lookup_workers=1,
        num_load_workers=2,
    )


def _config_dict(**overrides) -> dict[str, object]:
    config: dict[str, object] = {
        "device_path": "/tmp/raw-block-test-device",
        "slot_bytes": 64 * 1024,
        "use_odirect": False,
    }
    config.update(overrides)
    return config


def test_raw_block_l2_adapter_config_default_io_engine():
    config = RawBlockL2AdapterConfig.from_dict(_config_dict())

    assert config.io_engine == "posix"


@pytest.mark.parametrize("io_engine", ["posix", "io_uring"])
def test_raw_block_l2_adapter_config_accepts_io_engine_values(io_engine):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine=io_engine))

    assert config.io_engine == io_engine


def test_raw_block_l2_adapter_config_rejects_invalid_io_engine():
    with pytest.raises(ValueError, match="io_engine"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(io_engine="uring"))


@pytest.mark.parametrize("legacy_key", ["use_iouring", "use_uring"])
def test_raw_block_l2_adapter_config_legacy_use_uring_maps_to_iouring(legacy_key):
    config = RawBlockL2AdapterConfig.from_dict(_config_dict(**{legacy_key: True}))

    assert config.io_engine == "io_uring"


def test_raw_block_l2_adapter_config_explicit_io_engine_wins_over_legacy_flag():
    config = RawBlockL2AdapterConfig.from_dict(
        _config_dict(io_engine="posix", use_iouring=True)
    )

    assert config.io_engine == "posix"


def test_raw_block_l2_adapter_config_validates_iouring_queue_depth():
    with pytest.raises(ValueError, match="iouring_queue_depth"):
        RawBlockL2AdapterConfig.from_dict(_config_dict(iouring_queue_depth=0))


def _run_store(adapter: RawBlockL2Adapter, keys, objects) -> bool:
    task_id = adapter.submit_store_task(keys, objects)
    assert _wait_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert task_id in completed
    return completed[task_id]


def _run_lookup(adapter: RawBlockL2Adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys)
    assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
    return task_id, adapter.query_lookup_and_lock_result(task_id)


def _run_load(adapter: RawBlockL2Adapter, keys, objects):
    task_id = adapter.submit_load_task(keys, objects)
    assert _wait_event_fd(adapter.get_load_event_fd())
    return task_id, adapter.query_load_result(task_id)


@requires_raw_block_ext
def test_raw_block_l2_adapter_store_lookup_load_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key1 = _create_object_key(1)
            key_miss = _create_object_key(2)
            key3 = _create_object_key(3)
            obj1 = _create_memory_obj(fill_value=1.0)
            obj3 = _create_memory_obj(fill_value=3.0)

            assert _run_store(adapter, [key1, key3], [obj1, obj3]) is True

            lookup_task_id, lookup_bitmap = _run_lookup(
                adapter,
                [key1, key_miss, key3],
            )
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_lookup_and_lock_result(lookup_task_id) is None

            load_buffers = [
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
                _create_memory_obj(fill_value=0.0),
            ]
            load_task_id, load_bitmap = _run_load(
                adapter,
                [key1, key_miss, key3],
                load_buffers,
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 2]
            assert adapter.query_load_result(load_task_id) is None
            assert torch.equal(load_buffers[0].tensor, obj1.tensor)
            assert torch.equal(load_buffers[2].tensor, obj3.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            adapter.submit_unlock([key1, key_miss, key3])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_delete_respects_lock_until_unlock():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(11)
            obj = _create_memory_obj(fill_value=11.0)
            assert _run_store(adapter, [key], [obj]) is True

            _, bitmap = _run_lookup(adapter, [key])
            assert bitmap is not None
            assert bitmap.get_indices_list() == [0]

            adapter.delete([key])
            _, still_present = _run_lookup(adapter, [key])
            assert still_present is not None
            assert still_present.get_indices_list() == [0]
            adapter.submit_unlock([key, key])

            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
            adapter.submit_unlock([key])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_uses_global_eviction_accounting():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        slot_bytes = 64 * 1024
        capacity_bytes = (1 * 1024 * 1024) + slot_bytes
        adapter = RawBlockL2Adapter(
            _make_config(
                dev_path,
                slot_bytes=slot_bytes,
                capacity_bytes=capacity_bytes,
            )
        )
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key1 = _create_object_key(21)
            key2 = _create_object_key(22)
            obj1 = _create_memory_obj(fill_value=21.0)
            obj2 = _create_memory_obj(fill_value=22.0)

            assert _run_store(adapter, [key1], [obj1]) is True
            assert _run_store(adapter, [key2], [obj2]) is False

            assert listener.stored == [[key1]]
            assert listener.deleted == []

            usage = adapter.get_usage()
            assert usage.total_bytes_used == obj1.get_size()
            assert usage.total_capacity_bytes == slot_bytes
            assert 0.0 < usage.usage_fraction <= 1.0
            assert adapter.supports_global_eviction is True

            status = adapter.report_status()
            assert status["is_healthy"] is True
            assert status["type"] == "RawBlockL2Adapter"
            assert status["core"]["usable_capacity_bytes"] == slot_bytes

            _, bitmap1 = _run_lookup(adapter, [key1])
            assert bitmap1 is not None
            assert bitmap1.get_indices_list() == [0]
            _, bitmap2 = _run_lookup(adapter, [key2])
            assert bitmap2 is not None
            assert bitmap2.get_indices_list() == []
            adapter.submit_unlock([key1, key2])

            adapter.delete([key1])
            assert listener.deleted[-1] == [key1]
            assert adapter.get_usage().total_bytes_used == 0

            assert _run_store(adapter, [key2], [obj2]) is True
            assert listener.stored[-1] == [key2]
            _, bitmap_after_delete = _run_lookup(adapter, [key1, key2])
            assert bitmap_after_delete is not None
            assert bitmap_after_delete.get_indices_list() == [1]
            adapter.submit_unlock([key1, key2])
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_duplicate_store_batch_counts_once():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        listener = _RecordingListener()
        adapter.register_listener(listener)

        try:
            key = _create_object_key(25)
            obj_a = _create_memory_obj(size=512, fill_value=25.0)
            obj_b = _create_memory_obj(size=1024, fill_value=26.0)

            assert _run_store(adapter, [key, key], [obj_a, obj_b]) is True
            assert _run_store(adapter, [key], [obj_b]) is True

            assert listener.stored == [[key]]
            assert adapter.get_usage().total_bytes_used == obj_a.get_size()
            assert adapter.report_status()["core"]["indexed_key_count"] == 1

            accounting = adapter.report_status()["core"]["io_accounting"]
            assert accounting["store_attempted_count"] == 3
            assert accounting["store_attempted_logical_bytes"] == (
                obj_a.get_size() + 2 * obj_b.get_size()
            )
            assert accounting["store_committed_count"] == 1
            assert accounting["store_committed_logical_bytes"] == obj_a.get_size()
            assert accounting["store_existing_hit_count"] == 2
            assert accounting["store_existing_hit_logical_bytes"] == (
                2 * obj_a.get_size()
            )
            assert accounting["media_write_logical_bytes"] == obj_a.get_size()
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_delete_subtracts_exact_metadata_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key_small = _create_object_key(261)
            key_large = _create_object_key(262)
            obj_small = _create_memory_obj(size=512, fill_value=26.1)
            obj_large = _create_memory_obj(size=1024, fill_value=26.2)
            assert _run_store(
                adapter,
                [key_small, key_large],
                [obj_small, obj_large],
            ) is True

            assert adapter.get_usage().total_bytes_used == (
                obj_small.get_size() + obj_large.get_size()
            )

            adapter.delete([key_small])
            assert adapter.get_usage().total_bytes_used == obj_large.get_size()

            adapter.delete([key_large])
            assert adapter.get_usage().total_bytes_used == 0
            assert adapter.report_status()["core"]["indexed_key_count"] == 0
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_load_hit_miss_io_accounting():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            key = _create_object_key(251)
            missing_key = _create_object_key(252)
            obj = _create_memory_obj(size=512, fill_value=25.1)
            assert _run_store(adapter, [key], [obj]) is True

            before = adapter.report_status()["core"]["io_accounting"]
            load_buffers = [
                _create_memory_obj(size=512, fill_value=0.0),
                _create_memory_obj(size=512, fill_value=0.0),
            ]
            _, loaded = _run_load(adapter, [key, missing_key], load_buffers)
            assert loaded is not None
            assert loaded.get_indices_list() == [0]
            assert torch.equal(load_buffers[0].tensor, obj.tensor)
            assert torch.count_nonzero(load_buffers[1].tensor) == 0

            after = adapter.report_status()["core"]["io_accounting"]
            assert after["load_attempted_count"] - before["load_attempted_count"] == 2
            assert after["load_index_hit_count"] - before["load_index_hit_count"] == 1
            assert (
                after["load_index_hit_logical_bytes"]
                - before["load_index_hit_logical_bytes"]
                == obj.get_size()
            )
            assert (
                after["media_read_logical_bytes"]
                - before["media_read_logical_bytes"]
                == obj.get_size()
            )
            assert adapter.get_usage().total_bytes_used == obj.get_size()
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_l1_hit_does_not_touch_raw_block_accounting():
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.storage_manager import StorageManager

    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(16 * 1024 * 1024)

        config = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=16 * 1024 * 1024,
                    use_lazy=False,
                    init_size_in_bytes=16 * 1024 * 1024,
                    align_bytes=4096,
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
            l2_adapter_config=L2AdaptersConfig(
                adapters=[
                    _make_config(
                        dev_path,
                        slot_bytes=64 * 1024,
                        meta_total_bytes=1 * 1024 * 1024,
                    )
                ],
            ),
        )
        storage_manager = StorageManager(config)
        try:
            def core_status() -> dict:
                return storage_manager.report_status()["l2_adapters"][0]["core"]

            key = _create_object_key(253)
            layout = MemoryLayoutDesc(
                shapes=[torch.Size([128])],
                dtypes=[torch.float32],
            )

            reserved = storage_manager.reserve_write([key], layout, mode="new")
            assert list(reserved) == [key]
            reserved[key].tensor.fill_(25.3)
            storage_manager.finish_write([key])

            assert _wait_for_condition(
                lambda: core_status()["indexed_key_count"] == 1
            )
            before = core_status()["io_accounting"]
            assert before["store_committed_count"] == 1
            assert before["media_write_logical_bytes"] == reserved[key].get_size()

            handle = storage_manager.submit_prefetch_task([key], layout)
            assert handle.prefetch_request_id == -1
            assert storage_manager.query_prefetch_status(handle) == 1

            after_prefetch = core_status()["io_accounting"]
            assert after_prefetch == before

            with storage_manager.read_prefetched_results([key]) as objs:
                assert objs is not None
                assert torch.equal(objs[0].tensor, reserved[key].tensor)
            storage_manager.finish_read_prefetched([key])

            after_read = core_status()["io_accounting"]
            assert after_read == before
        finally:
            storage_manager.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_listener_errors_do_not_block_eventfds():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        adapter.register_listener(_FailingListener())

        try:
            key = _create_object_key(29)
            obj = _create_memory_obj(fill_value=29.0)

            store_task_id = adapter.submit_store_task([key], [obj])
            assert _wait_event_fd(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_task_id] is True

            load_buffer = _create_memory_obj(fill_value=0.0)
            load_task_id = adapter.submit_load_task([key], [load_buffer])
            assert _wait_event_fd(adapter.get_load_event_fd())
            load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            adapter.delete([key])
            _, after_delete = _run_lookup(adapter, [key])
            assert after_delete is not None
            assert after_delete.get_indices_list() == []
        finally:
            adapter.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_from_checkpoint():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key1 = _create_object_key(31)
        key2 = _create_object_key(32)
        obj1 = _create_memory_obj(size=512, fill_value=31.0)
        obj2 = _create_memory_obj(size=1024, fill_value=32.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key1, key2], [obj1, obj2]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            expected_usage = obj1.get_size() + obj2.get_size()
            assert adapter2.get_usage().total_bytes_used == expected_usage
            assert adapter2.report_status()["core"]["indexed_key_count"] == 2

            _, lookup_bitmap = _run_lookup(adapter2, [key1, key2])
            assert lookup_bitmap is not None
            assert lookup_bitmap.get_indices_list() == [0, 1]

            load_buffer1 = _create_memory_obj(size=512, fill_value=0.0)
            load_buffer2 = _create_memory_obj(size=1024, fill_value=0.0)
            _, load_bitmap = _run_load(
                adapter2,
                [key1, key2],
                [load_buffer1, load_buffer2],
            )
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0, 1]
            assert torch.equal(load_buffer1.tensor, obj1.tensor)
            assert torch.equal(load_buffer2.tensor, obj2.tensor)
            assert adapter2.get_usage().total_bytes_used == expected_usage
            adapter2.submit_unlock([key1, key2])
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovery_seeds_usage_by_cache_salt():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(33, cache_salt="u1")
        obj = _create_memory_obj(fill_value=33.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == obj.get_size()
            assert dict(usage.bytes_by_cache_salt) == {"u1": obj.get_size()}

            adapter2.delete([key])
            usage = adapter2.get_usage()
            assert usage.total_bytes_used == 0
            assert dict(usage.bytes_by_cache_salt) == {}
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovered_keys_seed_l2_eviction_state():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(34)
        obj = _create_memory_obj(fill_value=34.0)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            state = L2AdapterEvictionState(
                adapter2,
                EvictionConfig(eviction_policy="LRU", eviction_ratio=1.0),
            )
            assert state.eviction_policy.get_eviction_candidates(1) == [key]

            actions = state.eviction_policy.get_eviction_actions(1.0)
            assert len(actions) == 1
            assert actions[0].keys == [key]
            adapter2.delete(actions[0].keys)

            assert adapter2.get_usage().total_bytes_used == 0
            assert state.eviction_policy.get_eviction_candidates(1) == []
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_recovers_unknown_checkpoint_dtype():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        config = _make_config(dev_path)
        key = _create_object_key(35)
        obj = _create_complex_memory_obj(fill_value=1 + 2j)

        adapter1 = RawBlockL2Adapter(config)
        try:
            assert _run_store(adapter1, [key], [obj]) is True
        finally:
            adapter1.close()

        adapter2 = RawBlockL2Adapter(config)
        try:
            load_buffer = _create_complex_memory_obj(fill_value=0j)
            _, load_bitmap = _run_load(adapter2, [key], [load_buffer])
            assert load_bitmap is not None
            assert load_bitmap.get_indices_list() == [0]
            assert load_buffer.metadata.dtype is torch.complex64
            assert torch.equal(load_buffer.tensor, obj.tensor)
        finally:
            adapter2.close()


@requires_raw_block_ext
def test_raw_block_l2_adapter_error_bitmaps_keep_submitted_size():
    with tempfile.TemporaryDirectory() as td:
        dev_path = os.path.join(td, "dev.bin")
        with open(dev_path, "wb") as f:
            f.truncate(8 * 1024 * 1024)

        adapter = RawBlockL2Adapter(_make_config(dev_path))
        try:
            keys = [_create_object_key(41), _create_object_key(42)]
            objects = [_create_memory_obj(), _create_memory_obj()]

            with patch.object(
                adapter, "_run_lookup_task", side_effect=RuntimeError("lookup failed")
            ):
                lookup_task_id = adapter.submit_lookup_and_lock_task(keys)
                assert _wait_event_fd(adapter.get_lookup_and_lock_event_fd())
                lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task_id)
            assert lookup_bitmap is not None
            assert str(lookup_bitmap) == "00"

            with patch.object(
                adapter, "_run_load_task", side_effect=RuntimeError("load failed")
            ):
                load_task_id = adapter.submit_load_task(keys, objects)
                assert _wait_event_fd(adapter.get_load_event_fd())
                load_bitmap = adapter.query_load_result(load_task_id)
            assert load_bitmap is not None
            assert str(load_bitmap) == "00"
        finally:
            adapter.close()
