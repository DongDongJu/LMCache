# SPDX-License-Identifier: Apache-2.0
"""
Tests for the DAX MP L2 adapter.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import select
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import (
    CapacitySnapshot,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchRequestSpec,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L2AdapterListener
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage, L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.dax_l2_adapter import (
    DaxDeviceConfig,
    DaxL2Adapter,
    DaxL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.dax_physical import (
    DaxPhysicalMode,
    DaxPhysicalState,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2ReconfigurableAdapter,
    L2ReconfigureError,
)
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import AdHocMemoryAllocator
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.mp_observability.event import Event
from lmcache.v1.mp_observability.event_bus import EventBus
from lmcache.v1.platform import consume_fd
import lmcache.v1.distributed.l2_adapters.dax_l2_adapter as dax_l2_adapter_module

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])

_DEFAULT_SHAPE = torch.Size([2, 4, 8])


class _RecordingListener(L2AdapterListener):
    def __init__(self):
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []

    def on_l2_keys_stored(self, keys: list[ObjectKey], sizes: list[int]):
        self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]):
        self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]):
        self.deleted.append(list(keys))


def create_object_key(
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


def create_memory_obj(
    *,
    shape: torch.Size = _DEFAULT_SHAPE,
    dtype: torch.dtype = torch.bfloat16,
    fill_value: float = 0,
    fmt: MemoryFormat = MemoryFormat.KV_2LTD,
) -> MemoryObj:
    allocator = AdHocMemoryAllocator(device="cpu")
    obj = allocator.allocate([shape], [dtype], fmt=fmt)
    assert obj is not None
    assert obj.tensor is not None
    obj.tensor.fill_(fill_value)
    return obj


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    consume_fd(event_fd)
    return True


def wait_for_condition(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def make_adapter(
    tmp_path,
    *,
    slot_bytes: int = 4096,
    max_slots: int = 2,
    num_store_workers: int = 1,
    num_lookup_workers: int = 1,
    num_load_workers: int = 1,
) -> tuple[DaxL2Adapter, DaxL2AdapterConfig]:
    device_path = tmp_path / "dax.bin"
    arena_bytes = slot_bytes * max_slots
    with open(device_path, "wb") as fout:
        fout.truncate(arena_bytes)

    config = DaxL2AdapterConfig(
        devices=[
            DaxDeviceConfig(
                device_path=str(device_path),
                max_dax_size_gb=arena_bytes / (1024**3),
            )
        ],
        slot_bytes=slot_bytes,
        num_store_workers=num_store_workers,
        num_lookup_workers=num_lookup_workers,
        num_load_workers=num_load_workers,
    )
    return DaxL2Adapter(config), config


def make_hotplug_adapter(tmp_path, *, slot_bytes: int = 2048) -> DaxL2Adapter:
    devices = []
    for i in range(2):
        device_path = tmp_path / f"dax_hotplug_{i}.bin"
        with open(device_path, "wb") as fout:
            fout.truncate(slot_bytes * 2)
        devices.append(
            DaxDeviceConfig(
                device_path=str(device_path),
                max_dax_size_gb=(slot_bytes * 2) / (1024**3),
            )
        )

    return DaxL2Adapter(
        DaxL2AdapterConfig(
            devices=devices,
            hotplug_enabled=True,
            slot_bytes=slot_bytes,
            num_store_workers=1,
            num_lookup_workers=1,
            num_load_workers=1,
        )
    )


def store_and_wait(adapter: DaxL2Adapter, key: ObjectKey, obj: MemoryObj) -> None:
    task_id = adapter.submit_store_task([key], [obj])
    assert wait_for_event_fd(adapter.get_store_event_fd())
    completed = adapter.pop_completed_store_tasks()
    assert completed[task_id].is_successful()


def bitmap_to_bools(bitmap: Bitmap, size: int) -> list[bool]:
    return [bitmap.test(i) for i in range(size)]


def lookup_and_wait(adapter: DaxL2Adapter, keys: list[ObjectKey]) -> list[bool]:
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
    bitmap = adapter.query_lookup_and_lock_result(task_id)
    assert bitmap is not None
    return bitmap_to_bools(bitmap, len(keys))


def load_and_wait(
    adapter: DaxL2Adapter,
    keys: list[ObjectKey],
    objs: list[MemoryObj],
) -> list[bool]:
    task_id = adapter.submit_load_task(keys, objs)
    assert wait_for_event_fd(adapter.get_load_event_fd())
    bitmap = adapter.query_load_result(task_id)
    assert bitmap is not None
    return bitmap_to_bools(bitmap, len(keys))


def test_dax_adapter_registers_and_has_distinct_eventfds(tmp_path):
    adapter, _ = make_adapter(tmp_path)
    try:
        assert "dax" in get_registered_l2_adapter_types()
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()
        assert len({store_fd, lookup_fd, load_fd}) == 3
    finally:
        adapter.close()


def test_dax_hotplug_remove_migrate_preserves_loadability(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    listener = _RecordingListener()
    adapter.register_listener(listener)
    obj = create_memory_obj(fill_value=9)
    target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(80)
        store_and_wait(adapter, key, obj)
        source_path = adapter.hotplug_status()["devices"][0]["device_path"]

        result = adapter.reconfigure(
            "remove",
            {
                "device_path": source_path,
                "mode": "migrate",
            },
        )

        assert result["state"] == "removed"
        assert result["moved_keys"] == 1
        assert result["deleted_keys"] == 0
        assert result["source_slots_freed"] == 1
        assert listener.deleted == []
        assert lookup_and_wait(adapter, [key]) == [True]
        assert load_and_wait(adapter, [key], [target]) == [True]
        assert target.tensor is not None
        assert torch.all(target.tensor == 9)
        adapter.submit_unlock([key])
    finally:
        obj.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_dax_adapter_implements_generic_reconfigure_status(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    try:
        assert isinstance(adapter, L2ReconfigurableAdapter)
        status = adapter.reconfigure("status", {})
        assert status == {
            "backend": "dax",
            "supported_operations": ["status", "add", "remove", "resize"],
            "status": adapter.hotplug_status(),
        }
    finally:
        adapter.close()


def test_dax_hotplug_drain_stops_new_writes_to_source_device(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    obj0 = create_memory_obj(fill_value=1)
    obj1 = create_memory_obj(fill_value=2)
    try:
        store_and_wait(adapter, create_object_key(81), obj0)
        devices_before = adapter.hotplug_status()["devices"]
        source_path = devices_before[0]["device_path"]
        source_live_slots = devices_before[0]["live_slot_count"]

        result = adapter.hotplug_remove_device(source_path, "drain")

        assert result["status"] == "ok"
        assert result["state"] == "draining"
        store_and_wait(adapter, create_object_key(82), obj1)
        devices = adapter.hotplug_status()["devices"]
        assert devices[0]["live_slot_count"] == source_live_slots
        assert devices[1]["live_slot_count"] == 1
    finally:
        obj0.ref_count_down()
        obj1.ref_count_down()
        adapter.close()


def test_dax_hotplug_remove_blocked_restores_active_state(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    obj = create_memory_obj(fill_value=5)
    try:
        key = create_object_key(84)
        store_and_wait(adapter, key, obj)
        source_path = adapter.hotplug_status()["devices"][0]["device_path"]
        assert lookup_and_wait(adapter, [key]) == [True]

        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_remove_device(source_path, "migrate")

        assert exc_info.value.status_code == 409
        assert adapter.hotplug_status()["devices"][0]["state"] == "active"
        adapter.submit_unlock([key])
    finally:
        obj.ref_count_down()
        adapter.close()


def test_dax_hotplug_remove_migrate_rejects_duplicate_destination_key(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    source_obj = create_memory_obj(fill_value=1)
    duplicate_obj = create_memory_obj(fill_value=9)
    target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(83)
        store_and_wait(adapter, key, source_obj)
        source_path = adapter.hotplug_status()["devices"][0]["device_path"]
        destination = adapter._devices[1]
        assert destination.core.put_many([key], [duplicate_obj]) == [True]

        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_remove_device(source_path, "migrate")

        assert exc_info.value.status_code == 409
        assert adapter.hotplug_status()["devices"][0]["state"] == "active"
        assert lookup_and_wait(adapter, [key]) == [True]
        assert load_and_wait(adapter, [key], [target]) == [True]
        assert target.tensor is not None
        assert torch.all(target.tensor == 1)
        adapter.submit_unlock([key])
    finally:
        source_obj.ref_count_down()
        duplicate_obj.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_dax_hotplug_remove_evict_notifies_logical_delete(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    listener = _RecordingListener()
    adapter.register_listener(listener)
    obj = create_memory_obj(fill_value=1)
    try:
        key = create_object_key(85)
        store_and_wait(adapter, key, obj)
        source_path = adapter.hotplug_status()["devices"][0]["device_path"]

        result = adapter.hotplug_remove_device(source_path, "evict")

        assert result["deleted_keys"] == 1
        assert result["source_slots_freed"] == 1
        assert listener.deleted == [[key]]
    finally:
        obj.ref_count_down()
        adapter.close()


def test_dax_hotplug_evict_tombstone_does_not_poison_adapter_health(tmp_path):
    """A controlled remove leaves a ``removed`` tombstone whose own
    ``is_healthy`` is ``False``; the adapter aggregate must ignore it, both
    right after the remove and after the same path is added back."""
    adapter = make_hotplug_adapter(tmp_path)
    try:
        devices = adapter.hotplug_status()["devices"]
        source_path = devices[1]["device_path"]
        source_size = devices[1]["max_dax_size_bytes"]

        result = adapter.hotplug_remove_device(source_path, "evict")
        assert result["state"] == "removed"

        after_remove = adapter.report_status()
        tombstone = after_remove["devices"][1]
        assert tombstone["state"] == "removed"
        assert tombstone["is_healthy"] is False
        assert after_remove["is_healthy"] is True
        assert after_remove["closing"] is False
        # Capacity already excluded the tombstone; health now agrees with it.
        assert after_remove["max_dax_size_bytes"] == devices[0]["max_dax_size_bytes"]

        readded = adapter.hotplug_add_device(source_path, source_size)
        assert readded["device"]["state"] == "active"
        after_add = adapter.report_status()
        assert after_add["is_healthy"] is True
        assert [d["state"] for d in after_add["devices"]] == [
            "active",
            "removed",
            "active",
        ]
    finally:
        adapter.close()


def test_dax_hotplug_add_sanitizes_mapping_errors(tmp_path):
    adapter = DaxL2Adapter(
        DaxL2AdapterConfig(
            devices=[],
            hotplug_enabled=True,
            slot_bytes=2048,
            num_store_workers=1,
            num_lookup_workers=1,
            num_load_workers=1,
        )
    )
    missing_path = str(tmp_path / "missing_dax.bin")
    try:
        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_add_device(missing_path, 2048)

        assert exc_info.value.status_code == 400
        assert exc_info.value.payload == {"error": "failed to map DAX device"}
        assert missing_path not in str(exc_info.value.payload)
    finally:
        adapter.close()


def test_dax_lookup_batches_unmapped_keys_by_device(tmp_path, monkeypatch):
    adapter = make_hotplug_adapter(tmp_path)
    objs = [create_memory_obj(fill_value=i) for i in range(4)]
    keys = [create_object_key(500 + i) for i in range(4)]
    calls: list[tuple[int, int, bool]] = []

    try:
        task_id = adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[task_id].is_successful()

        with adapter._device_lock:
            adapter._key_to_device.clear()
            for entry in adapter._devices:
                original = entry.core.exists_many

                def wrapped_exists_many(
                    lookup_keys,
                    lock=False,
                    *,
                    device_id=entry.device_id,
                    original=original,
                ):
                    calls.append((device_id, len(lookup_keys), lock))
                    return original(lookup_keys, lock=lock)

                monkeypatch.setattr(entry.core, "exists_many", wrapped_exists_many)

        assert lookup_and_wait(adapter, keys) == [True, True, True, True]

        assert [count for _, count, _ in calls] == [4, 2]
        assert all(lock for _, _, lock in calls)
    finally:
        adapter.submit_unlock(keys)
        adapter.close()


class _FakeReconfigurableAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def reconfigure_status(self) -> dict:
        return {
            "backend": "fake",
            "supported_operations": ["flip"],
            "status": {"ready": True},
        }

    def reconfigure(self, operation: str, payload: dict[str, object]) -> dict:
        self.calls.append((operation, payload))
        return {"status": "ok", "operation": operation, "payload": payload}

    def get_usage(self) -> AdapterUsage:
        """Declare no capacity; reconfiguring still reports the compartment."""
        return AdapterUsage(total_bytes_used=0, total_capacity_bytes=0)


class _SerdeLikeWrapper:
    def __init__(self, inner_adapter: _FakeReconfigurableAdapter) -> None:
        self.inner_adapter = inner_adapter


class _FakeAdapterDescriptor:
    def __init__(self, type_name: str, shared: bool = False) -> None:
        self.type_name = type_name
        # The real AdapterDescriptor always carries its config; capacity
        # reporting reads ``shared`` off it.
        self.config = SimpleNamespace(shared=shared)


class _RecordingBus:
    """Captures capacity-change events instead of publishing them."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def publish(self, event: Event) -> None:
        self.events.append(event)


def _wire_capacity_publishing(sm: StorageManager) -> None:
    """Give a bare StorageManager the state its capacity publish needs.

    Reconfiguring an adapter also announces the new topology, which reads
    the L1 config, the adapter descriptors, and the event bus.

    Args:
        sm: The partially-constructed storage manager to wire up.
    """
    sm._lifecycle_lock = threading.Lock()
    sm._event_bus = cast(EventBus, _RecordingBus())
    # Declares no L1, keeping these tests about the L2 path.
    sm._l1_config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=0, use_lazy=True)
    )
    if not hasattr(sm, "_adapter_descriptors"):
        sm._adapter_descriptors = {
            adapter_id: cast(AdapterDescriptor, _FakeAdapterDescriptor("fake"))
            for adapter_id in sm._l2_adapters
        }


def test_storage_manager_routes_generic_l2_reconfigure_to_adapter():
    sm = StorageManager.__new__(StorageManager)
    adapter = _FakeReconfigurableAdapter()
    sm._adapters_lock = threading.Lock()
    sm._l2_adapters = {0: cast(L2AdapterInterface, adapter)}
    # Reconfiguring changes capacity, so the call also announces the new
    # topology; that needs enough state to build a capacity snapshot.
    _wire_capacity_publishing(sm)

    result = sm.reconfigure_l2_adapter(0, "flip", {"enabled": True})

    assert result == {
        "status": "ok",
        "operation": "flip",
        "payload": {"enabled": True},
        "adapter_index": 0,
    }
    assert adapter.calls == [("flip", {"enabled": True})]


def test_storage_manager_finds_serde_wrapped_reconfigurable_adapter():
    sm = StorageManager.__new__(StorageManager)
    sm._adapters_lock = threading.Lock()
    sm._l2_adapters = {
        0: cast(L2AdapterInterface, _SerdeLikeWrapper(_FakeReconfigurableAdapter()))
    }
    sm._adapter_descriptors = {0: _FakeAdapterDescriptor("configured_fake")}

    status = sm.get_l2_adapter_reconfigure_status()

    assert status["enabled"] is True
    assert status["num_adapters"] == 1
    assert status["adapters"][0]["backend"] == "configured_fake"
    assert status["adapters"][0]["adapter_index"] == 0
    assert status["adapters"][0]["l2_adapter_index"] == 0


def test_dax_adapter_store_lookup_load_and_one_shot_results(tmp_path):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=4)
    listener = _RecordingListener()
    adapter.register_listener(listener)

    obj0 = create_memory_obj(fill_value=3)
    obj2 = create_memory_obj(fill_value=7)
    miss_target = create_memory_obj(fill_value=0)
    load_target0 = create_memory_obj(fill_value=0)
    load_target2 = create_memory_obj(fill_value=0)
    try:
        key0 = create_object_key(10)
        key1 = create_object_key(11)
        key2 = create_object_key(12)

        store_task = adapter.submit_store_task([key0, key2], [obj0, obj2])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert completed[store_task].is_successful()
        assert adapter.pop_completed_store_tasks() == {}
        assert listener.stored == [[key0, key2]]

        lookup_task = adapter.submit_lookup_and_lock_task(
            [key0, key1, key2], {0: _EMPTY_LAYOUT}
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        lookup_bitmap = adapter.query_lookup_and_lock_result(lookup_task)
        assert lookup_bitmap is not None
        assert bitmap_to_bools(lookup_bitmap, 3) == [True, False, True]
        assert adapter.query_lookup_and_lock_result(lookup_task) is None

        load_task = adapter.submit_load_task(
            [key0, key1, key2],
            [load_target0, miss_target, load_target2],
        )
        assert wait_for_event_fd(adapter.get_load_event_fd())
        load_bitmap = adapter.query_load_result(load_task)
        assert load_bitmap is not None
        assert bitmap_to_bools(load_bitmap, 3) == [True, False, True]
        assert adapter.query_load_result(load_task) is None

        assert load_target0.tensor is not None
        assert load_target2.tensor is not None
        assert miss_target.tensor is not None
        assert torch.all(load_target0.tensor == 3)
        assert torch.all(load_target2.tensor == 7)
        assert torch.all(miss_target.tensor == 0)
        assert listener.accessed == [[key0, key2]]
    finally:
        obj0.ref_count_down()
        obj2.ref_count_down()
        miss_target.ref_count_down()
        load_target0.ref_count_down()
        load_target2.ref_count_down()
        adapter.close()


def test_dax_adapter_unlock_refcount_and_delete_skips_locked_keys(tmp_path):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=2)
    listener = _RecordingListener()
    adapter.register_listener(listener)

    obj = create_memory_obj(fill_value=5)
    try:
        key = create_object_key(20)
        store_and_wait(adapter, key, obj)

        first_lookup = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(first_lookup) is not None

        second_lookup = adapter.submit_lookup_and_lock_task([key], {0: _EMPTY_LAYOUT})
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        assert adapter.query_lookup_and_lock_result(second_lookup) is not None

        adapter.delete([key])
        assert listener.deleted == []
        assert adapter.report_status()["locked_key_count"] == 1

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert listener.deleted == []

        adapter.submit_unlock([key])
        adapter.delete([key])
        assert listener.deleted == [[key]]
    finally:
        obj.ref_count_down()
        adapter.close()


def test_dax_adapter_usage_and_status_track_pending_free_slots(tmp_path):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=2)
    obj = create_memory_obj(fill_value=9)
    try:
        assert adapter.supports_global_eviction is True
        key = create_object_key(30)
        store_and_wait(adapter, key, obj)

        reservations, hits = adapter._core.reserve_reads([key], prefix_only=False)
        assert hits == [True]

        adapter.delete([key])
        usage = adapter.get_usage()
        assert usage.usage_fraction == pytest.approx(0.5)
        assert usage.total_bytes_used == 2048

        status = adapter.report_status()
        assert status["live_slot_count"] == 1
        assert status["borrowed_slot_count"] == 1
        assert status["locked_key_count"] == 0
        assert status["inflight_store_tasks"] == 0
        assert status["inflight_lookup_tasks"] == 0
        assert status["inflight_load_tasks"] == 0
        _, usage_after_eviction = adapter._core.usage()
        assert usage_after_eviction == pytest.approx(0.0)
        assert status["supports_restart_recovery"] is False

        adapter._core.finalize_reads(reservations, set())
        assert adapter.get_usage().usage_fraction == pytest.approx(0.0)
        assert adapter._core.usage()[1] == pytest.approx(0.0)
    finally:
        obj.ref_count_down()
        adapter.close()


def test_dax_adapter_full_arena_does_not_evict_internally(tmp_path):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=2)
    listener = _RecordingListener()
    adapter.register_listener(listener)

    obj0 = create_memory_obj(fill_value=1)
    obj1 = create_memory_obj(fill_value=2)
    obj2 = create_memory_obj(fill_value=3)
    try:
        key0 = create_object_key(31)
        key1 = create_object_key(32)
        key2 = create_object_key(33)

        store_and_wait(adapter, key0, obj0)
        store_and_wait(adapter, key1, obj1)

        task_id = adapter.submit_store_task([key2], [obj2])
        assert wait_for_event_fd(adapter.get_store_event_fd())
        completed = adapter.pop_completed_store_tasks()
        assert not completed[task_id].is_successful()
        assert listener.stored == [[key0], [key1]]
        assert adapter.get_usage().usage_fraction == pytest.approx(1.0)

        lookup_task = adapter.submit_lookup_and_lock_task(
            [key0, key1, key2], {0: _EMPTY_LAYOUT}
        )
        assert wait_for_event_fd(adapter.get_lookup_and_lock_event_fd())
        bitmap = adapter.query_lookup_and_lock_result(lookup_task)
        assert bitmap is not None
        assert bitmap_to_bools(bitmap, 3) == [True, True, False]
        adapter.submit_unlock([key0, key1])
    finally:
        obj0.ref_count_down()
        obj1.ref_count_down()
        obj2.ref_count_down()
        adapter.close()


def test_dax_adapter_usage_tracks_cache_salt_by_slot(tmp_path):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=4)

    obj0 = create_memory_obj(fill_value=1)
    obj1 = create_memory_obj(fill_value=2)
    obj2 = create_memory_obj(fill_value=3)
    try:
        alice0 = create_object_key(34, cache_salt="alice")
        bob = create_object_key(35, cache_salt="bob")
        alice1 = create_object_key(36, cache_salt="alice")

        store_and_wait(adapter, alice0, obj0)
        store_and_wait(adapter, bob, obj1)
        store_and_wait(adapter, alice1, obj2)

        usage = adapter.get_usage()
        assert usage.total_bytes_used == 2048 * 3
        assert dict(usage.bytes_by_cache_salt) == {
            "alice": 2048 * 2,
            "bob": 2048,
        }

        adapter.delete([alice0])
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 2048 * 2
        assert dict(usage.bytes_by_cache_salt) == {
            "alice": 2048,
            "bob": 2048,
        }

        adapter.delete([alice1])
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 2048
        assert dict(usage.bytes_by_cache_salt) == {"bob": 2048}
    finally:
        obj0.ref_count_down()
        obj1.ref_count_down()
        obj2.ref_count_down()
        adapter.close()


def test_dax_adapter_close_waits_for_inflight_tasks(tmp_path, monkeypatch):
    adapter, _ = make_adapter(tmp_path, slot_bytes=2048, max_slots=2)
    obj = create_memory_obj(fill_value=2)
    target = create_memory_obj(fill_value=0)
    try:
        key = create_object_key(40)
        store_and_wait(adapter, key, obj)

        load_started = threading.Event()
        allow_load = threading.Event()
        close_returned = threading.Event()
        original_load = adapter._core.load_many_into

        def _blocking_load_many_into(keys, objs):
            load_started.set()
            assert allow_load.wait(timeout=2)
            return original_load(keys, objs)

        monkeypatch.setattr(adapter._core, "load_many_into", _blocking_load_many_into)

        adapter.submit_load_task([key], [target])
        assert load_started.wait(timeout=2)

        closer = threading.Thread(
            target=lambda: (adapter.close(), close_returned.set())
        )
        closer.start()
        time.sleep(0.05)
        assert not close_returned.is_set()

        allow_load.set()
        closer.join(timeout=2)
        assert not closer.is_alive()
        assert close_returned.is_set()
    finally:
        obj.ref_count_down()
        target.ref_count_down()
        adapter.close()


def test_dax_adapter_restart_is_volatile_only(tmp_path):
    adapter, config = make_adapter(tmp_path, slot_bytes=2048, max_slots=2)
    obj = create_memory_obj(fill_value=4)
    try:
        key = create_object_key(50)
        store_and_wait(adapter, key, obj)
        adapter.close()

        reopened = DaxL2Adapter(config)
        try:
            lookup_task = reopened.submit_lookup_and_lock_task(
                [key], {0: _EMPTY_LAYOUT}
            )
            assert wait_for_event_fd(reopened.get_lookup_and_lock_event_fd())
            bitmap = reopened.query_lookup_and_lock_result(lookup_task)
            assert bitmap is not None
            assert bitmap_to_bools(bitmap, 1) == [False]
        finally:
            reopened.close()
    finally:
        obj.ref_count_down()
        adapter.close()


def test_storage_manager_dax_adapter_roundtrip(tmp_path):
    shape = torch.Size([2, 4, 8])
    dtype = torch.bfloat16
    slot_bytes = shape.numel() * dtype.itemsize

    device_path = tmp_path / "sm_dax.bin"
    with open(device_path, "wb") as fout:
        fout.truncate(slot_bytes * 4)

    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20,
                use_lazy=False,
                init_size_in_bytes=1 << 20,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=60,
            read_ttl_seconds=60,
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig(
            [
                DaxL2AdapterConfig(
                    devices=[
                        DaxDeviceConfig(
                            device_path=str(device_path),
                            max_dax_size_gb=(slot_bytes * 4) / (1024**3),
                        )
                    ],
                    slot_bytes=slot_bytes,
                )
            ]
        ),
        store_policy="skip_l1",
    )

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    key = create_object_key(60)
    sm = StorageManager(storage_config)

    try:
        adapter = sm._l2_adapters[0]
        assert isinstance(adapter, DaxL2Adapter)

        reserved = sm.reserve_write([key], layout, mode="new")
        assert key in reserved
        assert reserved[key].tensor is not None
        reserved[key].tensor.fill_(11)
        sm.finish_write([key])

        assert wait_for_condition(
            lambda: (
                sm.report_status()["l1_manager"]["total_object_count"] == 0
                and adapter.get_usage().usage_fraction > 0
            ),
            timeout=5.0,
        )

        handle = sm.submit_prefetch_task(PrefetchRequestSpec([key], {0: layout}))
        assert wait_for_condition(
            lambda: sm.query_prefetch_lookup_hits(handle) is not None,
            timeout=5.0,
        )
        lookup_hits = sm.query_prefetch_lookup_hits(handle)
        assert lookup_hits == 1

        final_result: dict[str, int | None] = {"value": None}

        def _capture_prefetch_result() -> bool:
            result = sm.query_prefetch_status(handle)
            if result is None:
                return False
            final_result["value"] = result.count_leading_ones()
            return True

        assert wait_for_condition(_capture_prefetch_result, timeout=5.0)
        final_hits = final_result["value"]
        assert final_hits == 1

        with sm.read_prefetched_results([key]) as results:
            assert results is not None
            assert len(results) == 1
            assert results[0].tensor is not None
            assert torch.all(results[0].tensor == 11)

        assert adapter.report_status()["locked_key_count"] == 0
        assert sm._l1_manager.delete([key])[key] == L1Error.KEY_IS_LOCKED

        sm.finish_read_prefetched([key])
        assert sm._l1_manager.delete([key])[key] == L1Error.KEY_NOT_EXIST
    finally:
        sm.close()


def test_storage_manager_dax_adapter_uses_global_l2_eviction(tmp_path):
    shape = torch.Size([2, 4, 8])
    dtype = torch.bfloat16
    slot_bytes = shape.numel() * dtype.itemsize

    device_path = tmp_path / "sm_dax_eviction.bin"
    with open(device_path, "wb") as fout:
        fout.truncate(slot_bytes * 2)

    dax_config = DaxL2AdapterConfig(
        devices=[
            DaxDeviceConfig(
                device_path=str(device_path),
                max_dax_size_gb=(slot_bytes * 2) / (1024**3),
            )
        ],
        slot_bytes=slot_bytes,
    )
    dax_config.eviction_config = EvictionConfig(
        eviction_policy="LRU",
        trigger_watermark=0.5,
        eviction_ratio=0.5,
    )

    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20,
                use_lazy=False,
                init_size_in_bytes=1 << 20,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=60,
            read_ttl_seconds=60,
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig([dax_config]),
        store_policy="skip_l1",
    )

    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    sm = StorageManager(storage_config)

    try:
        sm._l2_eviction_controller.stop()
        adapter = sm._l2_adapters[0]
        assert isinstance(adapter, DaxL2Adapter)
        assert adapter.supports_global_eviction is True
        assert len(sm._l2_eviction_controller._adapter_states) == 1

        key0 = create_object_key(70)
        key1 = create_object_key(71)
        key2 = create_object_key(72)

        def _write_key(key: ObjectKey, fill_value: int, usage_fraction: float) -> None:
            reserved = sm.reserve_write([key], layout, mode="new")
            assert key in reserved
            assert reserved[key].tensor is not None
            reserved[key].tensor.fill_(fill_value)
            sm.finish_write([key])
            assert wait_for_condition(
                lambda: (
                    adapter.get_usage().usage_fraction == pytest.approx(usage_fraction)
                ),
                timeout=5.0,
            )

        _write_key(key0, 1, 0.5)
        _write_key(key1, 2, 1.0)
        assert adapter.get_usage().usage_fraction == pytest.approx(1.0)

        eviction_state = sm._l2_eviction_controller._adapter_states[0]
        sm._l2_eviction_controller._check_and_evict(eviction_state)
        assert adapter.get_usage().usage_fraction == pytest.approx(0.5)

        hits = adapter._core.exists_many([key0, key1], lock=True)
        assert hits == [False, True]
        adapter.submit_unlock([key1])

        _write_key(key2, 3, 1.0)
        assert adapter.get_usage().usage_fraction == pytest.approx(1.0)
    finally:
        sm.close()


def test_dax_adapter_from_dict_starts_with_zero_devices(tmp_path):
    """The hotplug-only form: no device at startup, capacity arrives via add."""
    slot_bytes = 2048
    config = DaxL2AdapterConfig.from_dict(
        {"slot_bytes": slot_bytes, "hotplug_enabled": True, "devices": []}
    )
    assert config.devices == []
    assert config.device_path == ""
    adapter = DaxL2Adapter(config)
    try:
        status = adapter.report_status()
        assert status["is_healthy"] is True
        assert status["hotplug_enabled"] is True
        assert status["num_devices"] == 0
        assert status["max_dax_size_bytes"] == 0
        assert status["device_path"] == ""
        hotplug = adapter.hotplug_status()
        assert hotplug["devices"] == []
        assert hotplug["total_capacity_bytes"] == 0
        assert adapter.get_usage().total_capacity_bytes == 0

        device_path = tmp_path / "late_dax.bin"
        with open(device_path, "wb") as fout:
            fout.truncate(slot_bytes * 4)
        added = adapter.hotplug_add_device(str(device_path), slot_bytes * 4)
        assert added["status"] == "ok"
        assert added["device"]["state"] == "active"
        assert added["device"]["index"] == 0

        hotplug = adapter.hotplug_status()
        assert [d["state"] for d in hotplug["devices"]] == ["active"]
        assert hotplug["total_capacity_bytes"] == slot_bytes * 4
        assert adapter.report_status()["max_dax_size_bytes"] == slot_bytes * 4
    finally:
        adapter.close()


def test_dax_adapter_from_dict_rejects_zero_devices_without_hotplug():
    with pytest.raises(ValueError, match="hotplug_enabled"):
        DaxL2AdapterConfig.from_dict(
            {"slot_bytes": 2048, "hotplug_enabled": False, "devices": []}
        )


def test_storage_manager_starts_dax_adapter_with_zero_devices(tmp_path):
    """A DAX adapter with no devices is created, reconfigurable, and declares
    a zero-capacity ``l2/dax`` compartment until the first device is added."""
    slot_bytes = 2048
    storage_config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20,
                use_lazy=False,
                init_size_in_bytes=1 << 20,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=60,
            read_ttl_seconds=60,
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig(
            [
                DaxL2AdapterConfig(
                    devices=[],
                    hotplug_enabled=True,
                    slot_bytes=slot_bytes,
                )
            ]
        ),
    )
    sm = StorageManager(storage_config)
    bus = _RecordingBus()
    sm._event_bus = cast(EventBus, bus)
    try:
        status = sm.report_status()
        assert status["is_healthy"] is True
        assert status["num_l2_adapters"] == 1
        dax = status["l2_adapters"][0]
        assert dax["type"] == "dax"
        assert dax["is_healthy"] is True
        assert dax["hotplug_enabled"] is True
        assert dax["num_devices"] == 0
        assert dax["max_dax_size_bytes"] == 0

        reconf = sm.get_l2_adapter_reconfigure_status()
        assert reconf["enabled"] is True
        assert reconf["num_adapters"] == 1
        assert reconf["adapters"][0]["status"]["devices"] == []
        assert reconf["adapters"][0]["status"]["hotplug_enabled"] is True

        sm.publish_capacity()
        declared = _dax_capacity(bus.events[-1])
        assert declared == 0

        device_path = tmp_path / "late_dax.bin"
        with open(device_path, "wb") as fout:
            fout.truncate(slot_bytes * 4)
        result = sm.reconfigure_l2_adapter(
            0, "add", {"device_path": str(device_path), "size_bytes": slot_bytes * 4}
        )
        assert result["status"] == "ok"
        assert result["device"]["state"] == "active"
        assert result["adapter_index"] == 0

        reconf = sm.get_l2_adapter_reconfigure_status()
        devices = reconf["adapters"][0]["status"]["devices"]
        assert [(d["device_path"], d["state"]) for d in devices] == [
            (str(device_path), "active")
        ]
        assert sm.report_status()["l2_adapters"][0]["max_dax_size_bytes"] == (
            slot_bytes * 4
        )
        assert _dax_capacity(bus.events[-1]) == slot_bytes * 4
    finally:
        sm.close()


def _dax_capacity(event: Event) -> int:
    """Return the ``l2/dax`` capacity declared by one capacity-changed event."""
    snapshot: CapacitySnapshot = event.metadata["snapshot"]
    modules = [
        m for m in snapshot.modules if m.tier.name == "L2" and m.backend == "dax"
    ]
    assert len(modules) == 1, snapshot.modules
    return int(modules[0].capacity_bytes)


# ---------------------------------------------------------------------------
# Presence watcher, read-only physical probe, and the hotplug add gate.
# ---------------------------------------------------------------------------


def _watch_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("dax-l2-watch")]


def _fake_probe(
    mode: DaxPhysicalMode, *, size_bytes: int = 0, probed_at: float = 1.0
) -> Callable[..., DaxPhysicalState]:
    """Return a ``probe_device`` stand-in reporting ``mode`` for any path."""

    def _probe(path: str, **kwargs: object) -> DaxPhysicalState:
        return DaxPhysicalState(
            device_path=path,
            mode=mode,
            present=mode is not DaxPhysicalMode.ABSENT,
            major=249,
            minor=2,
            kernel_name="dax2.1",
            driver="device_dax" if mode is DaxPhysicalMode.DEVDAX else "",
            size_bytes=size_bytes,
            align_bytes=2097152,
            probed_at=probed_at,
            detail="",
        )

    return _probe


def make_watching_adapter(
    tmp_path: Path, *, interval_seconds: float = 0.02
) -> tuple[DaxL2Adapter, Path]:
    """Create a hotplug-only adapter watching ``tmp_path / "devs"``."""
    directory = tmp_path / "devs"
    directory.mkdir()
    config = DaxL2AdapterConfig.from_dict(
        {
            "slot_bytes": 2048,
            "hotplug_enabled": True,
            "devices": [],
            "watch_directory": str(directory),
            "watch_interval_seconds": interval_seconds,
        }
    )
    return DaxL2Adapter(config), directory


def test_dax_config_from_dict_parses_watch_fields():
    config = DaxL2AdapterConfig.from_dict(
        {
            "slot_bytes": 2048,
            "hotplug_enabled": True,
            "devices": [],
            "watch_directory": " /dev/dax-cxl/ns_pod ",
            "watch_interval_seconds": 2,
        }
    )
    assert config.watch_directory == "/dev/dax-cxl/ns_pod"
    assert config.watch_interval_seconds == 2.0
    assert "watch_directory" in DaxL2AdapterConfig.help()
    assert "watch_interval_seconds" in DaxL2AdapterConfig.help()


def test_dax_config_watch_defaults_to_disabled():
    config = DaxL2AdapterConfig.from_dict(
        {"slot_bytes": 2048, "hotplug_enabled": True, "devices": []}
    )
    assert config.watch_directory == ""
    assert config.watch_interval_seconds == 1.0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"watch_directory": 5}, "watch_directory must be a string"),
        ({"watch_directory": "relative/dir"}, "absolute"),
        ({"watch_interval_seconds": 0}, "watch_interval_seconds"),
        ({"watch_interval_seconds": -0.5}, "watch_interval_seconds"),
        ({"watch_interval_seconds": "1"}, "watch_interval_seconds"),
        ({"watch_interval_seconds": float("nan")}, "positive finite"),
        ({"watch_interval_seconds": float("inf")}, "positive finite"),
    ],
)
def test_dax_config_from_dict_rejects_invalid_watch_fields(overrides, match):
    payload = {"slot_bytes": 2048, "hotplug_enabled": True, "devices": []}
    payload.update(overrides)
    with pytest.raises(ValueError, match=match):
        DaxL2AdapterConfig.from_dict(payload)


def test_dax_config_constructor_validates_watch_fields():
    with pytest.raises(ValueError, match="absolute"):
        DaxL2AdapterConfig(
            devices=[], hotplug_enabled=True, slot_bytes=2048, watch_directory="x"
        )
    with pytest.raises(ValueError, match="watch_interval_seconds"):
        DaxL2AdapterConfig(
            devices=[],
            hotplug_enabled=True,
            slot_bytes=2048,
            watch_directory="/dev/dax-cxl/ns_pod",
            watch_interval_seconds=0.0,
        )
    for non_finite in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive finite"):
            DaxL2AdapterConfig(
                devices=[],
                hotplug_enabled=True,
                slot_bytes=2048,
                watch_directory="/dev/dax-cxl/ns_pod",
                watch_interval_seconds=non_finite,
            )


def test_dax_watcher_disabled_reports_enabled_false_and_no_thread(tmp_path):
    adapter = make_hotplug_adapter(tmp_path)
    try:
        assert adapter.hotplug_status()["watcher"] == {"enabled": False}
        assert adapter.reconfigure("status", {})["status"]["watcher"] == {
            "enabled": False
        }
        assert _watch_threads() == []
    finally:
        adapter.close()


def test_dax_watcher_status_lists_directory_and_tracks_new_files(tmp_path):
    directory = tmp_path / "devs"
    directory.mkdir()
    first = directory / "a.bin"
    first.write_bytes(b"\0" * 4096)
    adapter = DaxL2Adapter(
        DaxL2AdapterConfig.from_dict(
            {
                "slot_bytes": 2048,
                "hotplug_enabled": True,
                "devices": [],
                "watch_directory": str(directory),
                "watch_interval_seconds": 0.02,
            }
        )
    )
    try:

        def _present_paths() -> list[str]:
            watcher = adapter.hotplug_status()["watcher"]
            return [d["device_path"] for d in watcher["present_devices"]]

        assert wait_for_condition(lambda: _present_paths() == [str(first)])
        watcher = adapter.hotplug_status()["watcher"]
        assert watcher["enabled"] is True
        assert watcher["directory"] == str(directory)
        assert watcher["interval_seconds"] == 0.02
        assert watcher["last_scan_at"] > 0
        present = watcher["present_devices"][0]
        assert present["mode"] == "not-a-device"
        assert present["present"] is True
        assert present["size_bytes"] == 4096

        second = directory / "b.bin"
        second.write_bytes(b"\0" * 2048)
        assert wait_for_condition(lambda: _present_paths() == [str(first), str(second)])
        # Presence is reported only: nothing was attached.
        assert adapter.hotplug_status()["devices"] == []
        assert adapter.report_status()["num_devices"] == 0
    finally:
        adapter.close()


def test_dax_watcher_thread_stops_on_close(tmp_path):
    adapter, _ = make_watching_adapter(tmp_path)
    try:
        assert wait_for_condition(lambda: len(_watch_threads()) == 1)
    finally:
        adapter.close()
    assert wait_for_condition(lambda: _watch_threads() == [])
    assert adapter.hotplug_status()["watcher"]["enabled"] is True


def test_dax_status_and_add_response_carry_physical(tmp_path):
    adapter = make_hotplug_adapter(tmp_path, slot_bytes=2048)
    try:
        devices = adapter.hotplug_status()["devices"]
        assert len(devices) == 2
        for device in devices:
            physical = device["physical"]
            assert physical["device_path"] == device["device_path"]
            assert physical["mode"] == "not-a-device"
            assert physical["present"] is True
            assert physical["size_bytes"] == 2048 * 2
            assert physical["probed_at"] > 0

        reported = adapter.report_status()["devices"]
        assert [d["physical"]["mode"] for d in reported] == ["not-a-device"] * 2

        new_path = tmp_path / "late.bin"
        with open(new_path, "wb") as fout:
            fout.truncate(2048 * 2)
        added = adapter.hotplug_add_device(str(new_path), 2048 * 2)
        assert added["device"]["physical"]["mode"] == "not-a-device"
        assert added["device"]["physical"]["device_path"] == str(new_path)
        assert added["device"]["physical"]["size_bytes"] == 2048 * 2
    finally:
        adapter.close()


def test_dax_watcher_refreshes_attached_device_physical_from_snapshot(tmp_path):
    adapter, directory = make_watching_adapter(tmp_path)
    node = directory / "dax0.1"
    with open(node, "wb") as fout:
        fout.truncate(4096)
    try:
        added = adapter.hotplug_add_device(str(node), 4096)
        assert added["device"]["physical"]["size_bytes"] == 4096
        first_probe = added["device"]["physical"]["probed_at"]

        with open(node, "r+b") as fout:
            fout.truncate(8192)

        def _status_size() -> int:
            return adapter.hotplug_status()["devices"][0]["physical"]["size_bytes"]

        assert wait_for_condition(lambda: _status_size() == 8192)
        physical = adapter.hotplug_status()["devices"][0]["physical"]
        assert physical["probed_at"] >= first_probe
        assert physical["device_path"] == str(node)
        # The mapping itself is unchanged; only the probe block moved.
        assert adapter.hotplug_status()["devices"][0]["max_dax_size_bytes"] == 4096
    finally:
        adapter.close()


def test_dax_status_keeps_newer_attach_probe_over_older_scan(tmp_path, monkeypatch):
    """A watcher scan older than the attach-time probe never overwrites it."""
    adapter, directory = make_watching_adapter(tmp_path)
    node = directory / "dax0.1"
    with open(node, "wb") as fout:
        fout.truncate(4096)
    # Far newer than any real scan the watcher can take during this test.
    attach_probed_at = time.time() + 3600.0
    try:
        monkeypatch.setattr(
            dax_l2_adapter_module,
            "probe_device",
            _fake_probe(
                DaxPhysicalMode.NOT_A_DEVICE,
                size_bytes=12345,
                probed_at=attach_probed_at,
            ),
        )
        added = adapter.hotplug_add_device(str(node), 4096)
        assert added["device"]["physical"]["probed_at"] == attach_probed_at
        assert added["device"]["physical"]["size_bytes"] == 12345
        added_at = time.time()

        # Wait for a scan that started after the add so the snapshot lists
        # the node with a real (older) probe; it must not win.
        assert wait_for_condition(
            lambda: adapter.hotplug_status()["watcher"]["last_scan_at"] >= added_at
        )
        watched = adapter.hotplug_status()["watcher"]["present_devices"]
        assert [d["device_path"] for d in watched] == [str(node)]
        assert 0 < watched[0]["probed_at"] < attach_probed_at
        physical = adapter.hotplug_status()["devices"][0]["physical"]
        assert physical["probed_at"] == attach_probed_at
        assert physical["size_bytes"] == 12345
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("mode", "size_bytes", "message"),
    [
        (DaxPhysicalMode.SYSTEM_RAM, 0, "DAX device is bound to kmem (system-ram)"),
        (DaxPhysicalMode.UNBOUND, 0, "DAX device has no device_dax driver bound"),
        (DaxPhysicalMode.DEVDAX, 2048, "DAX device size 2048 < requested 4096"),
    ],
)
def test_dax_hotplug_add_gate_rejects_bad_physical_signal(
    tmp_path, monkeypatch, mode, size_bytes, message
):
    adapter = make_hotplug_adapter(tmp_path, slot_bytes=2048)
    new_path = tmp_path / "gated.bin"
    with open(new_path, "wb") as fout:
        fout.truncate(4096)
    try:
        before = adapter.hotplug_status()
        monkeypatch.setattr(
            dax_l2_adapter_module,
            "probe_device",
            _fake_probe(mode, size_bytes=size_bytes),
        )

        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_add_device(str(new_path), 4096)

        assert exc_info.value.status_code == 400
        assert exc_info.value.payload == {"error": message}
        assert str(new_path) not in str(exc_info.value.payload)
        after = adapter.hotplug_status()
        assert len(after["devices"]) == len(before["devices"]) == 2
        assert after["total_capacity_bytes"] == before["total_capacity_bytes"]
        assert str(new_path) not in [d["device_path"] for d in after["devices"]]
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("mode", "size_bytes"),
    [
        (DaxPhysicalMode.NOT_A_DEVICE, 0),
        (DaxPhysicalMode.ABSENT, 0),
        (DaxPhysicalMode.UNKNOWN, 0),
        (DaxPhysicalMode.DEVDAX, 0),
        (DaxPhysicalMode.DEVDAX, 4096),
        (DaxPhysicalMode.DEVDAX, 1 << 40),
    ],
)
def test_dax_hotplug_add_proceeds_on_inconclusive_probe(
    tmp_path, monkeypatch, mode, size_bytes
):
    adapter = make_hotplug_adapter(tmp_path, slot_bytes=2048)
    new_path = tmp_path / "open.bin"
    with open(new_path, "wb") as fout:
        fout.truncate(4096)
    try:
        monkeypatch.setattr(
            dax_l2_adapter_module,
            "probe_device",
            _fake_probe(mode, size_bytes=size_bytes),
        )

        added = adapter.hotplug_add_device(str(new_path), 4096)

        assert added["status"] == "ok"
        assert added["device"]["state"] == "active"
        assert added["device"]["physical"]["mode"] == mode.value
        assert added["device"]["physical"]["size_bytes"] == size_bytes
        assert len(adapter.hotplug_status()["devices"]) == 3
    finally:
        adapter.close()


def test_dax_hotplug_add_gate_applies_to_active_path_re_add(tmp_path, monkeypatch):
    """The gate runs before the already-active check.

    A healthy same-size re-add stays idempotent (``200`` with the existing
    entry); one whose probe now reports kmem is rejected with ``400`` and the
    existing entry is left untouched.
    """
    adapter = make_hotplug_adapter(tmp_path, slot_bytes=2048)
    try:
        before = adapter.hotplug_status()
        device = before["devices"][0]
        path = device["device_path"]
        size = device["max_dax_size_bytes"]

        re_added = adapter.hotplug_add_device(path, size)
        assert re_added["status"] == "ok"
        assert re_added["device"]["device_id"] == device["device_id"]
        assert len(adapter.hotplug_status()["devices"]) == 2

        monkeypatch.setattr(
            dax_l2_adapter_module,
            "probe_device",
            _fake_probe(DaxPhysicalMode.SYSTEM_RAM),
        )
        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_add_device(path, size)

        assert exc_info.value.status_code == 400
        assert exc_info.value.payload == {
            "error": "DAX device is bound to kmem (system-ram)"
        }
        after = adapter.hotplug_status()
        assert [(d["device_id"], d["state"]) for d in after["devices"]] == [
            (d["device_id"], d["state"]) for d in before["devices"]
        ]
        assert after["total_capacity_bytes"] == before["total_capacity_bytes"]
        assert after["devices"][0]["physical"]["mode"] == "not-a-device"
    finally:
        adapter.close()


def test_dax_hotplug_add_missing_path_still_fails_in_mapping(tmp_path):
    """ABSENT is inconclusive for the gate; the add fails in mmap as before."""
    adapter = make_hotplug_adapter(tmp_path, slot_bytes=2048)
    missing = tmp_path / "missing.bin"
    try:
        with pytest.raises(L2ReconfigureError) as exc_info:
            adapter.hotplug_add_device(str(missing), 4096)
        assert exc_info.value.status_code == 400
        assert exc_info.value.payload == {"error": "failed to map DAX device"}
        assert len(adapter.hotplug_status()["devices"]) == 2
    finally:
        adapter.close()
