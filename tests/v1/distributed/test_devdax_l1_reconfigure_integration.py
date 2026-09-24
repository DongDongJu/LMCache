# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration test for runtime Device-DAX L1 reconfiguration
(add/remove lifecycle against live mmap-backed devices). Run with:

    RUN_DEVDAX_L1_INTEGRATION=1 pytest -xvs \
        tests/v1/distributed/test_devdax_l1_reconfigure_integration.py

Uses isolated temporary files by default (the same open/fstat/mmap path);
point at real devices (>=3) with LMCACHE_TEST_DEVDAX_L1_PATHS=/dev/dax0.0,...
Slot size defaults to 2 MiB (DAX mapping granularity); override with
LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
import gc
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxRemoveMode,
)
from tests.v1.distributed.dax_test_utils import DeviceProvider

pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        os.environ.get("RUN_DEVDAX_L1_INTEGRATION") != "1"
        and os.environ.get("LMCACHE_TEST_REQUIRE_REAL_DEVDAX") != "1",
        reason="set RUN_DEVDAX_L1_INTEGRATION=1",
    ),
]


@pytest.fixture
def devices(tmp_path: Path) -> Iterator[DeviceProvider]:
    """Yield isolated file backing or three strictly validated CXL devices."""
    yield DeviceProvider(tmp_path, "l1")


def _layout(num_bytes: int) -> MemoryLayoutDesc:
    return MemoryLayoutDesc(shapes=[torch.Size([num_bytes])], dtypes=[torch.uint8])


def _key(seed: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=seed.to_bytes(4, "big") + b"\0" * 28,
        model_name="devdax-l1-reconfig-it",
        kv_rank=0,
    )


def _open_fd_count(path: str) -> int:
    """Return how many of this process's open fds point at ``path``."""
    count = 0
    for fd in os.listdir("/proc/self/fd"):
        try:
            if os.readlink(f"/proc/self/fd/{fd}") == path:
                count += 1
        except OSError:
            continue
    return count


def test_runtime_add_and_drain_remove_lifecycle(devices: DeviceProvider) -> None:
    """Runtime capacity drains without corrupting live entries or leaking fds."""
    SLOT_BYTES = devices.slot_bytes
    primary = devices.acquire(SLOT_BYTES)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=SLOT_BYTES,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
            devdax_path=primary,
        )
    )
    try:
        # The pool starts as a single primary arena that is actually mapped.
        statuses = manager.get_arena_statuses()
        assert [status.device_path for status in statuses] == [primary]
        assert statuses[0].is_primary is True
        assert _open_fd_count(primary) > 0

        # Fill the primary arena; the mapping is live shared memory.
        error, primary_objs = manager.allocate(_layout(SLOT_BYTES), count=1)
        assert error == L1Error.SUCCESS
        assert primary_objs[0].raw_tensor is not None
        primary_objs[0].raw_tensor.fill_(0xAB)
        assert torch.all(primary_objs[0].raw_tensor == 0xAB)

        # Primary is full, so allocation fails until we add capacity.
        error, empty = manager.allocate(_layout(SLOT_BYTES), count=1)
        assert error == L1Error.OUT_OF_MEMORY
        assert empty == []

        # Add a device at runtime and confirm it is mapped and serves overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = manager.add_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        assert added.is_primary is False
        assert _open_fd_count(overflow) > 0

        error, overflow_objs = manager.allocate(_layout(SLOT_BYTES), count=2)
        assert error == L1Error.SUCCESS
        assert len(overflow_objs) == 2
        assert overflow_objs[0].raw_tensor is not None
        overflow_objs[0].raw_tensor.fill_(0xCD)
        assert torch.all(overflow_objs[0].raw_tensor == 0xCD)
        assert overflow_objs[1].raw_tensor is not None
        overflow_objs[1].raw_tensor.fill_(0xEF)

        used, total = manager.get_memory_usage()
        assert total == 3 * SLOT_BYTES
        assert used == 3 * SLOT_BYTES

        # Drain-remove the overflow arena while its allocations are still live.
        removing = manager.remove_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2

        # A draining arena is excluded from new allocations.
        error, blocked = manager.allocate(_layout(SLOT_BYTES), count=1)
        assert error == L1Error.OUT_OF_MEMORY

        assert torch.all(overflow_objs[0].raw_tensor == 0xCD)
        assert torch.all(overflow_objs[1].raw_tensor == 0xEF)

        # Freeing the arena's last allocation unmaps it automatically.
        manager.free(overflow_objs)
        del overflow_objs
        assert [status.device_path for status in manager.get_arena_statuses()] == [
            primary
        ]
        assert _open_fd_count(overflow) == 0

        # An empty added arena is unmapped immediately on removal.
        third = devices.acquire(SLOT_BYTES)
        manager.add_device(third, SLOT_BYTES)
        removed = manager.remove_device(third, DevDaxRemoveMode.DRAIN)
        assert removed.state == DevDaxArenaState.REMOVED
        assert _open_fd_count(third) == 0

        # The primary arena backs get_l1_memory_desc and cannot be removed.
        with pytest.raises(ValueError, match="primary"):
            manager.remove_device(primary)

        manager.free(primary_objs)
        del primary_objs
    finally:
        manager.close()

    # Every device is unmapped once the manager is closed.
    assert _open_fd_count(primary) == 0

    # tmpfs devices are real files, so MAP_SHARED write-through is observable on
    # media; real Device-DAX char devices do not support read()/write() syscalls.
    if devices.backing == "file":
        with open(primary, "rb") as handle:
            assert handle.read(SLOT_BYTES) == bytes([0xAB]) * SLOT_BYTES


def test_kv_cache_drain_gates_device_removal(
    devices: DeviceProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal sentinel via the KV-cache path: a device requested for removal
    stays DRAINING (and readable) while KV entries live on it, and unmaps only
    after the last one is deleted."""
    SLOT_BYTES = devices.slot_bytes
    # Retain the real constructed manager to call its public lifecycle API.
    # L1Manager intentionally does not expose its memory manager.
    constructed: list[DevDaxL1MemoryManager] = []

    def create_manager(config: L1MemoryManagerConfig) -> DevDaxL1MemoryManager:
        manager = DevDaxL1MemoryManager(config)
        constructed.append(manager)
        return manager

    monkeypatch.setattr(
        "lmcache.v1.distributed.l1_manager.DevDaxL1MemoryManager", create_manager
    )
    primary = devices.acquire(SLOT_BYTES)
    l1 = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=SLOT_BYTES,
                use_lazy=False,
                shm_name="",
                align_bytes=4096,
                devdax_path=primary,
            )
        )
    )
    memory_manager = constructed[0]
    assert isinstance(memory_manager, DevDaxL1MemoryManager)
    try:
        # KV entry A fills the primary device.
        key_a, key_b, key_c = _key(1), _key(2), _key(3)
        write = l1.reserve_write([key_a], [False], _layout(SLOT_BYTES))
        assert write[key_a][0] == L1Error.SUCCESS
        write[key_a][1].tensor.fill_(0xA1)
        assert l1.finish_write([key_a])[key_a] == L1Error.SUCCESS
        del write

        # Add a device at runtime; KV entries B and C land on it as overflow.
        overflow = devices.acquire(2 * SLOT_BYTES)
        added = memory_manager.add_device(overflow, 2 * SLOT_BYTES)
        assert added.state == DevDaxArenaState.ACTIVE
        for key, fill in ((key_b, 0xB2), (key_c, 0xC3)):
            write = l1.reserve_write([key], [False], _layout(SLOT_BYTES))
            assert write[key][0] == L1Error.SUCCESS
            write[key][1].tensor.fill_(fill)
            assert l1.finish_write([key])[key] == L1Error.SUCCESS
            del write
        gc.collect()

        # Per-device usage: the primary holds A, the added device holds B and C.
        statuses = {s.device_path: s for s in memory_manager.get_arena_statuses()}
        assert statuses[primary].used_bytes == SLOT_BYTES
        assert statuses[primary].active_allocations == 1
        assert statuses[overflow].used_bytes == 2 * SLOT_BYTES
        assert statuses[overflow].active_allocations == 2
        assert statuses[overflow].free_bytes == 0

        # Request removal while B and C are still cached: the device drains.
        removing = memory_manager.remove_device(overflow, DevDaxRemoveMode.DRAIN)
        assert removing.state == DevDaxArenaState.DRAINING
        assert removing.active_allocations == 2
        assert _open_fd_count(overflow) > 0

        # KV cached on a draining device stays readable.
        for key, fill in ((key_b, 0xB2), (key_c, 0xC3)):
            read = l1.reserve_read([key])
            assert read[key][0] == L1Error.SUCCESS
            assert torch.all(read[key][1].tensor == fill)
            assert read[key][1].get_shapes() == _layout(SLOT_BYTES).shapes
            assert read[key][1].get_dtypes() == _layout(SLOT_BYTES).dtypes
            assert l1.finish_read([key])[key] == L1Error.SUCCESS
            del read
        gc.collect()

        # Deleting one of the two entries keeps the device mapped and draining.
        assert l1.delete([key_b])[key_b] == L1Error.SUCCESS
        gc.collect()
        statuses = {s.device_path: s for s in memory_manager.get_arena_statuses()}
        assert statuses[overflow].state == DevDaxArenaState.DRAINING
        assert statuses[overflow].active_allocations == 1
        assert _open_fd_count(overflow) > 0

        # Deleting the last entry on the device unmaps it automatically.
        assert l1.delete([key_c])[key_c] == L1Error.SUCCESS
        gc.collect()
        assert [s.device_path for s in memory_manager.get_arena_statuses()] == [primary]
        assert _open_fd_count(overflow) == 0

        # KV on the remaining device is untouched by the removal.
        read = l1.reserve_read([key_a])
        assert read[key_a][0] == L1Error.SUCCESS
        assert torch.all(read[key_a][1].tensor == 0xA1)
        assert l1.finish_read([key_a])[key_a] == L1Error.SUCCESS
        del read
        assert l1.delete([key_a])[key_a] == L1Error.SUCCESS
        gc.collect()
    finally:
        l1.close()

    assert _open_fd_count(primary) == 0


@pytest.mark.parametrize("hybrid", [False, True])
def test_capacity_reuse_and_batch_rollback(
    devices: DeviceProvider, hybrid: bool
) -> None:
    """OOM rolls back partial batches; freed capacity preserves surviving payloads."""
    slot = devices.slot_bytes
    path = devices.acquire(2 * slot)
    manager = DevDaxL1MemoryManager(
        L1MemoryManagerConfig(
            size_in_bytes=slot if hybrid else 2 * slot,
            devdax_size_in_bytes=2 * slot if hybrid else 0,
            devdax_path=path,
            use_lazy=False,
            shm_name="",
            align_bytes=4096,
        )
    )
    try:
        capacity = 3 if hybrid else 2
        error, objects = manager.allocate(_layout(slot), count=capacity)
        assert error == L1Error.SUCCESS
        for index, obj in enumerate(objects):
            assert obj.raw_tensor is not None
            obj.raw_tensor.copy_(torch.arange(slot, dtype=torch.uint8) + index)
            expected = (
                L1BackendType.DRAM if hybrid and index == 0 else L1BackendType.DEVDAX
            )
            assert manager.get_backend_type(obj) == expected
        assert manager.allocate(_layout(slot), count=1) == (L1Error.OUT_OF_MEMORY, [])
        manager.free(objects[-1:])
        objects.pop()
        del obj
        before = manager.get_memory_usage()
        assert manager.allocate(_layout(slot), count=2) == (L1Error.OUT_OF_MEMORY, [])
        assert manager.get_memory_usage() == before
        error, reused = manager.allocate(_layout(slot), count=1)
        assert error == L1Error.SUCCESS
        assert reused[0].raw_tensor is not None
        reused[0].raw_tensor.fill_(0xA5)
        for index, obj in enumerate(objects):
            assert torch.equal(
                obj.raw_tensor, torch.arange(slot, dtype=torch.uint8) + index
            )
        del obj
        assert torch.all(reused[0].raw_tensor == 0xA5)
        manager.free(objects + reused)
        objects.clear()
        reused.clear()
        assert manager.get_memory_usage()[0] == 0
        with pytest.raises(ValueError, match="already|duplicate"):
            manager.add_device(path, slot)
        with pytest.raises(OSError):
            manager.add_device(str(devices.workspace / "missing"), slot)
        assert len(manager.get_arena_statuses()) == 1
    finally:
        manager.close()
    assert _open_fd_count(path) == 0
