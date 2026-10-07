# SPDX-License-Identifier: Apache-2.0
"""A failed store aborts its reservations instead of waiting for their TTL."""

# Standard
from collections.abc import Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])


def key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "abort-test", 0)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageManager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    config = L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False, shm_name="")
    )
    manager = StorageManager(StorageManagerConfig(config, EvictionConfig("noop")))
    yield manager
    manager.close()


def test_abort_releases_staging_and_lets_the_key_be_written_again(
    store: StorageManager,
) -> None:
    reserved = store.reserve_write([key(1), key(2)], LAYOUT)
    assert set(reserved) == {key(1), key(2)}
    status = store.report_status()["l1_manager"]
    assert status["staging_object_count"] == 2

    store.abort_write_by_owner(store.prepare_write_completion(reserved))

    status = store.report_status()["l1_manager"]
    assert status["staging_object_count"] == 0
    assert status["memory_used_bytes"] == 0
    # Nothing became readable, and the keys can be stored again at once.
    result = store.query_prefetch_status(
        store.submit_prefetch_task(single_row_spec([key(1)], LAYOUT), skip_l2=True)
    )
    assert result is not None and result.l1_owners == {}
    again = store.reserve_write([key(1)], LAYOUT)
    assert set(again) == {key(1)}


def test_abort_rejects_unknown_owner(store: StorageManager) -> None:
    with pytest.raises(ValueError, match="registered L1 owner"):
        store.abort_write_by_owner([(10_000, [key(1)])])
