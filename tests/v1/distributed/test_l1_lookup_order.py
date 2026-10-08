# SPDX-License-Identifier: Apache-2.0
"""Lookups ask local L1s first and a remote (shared) L1 only for misses."""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    SharedL1Config,
    StorageManagerConfig,
)
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import single_row_spec
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.float32])


def key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "lookup-order", 0)


class RemoteAuthorityL1:
    """An embedded L1 whose config says it is shared, recording lookups.

    Lets the lookup-order contract be tested without an orchestrator.
    """

    def __init__(self, inner: L1Manager) -> None:
        self.inner = inner
        self.read_calls: list[list[ObjectKey]] = []

    @property
    def config(self) -> L1ManagerConfig:
        return replace(
            self.inner.config,
            shared=SharedL1Config("127.0.0.1:1", "pool", "/unused"),
        )

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        self.read_calls.append(list(keys))
        return self.inner.reserve_read(keys, read_locks)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[StorageManager]]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    created: list[StorageManager] = []
    yield created
    for store in reversed(created):
        store.close()


def _build(
    stores: list[StorageManager], remote_first: bool
) -> tuple[StorageManager, L1Manager, RemoteAuthorityL1]:
    configs = [
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False, shm_name=""),
            tag=tag,
        )
        for tag in ("_default", "remote")
    ]
    local = L1Manager(configs[0])
    remote = RemoteAuthorityL1(L1Manager(configs[1]))
    managers = (remote, local) if remote_first else (local, remote)
    store = StorageManager(
        StorageManagerConfig(configs[0], EvictionConfig("noop")),
        _l1_managers=managers,  # type: ignore[arg-type]
    )
    stores.append(store)
    return store, local, remote


@pytest.mark.parametrize("remote_first", [False, True])
def test_remote_l1_is_asked_only_for_local_misses(
    stores: list[StorageManager], remote_first: bool
) -> None:
    store, local, remote = _build(stores, remote_first)
    for manager, index in ((local, 1), (remote.inner, 2), (remote.inner, 1)):
        manager.reserve_write([key(index)], [False], LAYOUT)
        manager.finish_write([key(index)])

    result = store.query_prefetch_status(
        store.submit_prefetch_task(
            single_row_spec([key(1), key(2), key(3)], LAYOUT), skip_l2=True
        )
    )

    assert remote.read_calls == [[key(2), key(3)]]
    assert result is not None
    assert result.l1_owners == {
        key(1): local.l1_manager_id,
        key(2): remote.l1_manager_id,
    }
    # key(1) also lives in the remote L1 but no remote lock was taken for it.
    assert remote.inner.report_status()["read_locked_count"] == 1


def test_full_local_hit_never_calls_the_remote_l1(stores: list[StorageManager]) -> None:
    store, local, remote = _build(stores, remote_first=True)
    local.reserve_write([key(1)], [False], LAYOUT)
    local.finish_write([key(1)])

    result = store.query_prefetch_status(
        store.submit_prefetch_task(single_row_spec([key(1)], LAYOUT), skip_l2=True)
    )

    assert result is not None and result.l1_owners == {key(1): local.l1_manager_id}
    assert remote.read_calls == []
