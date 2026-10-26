# SPDX-License-Identifier: Apache-2.0
"""A shared Device-DAX L1 inside StorageManager: placement, owner routing,
backing ownership and L2 affinity."""

# Standard
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
    PrefetchResult,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_orchestrator.api import RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.distributed.shared_l1.utils import CAPACITY, key, shared_config
from tests.v1.distributed.utils import single_row_spec
from tests.v1.memory_orchestrator.harness import Orchestrator

pytestmark = pytest.mark.no_shared_allocator
StoreFactory = Callable[[str], StorageManager]
Usage = Callable[[], RegionUsage]
LAYOUT = MemoryLayoutDesc([torch.Size([1000])], [torch.uint8])


def prefetch(store: StorageManager, spec: PrefetchTaskSpec) -> PrefetchResult:
    """Submit a prefetch and wait for its result."""
    handle = store.submit_prefetch_task(spec)
    assert store.wait_prefetch_status(handle, timeout=10)
    result = store.query_prefetch_status(handle)
    assert result is not None
    return result


@pytest.mark.parametrize("shared_first", [False, True])
def test_shared_region_cannot_overlap_a_private_arena(
    orchestrator: Orchestrator, region: Path, shared_first: bool
) -> None:
    shared = shared_config(region, orchestrator.endpoint, "owner")
    private = DevDaxL1ManagerConfig(
        L1MemoryManagerConfig(
            CAPACITY, use_lazy=False, shm_name="", devdax_path=str(region)
        ),
        tag="private",
        eviction=EvictionConfig("noop"),
    )
    configs: list[L1ManagerConfig] = (
        [shared, private] if shared_first else [private, shared]
    )
    with pytest.raises(ValueError, match="already owned by an L1"):
        StorageManager(
            StorageManagerConfig(
                configs[0], EvictionConfig("noop"), l1_manager_configs=configs
            )
        )


def store_keys(
    store: StorageManager, keys: list[ObjectKey], layout: MemoryLayoutDesc
) -> None:
    for value, k in enumerate(keys, 1):
        reserved = store.reserve_write([k], layout)
        tensor = reserved[k].tensor
        assert tensor is not None
        tensor.fill_(value)
        store.finish_write_by_owner(store.prepare_write_completion(reserved))


@pytest.mark.parametrize("capacity", [2 * 4096])
def test_full_region_overflows_to_the_private_l1(
    make_store: StoreFactory, usage: Usage
) -> None:
    writer, peer = make_store("writer"), make_store("peer")
    # An aborted store leaves the key writable; its extent stays consumed.
    aborted = writer.reserve_write([key(1)], LAYOUT)
    writer.abort_write_by_owner(writer.prepare_write_completion(aborted))
    assert usage().writing == 0
    keys = [key(1), key(2)]
    # key(2) finds the region full and overflows to private DRAM.
    store_keys(writer, keys, LAYOUT)
    own = prefetch(writer, single_row_spec(keys + [key(3)], LAYOUT))
    remote = prefetch(peer, single_row_spec(keys + [key(3)], LAYOUT))
    assert own.hit_cells[0].get_indices_list() == [0, 1]
    assert remote.hit_cells[0].get_indices_list() == [0]
    assert own.l1_owners[key(1)] != own.l1_owners[key(2)]
    with writer.read_prefetched_results(keys, own.l1_owners) as objects:
        assert objects is not None
        assert all(torch.all(o.tensor == i) for i, o in enumerate(objects, 1))
    writer.finish_read_prefetched(keys, l1_owners=own.l1_owners)
    peer.finish_read_prefetched(keys[:1], l1_owners=remote.l1_owners)
    assert usage().read_leases == 0


@pytest.mark.parametrize("capacity", [2 * 4096])
def test_grouped_rows_split_between_private_and_shared_l1s(
    make_store: StoreFactory, usage: Usage
) -> None:
    other = MemoryLayoutDesc([torch.Size([500])], [torch.float16])
    rows = [
        GroupedObjectKeys(
            [key(10 * group + chunk) for chunk in range(2)], group, layout
        )
        for group, layout in enumerate((LAYOUT, other))
    ]
    writer = make_store("writer")
    # Group 1 fills the region; group 0 overflows to private DRAM.
    store_keys(writer, rows[1].keys, other)
    store_keys(writer, rows[0].keys, LAYOUT)
    result = prefetch(writer, PrefetchTaskSpec(rows))
    assert [b.get_indices_list() for b in result.hit_cells] == [[0, 1]] * 2
    owners = {result.l1_owners[k] for row in rows for k in row.keys}
    assert len(owners) == 2
    writer.finish_read_prefetched(
        [k for row in rows for k in row.keys], l1_owners=result.l1_owners
    )
    # A shared layout that is not the reader's releases every lock taken, the
    # private ones included.
    wrong = GroupedObjectKeys(rows[1].keys, 1, LAYOUT)
    with pytest.raises(ValueError, match="reader's layout"):
        writer.submit_prefetch_task(PrefetchTaskSpec([rows[0], wrong]))
    assert usage().read_leases == 0
    status = writer.report_status()["l1_managers"]["private"]
    assert status["read_locked_count"] == 0


def test_l2_adapter_cannot_bind_to_the_shared_l1(
    orchestrator: Orchestrator, region: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)
    monkeypatch.setattr(
        MockL2AdapterConfig, "get_wanted_l1_memory_tags", lambda self: (remote,)
    )
    shared = shared_config(region, orchestrator.endpoint, "owner")
    with pytest.raises(ValueError, match="bind only to private L1s"):
        StorageManager(
            StorageManagerConfig(
                shared,
                EvictionConfig("noop"),
                l2_adapter_config=L2AdaptersConfig([MockL2AdapterConfig(1, 1)]),
                l1_manager_configs=[shared],
            )
        )


def test_tag_filters_and_private_affinity_next_to_a_shared_l1(
    orchestrator: Orchestrator, region: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = shared_config(region, orchestrator.endpoint, "owner")
    private = L1ManagerConfig(
        L1MemoryManagerConfig(8192, use_lazy=False, shm_name=""), tag="_default"
    )
    # A default adapter wants DRAM/LOCAL first and binds to the private L1.
    store = StorageManager(
        StorageManagerConfig(
            shared,
            EvictionConfig("noop"),
            l2_adapter_config=L2AdaptersConfig([MockL2AdapterConfig(1, 1)]),
            l1_manager_configs=[shared, private],
        )
    )
    try:
        reserved = store.reserve_write([key(1)], LAYOUT)
        store.finish_write_by_owner(store.prepare_write_completion(reserved))
        remote = MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)
        local = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
        for allowed in (None, frozenset({remote})):
            result = prefetch(
                store, single_row_spec([key(1)], LAYOUT, allowed_l1_memory_tags=allowed)
            )
            assert result.l1_hit_cells[0].get_indices_list() == [0]
            store.finish_read_prefetched([key(1)], l1_owners=result.l1_owners)
        # Filters that exclude CXL/REMOTE never ask the orchestrator.
        lookups = Mock(side_effect=AssertionError("shared L1 was looked up"))
        monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", lookups)
        for allowed in (frozenset({local}), frozenset()):
            result = prefetch(
                store, single_row_spec([key(1)], LAYOUT, allowed_l1_memory_tags=allowed)
            )
            assert result.l1_hit_cells[0].get_indices_list() == []
    finally:
        store.close()
