# SPDX-License-Identifier: Apache-2.0
"""StorageManager builds the remote binding for any L1 with a shared section."""

# Standard
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.shared_l1.utils import shared_config
from tests.v1.distributed.utils import single_row_spec
from tests.v1.memory_orchestrator.harness import REGION_ID, running_orchestrator
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

CAPACITY = 32 * 4096


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
    for store in created:
        store.close()


def test_shared_dram_l1_is_served_by_the_remote_binding(
    tmp_path: Path, stores: list[StorageManager]
) -> None:
    region = tmp_path / "region"
    with running_orchestrator(tmp_path / "state", capacity_bytes=CAPACITY) as orch:
        l1 = shared_config(region, orch.endpoint, "server-a", size_bytes=CAPACITY)
        store = StorageManager(StorageManagerConfig(l1, EvictionConfig("noop")))
        stores.append(store)

        shared = store.report_status()["l1_manager"]["shared"]
        assert (shared["orchestrator"], shared["region_id"], shared["client_id"]) == (
            orch.endpoint,
            REGION_ID,
            "server-a",
        )
        assert (shared["medium"], shared["path"]) == ("dram", str(region))
        assert shared["orchestrator_reachable"] is True
        # The file medium created the region file at the orchestrator's capacity.
        assert region.stat().st_size == CAPACITY


def test_embedded_l1_is_not_remote(stores: list[StorageManager]) -> None:
    config = L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False, shm_name="")
    )
    store = StorageManager(StorageManagerConfig(config, EvictionConfig("noop")))
    stores.append(store)

    assert "shared" not in store.report_status()["l1_manager"]


@pytest.mark.parametrize("shared_first", [False, True])
def test_shared_file_cannot_alias_a_private_arena(
    tmp_path: Path, shared_first: bool
) -> None:
    capacity = 2 << 20
    region = tmp_path / "region"
    with region.open("wb") as backing:
        backing.truncate(capacity)
    with running_orchestrator(tmp_path / "state", capacity_bytes=capacity) as orch:
        shared = shared_config(region, orch.endpoint, "shared", size_bytes=capacity)
        private = DevDaxL1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=capacity,
                use_lazy=False,
                shm_name="",
                devdax_path=str(region),
            ),
            tag="private",
        )
        configs = [shared, private] if shared_first else [private, shared]
        config = StorageManagerConfig(
            configs[0], EvictionConfig("noop"), l1_manager_configs=configs
        )
        with pytest.raises(ValueError, match="already owned by an L1"):
            StorageManager(config)


def test_shared_overflow_stays_private_and_reads_keep_their_owner(
    tmp_path: Path,
) -> None:
    with (
        running_orchestrator(tmp_path / "state", capacity_bytes=4096) as orch,
        ExitStack() as stack,
    ):
        stores = []
        for client_id in ("writer", "peer"):
            shared = shared_config(
                tmp_path / "region", orch.endpoint, client_id, size_bytes=4096
            )
            private = L1ManagerConfig(
                L1MemoryManagerConfig(size_in_bytes=8192, use_lazy=False, shm_name=""),
                tag="private",
            )
            store = StorageManager(
                StorageManagerConfig(
                    shared, EvictionConfig("noop"), l1_manager_configs=[shared, private]
                )
            )
            stack.callback(store.close)
            stores.append(store)
        writer, peer = stores
        layout = MemoryLayoutDesc([torch.Size([1000])], [torch.uint8])
        keys = [ObjectKey(ObjectKey.IntHash2Bytes(i), "overflow", 0) for i in (1, 2)]
        for value, key in enumerate(keys, 1):
            reserved = writer.reserve_write([key], layout)
            tensor = reserved[key].tensor
            assert tensor is not None
            tensor.fill_(value)
            writer.finish_write_by_owner(writer.prepare_write_completion(reserved))
        spec = single_row_spec(keys, layout)
        own = writer.query_prefetch_status(
            writer.submit_prefetch_task(spec, skip_l2=True)
        )
        remote = peer.query_prefetch_status(
            peer.submit_prefetch_task(spec, skip_l2=True)
        )
        assert own is not None and remote is not None
        assert own.hit_cells[0].get_indices_list() == [0, 1]
        assert remote.hit_cells[0].get_indices_list() == [0]
        assert own.l1_owners[keys[0]] != own.l1_owners[keys[1]]
        with writer.read_prefetched_results(keys, own.l1_owners) as objects:
            assert objects is not None
            assert all(torch.all(obj.tensor == i) for i, obj in enumerate(objects, 1))
        writer.finish_read_prefetched(keys, l1_owners=own.l1_owners)
        peer.finish_read_prefetched(keys[:1], l1_owners=remote.l1_owners)
        assert all(
            state["read_locked_count"] == 0
            for store in stores
            for state in store.report_status()["l1_managers"].values()
        )
