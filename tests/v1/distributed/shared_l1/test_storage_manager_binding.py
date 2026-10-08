# SPDX-License-Identifier: Apache-2.0
"""StorageManager builds the remote binding for any L1 with a shared section."""

# Standard
from collections.abc import Iterator
from pathlib import Path
import json

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.shared_l1.utils import REGION_ID, running_orchestrator
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
        spec = {
            "type": "DRAM",
            "tag": "pool",
            "size_gb": CAPACITY / (1 << 30),
            "path": str(region),
            "shared": {
                "orchestrator": orch.endpoint,
                "region_id": REGION_ID,
                "client_id": "server-a",
            },
        }
        store = StorageManager(
            parse_args(["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)])
        )
        stores.append(store)

        assert store.has_remote_l1()
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


def test_shared_devdax_l1_maps_a_device_not_a_file(
    tmp_path: Path, stores: list[StorageManager]
) -> None:
    region = tmp_path / "region"
    region.write_bytes(b"\0" * CAPACITY)
    with running_orchestrator(tmp_path / "state", capacity_bytes=CAPACITY) as orch:
        spec = {
            "type": "DEVDAX",
            "tag": "cxl",
            "size_gb": CAPACITY / (1 << 30),
            "path": str(region),
            "shared": {
                "orchestrator": orch.endpoint,
                "region_id": REGION_ID,
                "client_id": "server-a",
            },
        }
        with pytest.raises(ValueError, match="character device"):
            StorageManager(
                parse_args(
                    ["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)]
                )
            )


def test_embedded_l1_is_not_remote(stores: list[StorageManager]) -> None:
    config = L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False, shm_name="")
    )
    store = StorageManager(StorageManagerConfig(config, EvictionConfig("noop")))
    stores.append(store)

    assert not store.has_remote_l1()
    assert "shared" not in store.report_status()["l1_manager"]
