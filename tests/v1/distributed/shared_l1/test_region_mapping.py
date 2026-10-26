# SPDX-License-Identifier: Apache-2.0
"""DevDaxL1MemoryManager in shared mode: one mapping, bounds-checked views."""

# Standard
from collections.abc import Iterator
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.memory_manager.devdax_l1_memory_manager import (
    DevDaxL1MemoryManager,
)
from lmcache.v1.memory_orchestrator.api import RegionContract
from tests.v1.distributed.shared_l1.utils import CAPACITY, region_file

pytestmark = pytest.mark.no_shared_allocator
ALIGN = OFFSET = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.uint8])


@pytest.fixture
def region(tmp_path: Path) -> Path:
    return region_file(tmp_path / "region", OFFSET + CAPACITY)


@pytest.fixture
def mapping(region: Path) -> Iterator[DevDaxL1MemoryManager]:
    contract = RegionContract(
        region_id="region",
        region_epoch=1,
        capacity_bytes=CAPACITY,
        alignment_bytes=ALIGN,
        layout_fingerprint=b"",
        visibility_mode="coherent",
        max_batch_entries=0,
        reset_required=False,
    )
    config = L1MemoryManagerConfig(
        CAPACITY,
        use_lazy=False,
        shm_name="",
        devdax_path=str(region),
        devdax_offset_in_bytes=OFFSET,
    )
    manager = DevDaxL1MemoryManager(config, shared_region=contract)
    yield manager
    manager.close()


def test_views_alias_the_region_after_the_device_offset(
    mapping: DevDaxL1MemoryManager, region: Path
) -> None:
    first = mapping.view(ALIGN, ALIGN, LAYOUT)
    second = mapping.view(ALIGN, ALIGN, LAYOUT)
    assert first.tensor is not None and second.tensor is not None
    first.tensor.fill_(5)
    assert torch.all(second.tensor == 5)
    assert first.get_l1_manager() is None and first.parent() is None
    with open(region, "rb") as device:
        device.seek(OFFSET + ALIGN)
        assert device.read(first.get_size()) == bytes([5]) * first.get_size()
    assert mapping.owns_range(str(region), OFFSET, 1)
    assert not mapping.owns_range(str(region), 0, OFFSET)


@pytest.mark.parametrize(
    "offset,length",
    # LAYOUT's payload is 1024 bytes, so a 512-byte extent is short.
    [(-ALIGN, ALIGN), (1, ALIGN), (CAPACITY, ALIGN), (0, 512)],
    ids=["negative", "misaligned", "outside", "short"],
)
def test_invalid_extents_are_rejected(
    mapping: DevDaxL1MemoryManager, offset: int, length: int
) -> None:
    with pytest.raises(ValueError, match="does not fit"):
        mapping.view(offset, length, LAYOUT)
