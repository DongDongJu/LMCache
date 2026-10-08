# SPDX-License-Identifier: Apache-2.0
"""Media: mapping checks, bounds-checked views, and the choice per L1 type."""

# Standard
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.shared_l1.media import (
    DevDaxMedium,
    FileMedium,
    open_medium,
)
from tests.v1.distributed.shared_l1.utils import shared_config

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
CAPACITY = 16 * ALIGN
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])  # 4000 B


@pytest.fixture
def backing(tmp_path: Path) -> Path:
    path = tmp_path / "region"
    path.write_bytes(b"\0" * CAPACITY)
    return path


def test_views_alias_the_mapping_at_the_given_offset(backing: Path) -> None:
    medium = FileMedium(str(backing), CAPACITY, ALIGN)
    try:
        first = medium.view(0, ALIGN, LAYOUT)
        second = medium.view(3 * ALIGN, ALIGN, LAYOUT)
        assert first.get_size() == 4000
        assert first.tensor is not None and second.tensor is not None
        assert first.tensor.shape == torch.Size([2, 1000])
        assert second.data_ptr == medium.address(3 * ALIGN)
        second.tensor.fill_(1.5)
        # A second view of the same extent sees the bytes: one mapping.
        again = medium.view(3 * ALIGN, ALIGN, LAYOUT)
        assert again.tensor is not None
        assert torch.all(again.tensor == 1.5)
        assert torch.all(first.tensor == 0)
    finally:
        del first, second, again
        medium.close()
    # The bytes reached the file, which another process would map.
    data = backing.read_bytes()
    assert data[3 * ALIGN : 3 * ALIGN + 4000] != b"\0" * 4000


@pytest.mark.parametrize(
    ("offset", "length"),
    [
        (100, ALIGN),  # misaligned
        (0, 2048),  # shorter than the layout
        (CAPACITY - ALIGN, 2 * ALIGN),  # leaves the region
        (CAPACITY, ALIGN),  # starts past the end
    ],
)
def test_bad_extents_are_rejected(backing: Path, offset: int, length: int) -> None:
    medium = FileMedium(str(backing), CAPACITY, ALIGN)
    try:
        with pytest.raises(ValueError, match="does not fit"):
            medium.view(offset, length, LAYOUT)
    finally:
        medium.close()


def test_file_medium_creates_or_extends_the_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    medium = FileMedium(str(missing), CAPACITY, ALIGN)
    try:
        assert missing.stat().st_size == CAPACITY
        assert medium.backend_type is L1BackendType.DRAM
        assert medium.capacity_bytes == CAPACITY
        assert medium.data_path in ("direct-pinned", "pageable-staging")
    finally:
        medium.close()
    small = tmp_path / "small"
    small.write_bytes(b"\0" * ALIGN)
    FileMedium(str(small), CAPACITY, ALIGN).close()
    assert small.stat().st_size == CAPACITY


def test_file_medium_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        FileMedium(str(tmp_path), CAPACITY, ALIGN)


def test_devdax_medium_rejects_a_regular_file(backing: Path) -> None:
    with pytest.raises(ValueError, match="character device"):
        DevDaxMedium(str(backing), CAPACITY, ALIGN)


def test_owns_device_matches_aliases_only(backing: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(backing)
    other = tmp_path / "other"
    other.write_bytes(b"\0" * CAPACITY)
    medium = FileMedium(str(backing), CAPACITY, ALIGN)
    try:
        assert medium.owns_device(str(alias))
        assert not medium.owns_device(str(other))
        assert not medium.owns_device(str(tmp_path / "missing"))
    finally:
        medium.close()


def test_open_medium_follows_the_l1_type(backing: Path) -> None:
    dram = open_medium(
        shared_config(backing, "127.0.0.1:1", "server-a", size_bytes=CAPACITY),
        CAPACITY,
        ALIGN,
    )
    try:
        assert isinstance(dram, FileMedium)
        assert dram.path == str(backing)
    finally:
        dram.close()
    devdax = shared_config(
        backing, "127.0.0.1:1", "server-a", size_bytes=CAPACITY, l1_type="DEVDAX"
    )
    with pytest.raises(ValueError, match="character device"):
        open_medium(devdax, CAPACITY, ALIGN)
