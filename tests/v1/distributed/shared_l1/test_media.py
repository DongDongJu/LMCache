# SPDX-License-Identifier: Apache-2.0
"""Media: mapping checks, bounds-checked views, and the choice per L1 type."""

# Standard
from pathlib import Path
import mmap
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.shared_l1 import media
from lmcache.v1.distributed.shared_l1.media import RegionMedium

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
CAPACITY = 16 * ALIGN
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])  # 4000 B


@pytest.fixture
def backing(tmp_path: Path) -> Path:
    path = tmp_path / "region"
    path.write_bytes(b"\0" * CAPACITY)
    return path


def _medium(
    path: Path, backend_type: L1BackendType = L1BackendType.DRAM
) -> RegionMedium:
    return RegionMedium(str(path), CAPACITY, ALIGN, backend_type)


def test_views_alias_the_mapping_at_the_given_offset(backing: Path) -> None:
    medium = _medium(backing)
    try:
        first = medium.view(0, ALIGN, LAYOUT)
        second = medium.view(3 * ALIGN, ALIGN, LAYOUT)
        assert first.get_size() == 4000
        assert first.tensor is not None and second.tensor is not None
        assert first.tensor.shape == torch.Size([2, 1000])
        assert second.data_ptr == first.data_ptr + 3 * ALIGN
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
    medium = _medium(backing)
    try:
        with pytest.raises(ValueError, match="does not fit"):
            medium.view(offset, length, LAYOUT)
    finally:
        medium.close()


def test_file_medium_creates_or_extends_the_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    medium = _medium(missing)
    try:
        assert missing.stat().st_size == CAPACITY
        assert medium.backend_type is L1BackendType.DRAM
        assert medium.capacity_bytes == CAPACITY
        assert medium.data_path == "direct-pinned"
    finally:
        medium.close()
    small = tmp_path / "small"
    small.write_bytes(b"\0" * ALIGN)
    _medium(small).close()
    assert small.stat().st_size == CAPACITY


def test_media_refuse_the_wrong_backing(monkeypatch: pytest.MonkeyPatch) -> None:
    # /dev/null opens like a file but is a character device.
    with pytest.raises(ValueError, match="regular file"):
        _medium(Path("/dev/null"))
    monkeypatch.setattr(Path, "read_text", lambda path: str(CAPACITY // 2))
    with pytest.raises(ValueError, match="less than the region capacity"):
        _medium(Path("/dev/null"), L1BackendType.DEVDAX)


def test_owns_device_matches_aliases_only(backing: Path, tmp_path: Path) -> None:
    alias = tmp_path / "alias"
    alias.symlink_to(backing)
    other = tmp_path / "other"
    other.write_bytes(b"\0" * CAPACITY)
    medium = _medium(backing)
    try:
        assert medium.owns_device(str(alias))
        assert not medium.owns_device(str(other))
        assert not medium.owns_device(str(tmp_path / "missing"))
    finally:
        medium.close()


def test_mapping_follows_the_l1_type(backing: Path) -> None:
    dram = _medium(backing)
    try:
        assert dram.backend_type is L1BackendType.DRAM
        assert dram.path == str(backing)
    finally:
        dram.close()
    with pytest.raises(ValueError, match="character device"):
        _medium(backing, L1BackendType.DEVDAX)


def test_gds_medium_is_rejected(backing: Path) -> None:
    with pytest.raises(ValueError, match="DRAM or DEVDAX"):
        _medium(backing, L1BackendType.GDS)


def test_unreadable_device_size_is_rejected() -> None:
    with pytest.raises(OSError):
        _medium(Path("/dev/null"), L1BackendType.DEVDAX)


def test_failed_gpu_registration_closes_the_mapping(
    backing: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings: list[tuple[int, mmap.mmap]] = []
    open_mapping = media.open_devdax_mapping

    def track(path: str, size: int) -> tuple[int, mmap.mmap, object, torch.Tensor]:
        result = open_mapping(path, size)
        mappings.append(result[:2])
        return result

    monkeypatch.setattr(media, "open_devdax_mapping", track)
    monkeypatch.setattr(media.current_device_spec, "pin_memory", lambda *args: False)
    with pytest.raises(RuntimeError, match="requires GPU-registered memory"):
        _medium(backing)
    fd, mapping = mappings.pop()
    assert mapping.closed
    with pytest.raises(OSError):
        os.fstat(fd)
