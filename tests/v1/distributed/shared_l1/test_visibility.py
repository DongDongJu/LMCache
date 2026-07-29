# SPDX-License-Identifier: Apache-2.0
"""Focused tests for shared-L1 visibility boundaries."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import gc
import mmap
import os
import threading
import warnings

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.shared_l1 import (
    InMemorySharedL1Pool,
    SharedL1Error,
    SharedMemoryRegion,
)

_ALIGNMENT = 4096
_CAPACITY = 64 * 1024


class _RecordingVisibility:
    """Record exact publish/acquire ranges without touching CPU cache state."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, int, int]] = []

    @property
    def granularity(self) -> int:
        """Return a conservative cache-line visibility granularity."""
        return 64

    def publish(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        os.fstat(device_fd)
        assert mapped_address > 0
        self.calls.append(
            ("publish", device_offset, length, generation, mapped_address)
        )

    def acquire(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        os.fstat(device_fd)
        assert mapped_address > 0
        self.calls.append(
            ("acquire", device_offset, length, generation, mapped_address)
        )


class _FailingPublishVisibility(_RecordingVisibility):
    """Inject a publish failure before metadata may become readable."""

    def publish(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        raise RuntimeError("injected publish failure")


class _FailingAcquireVisibility(_RecordingVisibility):
    """Inject an acquire failure before any payload view is exposed."""

    def acquire(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        raise RuntimeError("injected acquire failure")


class _BlockingPublishVisibility(_RecordingVisibility):
    """Hold publish open so close serialization can be observed."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def publish(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release publish")
        super().publish(
            device_fd=device_fd,
            mapped_address=mapped_address,
            device_offset=device_offset,
            length=length,
            generation=generation,
        )


class _LargeGranularityVisibility(_RecordingVisibility):
    """Require a unit that an ordinary process mapping cannot align to."""

    @property
    def granularity(self) -> int:
        """Return a deliberately unrepresentable process mapping alignment."""
        return 1 << 62


def _create_region(path: Path) -> None:
    with path.open("wb") as region_file:
        region_file.truncate(_CAPACITY)


def _open_descriptors_for(path: Path) -> set[str]:
    """Return process descriptors that currently resolve to ``path``."""
    expected = path.resolve()
    descriptors: set[str] = set()
    for descriptor in Path("/proc/self/fd").iterdir():
        try:
            if descriptor.resolve() == expected:
                descriptors.add(descriptor.name)
        except OSError:
            # Descriptors owned by unrelated threads may close while procfs is
            # being inspected.
            continue
    return descriptors


def test_character_mapping_requires_explicit_visibility() -> None:
    pool = InMemorySharedL1Pool(
        "character-visibility-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )

    with pytest.raises(SharedL1Error, match="visibility"):
        SharedMemoryRegion("/dev/zero", pool.region_contract())


def test_visibility_rejects_sub_granularity_allocation_alignment(
    tmp_path: Path,
) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "visibility-alignment-test",
        _CAPACITY,
        32,
        "layout-v1",
    )

    with pytest.raises(SharedL1Error, match="cannot isolate"):
        SharedMemoryRegion(
            region_path,
            pool.region_contract(),
            visibility=_RecordingVisibility(),
        )


def test_visibility_rejects_misaligned_mapped_base(tmp_path: Path) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    visibility = _LargeGranularityVisibility()
    pool = InMemorySharedL1Pool(
        "visibility-base-alignment-test",
        _CAPACITY,
        visibility.granularity,
        "layout-v1",
    )

    with pytest.raises(SharedL1Error, match="mapped base address"):
        SharedMemoryRegion(
            region_path,
            pool.region_contract(),
            visibility=visibility,
        )


def test_character_mapping_publishes_and_acquires_exact_object_range() -> None:
    pool = InMemorySharedL1Pool(
        "character-visibility-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    contract = pool.region_contract()
    abandoned = pool.reserve_write("abandoned", 1)
    pool.abort_write(abandoned)
    reservation = pool.reserve_write("object", 1024)
    payload = bytes(
        (index * 37 + 19) % 256 for index in range(reservation.handle.length)
    )
    visibility = _RecordingVisibility()
    mapping_offset = mmap.PAGESIZE

    with SharedMemoryRegion(
        "/dev/zero",
        contract,
        mapping_offset,
        visibility=visibility,
    ) as region:
        region.write(reservation.handle, payload)
        region.publish(reservation.handle)
        pool.finish_write(reservation)
        read = pool.reserve_read("object", reservation.handle)
        assert read is not None
        with region.read_view(read.handle) as observed:
            assert observed == payload
        pool.finish_read(read)
        with pytest.raises(SharedL1Error, match="publish"):
            region.flush()

    assert [
        (operation, offset, length, generation)
        for operation, offset, length, generation, _ in visibility.calls
    ] == [
        (
            "publish",
            mapping_offset + reservation.handle.offset,
            reservation.handle.length,
            reservation.handle.generation,
        ),
        (
            "acquire",
            mapping_offset + reservation.handle.offset,
            reservation.handle.length,
            reservation.handle.generation,
        ),
    ]


def test_character_mapping_publish_failure_leaves_object_unreadable() -> None:
    pool = InMemorySharedL1Pool(
        "character-visibility-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    contract = pool.region_contract()
    reservation = pool.reserve_write("object", 1024)

    with SharedMemoryRegion(
        "/dev/zero",
        contract,
        visibility=_FailingPublishVisibility(),
    ) as region:
        region.write(reservation.handle, bytes(reservation.handle.length))
        with pytest.raises(RuntimeError, match="publish"):
            region.publish(reservation.handle)
        assert pool.reserve_read("object", reservation.handle) is None
        pool.abort_write(reservation)


def test_character_mapping_acquire_failure_exposes_no_view() -> None:
    pool = InMemorySharedL1Pool(
        "character-visibility-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    contract = pool.region_contract()
    reservation = pool.reserve_write("object", 1024)
    visibility = _FailingAcquireVisibility()

    with SharedMemoryRegion(
        "/dev/zero",
        contract,
        visibility=visibility,
    ) as region:
        region.write(reservation.handle, bytes(reservation.handle.length))
        region.publish(reservation.handle)
        pool.finish_write(reservation)
        read = pool.reserve_read("object", reservation.handle)
        assert read is not None
        with pytest.raises(RuntimeError, match="acquire"):
            with region.read_view(read.handle):
                pass
        pool.abort_read(read)

    objects = pool.snapshot()["objects"]
    assert isinstance(objects, dict)
    record = objects["object"]
    assert isinstance(record, dict)
    assert record["active_readers"] == 0


def test_region_close_is_idempotent_and_methods_fail_closed(
    tmp_path: Path,
) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "close-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    handle = pool.reserve_write("object", 4096).handle
    region = SharedMemoryRegion(region_path, pool.region_contract())
    region.close()

    sentinel_fd = os.open("/dev/null", os.O_RDONLY)
    try:
        region.close()
        os.fstat(sentinel_fd)
    finally:
        os.close(sentinel_fd)

    with pytest.raises(SharedL1Error, match="closed"):
        region.write(handle, bytes(handle.length))
    with pytest.raises(SharedL1Error, match="closed"):
        region.publish(handle)
    with pytest.raises(SharedL1Error, match="closed"):
        region.acquire(handle)
    with pytest.raises(SharedL1Error, match="closed"):
        region.read(handle)
    with pytest.raises(SharedL1Error, match="closed"):
        with region.read_view(handle):
            pass
    with pytest.raises(SharedL1Error, match="closed"):
        region.flush()


def test_region_close_rejects_active_zero_copy_view(tmp_path: Path) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "active-view-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    handle = pool.reserve_write("object", 4096).handle

    with SharedMemoryRegion(region_path, pool.region_contract()) as region:
        with region.read_view(handle):
            with pytest.raises(SharedL1Error, match="active zero-copy"):
                region.close()


def test_region_close_waits_for_visibility_publish(tmp_path: Path) -> None:
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "concurrent-close-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    handle = pool.reserve_write("object", 4096).handle
    visibility = _BlockingPublishVisibility()
    region = SharedMemoryRegion(
        region_path,
        pool.region_contract(),
        visibility=visibility,
    )
    publish_errors: list[BaseException] = []
    close_errors: list[BaseException] = []
    close_completed = threading.Event()

    def publish() -> None:
        try:
            region.publish(handle)
        except BaseException as error:
            publish_errors.append(error)

    def close() -> None:
        try:
            region.close()
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_completed.set()

    publish_thread = threading.Thread(target=publish)
    close_thread = threading.Thread(target=close)
    publish_thread.start()
    assert visibility.started.wait(timeout=5)
    close_thread.start()
    assert not close_completed.wait(timeout=0.1)
    visibility.release.set()
    publish_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert not publish_thread.is_alive()
    assert not close_thread.is_alive()
    assert publish_errors == []
    assert close_errors == []


def test_unclosed_region_releases_descriptor_during_gc(tmp_path: Path) -> None:
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        pytest.skip("descriptor-count assertion requires procfs")
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "gc-close-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    region = SharedMemoryRegion(region_path, pool.region_contract())
    assert _open_descriptors_for(region_path)

    with pytest.warns(ResourceWarning, match="unclosed SharedMemoryRegion"):
        del region
        gc.collect()

    assert not _open_descriptors_for(region_path)


def test_gc_cleanup_survives_resource_warning_as_error(tmp_path: Path) -> None:
    proc_fds = Path("/proc/self/fd")
    if not proc_fds.is_dir():
        pytest.skip("descriptor-count assertion requires procfs")
    region_path = tmp_path / "payload.bin"
    _create_region(region_path)
    pool = InMemorySharedL1Pool(
        "gc-warning-error-test",
        _CAPACITY,
        _ALIGNMENT,
        "layout-v1",
    )
    region = SharedMemoryRegion(region_path, pool.region_contract())
    assert _open_descriptors_for(region_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        del region
        gc.collect()

    assert not _open_descriptors_for(region_path)
