# SPDX-License-Identifier: Apache-2.0
"""
Tests for read-only Device-DAX inspection and the presence watcher.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
import builtins
import errno
import io
import os
import stat as stat_module
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.dax_physical import (
    DaxDeviceWatcher,
    DaxPhysicalMode,
    DaxPhysicalState,
    DaxWatcherSnapshot,
    probe_device,
    scan_directory,
)

_MAJOR = 249
_SIZE = 274877906944
_ALIGN = 2097152


def wait_for_condition(
    predicate: Callable[[], bool], timeout: float = 5.0, interval: float = 0.01
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _char_stat(major: int, minor: int) -> os.stat_result:
    """Build a stat result describing a character device ``major:minor``."""
    return os.stat_result(
        (stat_module.S_IFCHR | 0o660, 0, 0, 1, 0, 0, 0, 0, 0, 0),
        {"st_rdev": os.makedev(major, minor)},
    )


def _stat_table(
    table: dict[str, os.stat_result],
) -> Callable[[str], os.stat_result]:
    """Return a ``stat`` stand-in serving ``table`` and ENOENT otherwise."""

    def _stat(path: str) -> os.stat_result:
        try:
            return table[path]
        except KeyError:
            raise FileNotFoundError(errno.ENOENT, "No such file", path) from None

    return _stat


def _make_sysfs_device(
    sysfs_root: Path,
    minor: int,
    kernel_name: str,
    *,
    subsystem: str = "dax",
    driver: str = "device_dax",
    size: str = str(_SIZE),
    align: str = str(_ALIGN),
) -> Path:
    """Create one fake dax bus device under a fake sysfs root.

    ``driver=""`` leaves the device unbound (no ``driver`` symlink);
    ``size``/``align`` are file contents (``""`` omits the file).
    """
    device_dir = sysfs_root / "devices" / "platform" / "hmem.2" / kernel_name
    device_dir.mkdir(parents=True)
    subsystem_dir = sysfs_root / "bus" / subsystem
    subsystem_dir.mkdir(parents=True, exist_ok=True)
    (device_dir / "subsystem").symlink_to(subsystem_dir)
    if driver:
        driver_dir = sysfs_root / "bus" / "dax" / "drivers" / driver
        driver_dir.mkdir(parents=True, exist_ok=True)
        (device_dir / "driver").symlink_to(driver_dir)
    if size:
        (device_dir / "size").write_text(size + "\n")
    if align:
        (device_dir / "align").write_text(align + "\n")
    char_dir = sysfs_root / "dev" / "char"
    char_dir.mkdir(parents=True, exist_ok=True)
    (char_dir / f"{_MAJOR}:{minor}").symlink_to(device_dir)
    return device_dir


def test_probe_devdax_device_reports_driver_size_and_align(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 2, "dax2.1")
    device_path = "/dev/dax-cxl/ns_pod/dax0.1"

    state = probe_device(
        device_path,
        sysfs_root=str(sysfs),
        stat=_stat_table({device_path: _char_stat(_MAJOR, 2)}),
        clock=lambda: 123.5,
    )

    assert state == DaxPhysicalState(
        device_path=device_path,
        mode=DaxPhysicalMode.DEVDAX,
        present=True,
        major=_MAJOR,
        minor=2,
        kernel_name="dax2.1",
        driver="device_dax",
        size_bytes=_SIZE,
        align_bytes=_ALIGN,
        probed_at=123.5,
        detail="",
    )
    assert state.as_dict()["mode"] == "devdax"
    assert state.as_dict()["kernel_name"] == "dax2.1"


def test_probe_kmem_device_is_system_ram(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 3, "dax2.2", driver="kmem")

    state = probe_device(
        "/dev/dax2.2",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/dax2.2": _char_stat(_MAJOR, 3)}),
    )

    assert state.mode is DaxPhysicalMode.SYSTEM_RAM
    assert state.driver == "kmem"
    assert state.present is True
    assert state.size_bytes == _SIZE


def test_probe_driverless_device_is_unbound(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 0, "dax2.0", driver="", size="0")

    state = probe_device(
        "/dev/dax2.0",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/dax2.0": _char_stat(_MAJOR, 0)}),
    )

    assert state.mode is DaxPhysicalMode.UNBOUND
    assert state.driver == ""
    assert state.kernel_name == "dax2.0"
    assert state.size_bytes == 0
    assert state.align_bytes == _ALIGN
    assert state.detail == ""


def test_probe_unrecognised_driver_is_unknown(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 4, "dax2.3", driver="fsdev_dax")

    state = probe_device(
        "/dev/dax2.3",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/dax2.3": _char_stat(_MAJOR, 4)}),
    )

    assert state.mode is DaxPhysicalMode.UNKNOWN
    assert state.driver == "fsdev_dax"
    assert "fsdev_dax" in state.detail


def test_probe_regular_file_is_not_a_device(tmp_path):
    arena = tmp_path / "arena.bin"
    with open(arena, "wb") as fout:
        fout.truncate(8192)

    state = probe_device(str(arena))

    assert state.mode is DaxPhysicalMode.NOT_A_DEVICE
    assert state.present is True
    assert state.size_bytes == 8192
    assert (state.major, state.minor) == (0, 0)
    assert state.detail == "not a character device"


def test_probe_dev_null_is_not_a_dax_device():
    """A real non-dax char device resolves through the real sysfs."""
    if not os.path.isdir("/sys/dev/char/1:3"):
        pytest.skip("/sys/dev/char/1:3 is not visible in this environment")

    state = probe_device("/dev/null")

    assert state.mode is DaxPhysicalMode.NOT_A_DEVICE
    assert (state.major, state.minor) == (1, 3)
    assert state.detail == "subsystem mem"


def test_probe_non_dax_subsystem_is_not_a_device(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 7, "null", subsystem="mem", driver="")

    state = probe_device(
        "/dev/null-alike",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/null-alike": _char_stat(_MAJOR, 7)}),
    )

    assert state.mode is DaxPhysicalMode.NOT_A_DEVICE
    assert state.kernel_name == "null"
    assert state.detail == "subsystem mem"


def test_probe_missing_path_is_absent(tmp_path):
    state = probe_device(str(tmp_path / "missing"), clock=lambda: 7.0)

    assert state.mode is DaxPhysicalMode.ABSENT
    assert state.present is False
    assert state.probed_at == 7.0
    assert state.detail == "path does not exist"


def test_probe_stat_error_other_than_missing_is_unknown():
    def _denied(path: str) -> os.stat_result:
        raise PermissionError(errno.EACCES, "Permission denied", path)

    state = probe_device("/dev/secret", stat=_denied)

    assert state.mode is DaxPhysicalMode.UNKNOWN
    assert state.present is False
    assert "Permission denied" in state.detail


def test_probe_char_device_without_sysfs_entry_is_unknown(tmp_path):
    sysfs = tmp_path / "sys"
    (sysfs / "dev" / "char").mkdir(parents=True)

    state = probe_device(
        "/dev/dax9.9",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/dax9.9": _char_stat(_MAJOR, 9)}),
    )

    assert state.mode is DaxPhysicalMode.UNKNOWN
    assert state.present is True
    assert (state.major, state.minor) == (_MAJOR, 9)
    assert state.detail == "no sysfs entry"


def test_probe_unreadable_size_and_align_are_zero(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 5, "dax2.4", size="garbage", align="")

    state = probe_device(
        "/dev/dax2.4",
        sysfs_root=str(sysfs),
        stat=_stat_table({"/dev/dax2.4": _char_stat(_MAJOR, 5)}),
    )

    assert state.mode is DaxPhysicalMode.DEVDAX
    assert state.size_bytes == 0
    assert state.align_bytes == 0


def _spy_open(real: Callable[..., object], opened: list[str]) -> Callable[..., object]:
    """Wrap one ``open``-like entry point, recording every path it is given."""

    def _wrapped(*args: object, **kwargs: object) -> object:
        target = args[0] if args else kwargs.get("file", kwargs.get("path"))
        if isinstance(target, (str, bytes, os.PathLike)):
            opened.append(os.fsdecode(target))
        return real(*args, **kwargs)

    return _wrapped


def test_probe_never_opens_the_candidate_path(tmp_path, monkeypatch):
    """The probe must be a pure stat/sysfs read: no ``open()`` of the path.

    Every open entry point is spied (``builtins.open``/``io.open``, which
    ``open()`` calls resolve to, and ``os.open``), and the candidate exists
    on disk so an accidental open would succeed silently and only the spy
    would catch it. The sysfs reads must show up, so the spy cannot go
    vacuous unnoticed.
    """
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 2, "dax2.1")
    candidate = tmp_path / "dax0.1"
    candidate.write_bytes(b"")
    device_path = str(candidate)
    opened: list[str] = []
    monkeypatch.setattr(builtins, "open", _spy_open(builtins.open, opened))
    monkeypatch.setattr(io, "open", _spy_open(io.open, opened))
    monkeypatch.setattr(os, "open", _spy_open(os.open, opened))

    state = probe_device(
        device_path,
        sysfs_root=str(sysfs),
        stat=_stat_table({device_path: _char_stat(_MAJOR, 2)}),
    )

    assert state.mode is DaxPhysicalMode.DEVDAX
    assert state.size_bytes == _SIZE
    assert device_path not in opened
    # Positive control: the sysfs attribute reads went through the spies.
    assert any(path.endswith("/size") for path in opened)
    assert all(path.startswith(str(sysfs)) for path in opened), opened


def test_as_dict_is_json_friendly_and_round_trips_fields():
    state = DaxPhysicalState(
        device_path="/dev/x",
        mode=DaxPhysicalMode.UNBOUND,
        present=True,
        major=1,
        minor=2,
        kernel_name="dax1.2",
        driver="",
        size_bytes=3,
        align_bytes=4,
        probed_at=5.0,
        detail="",
    )

    assert state.as_dict() == {
        "device_path": "/dev/x",
        "mode": "unbound",
        "present": True,
        "major": 1,
        "minor": 2,
        "kernel_name": "dax1.2",
        "driver": "",
        "size_bytes": 3,
        "align_bytes": 4,
        "probed_at": 5.0,
        "detail": "",
    }


def test_scan_directory_sorts_by_name_and_skips_subdirectories(tmp_path):
    directory = tmp_path / "devs"
    directory.mkdir()
    for name, size in (("b.bin", 1024), ("a.bin", 2048), ("c.bin", 512)):
        with open(directory / name, "wb") as fout:
            fout.truncate(size)
    (directory / "subdir").mkdir()

    states = scan_directory(str(directory))

    assert [state.device_path for state in states] == [
        str(directory / "a.bin"),
        str(directory / "b.bin"),
        str(directory / "c.bin"),
    ]
    assert [state.size_bytes for state in states] == [2048, 1024, 512]
    assert all(state.mode is DaxPhysicalMode.NOT_A_DEVICE for state in states)


def test_scan_directory_missing_or_file_returns_empty(tmp_path):
    assert scan_directory(str(tmp_path / "missing")) == []

    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    assert scan_directory(str(not_a_dir)) == []


def test_scan_directory_forwards_sysfs_root_and_stat(tmp_path):
    sysfs = tmp_path / "sys"
    _make_sysfs_device(sysfs, 2, "dax2.1")
    directory = tmp_path / "devs"
    directory.mkdir()
    node = directory / "dax0.1"
    node.write_text("")

    states = scan_directory(
        str(directory),
        sysfs_root=str(sysfs),
        stat=_stat_table({str(node): _char_stat(_MAJOR, 2)}),
        clock=lambda: 42.0,
    )

    assert len(states) == 1
    assert states[0].mode is DaxPhysicalMode.DEVDAX
    assert states[0].device_path == str(node)
    assert states[0].probed_at == 42.0


def _watch_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("dax-l2-watch")]


def _state(path: str, probed_at: float = 0.0) -> DaxPhysicalState:
    return DaxPhysicalState(
        device_path=path,
        mode=DaxPhysicalMode.DEVDAX,
        present=True,
        probed_at=probed_at,
    )


def test_watcher_snapshot_is_empty_before_first_scan():
    watcher = DaxDeviceWatcher("/dev/dax-cxl/ns_pod", 1.0)

    snapshot = watcher.snapshot()

    assert snapshot == DaxWatcherSnapshot(
        directory="/dev/dax-cxl/ns_pod",
        interval_seconds=1.0,
        last_scan_at=0.0,
        devices=(),
    )
    assert snapshot.find("/dev/dax-cxl/ns_pod/dax0.1") is None
    assert watcher.enabled is True
    watcher.stop()


def test_watcher_publishes_scans_and_stops_thread():
    scans = 0
    seen: list[tuple[str, str]] = []
    ticks = iter(range(100, 200))

    def _scan(
        directory: str, *, sysfs_root: str, clock: Callable[[], float]
    ) -> list[DaxPhysicalState]:
        nonlocal scans
        scans += 1
        # Record rather than assert: this runs on the watcher thread, where
        # a failed assertion would be swallowed by the scan error handler.
        seen.append((directory, sysfs_root))
        return [_state("/watched/a", probed_at=clock()), _state("/watched/b")]

    watcher = DaxDeviceWatcher(
        "/watched",
        0.01,
        sysfs_root="/fake-sys",
        clock=lambda: float(next(ticks)),
        scan=_scan,
    )
    assert _watch_threads() == []
    watcher.start()
    watcher.start()  # idempotent
    try:
        assert wait_for_condition(lambda: watcher.snapshot().last_scan_at > 0)
        assert seen[0] == ("/watched", "/fake-sys")
        assert len(_watch_threads()) == 1
        snapshot = watcher.snapshot()
        assert [d.device_path for d in snapshot.devices] == [
            "/watched/a",
            "/watched/b",
        ]
        found = snapshot.find("/watched/b")
        assert found is not None
        assert found.device_path == "/watched/b"
        assert snapshot.find("/watched/c") is None
        status = watcher.status()
        assert status["enabled"] is True
        assert status["directory"] == "/watched"
        assert status["interval_seconds"] == 0.01
        assert status["last_scan_at"] == snapshot.last_scan_at
        assert [d["device_path"] for d in status["present_devices"]] == [
            "/watched/a",
            "/watched/b",
        ]
        assert wait_for_condition(lambda: scans >= 3)
    finally:
        watcher.stop()
    assert wait_for_condition(lambda: _watch_threads() == [])
    watcher.stop()  # idempotent


def test_watcher_keeps_previous_snapshot_when_scan_raises():
    calls = 0

    def _scan(
        directory: str, *, sysfs_root: str, clock: Callable[[], float]
    ) -> list[DaxPhysicalState]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return [_state("/watched/only")]
        raise OSError("sysfs went away")

    watcher = DaxDeviceWatcher("/watched", 0.01, scan=_scan)
    watcher.start()
    try:
        assert wait_for_condition(lambda: calls >= 3)
        snapshot = watcher.snapshot()
        assert [d.device_path for d in snapshot.devices] == ["/watched/only"]
        assert snapshot.last_scan_at > 0
        assert len(_watch_threads()) == 1
    finally:
        watcher.stop()


def test_watcher_scans_real_directory_and_sees_new_entries(tmp_path):
    directory = tmp_path / "devs"
    directory.mkdir()
    (directory / "a.bin").write_bytes(b"\0" * 16)
    watcher = DaxDeviceWatcher(str(directory), 0.01)
    watcher.start()
    try:
        assert wait_for_condition(
            lambda: (
                [d.device_path for d in watcher.snapshot().devices]
                == [str(directory / "a.bin")]
            )
        )
        (directory / "b.bin").write_bytes(b"\0" * 32)
        assert wait_for_condition(
            lambda: (
                [d.device_path for d in watcher.snapshot().devices]
                == [str(directory / "a.bin"), str(directory / "b.bin")]
            )
        )
        found = watcher.snapshot().find(str(directory / "b.bin"))
        assert found is not None
        assert found.mode is DaxPhysicalMode.NOT_A_DEVICE
        assert found.size_bytes == 32
    finally:
        watcher.stop()


def test_disabled_watcher_has_no_thread_and_reports_disabled():
    watcher = DaxDeviceWatcher("", 1.0)

    assert watcher.enabled is False
    watcher.start()
    assert _watch_threads() == []
    assert watcher.status() == {"enabled": False}
    assert watcher.snapshot().devices == ()
    watcher.stop()


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_watcher_rejects_non_positive_interval(interval):
    with pytest.raises(ValueError, match="interval_seconds"):
        DaxDeviceWatcher("/watched", interval)


def test_watcher_rejects_relative_directory():
    with pytest.raises(ValueError, match="absolute"):
        DaxDeviceWatcher("relative/dir", 1.0)
