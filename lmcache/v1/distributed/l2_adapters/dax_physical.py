# SPDX-License-Identifier: Apache-2.0
"""Read-only Device-DAX inspection for the MP DAX L2 adapter.

This module answers one question without touching the device: *if LMCache
mapped this path right now, would the kernel give it Device-DAX memory?* It
does so purely from ``stat(2)`` on the path and plain file reads under
``/sys``; it never ``open()``s the candidate device (opening an unbound dax
node forks ``modprobe`` on the host and can stall for seconds) and never
writes anywhere.

``probe_device`` classifies one path, ``scan_directory`` classifies every
entry of one directory, and ``DaxDeviceWatcher`` runs ``scan_directory`` on a
daemon thread and publishes an immutable ``DaxWatcherSnapshot`` that readers
can consult without any I/O. Presence and mode are *reported* only: nothing
in this module attaches a device, because physical presence says nothing
about ownership under the outside allocator.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from stat import S_ISCHR, S_ISREG
from typing import Protocol
import math
import os
import threading
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_DAX_SUBSYSTEM = "dax"
_DEVDAX_DRIVER = "device_dax"
_KMEM_DRIVER = "kmem"
_WATCH_THREAD_NAME = "dax-l2-watch"
_STOP_JOIN_TIMEOUT_SECONDS = 5.0


class DaxPhysicalMode(str, Enum):
    """How the kernel currently exposes a candidate DAX path."""

    DEVDAX = "devdax"
    """Bound to ``device_dax``: mappable by LMCache."""

    SYSTEM_RAM = "system-ram"
    """Bound to ``kmem``: the range is System RAM and must never be mapped."""

    UNBOUND = "unbound"
    """A dax bus device with no driver bound: not usable (yet)."""

    NOT_A_DEVICE = "not-a-device"
    """A regular file or a non-dax character device (tests, ``/dev/null``)."""

    ABSENT = "absent"
    """The path does not exist."""

    UNKNOWN = "unknown"
    """sysfs is unreadable or another ``OSError`` occurred; see ``detail``."""


@dataclass(frozen=True)
class DaxPhysicalState:
    """Result of one read-only probe of a DAX candidate path.

    ``present`` is ``True`` exactly when ``stat`` succeeded on
    ``device_path``. ``major``/``minor`` are the character-device numbers
    (``0`` when the path is not a character device). ``kernel_name`` is the
    dax bus device name resolved through ``/sys/dev/char`` (e.g. ``dax2.1``),
    ``driver`` is the bound driver's basename (``device_dax``, ``kmem``, or
    ``""`` when unbound or unknown), ``size_bytes``/``align_bytes`` come from
    sysfs for dax devices (``size_bytes`` is ``st_size`` for a regular file)
    and are ``0`` when unknown. ``detail`` is a human-readable reason for
    ``UNKNOWN``, ``NOT_A_DEVICE`` and ``ABSENT`` and empty otherwise.
    ``probed_at`` is the ``clock`` value taken when the probe started;
    every probe sets it, so the dataclass default ``0.0`` means "never
    probed" and always loses a freshness comparison against a real probe.
    """

    device_path: str
    mode: DaxPhysicalMode
    present: bool
    major: int = 0
    minor: int = 0
    kernel_name: str = ""
    driver: str = ""
    size_bytes: int = 0
    align_bytes: int = 0
    probed_at: float = 0.0
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly view of this state.

        Returns:
            A dictionary with every field of this dataclass, where ``mode``
            is the enum's string value (e.g. ``"devdax"``).
        """
        return {
            "device_path": self.device_path,
            "mode": self.mode.value,
            "present": self.present,
            "major": self.major,
            "minor": self.minor,
            "kernel_name": self.kernel_name,
            "driver": self.driver,
            "size_bytes": self.size_bytes,
            "align_bytes": self.align_bytes,
            "probed_at": self.probed_at,
            "detail": self.detail,
        }


def _read_sysfs_int(path: str) -> int:
    """Read one integer sysfs attribute.

    Args:
        path: sysfs attribute file to read (e.g. ``<device>/size``).

    Returns:
        The parsed integer, or ``0`` when the file is missing, unreadable,
        or does not contain an integer. Callers treat ``0`` as "unknown",
        which is why the add gate only acts on a non-zero size.
    """
    try:
        with open(path, encoding="ascii") as attribute:
            return int(attribute.read().strip())
    except (OSError, ValueError):
        return 0


def probe_device(
    path: str,
    *,
    sysfs_root: str = "/sys",
    stat: Callable[[str], os.stat_result] = os.stat,
    clock: Callable[[], float] = time.time,
) -> DaxPhysicalState:
    """Classify one path as a Device-DAX candidate without opening it.

    The probe is a sequence of plain reads: ``stat(path)``, then, for a
    character device, ``/sys/dev/char/<major>:<minor>`` is resolved to the
    dax bus device directory whose ``subsystem``, ``driver``, ``size`` and
    ``align`` entries decide the mode. The candidate path itself is never
    opened. This function never raises for filesystem errors; they are
    folded into ``DaxPhysicalMode.UNKNOWN`` with a ``detail``.

    Args:
        path: Candidate device path to inspect.
        sysfs_root: Root of the sysfs mount, overridable for tests.
        stat: ``stat`` implementation returning an ``os.stat_result``,
            overridable for tests that need a synthetic character device.
        clock: Wall-clock source used for ``probed_at``.

    Returns:
        The probed ``DaxPhysicalState``. ``mode`` is ``ABSENT`` when the path
        does not exist, ``NOT_A_DEVICE`` for regular files and non-dax
        character devices, ``UNBOUND``/``DEVDAX``/``SYSTEM_RAM`` for dax bus
        devices according to their bound driver, and ``UNKNOWN`` when sysfs
        cannot be read or the driver is unrecognised.
    """
    probed_at = clock()
    try:
        st = stat(path)
    except FileNotFoundError:
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.ABSENT,
            present=False,
            probed_at=probed_at,
            detail="path does not exist",
        )
    except OSError as exc:
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.UNKNOWN,
            present=False,
            probed_at=probed_at,
            detail=f"stat failed: {exc.strerror or exc}",
        )

    if not S_ISCHR(st.st_mode):
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.NOT_A_DEVICE,
            present=True,
            size_bytes=int(st.st_size) if S_ISREG(st.st_mode) else 0,
            probed_at=probed_at,
            detail="not a character device",
        )

    major = os.major(st.st_rdev)
    minor = os.minor(st.st_rdev)
    sysdir = os.path.realpath(
        os.path.join(sysfs_root, "dev", "char", f"{major}:{minor}")
    )
    if not os.path.isdir(sysdir):
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.UNKNOWN,
            present=True,
            major=major,
            minor=minor,
            probed_at=probed_at,
            detail="no sysfs entry",
        )

    kernel_name = os.path.basename(sysdir)
    subsystem_link = os.path.join(sysdir, "subsystem")
    if not os.path.exists(subsystem_link):
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.UNKNOWN,
            present=True,
            major=major,
            minor=minor,
            kernel_name=kernel_name,
            probed_at=probed_at,
            detail="no sysfs subsystem entry",
        )
    subsystem = os.path.basename(os.path.realpath(subsystem_link))
    if subsystem != _DAX_SUBSYSTEM:
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.NOT_A_DEVICE,
            present=True,
            major=major,
            minor=minor,
            kernel_name=kernel_name,
            probed_at=probed_at,
            detail=f"subsystem {subsystem}",
        )

    try:
        driver = os.path.basename(os.readlink(os.path.join(sysdir, "driver")))
    except FileNotFoundError:
        driver = ""
    except OSError as exc:
        return DaxPhysicalState(
            device_path=path,
            mode=DaxPhysicalMode.UNKNOWN,
            present=True,
            major=major,
            minor=minor,
            kernel_name=kernel_name,
            probed_at=probed_at,
            detail=f"driver link unreadable: {exc.strerror or exc}",
        )

    size_bytes = _read_sysfs_int(os.path.join(sysdir, "size"))
    align_bytes = _read_sysfs_int(os.path.join(sysdir, "align"))

    detail = ""
    if driver == "":
        mode = DaxPhysicalMode.UNBOUND
    elif driver == _DEVDAX_DRIVER:
        mode = DaxPhysicalMode.DEVDAX
    elif driver == _KMEM_DRIVER:
        mode = DaxPhysicalMode.SYSTEM_RAM
    else:
        mode = DaxPhysicalMode.UNKNOWN
        detail = f"unrecognised dax driver {driver}"

    return DaxPhysicalState(
        device_path=path,
        mode=mode,
        present=True,
        major=major,
        minor=minor,
        kernel_name=kernel_name,
        driver=driver,
        size_bytes=size_bytes,
        align_bytes=align_bytes,
        probed_at=probed_at,
        detail=detail,
    )


def scan_directory(
    directory: str,
    *,
    sysfs_root: str = "/sys",
    stat: Callable[[str], os.stat_result] = os.stat,
    clock: Callable[[], float] = time.time,
) -> list[DaxPhysicalState]:
    """Probe every non-directory entry of one directory (non-recursive).

    Entries are probed with ``probe_device`` in ascending name order, so the
    result is deterministic for a given directory content. Subdirectories are
    skipped. A missing or unreadable directory yields an empty list (logged
    at debug level); this function never raises for filesystem errors.

    Args:
        directory: Directory whose entries are candidate device paths.
        sysfs_root: Root of the sysfs mount, overridable for tests.
        stat: ``stat`` implementation forwarded to ``probe_device``.
        clock: Wall-clock source forwarded to ``probe_device``.

    Returns:
        One ``DaxPhysicalState`` per non-directory entry, sorted by entry
        name; each ``device_path`` is ``os.path.join(directory, name)``.
    """
    try:
        with os.scandir(directory) as entries:
            sorted_entries = sorted(entries, key=lambda entry: entry.name)
    except OSError as exc:
        logger.debug("DAX watch directory %s is not readable: %s", directory, exc)
        return []

    states: list[DaxPhysicalState] = []
    for entry in sorted_entries:
        is_dir = False
        try:
            is_dir = entry.is_dir()
        except OSError:
            is_dir = False
        if is_dir:
            continue
        states.append(
            probe_device(entry.path, sysfs_root=sysfs_root, stat=stat, clock=clock)
        )
    return states


class DaxScanFn(Protocol):
    """Signature of the directory scan used by ``DaxDeviceWatcher``."""

    def __call__(
        self,
        directory: str,
        *,
        sysfs_root: str,
        clock: Callable[[], float],
    ) -> list[DaxPhysicalState]:
        """Probe ``directory`` and return one state per candidate entry."""
        ...


@dataclass(frozen=True)
class DaxWatcherSnapshot:
    """Immutable result of the most recent watcher scan.

    ``last_scan_at`` is ``0.0`` and ``devices`` is empty until the first
    scan completes. ``devices`` keeps ``scan_directory`` ordering.
    """

    directory: str
    interval_seconds: float
    last_scan_at: float
    devices: tuple[DaxPhysicalState, ...]

    def find(self, device_path: str) -> DaxPhysicalState | None:
        """Return the probed state for an exact ``device_path`` match.

        This is a pure in-memory lookup and performs no I/O, so it is safe to
        call while holding the adapter's device lock.

        Args:
            device_path: Path to look up; compared byte-for-byte with the
                paths produced by the scan (``os.path.join(directory, name)``).

        Returns:
            The matching state, or ``None`` when the last scan did not list
            that path (including before the first scan).
        """
        for state in self.devices:
            if state.device_path == device_path:
                return state
        return None

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-friendly status document for an enabled watcher.

        Returns:
            ``{"enabled": True, "directory", "interval_seconds",
            "last_scan_at", "present_devices": [DaxPhysicalState.as_dict()...]}``.
        """
        return {
            "enabled": True,
            "directory": self.directory,
            "interval_seconds": self.interval_seconds,
            "last_scan_at": self.last_scan_at,
            "present_devices": [state.as_dict() for state in self.devices],
        }


class DaxDeviceWatcher:
    """Daemon thread that periodically scans one directory for DAX devices.

    The watcher only *reports*: it publishes an immutable
    ``DaxWatcherSnapshot`` after each scan and never attaches, opens, or
    otherwise touches a device. Snapshot replacement is a single attribute
    assignment under a small lock, so ``snapshot()`` never blocks on a scan
    in progress. A scan that raises is logged and the previous snapshot is
    kept.

    An empty ``directory`` creates a *disabled* watcher: ``start()`` is a
    no-op, no thread exists, ``snapshot()`` stays empty and ``status()``
    reports ``{"enabled": False}``. This lets the owning adapter always hold
    a watcher instead of an optional one.
    """

    def __init__(
        self,
        directory: str,
        interval_seconds: float,
        *,
        sysfs_root: str = "/sys",
        clock: Callable[[], float] = time.time,
        scan: DaxScanFn = scan_directory,
    ) -> None:
        """Create a watcher; call ``start()`` to begin scanning.

        Args:
            directory: Absolute directory to scan, or ``""`` for a disabled
                watcher.
            interval_seconds: Delay between the end of one scan and the start
                of the next; must be a positive finite number (``NaN`` would
                make the loop spin and ``inf`` cannot be waited on).
            sysfs_root: Root of the sysfs mount, overridable for tests.
            clock: Wall-clock source for ``last_scan_at`` and ``probed_at``.
            scan: Directory scan implementation; defaults to
                ``scan_directory`` and is overridable for tests.

        Raises:
            ValueError: If ``interval_seconds`` is not a positive finite
                number or ``directory`` is non-empty and not absolute.
        """
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive finite number")
        if directory and not os.path.isabs(directory):
            raise ValueError("directory must be an absolute path or empty")
        self._directory = directory
        self._interval_seconds = float(interval_seconds)
        self._sysfs_root = sysfs_root
        self._clock = clock
        self._scan = scan
        # Guards _snapshot and _started.
        self._lock = threading.Lock()
        self._started = False
        self._stop_flag = threading.Event()
        self._snapshot = DaxWatcherSnapshot(
            directory=directory,
            interval_seconds=self._interval_seconds,
            last_scan_at=0.0,
            devices=(),
        )
        self._thread = threading.Thread(
            target=self._run,
            name=_WATCH_THREAD_NAME,
            daemon=True,
        )

    @property
    def enabled(self) -> bool:
        """Whether this watcher has a directory to scan."""
        return bool(self._directory)

    def start(self) -> None:
        """Start the scan thread; a no-op when disabled or already started."""
        if not self.enabled:
            return
        with self._lock:
            if self._started:
                return
            self._started = True
        logger.info(
            "Starting DAX device watcher on %s every %.3fs",
            self._directory,
            self._interval_seconds,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the scan thread to exit and join it with a bounded wait.

        Safe to call before ``start()`` and more than once. If the thread is
        still alive after the bounded join a warning is logged; the thread is
        a daemon, so it cannot keep the process alive.
        """
        self._stop_flag.set()
        if not self._thread.is_alive():
            return
        self._thread.join(timeout=_STOP_JOIN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning(
                "DAX device watcher on %s did not stop within %.1fs",
                self._directory,
                _STOP_JOIN_TIMEOUT_SECONDS,
            )

    def snapshot(self) -> DaxWatcherSnapshot:
        """Return the most recently published snapshot without blocking.

        Returns:
            The latest ``DaxWatcherSnapshot``; empty (``last_scan_at == 0.0``)
            before the first scan completes or when the watcher is disabled.
        """
        with self._lock:
            return self._snapshot

    def status(self) -> dict[str, object]:
        """Return the JSON-friendly watcher status for ``hotplug_status``.

        Returns:
            ``{"enabled": False}`` for a disabled watcher, otherwise
            ``DaxWatcherSnapshot.as_dict()`` of the latest snapshot.
        """
        if not self.enabled:
            return {"enabled": False}
        return self.snapshot().as_dict()

    def _run(self) -> None:
        while not self._stop_flag.is_set():
            self._scan_once()
            self._stop_flag.wait(self._interval_seconds)

    def _scan_once(self) -> None:
        """Run one scan and publish it; keep the old snapshot on failure."""
        try:
            scanned_at = self._clock()
            devices = self._scan(
                self._directory,
                sysfs_root=self._sysfs_root,
                clock=self._clock,
            )
        except Exception:
            logger.exception(
                "DAX device watcher scan of %s failed; keeping previous snapshot",
                self._directory,
            )
            return
        snapshot = DaxWatcherSnapshot(
            directory=self._directory,
            interval_seconds=self._interval_seconds,
            last_scan_at=scanned_at,
            devices=tuple(devices),
        )
        with self._lock:
            self._snapshot = snapshot
