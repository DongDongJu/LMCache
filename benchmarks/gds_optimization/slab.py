# SPDX-License-Identifier: Apache-2.0
"""Private regular-file allocation, full preparation and exclusive reuse lease."""

# Standard
from dataclasses import dataclass
from pathlib import Path
import fcntl
import hashlib
import json
import os
import stat
import time

# Third Party
import numpy as np

CHUNK_BYTES = 16 << 20


def pattern_bytes(offset: int, size: int) -> bytes:
    """Return nonzero splitmix64 bytes at an eight-byte-aligned file range.

    Args:
        offset: Absolute aligned byte offset, with preparation seed zero.
        size: Aligned number of bytes to generate.
    """
    words = np.arange(offset // 8, (offset + size) // 8, dtype=np.uint64)
    words += np.uint64(0x9E3779B97F4A7C15)
    words = (words ^ (words >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    words = (words ^ (words >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    words = (words ^ (words >> np.uint64(31))) | np.uint64(0x0101010101010101)
    return words.astype("<u8", copy=False).tobytes()


def save_json(path: Path, value: object) -> None:
    """Atomically write a controller-owned JSON artifact, preserving newlines."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


@dataclass
class SlabLease:
    """Hold a nonblocking exclusive flock through every owned server's lifetime.

    Use create/open_existing constructors and close only after server teardown.
    A marker proves this helper prepared the inode; every reopen checks identity.
    """

    path: Path
    fd: int
    size: int
    device: int
    inode: int

    @classmethod
    def create(cls, directory: Path, size: int, report: Path) -> "SlabLease":
        """Allocate and fully prepare a new private slab, outside MP timing.

        Args:
            directory: Fresh directory under the explicit scratch root.
            size: Positive, 4 KiB-aligned capacity in bytes.
            report: New preparation report; failed preparation is retained.

        Raises:
            OSError: Allocation, locking or I/O fails.
            ValueError: Capacity is invalid.
            RuntimeError: A short transfer or full-byte comparison fails.
        """
        if size <= 0 or size % 4096:
            raise ValueError("Slab capacity must be positive and 4 KiB aligned")
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        path = directory / "lmcache_gds_slab.bin"
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.posix_fallocate(fd, 0, size)
            state = os.fstat(fd)
            lease = cls(path, fd, size, state.st_dev, state.st_ino)
            lease.prepare(report)
            save_json(lease.marker, {**lease.identity(), "format": 1, "prepared": True})
            return lease
        except BaseException:
            os.close(fd)
            raise

    @classmethod
    def open_existing(cls, path: Path, scratch_root: Path, size: int) -> "SlabLease":
        """Lock a previously prepared owned inode, rejecting foreign/live slabs.

        Args:
            path: Exact regular slab path created by this package.
            scratch_root: Explicit root containing its private parent directory.
            size: Expected capacity matching the benchmark configuration.

        Raises:
            ValueError: Path, ownership or preparation identity does not match.
            BlockingIOError: Another benchmark owns the file lease.
            OSError: File or ownership evidence cannot be read.

        The server inherits this locked descriptor, so an orphaned server keeps
        reuse blocked even if its controller exits unexpectedly.
        """
        path = path.absolute()
        if path.is_symlink() or path.resolve() != path:
            raise ValueError("Reuse does not accept symlink slab paths")
        if not path.is_relative_to(scratch_root.resolve()):
            raise ValueError("Reuse slab is outside the explicit scratch root")
        state = path.lstat()
        if not stat.S_ISREG(state.st_mode):
            raise ValueError("Reuse requires a regular file")
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease = cls(path, fd, size, state.st_dev, state.st_ino)
            lease.validate()
            if not stat.S_ISREG(lease.marker.lstat().st_mode):
                raise ValueError("Preparation marker must be a regular file")
            expected = {**lease.identity(), "format": 1, "prepared": True}
            if json.loads(lease.marker.read_text()) != expected:
                raise ValueError("Slab preparation identity changed")
            return lease
        except BaseException:
            os.close(fd)
            raise

    @property
    def marker(self) -> Path:
        """Return the benchmark-only inode preparation marker path."""
        return self.path.with_suffix(".benchmark.json")

    def identity(self) -> dict[str, int | str]:
        """Return the identity required by the benchmark server's reopen hook."""
        return {
            "path": str(self.path),
            "size": self.size,
            "device": self.device,
            "inode": self.inode,
        }

    def validate(self) -> None:
        """Reject replaced, resized, linked, nonprivate or unowned allocations."""
        parent = self.path.parent.lstat()
        if (
            not stat.S_ISDIR(parent.st_mode)
            or parent.st_uid != os.getuid()
            or stat.S_IMODE(parent.st_mode) & 0o077
        ):
            raise ValueError("Slab parent must be an owned private directory")
        for state in (self.path.lstat(), os.fstat(self.fd)):
            if (
                not stat.S_ISREG(state.st_mode)
                or state.st_uid != os.getuid()
                or state.st_nlink != 1
                or (state.st_dev, state.st_ino, state.st_size)
                != (self.device, self.inode, self.size)
            ):
                raise ValueError("Slab identity, size or ownership changed")

    def prepare(self, report: Path) -> None:
        """Write, fsync and compare every byte; never part of timed durability.

        The readback is a buffered filesystem coherency check, not independent
        media-read proof. All extents are explicitly written before direct I/O.
        """
        started = time.monotonic_ns()
        written = verified = 0
        expected_hash, actual_hash = hashlib.sha256(), hashlib.sha256()
        result: dict[str, object] = {
            **self.identity(),
            "status": "running",
            "seed": 0,
            "pattern": "splitmix64_absolute_word_generation_v1",
            "outside_measurement": True,
            "verification": "full buffered read after fsync, byte compare and SHA256",
        }
        try:
            for offset in range(0, self.size, CHUNK_BYTES):
                self.validate()
                data = pattern_bytes(offset, min(CHUNK_BYTES, self.size - offset))
                count = os.pwrite(self.fd, data, offset)
                if count != len(data):
                    raise RuntimeError(f"Short preparation write at {offset}: {count}")
                written += count
                expected_hash.update(data)
            os.fsync(self.fd)
            for offset in range(0, self.size, CHUNK_BYTES):
                self.validate()
                size = min(CHUNK_BYTES, self.size - offset)
                data = os.pread(self.fd, size, offset)
                if data != pattern_bytes(offset, size):
                    raise RuntimeError(
                        f"Full preparation readback mismatch at {offset}"
                    )
                verified += len(data)
                actual_hash.update(data)
            result["status"] = "passed"
        except (OSError, ValueError, RuntimeError) as error:
            result.update(status="failed", error=str(error))
            raise
        finally:
            result.update(
                written_bytes=written,
                verified_bytes=verified,
                expected_sha256=expected_hash.hexdigest(),
                actual_sha256=actual_hash.hexdigest(),
                total_seconds=(time.monotonic_ns() - started) / 1e9,
            )
            save_json(report, result)

    def close(self) -> None:
        """Release the exclusive lease after all owned server processes exit."""
        os.close(self.fd)
