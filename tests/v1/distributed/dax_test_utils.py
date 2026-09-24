# SPDX-License-Identifier: Apache-2.0
"""Isolated file fixtures and strict CXL Device-DAX preflight (no torch imports)."""

# Standard
from pathlib import Path
import json
import math
import mmap
import os
import stat


def inspect_device(path: str) -> dict:
    """Return sysfs identity, capacity and alignment for a CXL devdax path.

    Args:
        path: Character-device path, including a possible symlink alias.
    Returns:
        JSON-compatible identity and topology record.
    Raises:
        ValueError: If the path is not CXL-backed Device-DAX in devdax mode.
        OSError: If the path or required sysfs attributes cannot be read.
    """
    device = Path(path).resolve(strict=True)
    info = device.stat()
    if not stat.S_ISCHR(info.st_mode):
        raise ValueError(f"{path}: expected a DAX character device")
    major, minor = os.major(info.st_rdev), os.minor(info.st_rdev)
    sysfs = Path(f"/sys/dev/char/{major}:{minor}").resolve(strict=True)
    dax = Path("/sys/bus/dax/devices") / sysfs.name
    if not dax.exists() or dax.resolve() != sysfs:
        raise ValueError(f"{path}: not a sysfs DAX device")
    driver = (dax / "driver").resolve(strict=True).name
    regions = [
        p for p in sysfs.parents if (Path("/sys/bus/cxl/devices") / p.name).exists()
    ]
    if driver != "device_dax" or not regions:
        raise ValueError(f"{path}: expected CXL parent and device_dax driver")
    return {
        "path": str(device),
        "major": major,
        "minor": minor,
        "size": int((dax / "size").read_text(), 0),
        "alignment": int((dax / "align").read_text(), 0),
        "mode": "devdax",
        "driver": driver,
        "parent": str(regions[0]),
        "sysfs": str(sysfs),
    }


class DeviceProvider:
    """Provide serial, preflighted mappings or files owned by pytest's tmp_path."""

    def __init__(self, workspace: Path, suite: str, count: int = 3) -> None:
        """Validate all supplied paths before any mmap writes.

        Args:
            workspace: Unique per-test directory for ordinary file fixtures.
            suite: ``l1`` or ``l2``; chooses the existing path environment input.
            count: Minimum number of independently usable devices.
        Raises:
            ValueError: For missing, duplicate, insufficient or invalid devices.
            OSError: For inaccessible devices or manifests.
        """
        self.workspace = workspace
        self.index = 0
        strict = os.environ.get("LMCACHE_TEST_REQUIRE_REAL_DEVDAX", "0")
        if strict not in ("0", "1"):
            raise ValueError("LMCACHE_TEST_REQUIRE_REAL_DEVDAX must be 0 or 1")
        self.strict = strict == "1"
        if self.strict and os.environ.get("PYTEST_XDIST_WORKER"):
            raise ValueError("real Device-DAX fixtures must run serially")
        env = f"LMCACHE_TEST_DEVDAX_{suite.upper()}_PATHS"
        self.paths = [
            p.strip() for p in os.environ.get(env, "").split(",") if p.strip()
        ]
        self.records = [inspect_device(p) for p in self.paths]
        identities = {(r["major"], r["minor"]) for r in self.records}
        if len(identities) != len(self.paths):
            raise ValueError("duplicate Device-DAX devices (including aliases)")
        if (self.strict or self.paths) and len(self.paths) < count:
            raise ValueError(f"{env}: requires at least {count} Device-DAX devices")
        if self.strict:
            if not os.environ.get("LMCACHE_TEST_DEVDAX_MANIFEST"):
                raise ValueError(
                    "LMCACHE_TEST_DEVDAX_MANIFEST is required in strict mode"
                )
            manifest = json.loads(
                Path(os.environ["LMCACHE_TEST_DEVDAX_MANIFEST"]).read_text()
            )
            for record in self.records:
                if record not in manifest:
                    raise ValueError(f"device does not match guest manifest: {record}")
        requested = int(
            os.environ.get("LMCACHE_TEST_DEVDAX_L1_SLOT_BYTES", str(2 << 20))
        )
        if requested <= 0:
            raise ValueError("slot size must be positive")
        self.slot_bytes = math.lcm(
            requested, mmap.PAGESIZE, *(r["alignment"] for r in self.records)
        )
        self.backing = "cxl-devdax" if self.paths else "file"
        print(f"DAX fixture: backing={self.backing}, slot_bytes={self.slot_bytes}")

    def acquire(self, size_in_bytes: int) -> str:
        """Return a fresh path and probe an aligned shared mapping over its size.

        Args:
            size_in_bytes: Positive mapping length, a multiple of slot_bytes.
        Returns:
            Device path or an isolated file path.
        Raises:
            ValueError: For exhausted devices, insufficient capacity or alignment.
            OSError: If creation or mmap fails. Character devices are never truncated.
        """
        if size_in_bytes <= 0 or size_in_bytes % self.slot_bytes:
            raise ValueError("mapping length must be a positive multiple of slot_bytes")
        if self.paths:
            if self.index >= len(self.paths):
                raise ValueError("not enough Device-DAX devices")
            record = self.records[self.index]
            if record["size"] < size_in_bytes:
                raise ValueError("insufficient Device-DAX capacity")
            path = record["path"]
        else:
            path = str(self.workspace / f"dax-{self.index}")
            with open(path, "xb") as output:
                output.truncate(size_in_bytes)
        self.index += 1
        with open(path, "r+b", buffering=0) as device:
            with mmap.mmap(
                device.fileno(), size_in_bytes, flags=mmap.MAP_SHARED
            ) as view:
                payload = bytes(range(256)) * 16
                view[: len(payload)] = payload
                if view[: len(payload)] != payload:
                    raise ValueError("Device-DAX mmap byte round trip failed")
        return path
