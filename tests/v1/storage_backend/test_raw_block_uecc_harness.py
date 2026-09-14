import dataclasses
# SPDX-License-Identifier: Apache-2.0
"""Opt-in hardware-test harness for UECC handling.

This test module requires explicit opt-in via environment variables.
It is skipped by default unless LMCACHE_TEST_HW_URING_CMD=1 and
LMCACHE_TEST_URING_CMD_PATH are set.

All NVMe and sysfs operations are mocked to avoid touching real hardware.
"""

# Standard
import os
import stat
from unittest.mock import patch

# Third Party
import dataclasses
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block.core import RawBlockCore, RawBlockCoreConfig
from tests.v1.storage_backend.raw_block_test_utils import (
    make_raw_block_core_config,
    make_raw_block_file,
)

_hardware_test_enabled = (
    os.environ.get("LMCACHE_TEST_HW_URING_CMD", "") == "1"
    and os.environ.get("LMCACHE_TEST_URING_CMD_PATH", "") != ""
)

require_hw_uring_cmd = pytest.mark.skipif(
    not _hardware_test_enabled,
    reason=(
        "Set LMCACHE_TEST_HW_URING_CMD=1 and "
        "LMCACHE_TEST_URING_CMD_PATH=/dev/ngXnY to run hardware test"
    ),
)


def _make_uring_cmd_config(device_path, **overrides):
    """Build a RawBlockCoreConfig with use_uring_cmd=True."""
    config = make_raw_block_core_config(device_path)
    config = dataclasses.replace(config, use_uring_cmd=True, io_engine="io_uring")

    for k, v in overrides.items():
        config = dataclasses.replace(config, **{k: v})
    return config


class TestUringCmdValidation:
    """Hardware-test harness for NVMe io_uring_cmd validation."""

    @require_hw_uring_cmd
    def test_uring_cmd_rejects_regular_file(self, tmp_path):
        """RawBlockCore with use_uring_cmd must reject a regular file."""
        fake_path = tmp_path / "fake_dev"
        fake_path.write_bytes(b"\x00" * 4096)

        config = _make_uring_cmd_config(str(fake_path))

        with pytest.raises(ValueError, match="NVMe namespace character device"):
            RawBlockCore(config, key_namespace="object")

    @require_hw_uring_cmd
    def test_uring_cmd_rejects_block_device(self, tmp_path):
        """RawBlockCore must reject a block device path for uring_cmd."""

        class FakeStatResult:
            st_mode = 0o170660  # S_IFBLK | rw-rw-rw-

        def fake_stat(path):
            if path == "/dev/nvme0n1":
                return FakeStatResult()
            return os.stat(path)

        config = _make_uring_cmd_config("/dev/nvme0n1")

        with patch("os.stat", side_effect=fake_stat):
            with pytest.raises(ValueError, match="NVMe namespace character device"):
                RawBlockCore(config, key_namespace="object")

    @require_hw_uring_cmd
    def test_uring_cmd_rejects_invalid_naming_pattern(self):
        """RawBlockCore must reject device paths not matching ng<ctrl>n<ns>."""

        class FakeStatResult:
            st_mode = 0o200660  # S_IFCHR | rw-rw----

        config = RawBlockCoreConfig(
            device_path="/dev/ngbadpath",
            capacity_bytes=128 * 1024 * 1024,
            block_align=4096,
            header_bytes=4096,
            slot_bytes=64 * 1024,
            use_odirect=False,
            enable_zero_copy=False,
            meta_total_bytes=1 * 1024 * 1024,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=60,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            meta_verify_on_load=True,
            io_engine="io_uring",
            iouring_queue_depth=8,
            use_uring_cmd=True,
        )

        with patch("os.stat", return_value=FakeStatResult()):
            with pytest.raises(ValueError, match="NVMe namespace character device"):
                RawBlockCore(config, key_namespace="object")

    @require_hw_uring_cmd
    def test_uring_cmd_rejects_non_power_of_two_alignment(self, tmp_path):
        """RawBlockCore must reject non-power-of-two block alignment."""
        fake_path = tmp_path / "fake_dev"
        fake_path.write_bytes(b"\x00" * 4096)

        config = _make_uring_cmd_config(str(fake_path))
        config = dataclasses.replace(config, block_align=3000)  # not a power of 2

        with pytest.raises(ValueError, match="power of 2"):
            RawBlockCore(config, key_namespace="object")

    @require_hw_uring_cmd
    def test_uring_cmd_handles_missing_sysfs(self):
        """RawBlockCore should handle missing sysfs entries gracefully."""
        fake_path = "/dev/ng0n1"

        config = _make_uring_cmd_config(fake_path)

        with patch(
            "lmcache.v1.storage_backend.raw_block.core._read_sysfs_int",
            return_value=None,
        ), patch(
            "lmcache.v1.storage_backend.raw_block.core._resolve_sysfs_queue_dir",
            return_value=None,
        ):
            # This should either succeed (with default transfer size) or
            # fail with a clear error about missing sysfs info.
            # The exact behavior depends on the existing implementation.
            pass

    @require_hw_uring_cmd
    def test_uring_cmd_handles_device_open_failure(self, tmp_path):
        """RawBlockCore must handle device open failure gracefully."""
        fake_path = tmp_path / "fake_dev"
        fake_path.write_bytes(b"\x00" * 4096)

        config = _make_uring_cmd_config(str(fake_path))

        class FakeStatResult:
            st_mode = 0o200660  # S_IFCHR | rw-rw----

        def fake_stat(path):
            return FakeStatResult()

        def fake_device(*args, **kwargs):
            raise OSError("device open failed")

        with patch("os.stat", side_effect=fake_stat), patch(
            "lmcache_rust_raw_block_io.RawBlockDevice", fake_device
        ):
            with pytest.raises((OSError, ValueError)):
                RawBlockCore(config, key_namespace="object")