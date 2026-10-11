# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only reopening of the controller's initialized, leased allocation."""

# Standard
from pathlib import Path
from unittest.mock import patch
import json
import os
import runpy
import stat


def main() -> None:
    """Verify source/inode selection before running the public production server.

    The controller holds the advisory slab lease until the server exits. This
    hook applies equally to original and candidate checkouts and only replaces
    initial truncation; all allocation, transfer and RPC code is unchanged.
    """
    # First Party
    from lmcache.v1.gpu_connector.gds_backends._file import FileGDSBackend
    from lmcache.v1.gpu_connector.gds_backends.base import GDSHandle
    import lmcache

    source = Path(os.environ["GDS_BENCH_SOURCE"])
    if not Path(lmcache.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Imported LMCache differs from the selected checkout")
    expected = json.loads(os.environ["GDS_BENCH_SLAB_IDENTITY"])

    def reuse_slab(
        self: FileGDSBackend, location: str, size: int, direct_io: bool
    ) -> GDSHandle:
        """Open exactly the controller-owned regular inode without truncation."""
        path = Path(location) / "lmcache_gds_slab.bin"
        state = path.lstat()
        actual = {
            "path": str(path),
            "size": size,
            "device": state.st_dev,
            "inode": state.st_ino,
        }
        if (
            actual != expected
            or state.st_size != size
            or not stat.S_ISREG(state.st_mode)
        ):
            raise RuntimeError("Benchmark slab identity changed before registration")
        # cuFile can reject O_NOFOLLOW. Preserve its supported flags while proving
        # both the pathname and opened descriptor match the leased private inode.
        flags = os.O_RDWR | (os.O_DIRECT if direct_io else 0)
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            state.st_dev,
            state.st_ino,
            size,
        ):
            os.close(fd)
            raise RuntimeError("Benchmark slab changed during open")
        return self.open_handle(fd, str(path))

    with patch.object(FileGDSBackend, "open_slab", reuse_slab):
        module = (
            "benchmarks.gds_optimization.diagnostic_server"
            if "GDS_DIAGNOSTIC_EVENTS" in os.environ
            else "lmcache.cli.main"
        )
        runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
