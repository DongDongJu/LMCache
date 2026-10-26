# SPDX-License-Identifier: Apache-2.0
"""CPU cache maintenance around copies into and out of a shared region.

CXL 2.0 multi-host memory gives no cross-host cache coherence: a host may keep
a stale line of bytes another host just wrote, and an Intel host with DDIO may
hold DMA-written lines in its LLC instead of the device. ``software_fenced``
writes the range back after the device-to-host copy completes (publish) and
drops it before a host-to-device copy reads it (acquire). ``coherent`` does
nothing; it fits a region that only one host maps, or a coherent fabric.
"""

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported
from lmcache.v1.memory_orchestrator.api import VISIBILITY_MODES


class Visibility:
    """Publish/acquire fences for a region's ``visibility_mode``.

    Args:
        mode: ``"software_fenced"`` (CLFLUSHOPT over the range, then SFENCE, on
            both sides of every copy) or ``"coherent"`` (no maintenance).

    Raises:
        ValueError: The mode is unknown.
        RuntimeError: ``software_fenced`` on a CPU without CLFLUSHOPT.
    """

    def __init__(self, mode: str) -> None:
        if mode not in VISIBILITY_MODES:
            raise ValueError(f"Unknown shared-L1 visibility_mode {mode!r}")
        if mode == "software_fenced" and not cache_flush_supported():
            raise RuntimeError(
                "visibility_mode software_fenced needs CLFLUSHOPT, which this CPU "
                "does not implement"
            )
        self.mode = mode

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Make written ``(address, size)`` ranges visible to other hosts.

        Args:
            ranges: ``(address, size)`` pairs in this process whose
                device-to-host copies have completed.
        """
        self._flush(ranges)

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Drop this host's cached lines of ``(address, size)`` ranges.

        Args:
            ranges: ``(address, size)`` pairs a host-to-device copy will read.
        """
        self._flush(ranges)

    def _flush(self, ranges: list[tuple[int, int]]) -> None:
        # Each flush ends in an SFENCE, which orders it before whatever the
        # caller does next.
        if self.mode == "software_fenced":
            for address, size in ranges:
                cache_flush_range(address, size)
