# SPDX-License-Identifier: Apache-2.0
"""CPU cache maintenance around copies into and out of a shared region.

CXL 2.0 multi-host memory gives no cross-host cache coherence: a host may keep
a stale line of bytes another host just wrote, and an Intel host with DDIO may
hold DMA-written lines in its LLC instead of the device. ``software_fenced``
writes the range back after the device-to-host copy completes (publish) and
drops it before a host-to-device copy reads it (acquire). ``coherent`` does
nothing; it fits a region that only one host maps, or a coherent fabric.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported

# A batch of objects is flushed on this many threads: one 36 MB object takes
# longer to flush on one core than to copy over CXL.
_FLUSH_THREADS = 8


class Visibility(Protocol):
    """Publish/acquire hook named by the orchestrator's ``visibility_mode``."""

    @property
    def mode(self) -> str:
        """The ``visibility_mode`` this hook implements."""
        ...

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Make written ``(address, size)`` ranges visible to other hosts."""
        ...

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Drop this host's cached copies of ``(address, size)`` ranges."""
        ...

    def close(self) -> None:
        """Release worker threads."""
        ...


class SoftwareFencedVisibility:
    """CLFLUSHOPT over the range, then SFENCE, on both sides of every copy.

    Raises:
        RuntimeError: This CPU architecture has no cache-flush implementation.
    """

    def __init__(self) -> None:
        if not cache_flush_supported():
            raise RuntimeError(
                "visibility_mode software_fenced needs cache-line flushes, "
                "which this CPU architecture does not implement"
            )
        self._executor = ThreadPoolExecutor(
            max_workers=_FLUSH_THREADS, thread_name_prefix="shared-l1-flush"
        )

    @property
    def mode(self) -> str:
        """Return ``"software_fenced"``."""
        return "software_fenced"

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Write back ranges whose device-to-host copies have completed.

        Args:
            ranges: ``(address, size)`` pairs in this process.
        """
        self._flush(ranges)

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Invalidate ranges before a host-to-device copy reads them.

        Args:
            ranges: ``(address, size)`` pairs in this process.
        """
        self._flush(ranges)

    def close(self) -> None:
        """Stop the flush threads."""
        self._executor.shutdown(wait=True)

    def _flush(self, ranges: list[tuple[int, int]]) -> None:
        # Each flush ends in its own SFENCE; waiting on the futures orders
        # those fences before whatever the caller does next.
        if len(ranges) == 1:
            cache_flush_range(*ranges[0])
            return
        for future in [
            self._executor.submit(cache_flush_range, address, size)
            for address, size in ranges
        ]:
            future.result()


class CoherentVisibility:
    """No cache maintenance: hardware keeps every mapping coherent."""

    @property
    def mode(self) -> str:
        """Return ``"coherent"``."""
        return "coherent"

    def publish(self, ranges: list[tuple[int, int]]) -> None:
        """Do nothing; coherent mappings need no write-back."""

    def acquire(self, ranges: list[tuple[int, int]]) -> None:
        """Do nothing; coherent mappings hold no stale lines."""

    def close(self) -> None:
        """Do nothing; there are no threads."""


def create_visibility(mode: str) -> Visibility:
    """Build the hook for a region's ``visibility_mode``.

    Args:
        mode: ``"software_fenced"`` or ``"coherent"``.

    Returns:
        The matching hook.

    Raises:
        ValueError: The mode is unknown.
        RuntimeError: ``software_fenced`` on an unsupported CPU architecture.
    """
    if mode == "software_fenced":
        return SoftwareFencedVisibility()
    if mode == "coherent":
        return CoherentVisibility()
    raise ValueError(f"Unknown shared-L1 visibility_mode {mode!r}")
