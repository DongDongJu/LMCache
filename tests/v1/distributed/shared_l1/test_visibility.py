# SPDX-License-Identifier: Apache-2.0
"""The visibility fences and the native cache-line flush they use."""

# Standard
import ctypes

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported
from lmcache.v1.distributed.shared_l1 import visibility
from lmcache.v1.distributed.shared_l1.visibility import Visibility

flush_only = pytest.mark.skipif(
    not cache_flush_supported(), reason="needs the native x86-64 cache flush"
)


@flush_only
def test_flush_supports_unaligned_and_empty_ranges() -> None:
    buffer = ctypes.create_string_buffer(b"\x5a" * 10_000)
    base = ctypes.addressof(buffer)
    cache_flush_range(base, len(buffer))
    cache_flush_range(base + 3, 61)  # inside one line, unaligned
    cache_flush_range(base + 7, 0)
    # Flushing writes back and drops lines; it never changes bytes.
    assert buffer.raw[:-1] == b"\x5a" * 10_000


@flush_only
def test_software_fenced_flushes_batches_without_changing_bytes() -> None:
    hook = Visibility("software_fenced")
    buffers = [ctypes.create_string_buffer(bytes([i]) * 65536) for i in range(5)]
    ranges = [(ctypes.addressof(b), len(b)) for b in buffers]
    try:
        hook.publish(ranges)
        hook.acquire(ranges[:1])
        assert hook.mode == "software_fenced"
    finally:
        hook.close()
    for i, buffer in enumerate(buffers):
        assert buffer.raw[:-1] == bytes([i]) * 65536


def test_software_fenced_needs_the_native_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(visibility, "cache_flush_supported", lambda: False)
    with pytest.raises(RuntimeError, match="software_fenced"):
        Visibility("software_fenced")
    # Coherent needs no flush support and never flushes: address 0 is not read.
    coherent = Visibility("coherent")
    coherent.publish([(0, 64)])
    coherent.acquire([(0, 64)])
    coherent.close()


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="visibility_mode"):
        Visibility("hope")
