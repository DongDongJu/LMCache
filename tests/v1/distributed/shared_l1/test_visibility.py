# SPDX-License-Identifier: Apache-2.0
"""Visibility hooks and the native cache-line flush they use."""

# Standard
import ctypes
import platform

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import cache_flush_range, cache_flush_supported
from lmcache.v1.distributed.shared_l1.visibility import (
    CoherentVisibility,
    SoftwareFencedVisibility,
    create_visibility,
)

x86_only = pytest.mark.skipif(
    platform.machine() != "x86_64", reason="cache flush is implemented for x86-64"
)


@x86_only
def test_flush_supports_unaligned_and_empty_ranges() -> None:
    assert cache_flush_supported()
    buffer = ctypes.create_string_buffer(b"\x5a" * 10_000)
    base = ctypes.addressof(buffer)
    cache_flush_range(base, len(buffer))
    cache_flush_range(base + 3, 61)  # inside one line, unaligned
    cache_flush_range(base + 7, 0)
    # Flushing writes back and drops lines; it never changes bytes.
    assert buffer.raw[:-1] == b"\x5a" * 10_000


@x86_only
def test_software_fenced_flushes_batches_without_changing_bytes() -> None:
    hook = create_visibility("software_fenced")
    assert isinstance(hook, SoftwareFencedVisibility)
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


def test_coherent_needs_no_maintenance() -> None:
    hook = create_visibility("coherent")
    assert isinstance(hook, CoherentVisibility)
    hook.publish([(0, 64)])  # never dereferenced
    hook.acquire([(0, 64)])
    hook.close()
    assert hook.mode == "coherent"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="visibility_mode"):
        create_visibility("hope")
