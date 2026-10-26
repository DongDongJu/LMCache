# SPDX-License-Identifier: Apache-2.0
"""Native fences preserve bytes; coherent mappings perform no flush."""

# Standard
from unittest.mock import Mock
import ctypes

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import cache_flush_supported
from lmcache.v1.distributed.shared_l1 import visibility
from lmcache.v1.distributed.shared_l1.visibility import Visibility


@pytest.mark.skipif(not cache_flush_supported(), reason="requires CLFLUSHOPT")
def test_native_fences_preserve_bytes() -> None:
    buffers = [ctypes.create_string_buffer(bytes([i]) * 65536) for i in range(5)]
    ranges = [(ctypes.addressof(buf), len(buf)) for buf in buffers]
    fence = Visibility("software_fenced")
    fence.publish(ranges)
    fence.acquire(ranges[:1])
    assert all(buf.raw[:-1] == bytes([i]) * 65536 for i, buf in enumerate(buffers))


def test_unsupported_and_coherent_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    flush = Mock()
    monkeypatch.setattr(visibility, "cache_flush_range", flush)
    monkeypatch.setattr(visibility, "cache_flush_supported", lambda: False)
    with pytest.raises(RuntimeError, match="software_fenced"):
        Visibility("software_fenced")
    with pytest.raises(ValueError, match="visibility_mode"):
        Visibility("unknown")
    coherent = Visibility("coherent")
    coherent.publish([(0, 64)])
    coherent.acquire([(0, 64)])
    flush.assert_not_called()
