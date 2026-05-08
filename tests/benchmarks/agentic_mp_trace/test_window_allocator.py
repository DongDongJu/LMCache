# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from benchmarks.agentic_mp_trace.replay.fdp_policy import validate_ruh_ids
from benchmarks.agentic_mp_trace.replay.window_allocator import (
    allocate_windows,
    validate_windows,
)


def test_allocate_windows_and_detect_overlap():
    windows = allocate_windows(
        count=2,
        start_offset_bytes=4096,
        window_stride_bytes=8192,
        capacity_bytes=4096,
        block_align=4096,
    )
    assert windows[0].meta_magic == "AG000001"
    validate_windows(windows)

    overlap = allocate_windows(
        count=2,
        start_offset_bytes=4096,
        window_stride_bytes=4096,
        capacity_bytes=4096,
        block_align=4096,
    )
    validate_windows(overlap)


def test_ruh_count_validation():
    assert validate_ruh_ids([0, 1, 2, 3], ruh_count=4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        validate_ruh_ids([4], ruh_count=4)

