# SPDX-License-Identifier: Apache-2.0

"""Byte-window helpers for agentic record/replay jobs."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass


@dataclass(frozen=True)
class ByteWindow:
    index: int
    base_offset_bytes: int
    capacity_bytes: int
    meta_magic: str


def make_meta_magic(index: int, *, prefix: str = "AG") -> str:
    if index < 0 or index > 999_999:
        raise ValueError("index cannot fit in 8-byte meta_magic")
    return f"{prefix}{index:06d}"


def allocate_windows(
    *,
    count: int,
    start_offset_bytes: int,
    window_stride_bytes: int,
    capacity_bytes: int,
    block_align: int,
    meta_magic_prefix: str = "AG",
) -> list[ByteWindow]:
    windows = []
    for index in range(count):
        base = start_offset_bytes + index * window_stride_bytes
        if base % block_align:
            raise ValueError(f"base_offset_bytes={base} is not block aligned")
        if capacity_bytes % block_align:
            raise ValueError(f"capacity_bytes={capacity_bytes} is not block aligned")
        windows.append(
            ByteWindow(
                index=index,
                base_offset_bytes=base,
                capacity_bytes=capacity_bytes,
                meta_magic=make_meta_magic(index + 1, prefix=meta_magic_prefix),
            )
        )
    validate_windows(windows)
    return windows


def validate_windows(windows: list[ByteWindow]) -> None:
    ranges = sorted(
        (
            window.base_offset_bytes,
            window.base_offset_bytes + window.capacity_bytes,
            window,
        )
        for window in windows
    )
    for (_, prev_end, prev_window), (start, _, window) in zip(
        ranges,
        ranges[1:],
        strict=False,
    ):
        if start < prev_end:
            raise ValueError(
                f"byte windows overlap: {prev_window.index} and {window.index}"
            )

