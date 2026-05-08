# SPDX-License-Identifier: Apache-2.0

"""Build existing ``lmcache server`` commands for trace recording."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import json
import shlex


def as_command_prefix(value: Any, default: str) -> list[str]:
    if value is None:
        return [default]
    if isinstance(value, list):
        return [str(item) for item in value]
    return shlex.split(str(value))


def build_record_l2_adapter(
    *,
    device_path: str,
    slot_bytes: int,
    base_offset_bytes: int,
    capacity_bytes: int,
    meta_total_bytes: int,
    meta_magic: str,
    block_align: int,
    use_odirect: bool,
    use_uring: bool,
    use_uring_cmd: bool,
    use_fdp: bool,
    fdp_data_ruh_ids: list[int],
    fdp_metadata_ruh_ids: list[int],
) -> dict[str, Any]:
    adapter: dict[str, Any] = {
        "type": "raw_block",
        "device_path": device_path,
        "slot_bytes": int(slot_bytes),
        "base_offset_bytes": int(base_offset_bytes),
        "capacity_bytes": int(capacity_bytes),
        "block_align": int(block_align),
        "meta_total_bytes": int(meta_total_bytes),
        "meta_magic": meta_magic,
        "use_odirect": bool(use_odirect),
        "use_uring": bool(use_uring),
        "use_uring_cmd": bool(use_uring_cmd),
        "use_fdp": bool(use_fdp),
    }
    if use_fdp:
        adapter["fdp_data_ruh_ids"] = list(fdp_data_ruh_ids)
        adapter["fdp_metadata_ruh_ids"] = list(fdp_metadata_ruh_ids)
    return adapter


def build_lmcache_server_command(
    *,
    binary: Any,
    mp_port: int,
    http_port: int,
    l1_size_gb: float,
    eviction_policy: str,
    l2_store_policy: str,
    chunk_size: int,
    trace_output: str,
    l2_adapter: dict[str, Any],
    max_workers: int = 4,
    l1_align_bytes: int = 4096,
    disable_metrics: bool = True,
) -> list[str]:
    cmd = as_command_prefix(binary, "lmcache")
    cmd.extend(
        [
            "server",
            "--host",
            "localhost",
            "--port",
            str(mp_port),
            "--http-host",
            "127.0.0.1",
            "--http-port",
            str(http_port),
            "--chunk-size",
            str(chunk_size),
            "--max-workers",
            str(max_workers),
            "--l1-size-gb",
            str(l1_size_gb),
            "--l1-align-bytes",
            str(l1_align_bytes),
            "--eviction-policy",
            eviction_policy,
            "--l2-store-policy",
            l2_store_policy,
            "--trace-level",
            "storage",
            "--trace-output",
            trace_output,
            "--l2-adapter",
            json.dumps(l2_adapter, separators=(",", ":")),
        ]
    )
    if disable_metrics:
        cmd.append("--disable-metrics")
    return cmd

