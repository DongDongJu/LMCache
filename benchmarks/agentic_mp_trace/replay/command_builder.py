# SPDX-License-Identifier: Apache-2.0

"""Convert agentic manifests into FDP WAF replay configs."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import math

# First Party
from benchmarks.agentic_mp_trace.replay.fdp_policy import validate_replay_modes


MODE_ALIASES = {
    "no_fdp": "no_fdp",
    "fdp_mixed": "mixed",
    "fdp_separated": "separated",
}


def trace_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries = manifest.get("traces") or manifest.get("trace_manifest") or []
    if not isinstance(entries, list):
        raise ValueError("trace manifest must contain a traces list")
    return [entry for entry in entries if isinstance(entry, dict)]


def storage_class_for_entry(entry: dict[str, Any]) -> str:
    if entry.get("storage_class"):
        return str(entry["storage_class"])
    dataset = entry.get("dataset", {})
    family = str(dataset.get("adapter") or dataset.get("family") or "")
    defaults = {
        "tau_bench": "tool_agent",
        "swe_bench": "coding_agent",
        "webarena": "browser_agent",
        "mind2web": "browser_agent",
        "appworld": "tool_agent",
        "toolbench": "shared_prefix_heavy",
        "workarena": "workplace_agent",
        "gaia": "general_assistant",
    }
    return defaults.get(family, "tool_agent")


def trace_id_for_entry(entry: dict[str, Any], index: int) -> str:
    return str(entry.get("trace_id") or entry.get("name") or f"trace_{index:04d}")


def build_fdp_replay_config(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    replay_cfg = config["replay"]
    validate_replay_modes(replay_cfg)
    global_cfg = {
        "l2_store_policy": replay_cfg.get("l2_store_policy", "skip_l1"),
        "eviction_policy": replay_cfg.get("eviction_policy", "noop"),
        "disable_metrics": True,
        "quiet": True,
        "l1_align_bytes": replay_cfg.get("l1_align_bytes", 4096),
        "meta_total_bytes": replay_cfg.get(
            "default_meta_total_bytes",
            64 * 1024 * 1024,
        ),
        "use_odirect": replay_cfg.get("use_odirect", False),
        "use_uring": replay_cfg.get("use_uring", True),
        "use_uring_cmd": replay_cfg.get("use_uring_cmd", True),
        "output_root": replay_cfg.get("output_root", "/mnt/hc-ssd/agentic-replay"),
        "replay_binary": replay_cfg.get("replay_binary")
        or config.get("record", {}).get("lmcache_binary")
        or "lmcache",
    }
    fdp_config = {
        "device_path": replay_cfg.get("device_path", "/dev/ng1n1"),
        "block_device_path": replay_cfg.get("block_device_path", "/dev/nvme1n1"),
        "block_align": int(replay_cfg.get("block_align", 4096)),
        "global": global_cfg,
        "measurement": config.get("measurement", {}),
        "windows": {
            "start_offset_bytes": int(replay_cfg["start_offset_bytes"]),
            "window_stride_bytes": int(replay_cfg["window_stride_bytes"]),
            "default_capacity_bytes": int(replay_cfg["default_capacity_bytes"]),
            "auto_assign": True,
        },
        "modes": {
            "no_fdp": replay_cfg["modes"]["no_fdp"],
            "mixed": replay_cfg["modes"]["fdp_mixed"],
            "separated": replay_cfg["modes"]["fdp_separated"],
        },
        "workloads": [],
    }
    overrides = replay_cfg.get("trace_overrides", {})
    default_slot = int(replay_cfg.get("default_slot_bytes", 4 * 1024 * 1024))
    default_l1 = float(replay_cfg.get("replay_l1_size_gb", 1))
    for index, entry in enumerate(trace_entries(manifest)):
        storage_class = storage_class_for_entry(entry)
        override = overrides.get(storage_class, {})
        workload = {
            "name": trace_id_for_entry(entry, index),
            "class": storage_class,
            "trace_path": entry["trace_path"],
            "concurrency": int(entry.get("replay_concurrency", 1)),
            "slot_bytes": int(override.get("slot_bytes", default_slot)),
            "capacity_bytes": int(
                override.get(
                    "capacity_bytes",
                    replay_cfg.get("default_capacity_bytes"),
                )
            ),
            "l1_size_gb": float(override.get("l1_size_gb", default_l1)),
        }
        trace_stats = entry.get("trace_stats", {})
        if trace_stats.get("estimated_store_bytes") is not None:
            workload["estimated_store_bytes"] = int(
                trace_stats["estimated_store_bytes"]
            )
        fdp_config["workloads"].append(workload)
    return fdp_config


def estimate_iteration_bytes(fdp_config: dict[str, Any]) -> int:
    total = 0
    for workload in fdp_config.get("workloads", []):
        estimated = workload.get("estimated_store_bytes")
        if estimated is None:
            estimated = int(workload["capacity_bytes"])
        total += int(estimated) * int(workload.get("concurrency", 1))
    return max(1, total)


def target_host_write_bytes(
    *,
    fdp_config: dict[str, Any],
    replay_cfg: dict[str, Any],
) -> int:
    explicit = replay_cfg.get("min_measurement_host_write_bytes")
    if explicit is not None:
        return int(explicit)
    capacity = replay_cfg.get("test_region_capacity_bytes")
    if capacity is None:
        capacity = sum(
            int(workload["capacity_bytes"]) * int(workload.get("concurrency", 1))
            for workload in fdp_config.get("workloads", [])
        )
    return int(capacity) * int(replay_cfg.get("target_host_write_multiplier", 5))


def auto_iterations(
    *,
    requested_iterations: int,
    fdp_config: dict[str, Any],
    replay_cfg: dict[str, Any],
) -> int:
    if not bool(replay_cfg.get("auto_iterations_for_target", True)):
        return requested_iterations
    target = target_host_write_bytes(fdp_config=fdp_config, replay_cfg=replay_cfg)
    needed = math.ceil(target / estimate_iteration_bytes(fdp_config))
    return max(requested_iterations, needed)
