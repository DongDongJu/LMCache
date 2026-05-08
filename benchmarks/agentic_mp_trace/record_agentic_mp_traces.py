# SPDX-License-Identifier: Apache-2.0

"""Record real vLLM + LMCache MP storage traces from agentic workloads."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

# Third Party
import httpx  # noqa: E402

# First Party
from benchmarks.agentic_mp_trace.config import (  # noqa: E402
    load_yaml_config,
    write_json,
    write_text,
    write_yaml,
)
from benchmarks.agentic_mp_trace.datasets import AgenticRequest, get_adapter  # noqa: E402
from benchmarks.agentic_mp_trace.launchers.lmcache_server import (  # noqa: E402
    build_lmcache_server_command,
    build_record_l2_adapter,
)
from benchmarks.agentic_mp_trace.launchers.openai_client import (  # noqa: E402
    drive_requests,
)
from benchmarks.agentic_mp_trace.launchers.vllm_server import (  # noqa: E402
    build_vllm_command,
)
from benchmarks.agentic_mp_trace.replay.fdp_policy import (  # noqa: E402
    validate_ruh_ids,
)
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (  # noqa: E402
    analyze_trace_footprint,
    command_to_text,
    expand_ruh_ids,
)


def _wait_http(url: str, *, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
            if response.status_code < 500:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"timed out waiting for {url}: {last_error}")


def _terminate(proc: subprocess.Popen, *, timeout_s: int) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _record_jobs(config: dict[str, Any], suite_name: str) -> list[dict[str, Any]]:
    suite = config.get("record_suites", {}).get(suite_name)
    if not isinstance(suite, dict):
        raise ValueError(f"unknown record suite: {suite_name}")
    catalog = config["dataset_catalog"]
    profiles = config["model_profiles"]
    jobs = []
    for job in suite.get("jobs", []):
        dataset_ref = job["dataset_ref"]
        profile_name = job["model_profile"]
        merged = dict(job)
        merged["catalog"] = catalog[dataset_ref]
        merged["model"] = profiles[profile_name]
        merged["dataset_ref"] = dataset_ref
        merged["model_profile"] = profile_name
        merged["path"] = job.get("path") or catalog[dataset_ref].get("local_path")
        jobs.append(merged)
    return jobs


def _adapter_requests(
    *,
    job: dict[str, Any],
    strict_dataset_access: bool,
) -> tuple[list[AgenticRequest], list[dict[str, Any]], list[str], bool]:
    catalog_entry = job["catalog"]
    adapter = get_adapter(str(catalog_entry["adapter"]))
    result = adapter.load_requests(
        dataset_name=str(job["dataset_ref"]),
        catalog_entry=catalog_entry,
        job=job,
        strict_dataset_access=strict_dataset_access,
    )
    return (
        result.requests,
        [request.to_dict() for request in result.requests],
        result.warnings,
        result.skipped,
    )


def _job_ports(record_cfg: dict[str, Any], index: int) -> tuple[int, int, int]:
    base_mp = int(record_cfg.get("base_mp_port", 5555))
    base_http = int(record_cfg.get("base_http_port", 8080))
    base_vllm = int(record_cfg.get("base_vllm_port", 8000))
    stride = int(record_cfg.get("port_stride", 10))
    return base_mp + index * stride, base_http + index * stride, base_vllm + index


def _job_window(record_cfg: dict[str, Any], index: int) -> tuple[int, int, str]:
    windows = record_cfg["windows"]
    base = int(windows["start_offset_bytes"]) + index * int(
        windows["window_stride_bytes"]
    )
    capacity = int(windows["default_capacity_bytes"])
    return base, capacity, f"AR{index + 1:06d}"


def _build_job_commands(
    *,
    config: dict[str, Any],
    job: dict[str, Any],
    index: int,
    output_dir: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    record_cfg = config["record"]
    replay_cfg = config["replay"]
    model = job["model"]
    mp_port, http_port, vllm_port = _job_ports(record_cfg, index)
    base_offset, capacity, meta_magic = _job_window(record_cfg, index)
    ruh_count = int(replay_cfg.get("ruh_count", 4))
    use_fdp = bool(record_cfg.get("use_fdp", True))
    data_ruhs = validate_ruh_ids(
        expand_ruh_ids(record_cfg.get("fdp_data_ruhs", [0, 1, 2])),
        ruh_count=ruh_count,
    )
    metadata_ruhs = validate_ruh_ids(
        expand_ruh_ids(record_cfg.get("fdp_metadata_ruhs", [3])),
        ruh_count=ruh_count,
    )
    trace_path = os.path.join(output_dir, "traces", f"{job['name']}.lct")
    adapter = build_record_l2_adapter(
        device_path=replay_cfg.get("device_path", "/dev/ng1n1"),
        slot_bytes=int(record_cfg.get("slot_bytes", replay_cfg["default_slot_bytes"])),
        base_offset_bytes=base_offset,
        capacity_bytes=capacity,
        meta_total_bytes=int(
            record_cfg.get(
                "meta_total_bytes",
                replay_cfg.get("default_meta_total_bytes", 64 * 1024 * 1024),
            )
        ),
        meta_magic=meta_magic,
        block_align=int(replay_cfg.get("block_align", 4096)),
        use_odirect=bool(replay_cfg.get("use_odirect", False)),
        use_uring=bool(replay_cfg.get("use_uring", True)),
        use_uring_cmd=bool(replay_cfg.get("use_uring_cmd", True)),
        use_fdp=use_fdp,
        fdp_data_ruh_ids=data_ruhs,
        fdp_metadata_ruh_ids=metadata_ruhs,
    )
    lmcache_cmd = build_lmcache_server_command(
        binary=record_cfg.get("lmcache_binary", "lmcache"),
        mp_port=mp_port,
        http_port=http_port,
        l1_size_gb=float(model.get("record_l1_size_gb", 1)),
        eviction_policy=str(model.get("record_eviction_policy", "noop")),
        l2_store_policy=str(record_cfg.get("l2_store_policy", "skip_l1")),
        chunk_size=int(model["lmcache_chunk_size"]),
        trace_output=trace_path,
        l2_adapter=adapter,
        max_workers=int(record_cfg.get("max_workers", 4)),
        l1_align_bytes=int(replay_cfg.get("l1_align_bytes", 4096)),
        disable_metrics=bool(record_cfg.get("disable_metrics", True)),
    )
    vllm_cmd = build_vllm_command(
        binary=record_cfg.get("vllm_binary", "vllm"),
        model_id=str(model["model_id"]),
        vllm_port=vllm_port,
        mp_port=mp_port,
        max_model_len=int(model["max_model_len"]),
        tensor_parallel_size=int(model["tensor_parallel_size"]),
        gpu_memory_utilization=float(model["gpu_memory_utilization"]),
    )
    metadata = {
        "trace_path": trace_path,
        "mp_port": mp_port,
        "http_port": http_port,
        "vllm_port": vllm_port,
        "base_offset_bytes": base_offset,
        "capacity_bytes": capacity,
        "meta_magic": meta_magic,
        "l2_adapter": adapter,
    }
    return lmcache_cmd, vllm_cmd, metadata


def _trace_stats(trace_path: str) -> dict[str, Any]:
    footprint = analyze_trace_footprint(trace_path)
    return {
        "total_records": footprint.record_count,
        "reserve_write_count": footprint.store_count,
        "finish_write_count": footprint.store_count,
        "submit_prefetch_count": footprint.retrieve_prefetch_count,
        "read_prefetched_enter_count": 0,
        "estimated_store_bytes": footprint.estimated_total_store_bytes,
        "estimated_unique_keys": footprint.unique_object_key_count,
        "estimated_max_object_bytes": footprint.estimated_max_object_bytes,
        "duration_seconds": footprint.duration_seconds,
        "warnings": footprint.warnings,
    }


def _run_trace_info(config: dict[str, Any], trace_path: str, output_path: str) -> None:
    binary = config["record"].get("lmcache_binary", "lmcache")
    cmd = shlex.split(binary) if isinstance(binary, str) else list(binary)
    cmd.extend(["trace", "info", trace_path])
    result = subprocess.run(cmd, text=True, capture_output=True, check=False)
    write_text(output_path, result.stdout + result.stderr)


def _manifest_entry(
    *,
    job: dict[str, Any],
    metadata: dict[str, Any],
    trace_stats: dict[str, Any],
) -> dict[str, Any]:
    catalog = job["catalog"]
    model = job["model"]
    source_url = (
        catalog.get("repo_url")
        or catalog.get("hf_url")
        or catalog.get("homepage_url")
        or ""
    )
    data_url = (
        catalog.get("hf_url")
        or catalog.get("resource_page_url")
        or catalog.get("data_url")
        or source_url
    )
    return {
        "trace_id": job["name"],
        "trace_path": metadata["trace_path"],
        "storage_class": job["storage_class"],
        "dataset": {
            "name": job["dataset_ref"],
            "source_url": source_url,
            "data_url": data_url,
            "adapter": catalog["adapter"],
            "access": catalog.get("access", "unknown"),
            "subset": job.get("subset"),
            "split": job.get("split"),
        },
        "model": {
            "profile": job["model_profile"],
            "model_id": model["model_id"],
            "model_size_class": model.get("model_size_class"),
            "tensor_parallel_size": model["tensor_parallel_size"],
            "max_model_len": model["max_model_len"],
        },
        "lmcache": {
            "chunk_size": model["lmcache_chunk_size"],
            "record_l1_size_gb": model.get("record_l1_size_gb", 1),
            "record_eviction_policy": model.get("record_eviction_policy", "noop"),
        },
        "request_shape": {
            "concurrency": job["request_concurrency"],
            "request_rate_qps": job["request_rate_qps"],
            "num_tasks": job["num_tasks"],
            "turns_per_task": job["turns_per_task"],
            "workload_shape": job.get("workload_shape"),
        },
        "record_window": {
            "base_offset_bytes": metadata["base_offset_bytes"],
            "capacity_bytes": metadata["capacity_bytes"],
            "meta_magic": metadata["meta_magic"],
        },
        "trace_stats": trace_stats,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--record-suite", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-dataset-access", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    output_dir = os.path.abspath(args.output_dir)
    for child in ("request_logs", "server_logs", "traces", "trace_info"):
        os.makedirs(os.path.join(output_dir, child), exist_ok=True)

    jobs = _record_jobs(config, args.record_suite)
    manifest = {"traces": []}
    record_jobs = []
    commands = []
    warnings = []

    for index, job in enumerate(jobs):
        request_objects, requests, adapter_warnings, skipped = _adapter_requests(
            job=job,
            strict_dataset_access=args.strict_dataset_access,
        )
        warnings.extend(adapter_warnings)
        lmcache_cmd, vllm_cmd, metadata = _build_job_commands(
            config=config,
            job=job,
            index=index,
            output_dir=output_dir,
        )
        commands.extend([command_to_text(lmcache_cmd), command_to_text(vllm_cmd)])
        record_jobs.append(
            {
                "name": job["name"],
                "skipped": skipped,
                "request_count": len(requests),
                "lmcache_command": command_to_text(lmcache_cmd),
                "vllm_command": command_to_text(vllm_cmd),
                "metadata": metadata,
            }
        )
        write_json(
            os.path.join(output_dir, "request_logs", f"{job['name']}.requests.json"),
            requests,
        )
        if args.dry_run or skipped:
            stats = {
                "total_records": 0,
                "reserve_write_count": 0,
                "finish_write_count": 0,
                "submit_prefetch_count": 0,
                "read_prefetched_enter_count": 0,
                "estimated_store_bytes": None,
                "estimated_unique_keys": None,
            }
            manifest["traces"].append(
                _manifest_entry(job=job, metadata=metadata, trace_stats=stats)
            )
            continue

        lmcache_log = open(
            os.path.join(output_dir, "server_logs", f"{job['name']}.lmcache.log"),
            "w",
        )
        vllm_log = open(
            os.path.join(output_dir, "server_logs", f"{job['name']}.vllm.log"),
            "w",
        )
        lmcache_proc = subprocess.Popen(
            lmcache_cmd,
            stdout=lmcache_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        vllm_proc: subprocess.Popen | None = None
        try:
            _wait_http(
                f"http://127.0.0.1:{metadata['http_port']}/api/healthcheck",
                timeout_s=int(config["record"].get("startup_timeout_sec", 300)),
            )
            vllm_proc = subprocess.Popen(
                vllm_cmd,
                stdout=vllm_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_http(
                f"http://127.0.0.1:{metadata['vllm_port']}/v1/models",
                timeout_s=int(config["record"].get("startup_timeout_sec", 300)),
            )
            drive_requests(
                request_objects,
                base_url=f"http://127.0.0.1:{metadata['vllm_port']}",
                model_id=str(job["model"]["model_id"]),
                concurrency=int(job["request_concurrency"]),
                request_rate_qps=float(job["request_rate_qps"]),
                log_path=os.path.join(
                    output_dir,
                    "request_logs",
                    f"{job['name']}.responses.jsonl",
                ),
            )
        finally:
            if vllm_proc is not None:
                _terminate(
                    vllm_proc,
                    timeout_s=int(config["record"].get("shutdown_timeout_sec", 60)),
                )
            _terminate(
                lmcache_proc,
                timeout_s=int(config["record"].get("shutdown_timeout_sec", 60)),
            )
            lmcache_log.close()
            vllm_log.close()

        _run_trace_info(
            config,
            metadata["trace_path"],
            os.path.join(output_dir, "trace_info", f"{job['name']}.txt"),
        )
        manifest["traces"].append(
            _manifest_entry(
                job=job,
                metadata=metadata,
                trace_stats=_trace_stats(metadata["trace_path"]),
            )
        )

    write_yaml(os.path.join(output_dir, "record_config.resolved.yaml"), config)
    write_yaml(os.path.join(output_dir, "trace_manifest.yaml"), manifest)
    write_json(os.path.join(output_dir, "record_jobs.json"), record_jobs)
    write_text(
        os.path.join(output_dir, "record_commands.txt"),
        "\n".join(commands) + "\n",
    )
    summary = [
        "# Agentic MP Trace Recording Summary",
        "",
        f"- record_suite: `{args.record_suite}`",
        f"- jobs: {len(jobs)}",
        f"- dry_run: {args.dry_run}",
        f"- warnings: {len(warnings)}",
        "",
    ]
    summary.extend(f"- WARNING: {warning}" for warning in warnings)
    write_text(os.path.join(output_dir, "summary.md"), "\n".join(summary) + "\n")
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
