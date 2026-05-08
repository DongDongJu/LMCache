# SPDX-License-Identifier: Apache-2.0

"""Run randomized trace churn in a small number of fixed raw-block regions.

This runner is for WAF stress runs where the device should see a few long-lived
hot byte ranges instead of one independent byte window per trace. Each region
keeps a fixed base offset, capacity, meta magic, and slot size, while the trace
replay salt changes on every replay.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import argparse
import datetime as dt
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

# First Party
from benchmarks.agentic_mp_trace.config import (  # noqa: E402
    load_yaml_config,
    write_json,
    write_yaml,
)
from benchmarks.agentic_mp_trace.replay.command_builder import (  # noqa: E402
    MODE_ALIASES,
    build_fdp_replay_config,
)
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (  # noqa: E402
    WorkerSpec,
    _resolve_worker_ruhs,
    build_replay_command,
    command_to_text,
)


VALID_MODES = ("no_fdp", "fdp_mixed", "fdp_separated")
DEFAULT_RUN_ID = "region_churn001"


@dataclass(frozen=True)
class TraceChoice:
    name: str
    class_name: str
    trace_path: str
    l1_size_gb: float


@dataclass(frozen=True)
class RegionSpec:
    region_index: int
    base_offset_bytes: int
    capacity_bytes: int
    slot_bytes: int
    meta_magic: str


@dataclass
class RegionReplayResult:
    region_index: int
    trace_name: str
    trace_path: str
    iteration: int
    phase: str
    command: list[str]
    output_dir: str
    log_path: str
    jsonl_path: str
    exit_code: int
    records_failed: int | None
    started_at: str
    ended_at: str
    stopped_by_controller: bool = False


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _align_down(value: int, align: int) -> int:
    return value - (value % align)


def _device_size_bytes(block_device_path: str) -> int:
    name = os.path.basename(block_device_path)
    size_path = Path("/sys/block") / name / "size"
    if not size_path.exists():
        raise FileNotFoundError(f"cannot infer device size from {size_path}")
    return int(size_path.read_text().strip()) * 512


def _region_meta_magic(region_index: int) -> str:
    if region_index < 0 or region_index > 999_999:
        raise ValueError("region index cannot fit in 8-byte meta_magic")
    return f"RC{region_index + 1:06d}"


def allocate_regions(
    *,
    start_offset_bytes: int,
    usable_end_offset_bytes: int,
    region_count: int,
    block_align: int,
    region_capacity_bytes: int | None = None,
    slot_bytes: int,
) -> list[RegionSpec]:
    if region_count <= 0:
        raise ValueError("region_count must be > 0")
    if usable_end_offset_bytes <= start_offset_bytes:
        raise ValueError("usable end must be greater than start offset")
    if start_offset_bytes % block_align:
        raise ValueError("start_offset_bytes is not block aligned")

    usable_bytes = usable_end_offset_bytes - start_offset_bytes
    if region_capacity_bytes is None:
        region_capacity_bytes = _align_down(usable_bytes // region_count, block_align)
    else:
        region_capacity_bytes = int(region_capacity_bytes)
    if region_capacity_bytes <= 0:
        raise ValueError("region_capacity_bytes must be > 0")
    if region_capacity_bytes % block_align:
        raise ValueError("region_capacity_bytes is not block aligned")
    if slot_bytes % block_align:
        raise ValueError("region slot_bytes is not block aligned")
    if region_count * region_capacity_bytes > usable_bytes:
        raise ValueError(
            "region_count * region_capacity_bytes exceeds usable byte range"
        )

    regions = []
    for index in range(region_count):
        base = start_offset_bytes + index * region_capacity_bytes
        regions.append(
            RegionSpec(
                region_index=index,
                base_offset_bytes=base,
                capacity_bytes=region_capacity_bytes,
                slot_bytes=slot_bytes,
                meta_magic=_region_meta_magic(index),
            )
        )
    return regions


def _count_failed_jsonl(path: str) -> int | None:
    if not os.path.exists(path):
        return None
    failed = 0
    with open(path) as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bool(payload.get("failed", False)):
                failed += 1
    return failed


def _trace_choices(fdp_config: dict[str, Any]) -> list[TraceChoice]:
    choices = []
    for workload in fdp_config.get("workloads", []):
        choices.append(
            TraceChoice(
                name=str(workload["name"]),
                class_name=str(workload.get("class", "")),
                trace_path=str(workload["trace_path"]),
                l1_size_gb=float(workload.get("l1_size_gb", 1.0)),
            )
        )
    if not choices:
        raise ValueError("no trace choices found in manifest/config")
    return choices


def _max_slot_bytes(fdp_config: dict[str, Any]) -> int:
    slots = [int(workload["slot_bytes"]) for workload in fdp_config["workloads"]]
    return max(slots)


def _command_prefix(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(part) for part in value]
    raise TypeError("replay_binary must be a string or list")


def _is_python_cli_shim(prefix: list[str]) -> bool:
    if "-c" not in prefix:
        return False
    script_index = prefix.index("-c") + 1
    if script_index >= len(prefix):
        return False
    return "lmcache.cli.main" in prefix[script_index]


def _apply_replay_binary_override(
    fdp_config: dict[str, Any],
    *,
    replay_binary: str | None,
    keep_config_replay_binary: bool,
) -> None:
    global_cfg = fdp_config.setdefault("global", {})
    current_prefix = _command_prefix(global_cfg.get("replay_binary"))
    if replay_binary:
        global_cfg["replay_binary"] = _command_prefix(replay_binary)
        return
    if not keep_config_replay_binary and _is_python_cli_shim(current_prefix):
        global_cfg["replay_binary"] = ["uv", "run", "--no-sync", "lmcache"]


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_path = os.fspath(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        paths = existing.split(os.pathsep)
        if repo_path not in paths:
            env["PYTHONPATH"] = os.pathsep.join([repo_path, existing])
    else:
        env["PYTHONPATH"] = repo_path
    return env


def _make_worker_for_region(
    *,
    region: RegionSpec,
    trace: TraceChoice,
    fdp_config: dict[str, Any],
    mode: str,
    l1_size_gb: float | None,
) -> WorkerSpec:
    use_fdp, data_ruhs, metadata_ruhs = _resolve_worker_ruhs(
        fdp_config,
        {"class": trace.class_name},
        mode,
    )
    return WorkerSpec(
        name=trace.name,
        class_name=trace.class_name,
        trace_path=trace.trace_path,
        worker_index=region.region_index,
        worker_global_index=region.region_index,
        base_offset_bytes=region.base_offset_bytes,
        capacity_bytes=region.capacity_bytes,
        slot_bytes=region.slot_bytes,
        l1_size_gb=trace.l1_size_gb if l1_size_gb is None else l1_size_gb,
        meta_magic=region.meta_magic,
        use_fdp=use_fdp,
        fdp_data_ruh_ids=list(data_ruhs),
        fdp_metadata_ruh_ids=list(metadata_ruhs),
    )


def _run_replay(
    *,
    region: RegionSpec,
    trace: TraceChoice,
    fdp_config: dict[str, Any],
    mode: str,
    run_id: str,
    output_dir: str,
    phase: str,
    iteration: int,
    l1_size_gb: float | None,
    stop_event: threading.Event | None = None,
    terminate_on_stop: bool = False,
) -> RegionReplayResult:
    worker = _make_worker_for_region(
        region=region,
        trace=trace,
        fdp_config=fdp_config,
        mode=mode,
        l1_size_gb=l1_size_gb,
    )
    worker_dir = os.path.join(
        output_dir,
        "worker_logs",
        f"region_{region.region_index:03d}",
        f"{phase}_{iteration:06d}_{trace.name}",
    )
    os.makedirs(worker_dir, exist_ok=True)
    jsonl_path = os.path.join(worker_dir, "records.jsonl")
    log_path = os.path.join(worker_dir, "replay.log")
    cmd = build_replay_command(
        worker,
        fdp_config,
        mode=mode,
        run_id=run_id,
        iteration=iteration,
        worker_output_dir=worker_dir,
        jsonl_path=jsonl_path,
    )
    started_at = _utc_now()
    stopped_by_controller = False
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=_child_env(),
            text=True,
        )
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                break
            if terminate_on_stop and stop_event is not None and stop_event.is_set():
                stopped_by_controller = True
                proc.terminate()
                try:
                    exit_code = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    exit_code = proc.wait()
                break
            time.sleep(0.25)
    return RegionReplayResult(
        region_index=region.region_index,
        trace_name=trace.name,
        trace_path=trace.trace_path,
        iteration=iteration,
        phase=phase,
        command=cmd,
        output_dir=worker_dir,
        log_path=log_path,
        jsonl_path=jsonl_path,
        exit_code=exit_code,
        records_failed=_count_failed_jsonl(jsonl_path),
        started_at=started_at,
        ended_at=_utc_now(),
        stopped_by_controller=stopped_by_controller,
    )


def _run(cmd: list[str], *, binary: bool = False) -> tuple[int, str | bytes, str]:
    proc = subprocess.run(cmd, capture_output=True, text=not binary, check=False)
    stderr = proc.stderr if isinstance(proc.stderr, str) else proc.stderr.decode(
        "utf-8",
        "replace",
    )
    return proc.returncode, proc.stdout, stderr


def _parse_smart_host_bytes(stdout: str) -> int | None:
    match = re.search(r"Data Units Written\s*:\s*([0-9]+)", stdout)
    if not match:
        return None
    return int(match.group(1)) * 512_000


def sample_nvme(
    *,
    block_device_path: str,
    smart_binary: str,
    get_log_binary: str,
    use_sudo: bool,
) -> dict[str, Any]:
    sudo_prefix = ["sudo", "-n"] if use_sudo else []
    sample: dict[str, Any] = {"timestamp": _utc_now()}

    smart_cmd = sudo_prefix + [smart_binary, "smart-log", block_device_path, "-H"]
    rc, stdout, stderr = _run(smart_cmd)
    sample["smart_command"] = shlex.join(smart_cmd)
    sample["smart_returncode"] = rc
    sample["smart_stderr"] = stderr.strip()
    if isinstance(stdout, str):
        sample["host_write_bytes"] = _parse_smart_host_bytes(stdout)

    ocp_cmd = sudo_prefix + [
        get_log_binary,
        "get-log",
        block_device_path,
        "--log-id=0xc0",
        "--log-len=512",
        "--raw-binary",
    ]
    rc, raw, stderr = _run(ocp_cmd, binary=True)
    sample["ocp_c0_command"] = shlex.join(ocp_cmd)
    sample["ocp_c0_returncode"] = rc
    sample["ocp_c0_stderr"] = stderr.strip()
    if isinstance(raw, bytes) and len(raw) >= 32:
        sample["physical_media_write_bytes"] = int.from_bytes(raw[:16], "little")
        sample["physical_media_read_bytes"] = int.from_bytes(raw[16:32], "little")

    try:
        sample["sysfs_stat"] = [
            int(value)
            for value in Path(
                f"/sys/block/{os.path.basename(block_device_path)}/stat"
            )
            .read_text()
            .split()
        ]
    except Exception as exc:
        sample["sysfs_stat_error"] = str(exc)
    return sample


def _write_sample(
    path: Path,
    sample: dict[str, Any],
    baseline: dict[str, Any] | None,
    target_host_write_bytes: int | None,
) -> dict[str, Any]:
    host = sample.get("host_write_bytes")
    media = sample.get("physical_media_write_bytes")
    if baseline and host is not None:
        sample["host_write_bytes_delta"] = int(host) - int(
            baseline.get("host_write_bytes", host)
        )
    if baseline and media is not None:
        sample["media_write_bytes_delta"] = int(media) - int(
            baseline.get("physical_media_write_bytes", media)
        )
    if (
        sample.get("host_write_bytes_delta")
        and sample["host_write_bytes_delta"] > 0
        and sample.get("media_write_bytes_delta") is not None
    ):
        sample["waf"] = sample["media_write_bytes_delta"] / sample[
            "host_write_bytes_delta"
        ]
    if target_host_write_bytes and sample.get("host_write_bytes_delta") is not None:
        sample["target_pct"] = (
            100 * sample["host_write_bytes_delta"] / target_host_write_bytes
        )
    with path.open("a") as file_obj:
        file_obj.write(json.dumps(sample, sort_keys=True) + "\n")
    return sample


def _sampler_loop(
    *,
    output_dir: str,
    block_device_path: str,
    smart_binary: str,
    get_log_binary: str,
    use_sudo: bool,
    interval_seconds: int,
    post_samples: int,
    target_host_write_bytes: int | None,
    stop_event: threading.Event,
    replay_done: threading.Event,
    baseline_ready: threading.Event,
    measurement_start: threading.Event,
    latest: dict[str, Any],
) -> None:
    samples_path = Path(output_dir) / "nvme_samples.jsonl"
    log_path = Path(output_dir) / "sampler.log"
    baseline: dict[str, Any] | None = None

    def _log(text: str) -> None:
        with log_path.open("a") as file_obj:
            file_obj.write(text + "\n")
        print(text, flush=True)

    while not baseline_ready.is_set() and not stop_event.is_set():
        time.sleep(0.2)
    if stop_event.is_set():
        return

    baseline = sample_nvme(
        block_device_path=block_device_path,
        smart_binary=smart_binary,
        get_log_binary=get_log_binary,
        use_sudo=use_sudo,
    )
    baseline["label"] = "baseline_after_warmup"
    latest.clear()
    latest.update(
        _write_sample(samples_path, baseline, baseline, target_host_write_bytes)
    )
    measurement_start.set()
    _log(
        f"[{baseline['timestamp']}] baseline host={baseline.get('host_write_bytes')} "
        f"media={baseline.get('physical_media_write_bytes')}"
    )

    remaining_post = post_samples
    while not stop_event.is_set():
        time.sleep(interval_seconds)
        label = "post_replay_delay" if replay_done.is_set() else "measurement"
        sample = sample_nvme(
            block_device_path=block_device_path,
            smart_binary=smart_binary,
            get_log_binary=get_log_binary,
            use_sudo=use_sudo,
        )
        sample["label"] = label
        latest.clear()
        latest.update(_write_sample(samples_path, sample, baseline, target_host_write_bytes))
        _log(
            f"[{sample['timestamp']}] sample={label} "
            f"host_delta={sample.get('host_write_bytes_delta')} "
            f"media_delta={sample.get('media_write_bytes_delta')} "
            f"waf={sample.get('waf')} target_pct={sample.get('target_pct')}"
        )
        if (
            target_host_write_bytes is not None
            and sample.get("host_write_bytes_delta") is not None
            and sample["host_write_bytes_delta"] >= target_host_write_bytes
        ):
            stop_event.set()
        if replay_done.is_set():
            remaining_post -= 1
            if remaining_post <= 0:
                stop_event.set()


def _runner_loop(
    *,
    region: RegionSpec,
    traces: list[TraceChoice],
    fdp_config: dict[str, Any],
    mode: str,
    run_id: str,
    output_dir: str,
    rng_seed: int,
    warmup_runs: int,
    max_initial_stagger_seconds: float,
    min_pause_seconds: float,
    max_pause_seconds: float,
    l1_size_gb: float | None,
    stop_event: threading.Event,
    baseline_ready: threading.Event,
    measurement_start: threading.Event,
    warmup_barrier: threading.Barrier,
    results: list[RegionReplayResult],
    results_lock: threading.Lock,
) -> None:
    rng = random.Random(rng_seed)
    if max_initial_stagger_seconds > 0:
        time.sleep(rng.uniform(0, max_initial_stagger_seconds))

    local_iteration = 0
    for _ in range(warmup_runs):
        trace = rng.choice(traces)
        result = _run_replay(
            region=region,
            trace=trace,
            fdp_config=fdp_config,
            mode=mode,
            run_id=run_id,
            output_dir=output_dir,
            phase="warmup",
            iteration=local_iteration,
            l1_size_gb=l1_size_gb,
            stop_event=stop_event,
            terminate_on_stop=True,
        )
        with results_lock:
            results.append(result)
        local_iteration += 1

    barrier_index = warmup_barrier.wait()
    if barrier_index == 0:
        baseline_ready.set()
    measurement_start.wait()
    while not stop_event.is_set():
        trace = rng.choice(traces)
        result = _run_replay(
            region=region,
            trace=trace,
            fdp_config=fdp_config,
            mode=mode,
            run_id=run_id,
            output_dir=output_dir,
            phase="measurement",
            iteration=local_iteration,
            l1_size_gb=l1_size_gb,
            stop_event=stop_event,
            terminate_on_stop=True,
        )
        with results_lock:
            results.append(result)
        local_iteration += 1
        if max_pause_seconds > 0 and not stop_event.is_set():
            time.sleep(rng.uniform(min_pause_seconds, max_pause_seconds))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=VALID_MODES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--region-count", type=int, default=4)
    parser.add_argument("--region-capacity-bytes", type=int, default=None)
    parser.add_argument("--usable-end-offset-bytes", type=int, default=None)
    parser.add_argument("--region-slot-bytes", type=int, default=None)
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--warmup-runs-per-region", type=int, default=1)
    parser.add_argument("--target-host-write-bytes", type=int, default=None)
    parser.add_argument("--target-device-write-multiplier", type=int, default=None)
    parser.add_argument("--sampler-interval-seconds", type=int, default=60)
    parser.add_argument("--post-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-initial-stagger-seconds", type=float, default=0.0)
    parser.add_argument("--min-pause-seconds", type=float, default=0.0)
    parser.add_argument("--max-pause-seconds", type=float, default=0.0)
    parser.add_argument("--l1-size-gb", type=float, default=None)
    parser.add_argument(
        "--replay-binary",
        default=None,
        help=(
            "Override replay command prefix, for example "
            "'uv run --no-sync lmcache'."
        ),
    )
    parser.add_argument(
        "--keep-config-replay-binary",
        action="store_true",
        help=(
            "Use replay.replay_binary exactly as configured. By default this "
            "runner rewrites the old python -c lmcache.cli.main shim to "
            "'uv run --no-sync lmcache'."
        ),
    )
    parser.add_argument("--nvme-smart-binary", default="/usr/local/sbin/nvme")
    parser.add_argument("--nvme-get-log-binary", default="/usr/sbin/nvme")
    parser.add_argument("--no-sudo-nvme", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _summary(
    *,
    args: argparse.Namespace,
    regions: list[RegionSpec],
    results: list[RegionReplayResult],
    latest_sample: dict[str, Any],
    target_host_write_bytes: int | None,
) -> dict[str, Any]:
    failures = [
        result
        for result in results
        if (
            not result.stopped_by_controller
            and (result.exit_code != 0 or (result.records_failed or 0) > 0)
        )
    ]
    coordinated_stops = [
        result for result in results if result.stopped_by_controller
    ]
    measurement_results = [result for result in results if result.phase == "measurement"]
    return {
        "run_id": args.run_id,
        "mode": args.mode,
        "region_count": len(regions),
        "duration_seconds": args.duration_seconds,
        "target_host_write_bytes": target_host_write_bytes,
        "target_host_write_bytes_reached": bool(
            target_host_write_bytes is not None
            and latest_sample.get("host_write_bytes_delta") is not None
            and latest_sample["host_write_bytes_delta"] >= target_host_write_bytes
        ),
        "host_write_bytes_delta": latest_sample.get("host_write_bytes_delta"),
        "media_write_bytes_delta": latest_sample.get("media_write_bytes_delta"),
        "waf": latest_sample.get("waf"),
        "latest_sample": latest_sample,
        "warmup_runs_per_region": args.warmup_runs_per_region,
        "measurement_replays_completed": len(measurement_results),
        "replay_results_total": len(results),
        "failed_replay_count": len(failures),
        "coordinated_stop_replay_count": len(coordinated_stops),
        "regions": [asdict(region) for region in regions],
        "results": [asdict(result) for result in results],
    }


def _write_summary_md(path: str, summary: dict[str, Any]) -> None:
    lines = [
        "# Region Churn Replay Summary",
        "",
        f"- run_id: {summary['run_id']}",
        f"- mode: {summary['mode']}",
        f"- region_count: {summary['region_count']}",
        f"- measurement_replays_completed: {summary['measurement_replays_completed']}",
        f"- failed_replay_count: {summary['failed_replay_count']}",
        f"- coordinated_stop_replay_count: {summary['coordinated_stop_replay_count']}",
        f"- host_write_bytes_delta: {summary['host_write_bytes_delta']}",
        f"- media_write_bytes_delta: {summary['media_write_bytes_delta']}",
        f"- waf: {summary['waf']}",
        f"- target_host_write_bytes: {summary['target_host_write_bytes']}",
        f"- target_host_write_bytes_reached: {summary['target_host_write_bytes_reached']}",
        "",
        "## Regions",
        "",
    ]
    for region in summary["regions"]:
        lines.append(
            "- region_{region_index:03d}: base={base_offset_bytes} "
            "capacity={capacity_bytes} slot={slot_bytes} meta_magic={meta_magic}".format(
                **region
            )
        )
    Path(path).write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    manifest = load_yaml_config(args.trace_manifest)
    fdp_config = build_fdp_replay_config(
        config=config,
        manifest=manifest,
        mode=args.mode,
    )
    _apply_replay_binary_override(
        fdp_config,
        replay_binary=args.replay_binary,
        keep_config_replay_binary=args.keep_config_replay_binary,
    )
    mode = MODE_ALIASES[args.mode]
    replay_cfg = config["replay"]
    block_align = int(fdp_config.get("block_align", 4096))
    block_device_path = str(replay_cfg.get("block_device_path", "/dev/nvme1n1"))
    usable_end = args.usable_end_offset_bytes
    if usable_end is None:
        usable_end = _device_size_bytes(block_device_path)
    usable_end = _align_down(int(usable_end), block_align)
    start_offset = int(replay_cfg["start_offset_bytes"])
    slot_bytes = int(args.region_slot_bytes or _max_slot_bytes(fdp_config))
    regions = allocate_regions(
        start_offset_bytes=start_offset,
        usable_end_offset_bytes=usable_end,
        region_count=args.region_count,
        block_align=block_align,
        region_capacity_bytes=args.region_capacity_bytes,
        slot_bytes=slot_bytes,
    )
    traces = _trace_choices(fdp_config)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    write_yaml(os.path.join(output_dir, "replay_config.resolved.yaml"), fdp_config)
    write_yaml(os.path.join(output_dir, "replay_manifest.resolved.yaml"), manifest)
    write_json(os.path.join(output_dir, "regions.json"), [asdict(r) for r in regions])

    target_host_write_bytes = args.target_host_write_bytes
    if target_host_write_bytes is None and args.target_device_write_multiplier:
        device_size = _device_size_bytes(block_device_path)
        target_host_write_bytes = device_size * args.target_device_write_multiplier

    preview_commands = []
    for region in regions:
        trace = traces[region.region_index % len(traces)]
        worker = _make_worker_for_region(
            region=region,
            trace=trace,
            fdp_config=fdp_config,
            mode=mode,
            l1_size_gb=args.l1_size_gb,
        )
        worker_dir = os.path.join(
            output_dir,
            "worker_logs",
            f"region_{region.region_index:03d}",
            f"dry_run_000000_{trace.name}",
        )
        preview_commands.append(
            command_to_text(
                build_replay_command(
                    worker,
                    fdp_config,
                    mode=mode,
                    run_id=args.run_id,
                    iteration=0,
                    worker_output_dir=worker_dir,
                    jsonl_path=os.path.join(worker_dir, "records.jsonl"),
                )
            )
        )
    Path(os.path.join(output_dir, "commands.preview.txt")).write_text(
        "\n".join(preview_commands) + "\n"
    )
    if args.dry_run:
        print("\n".join(preview_commands))
        return 0

    stop_event = threading.Event()
    replay_done = threading.Event()
    baseline_ready = threading.Event()
    measurement_start = threading.Event()
    warmup_barrier = threading.Barrier(len(regions))
    latest_sample: dict[str, Any] = {}
    results: list[RegionReplayResult] = []
    results_lock = threading.Lock()

    sampler = threading.Thread(
        target=_sampler_loop,
        kwargs={
            "output_dir": output_dir,
            "block_device_path": block_device_path,
            "smart_binary": args.nvme_smart_binary,
            "get_log_binary": args.nvme_get_log_binary,
            "use_sudo": not args.no_sudo_nvme,
            "interval_seconds": args.sampler_interval_seconds,
            "post_samples": args.post_samples,
            "target_host_write_bytes": target_host_write_bytes,
            "stop_event": stop_event,
            "replay_done": replay_done,
            "baseline_ready": baseline_ready,
            "measurement_start": measurement_start,
            "latest": latest_sample,
        },
        daemon=True,
    )
    sampler.start()

    deadline = time.monotonic() + args.duration_seconds
    deadline_guard = threading.Thread(
        target=lambda: (time.sleep(max(0, deadline - time.monotonic())), stop_event.set()),
        daemon=True,
    )
    deadline_guard.start()

    runner_threads = []
    for region in regions:
        thread = threading.Thread(
            target=_runner_loop,
            kwargs={
                "region": region,
                "traces": traces,
                "fdp_config": fdp_config,
                "mode": mode,
                "run_id": args.run_id,
                "output_dir": output_dir,
                "rng_seed": args.seed + region.region_index * 1009,
                "warmup_runs": args.warmup_runs_per_region,
                "max_initial_stagger_seconds": args.max_initial_stagger_seconds,
                "min_pause_seconds": args.min_pause_seconds,
                "max_pause_seconds": args.max_pause_seconds,
                "l1_size_gb": args.l1_size_gb,
                "stop_event": stop_event,
                "baseline_ready": baseline_ready,
                "measurement_start": measurement_start,
                "warmup_barrier": warmup_barrier,
                "results": results,
                "results_lock": results_lock,
            },
        )
        thread.start()
        runner_threads.append(thread)

    try:
        for thread in runner_threads:
            thread.join()
    except KeyboardInterrupt:
        stop_event.set()
        for thread in runner_threads:
            thread.join()
    finally:
        replay_done.set()
    sampler.join()

    with results_lock:
        results_snapshot = list(results)
    write_json(
        os.path.join(output_dir, "replay_results.json"),
        [asdict(result) for result in results_snapshot],
    )
    summary = _summary(
        args=args,
        regions=regions,
        results=results_snapshot,
        latest_sample=dict(latest_sample),
        target_host_write_bytes=target_host_write_bytes,
    )
    write_json(os.path.join(output_dir, "summary.json"), summary)
    _write_summary_md(os.path.join(output_dir, "summary.md"), summary)
    return 1 if summary["failed_replay_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
