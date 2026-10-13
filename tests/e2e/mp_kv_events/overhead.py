# SPDX-License-Identifier: Apache-2.0
"""Matched targeted throughput/latency measurements, separate from routing gates."""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import argparse
import hashlib
import json
import re
import statistics
import time

# Third Party
from fixtures import fixture
from recovery import control
from response_oracle import validate_inference_response
from run import write_json

if TYPE_CHECKING:
    # Third Party
    from scenarios import Driver


def runner_source() -> dict[str, str]:
    """Fingerprint the actual host driver, independently of the worker image."""
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "overhead.py",
            "scenarios.py",
            "fixtures.py",
            "response_oracle.py",
            "recovery.py",
            "residency.py",
            "run.py",
            "versions.lock.yaml",
        )
    }


def process_usage(driver: "Driver", owner: str) -> dict:
    """Sample CPU times/RSS of only the supervisor's owned MP/engine processes."""
    script = """
from pathlib import Path
import json, psutil, time
active = {}
for line in Path('/artifacts/lifecycle.jsonl').read_text().splitlines():
    item = json.loads(line)
    if item['action'] == 'start':
        active[item['process']] = item['pid']
    elif item['action'] == 'stop':
        active.pop(item['process'], None)
result = {'observed_ns': time.time_ns()}
for name in ('mp', 'engine'):
    parent = psutil.Process(active[name])
    processes = [parent, *parent.children(recursive=True)]
    result[name] = {
        str(p.pid): {'created': p.create_time(),
                     'cpu_seconds': p.cpu_times().user + p.cpu_times().system,
                     'rss_bytes': p.memory_info().rss}
        for p in processes
    }
print(json.dumps(result))
"""
    return json.loads(driver.exec(owner, script).splitlines()[-1])


def run_overhead(driver: "Driver", samples: int = 30) -> dict:
    """Measure one enabled/disabled arm with the identical targeted token workload.

    Run both arms in fresh deployments from the same lock and hardware profile.
    Fixed targeting isolates event processing from the intended routing changes.
    Cold and repeated-prefix samples are reported separately. These measurements
    do not replace any unbiased routing scenario.
    """
    if samples < 20:
        raise ValueError("Overhead measurement requires at least twenty samples")
    runner_fingerprint = runner_source()
    owner = "a"
    chunk_size = driver.state["chunk_size"]
    for number in range(3):
        driver.request(fixture(7000 + number, chunk_size=chunk_size), owner)
    before = driver.snapshot("overhead-before")
    provenance = json.loads(
        driver.exec(
            owner,
            "from pathlib import Path; "
            "print(Path('/artifacts/provenance.json').read_text())",
        )
    )
    usage_before = process_usage(driver, owner)
    rows = []
    started = time.perf_counter()
    for index in range(samples):
        tokens = fixture(7100 + index, chunk_size=chunk_size)
        for phase in ("cold", "repeat"):
            result = driver.request(tokens, owner, max_tokens=16)
            first_token = result["first_token_ns"]
            if first_token is None:
                raise AssertionError("Missing actual first generated token arrival")
            rows.append(
                {
                    "phase": phase,
                    "frontend_request_id": result["frontend_request_id"],
                    "tokens": tokens,
                    "ttft_ms": (first_token - result["started_ns"]) / 1e6,
                    "duration_ms": (result["ended_ns"] - result["started_ns"]) / 1e6,
                    "saw_done": result["saw_done"],
                    "completion_tokens": result["completion_tokens"],
                    "output": result["output"],
                    "response": result["response"],
                }
            )
    elapsed = time.perf_counter() - started
    usage_after = process_usage(driver, owner)
    after = driver.snapshot("overhead-after")
    expected_subscribers = 0 if driver.state["events_disabled"] else 1
    for replica in ("a", "b"):
        observed = after[replica]["status"]["kv_events"]["subscribers"]
        if observed != expected_subscribers:
            raise AssertionError(
                f"Replica {replica}: expected {expected_subscribers} event "
                f"subscribers, observed {observed}"
            )
        if driver.state["events_disabled"] and after[replica]["native"]:
            raise AssertionError("Disabled arm emitted native GPU or CPU events")
    subscriber_log = after[owner]["engine_log"][len(before[owner]["engine_log"]) :]
    queue_depths = [
        int(value)
        for value in re.findall(r"MP_KV_SUBSCRIBER buffer_depth=(\d+)", subscriber_log)
    ]
    if not driver.state["events_disabled"] and not queue_depths:
        raise AssertionError("Enabled arm omitted subscriber queue-depth diagnostics")
    generated_tokens = sum(row["completion_tokens"] for row in rows)
    distributions = {}
    for phase in ("cold", "repeat"):
        values = [row["ttft_ms"] for row in rows if row["phase"] == phase]
        distributions[phase] = {
            "samples": len(values),
            "ttft_p50_ms": statistics.median(values),
            "ttft_p95_ms": statistics.quantiles(values, n=100, method="inclusive")[94],
        }
    cpu = {}
    for process in ("mp", "engine"):
        previous = usage_before[process]
        current = usage_after[process]
        if previous.keys() != current.keys() or any(
            previous[pid]["created"] != current[pid]["created"] for pid in previous
        ):
            raise AssertionError(
                "Owned process set changed during overhead measurement"
            )
        cpu[process] = {
            "mean_cores_used": sum(
                current[pid]["cpu_seconds"] - previous[pid]["cpu_seconds"]
                for pid in previous
            )
            / ((usage_after["observed_ns"] - usage_before["observed_ns"]) / 1e9),
            "rss_before_bytes": sum(p["rss_bytes"] for p in previous.values()),
            "rss_after_bytes": sum(p["rss_bytes"] for p in current.values()),
        }
    # Trace memory separately so allocation tracing does not perturb timings.
    control(driver, owner, "memory_start")
    try:
        memory_before = control(driver, owner, "memory")
        for index in range(20):
            driver.request(fixture(7500 + index, chunk_size=chunk_size), owner)
        driver.active_barrier(owner)
        memory_live = control(driver, owner, "memory")
        driver.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        driver.active_barrier(owner)
        memory_cleared = control(driver, owner, "memory")
    finally:
        control(driver, owner, "memory_stop")
    if runner_source() != runner_fingerprint:
        raise ValueError("Host runner source changed during overhead measurement")
    result = {
        "status": "PASS",
        "scope": (
            "one measured arm of combined native GPU and MP CPU KV-event overhead; "
            "compare only matching enabled/disabled arms"
        ),
        "events_enabled": not driver.state["events_disabled"],
        "workload": (
            "targeted replica A, concurrency=1, cold then repeat, max_tokens=16"
        ),
        "samples_per_phase": samples,
        "elapsed_seconds": elapsed,
        "request_throughput_per_second": len(rows) / elapsed,
        "generated_token_throughput_per_second": generated_tokens / elapsed,
        "latency": distributions,
        "process_usage": cpu,
        "peak_subscriber_buffer_depth": max(queue_depths, default=0),
        "subscriber_count": after[owner]["status"]["kv_events"]["subscribers"],
        "mp_before": before[owner]["status"]["kv_events"],
        "mp_after": after[owner]["status"]["kv_events"],
        "metadata_memory": {
            "before": memory_before,
            "live": memory_live,
            "cleared": memory_cleared,
            "cleared_scope": (
                "after L1 clear and one ordinary publication-barrier request; "
                "that request may create a fresh live binding/chunk, so this "
                "is not a zero-live-object baseline"
            ),
            "timed_separately": True,
        },
        "rows": rows,
        "deployment": driver.state,
        "gpu": provenance["gpu"],
        "source": provenance["source"],
        "runner_source_sha256": runner_fingerprint,
    }
    write_json(driver.root / "overhead.json", result)
    return result


def validate_matched_arms(enabled: dict, disabled: dict) -> None:
    """Reject incomparable configurations or incomplete/different token workloads."""
    if not enabled["events_enabled"] or disabled["events_enabled"]:
        raise ValueError("Expected enabled then disabled measurement arms")
    for field in (
        "status",
        "gpu",
        "source",
        "runner_source_sha256",
        "workload",
        "samples_per_phase",
    ):
        if enabled[field] != disabled[field]:
            raise ValueError(f"Unmatched overhead arms: {field}")
    if enabled["status"] != "PASS":
        raise ValueError("Cannot compare an incomplete measurement")
    for field in (
        "engine",
        "tp",
        "transport",
        "chunk_size",
        "native_block_size",
        "event_log_size",
        "node",
        "image",
        "external_runtime",
        "lock",
    ):
        if enabled["deployment"][field] != disabled["deployment"][field]:
            raise ValueError(f"Unmatched deployment field: {field}")
    for arm in (enabled, disabled):
        rows = arm["rows"]
        if arm["samples_per_phase"] < 20 or len(rows) != 2 * arm["samples_per_phase"]:
            raise ValueError("Incomplete overhead sample workload")
        for index, row in enumerate(rows):
            if row["phase"] != ("cold" if index % 2 == 0 else "repeat"):
                raise ValueError("Overhead sample phases changed")
            if index % 2 and row["tokens"] != rows[index - 1]["tokens"]:
                raise ValueError("Repeat sample changed its prompt")
            completed = validate_inference_response(
                row["response"], saw_done=row["saw_done"], max_tokens=16
            )
            if row["completion_tokens"] != completed:
                raise ValueError("Overhead token count disagrees with actual response")
    for enabled_row, disabled_row in zip(
        enabled["rows"], disabled["rows"], strict=True
    ):
        for field in ("phase", "tokens", "completion_tokens", "output"):
            if enabled_row[field] != disabled_row[field]:
                raise ValueError(f"Unmatched actual workload field: {field}")


def compare(enabled_path: Path, disabled_path: Path, output: Path) -> None:
    """Compare successful arms with identical source, GPUs, and workload pins."""
    enabled = json.loads(enabled_path.read_text())
    disabled = json.loads(disabled_path.read_text())
    validate_matched_arms(enabled, disabled)
    metrics = {
        name: {
            "disabled": disabled[name],
            "enabled": enabled[name],
            "change_percent": (enabled[name] / disabled[name] - 1) * 100,
        }
        for name in (
            "request_throughput_per_second",
            "generated_token_throughput_per_second",
        )
    }
    for phase in ("cold", "repeat"):
        for quantile in ("ttft_p50_ms", "ttft_p95_ms"):
            base = disabled["latency"][phase][quantile]
            value = enabled["latency"][phase][quantile]
            metrics[f"{phase}_{quantile}"] = {
                "disabled": base,
                "enabled": value,
                "change_percent": (value / base - 1) * 100,
            }
    write_json(
        output,
        {
            "status": "PASS",
            "scope": (
                "Observed combined native GPU and MP CPU KV-event overhead on "
                "one targeted worker at concurrency=1; "
                "fixed debug/trace configuration. This is not a routing-performance "
                "or broad throughput claim. Event-lag evidence is reported in T11."
            ),
            "metrics": metrics,
            "enabled_artifact": str(enabled_path.resolve()),
            "disabled_artifact": str(disabled_path.resolve()),
            "enabled_process_usage": enabled["process_usage"],
            "disabled_process_usage": disabled["process_usage"],
            "enabled_metadata_memory": enabled["metadata_memory"],
            "disabled_metadata_memory": disabled["metadata_memory"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("enabled", type=Path)
    parser.add_argument("disabled", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    compare(options.enabled, options.disabled, options.output)
