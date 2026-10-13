# SPDX-License-Identifier: Apache-2.0
"""Standalone real-vLLM admission probe; this is not a Kubernetes/router result."""

# Standard
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def http(path: str, body: dict | None = None, *, port: int = 8000) -> dict:
    """Record the actual HTTP status/body without treating an engine crash as 400."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return {"status": response.status, "body": response.read().decode()}
    except urllib.error.HTTPError as error:
        return {"status": error.code, "body": error.read().decode()}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"status": None, "transport_error": str(error)}


def run(output: Path, events_enabled: bool, multi_connector: bool = False) -> dict:
    """Launch a private MP and pinned vLLM API, then test request-level rejection."""
    output.mkdir(parents=True, exist_ok=True)
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
    processes: dict[str, subprocess.Popen] = {}
    logs = []
    commands = {
        "mp": [
            "lmcache",
            "server",
            "--port",
            "5555",
            "--http-port",
            "8080",
            "--l1-size-gb",
            "0.25",
            "--eviction-policy",
            "LRU",
            "--chunk-size",
            "256",
            "--kv-event-log-size",
            "4096" if events_enabled else "0",
        ],
        "observer": [
            sys.executable,
            "/opt/mp-kv-events/observe.py",
            "--endpoint",
            "tcp://127.0.0.1:20080",
            "--output",
            str(output / "native-events.jsonl"),
        ],
        "engine": [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model",
            model,
            "--revision",
            revision,
            "--tokenizer-revision",
            revision,
            "--tensor-parallel-size",
            "1",
            "--block-size",
            "16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--enforce-eager",
            "--disable-hybrid-kv-cache-manager",
            "--num-gpu-blocks-override",
            "257",
            "--gpu-memory-utilization",
            "0.1",
            "--prefix-caching-hash-algo",
            "sha256_cbor",
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lmcache.mp.server_urls": ["tcp://127.0.0.1:5555"],
                        "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                    },
                }
            ),
            "--kv-events-config",
            json.dumps(
                {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": "tcp://*:20080",
                    "enable_kv_cache_events": events_enabled,
                }
            ),
        ],
    }
    if multi_connector:
        slot = commands["engine"].index("--kv-transfer-config") + 1
        child = json.loads(commands["engine"][slot])
        commands["engine"][slot] = json.dumps(
            {
                "kv_connector": "MultiConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {"connectors": [child]},
            }
        )
    source_paths = [
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py",
        "vllm/v1/engine/input_processor.py",
        "vllm/v1/engine/async_llm.py",
        "vllm/v1/engine/llm_engine.py",
        "lmcache/integration/vllm/lmcache_mp_connector.py",
    ]
    result: dict = {
        "scope": (
            "standalone one-GPU direct API admission; not router or Kubernetes E2E"
        ),
        "events_enabled": events_enabled,
        "multi_connector": multi_connector,
        "source_sha256": {
            path: hashlib.sha256(
                Path("/usr/local/lib/python3.12/dist-packages", path).read_bytes()
            ).hexdigest()
            for path in source_paths
        },
        "commands": commands,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "lmcache", "torch")
        },
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
        ),
    }
    try:
        for name in ("mp", "observer", "engine"):
            stream = (output / f"{name}.log").open("w")
            logs.append(stream)
            processes[name] = subprocess.Popen(
                commands[name],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            if name == "mp":
                deadline = time.monotonic() + 60
                while http("/healthcheck", port=8080)["status"] != 200:
                    if (
                        processes[name].poll() is not None
                        or time.monotonic() > deadline
                    ):
                        raise RuntimeError("MP failed startup; inspect mp.log")
                    time.sleep(0.2)
        deadline = time.monotonic() + 300
        while http("/health")["status"] != 200:
            if processes["engine"].poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("vLLM failed startup; inspect engine.log")
            time.sleep(0.2)
        result["mp_status_after_registration"] = http("/status", port=8080)
        ordinary = {
            "model": model,
            "prompt": [1001] * 512 + [198],
            "max_tokens": 1,
            "temperature": 0,
        }
        result["ordinary_before"] = http("/v1/completions", ordinary)
        if result["ordinary_before"]["status"] != 200:
            raise RuntimeError("Ordinary inference failed before negative request")
        if events_enabled:
            deadline = time.monotonic() + 30
            while True:
                status = http("/status", port=8080)
                if (
                    status["status"] == 200
                    and json.loads(status["body"])["kv_events"]["next_seq"] >= 3
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Ordinary CPU KV never committed")
                time.sleep(0.05)
            while "events=2" not in (output / "engine.log").read_text():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Native subscriber did not receive ordinary CPU KV"
                    )
                time.sleep(0.05)
            result["publication_setup"] = http(
                "/v1/completions", {**ordinary, "prompt": [1004] * 256 + [198]}
            )
            while True:
                trace = output / "native-events.jsonl"
                native = (
                    [
                        json.loads(line)
                        for line in trace.read_text().splitlines()
                        if line
                    ]
                    if trace.exists()
                    else []
                )
                cpu = [
                    event
                    for record in native
                    for event in record["payload"][1]
                    if event["type"] == "BlockStored"
                    and event.get("medium") == "CPU"
                    and 1001 in event["token_ids"]
                ]
                gpu = [
                    event
                    for record in native
                    for event in record["payload"][1]
                    if event["type"] == "BlockStored"
                    and event.get("medium") == "GPU"
                    and 1001 in event["token_ids"]
                ]
                if cpu and gpu:
                    result["positive_cpu_events"] = len(cpu)
                    result["positive_gpu_events"] = len(gpu)
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Missing independent positive CPU/GPU publication"
                    )
                time.sleep(0.05)
        result["salted"] = http(
            "/v1/completions",
            {
                **ordinary,
                "prompt": [1002] * 512 + [198],
                "cache_salt": "unsupported-namespace",
            },
        )
        result["health_after"] = http("/health")
        result["ordinary_after"] = http(
            "/v1/completions", {**ordinary, "prompt": [1003] * 512 + [198]}
        )
        trace = output / "native-events.jsonl"
        native = (
            [json.loads(line) for line in trace.read_text().splitlines() if line]
            if trace.exists()
            else []
        )
        result["salted_cpu_announcements"] = [
            event
            for record in native
            for event in record["payload"][1]
            if event["type"] == "BlockStored"
            and event.get("medium") in ("CPU", "CPU_PINNED")
            and 1002 in event["token_ids"]
        ]
        expected_status = 400 if events_enabled else 200
        result["status"] = (
            "PASS"
            if (
                result["salted"]["status"] == expected_status
                and result["health_after"]["status"] == 200
                and result["ordinary_after"]["status"] == 200
                and not result["salted_cpu_announcements"]
                and (not events_enabled or "MP KV events" in result["salted"]["body"])
            )
            else "FAIL"
        )
    except (RuntimeError, subprocess.SubprocessError) as error:
        result.update(status="BLOCKED", reason=str(error))
    finally:
        for process in reversed(list(processes.values())):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
        for stream in logs:
            stream.close()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/artifacts"))
    parser.add_argument("--events-disabled", action="store_true")
    parser.add_argument("--multi-connector", action="store_true")
    args = parser.parse_args()
    result = run(args.output, not args.events_disabled, args.multi_connector)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
