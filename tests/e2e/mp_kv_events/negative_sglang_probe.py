# SPDX-License-Identifier: Apache-2.0
"""Standalone SGLang admission probe; this is not a Kubernetes/router result."""

# Standard
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def http(path: str, body: dict | None = None) -> dict:
    """Capture the real HTTP status, including a distinct transport failure."""
    request = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=180 if body is not None else 10
        ) as response:
            return {"status": response.status, "body": response.read().decode()}
    except urllib.error.HTTPError as error:
        return {"status": error.code, "body": error.read().decode()}
    except (urllib.error.URLError, TimeoutError) as error:
        return {"status": None, "transport_error": str(error)}


def run(output: Path, enabled: bool) -> dict:
    """Launch private MP/SGLang processes and verify request rejection and health."""
    output.mkdir(parents=True, exist_ok=True)
    config = output / "lmcache.json"
    config.write_text(
        json.dumps(
            {
                "mp_host": "127.0.0.1",
                "mp_port": 5555,
                "extra_config": {"lmcache.mp.mp_transfer_mode": "lmcache_driven"},
            }
        )
    )
    revision = "7ae557604adf67be50417f59c2c2f167def9a775"
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
            "64",
            "--kv-event-log-size",
            "4096" if enabled else "0",
        ],
        "engine": [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--model-path",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "--revision",
            revision,
            "--tp-size",
            "1",
            "--page-size",
            "16",
            "--max-running-requests",
            "1",
            "--max-total-tokens",
            "4096",
            "--context-length",
            "4096",
            "--mem-fraction-static",
            "0.1",
            "--enable-lmcache",
            "--lmcache-config-file",
            str(config),
            "--disable-cuda-graph",
            "--log-level",
            "debug",
        ],
    }
    if enabled:
        commands["engine"].extend(
            [
                "--kv-events-config",
                json.dumps(
                    {
                        "publisher": "zmq",
                        "endpoint": "tcp://*:20080",
                        "topic": "kv-events",
                        "format": "dynamo",
                    }
                ),
            ]
        )
    result: dict = {
        "scope": "standalone single replica; no routing claim",
        "events_enabled": enabled,
        "commands": commands,
        "sglang_version": importlib.metadata.version("sglang"),
        "torch_version": importlib.metadata.version("torch"),
        "requests": [],
    }
    processes: list[subprocess.Popen] = []
    logs = []
    try:
        for name, argv in commands.items():
            log = (output / (name + ".log")).open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
            if name == "mp":
                time.sleep(2)
        deadline = time.monotonic() + 300
        while http("/model_info")["status"] != 200:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("Probe process exited before engine readiness")
            if time.monotonic() >= deadline:
                raise RuntimeError("Engine startup exceeded 300 seconds")
            time.sleep(1)
        cases = {
            "cache_salt": "private-user",
            "extra_key": "private-key",
            "session_id": "session",
        }
        if not enabled:
            cases = {"cache_salt": "private-user"}
        for field, value in cases.items():
            body = {
                "rid": "negative-sglang-" + field,
                "input_ids": list(range(100, 165)),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                field: value,
            }
            response = http("/generate", body)
            result["requests"].append({"request": body, "response": response})
            assert response["status"] == (400 if enabled else 200), response
            if enabled:
                assert "LMCache MP KV events" in response["body"], response
            assert http("/model_info")["status"] == 200
        followup = http(
            "/generate",
            {
                "rid": "negative-sglang-valid-followup",
                "input_ids": list(range(100, 165)),
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            },
        )
        result["valid_followup"] = followup
        assert followup["status"] == 200, followup
        result["status"] = "PASS"
        return result
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = repr(error)
        raise
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in reversed(processes):
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for log in logs:
            log.close()
        (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-disabled", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve(), not args.events_disabled)))
