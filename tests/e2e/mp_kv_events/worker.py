# SPDX-License-Identifier: Apache-2.0
"""Test-only supervisor for an engine and its private MP in one GPU container."""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.request

# Third Party
import psutil


class Supervisor:
    """Own process groups and permit isolated engine/MP restarts in tests.

    Args:
        config: JSON launcher containing explicit argv and health URLs.
        logs: Directory receiving separate process logs and provenance.
    """

    def __init__(self, config: dict, logs: Path) -> None:
        self.config = config
        self.logs = logs
        self.processes: dict[str, subprocess.Popen] = {}
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.paused_processes: list[dict] = []
        logs.mkdir(parents=True, exist_ok=True)

    def start(self, name: str) -> None:
        """Start the named process and wait for its configured health endpoint."""
        spec = self.config[name]
        with (self.logs / f"{name}.log").open("a") as log:
            process = subprocess.Popen(
                spec["argv"],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.processes[name] = process
        with (self.logs / "lifecycle.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "time_ns": time.time_ns(),
                        "action": "start",
                        "process": name,
                        "pid": process.pid,
                        "argv": spec["argv"],
                    }
                )
                + "\n"
            )
        if "health" not in spec:
            return
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if not self.running:
                raise InterruptedError("Supervisor shutdown during startup")
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with {process.returncode}")
            try:
                with urllib.request.urlopen(spec["health"], timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.2)
        raise TimeoutError(f"{name} health deadline exceeded")

    def stop(self, name: str) -> None:
        """Stop only the named process group, allowing graceful CUDA cleanup."""
        process = self.processes.pop(name)
        if process.poll() is None:
            if name == "engine" and self.paused:
                self._resume_descendants(strict=False)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        with (self.logs / "lifecycle.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "time_ns": time.time_ns(),
                        "action": "stop",
                        "process": name,
                        "returncode": process.returncode,
                    }
                )
                + "\n"
            )

    def restart(self, name: str, wait_for_unhealthy_heartbeats: int = 0) -> None:
        """Restart one process, optionally observing real client health loss."""
        if name not in ("engine", "mp"):
            raise ValueError("Only engine and mp are restartable")
        if wait_for_unhealthy_heartbeats < 0 or wait_for_unhealthy_heartbeats > 3:
            raise ValueError("Expected at most one scheduler and two worker heartbeats")
        if wait_for_unhealthy_heartbeats and name != "mp":
            raise ValueError("Heartbeat outage observation applies only to MP")
        with self.lock:
            engine_log = self.logs / "engine.log"
            offset = engine_log.stat().st_size if engine_log.exists() else 0
            self.stop(name)
            try:
                if wait_for_unhealthy_heartbeats:
                    deadline = time.monotonic() + 60
                    while True:
                        with engine_log.open() as stream:
                            stream.seek(offset)
                            observed = stream.read().count(
                                "LMCache server is unhealthy — entering degraded mode"
                            )
                        if observed >= wait_for_unhealthy_heartbeats:
                            with (self.logs / "lifecycle.jsonl").open("a") as log:
                                log.write(
                                    json.dumps(
                                        {
                                            "time_ns": time.time_ns(),
                                            "action": "observed_unhealthy_heartbeats",
                                            "process": "mp",
                                            "count": observed,
                                            "required": wait_for_unhealthy_heartbeats,
                                            "engine_log_offset": offset,
                                        }
                                    )
                                    + "\n"
                                )
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "MP outage did not reach every client heartbeat"
                            )
                        time.sleep(0.1)
            finally:
                self.start(name)

    def _resume_descendants(self, *, strict: bool) -> None:
        """Resume only saved child identities, never a PID reused by another task."""
        disappeared = []
        for saved in self.paused_processes:
            try:
                child = psutil.Process(saved["pid"])
                if child.create_time() != saved["created"]:
                    disappeared.append(saved["pid"])
                    continue
                child.send_signal(signal.SIGCONT)
            except psutil.NoSuchProcess:
                disappeared.append(saved["pid"])
        self.paused = False
        self.paused_processes = []
        if strict and disappeared:
            raise RuntimeError(f"Paused inference processes exited: {disappeared}")

    def pause(self, paused: bool) -> None:
        """Pause inference descendants while the Dynamo parent renews discovery."""
        with self.lock:
            process = self.processes["engine"]
            if process.poll() is not None:
                raise RuntimeError("Cannot pause a terminated engine")
            if paused == self.paused:
                raise ValueError("Engine already has the requested pause state")
            if paused:
                children = psutil.Process(process.pid).children(recursive=True)
                identities = [
                    {
                        "pid": child.pid,
                        "created": child.create_time(),
                        "name": child.name(),
                        "command": child.cmdline(),
                    }
                    for child in children
                    if not any("resource_tracker" in part for part in child.cmdline())
                ]
                if not identities:
                    raise RuntimeError("No inference descendants found to pause")
                self.paused_processes = identities
                self.paused = True
                try:
                    for saved in identities:
                        child = psutil.Process(saved["pid"])
                        if child.create_time() != saved["created"]:
                            raise RuntimeError("Inference process identity changed")
                        child.send_signal(signal.SIGSTOP)
                except Exception:
                    self._resume_descendants(strict=False)
                    raise
            else:
                identities = list(self.paused_processes)
                self._resume_descendants(strict=True)
            with (self.logs / "lifecycle.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "time_ns": time.time_ns(),
                            "action": "pause" if paused else "resume",
                            "process": "engine",
                            "scope": "inference descendants",
                            "discovery_parent_pid": process.pid,
                            "children": identities,
                        }
                    )
                    + "\n"
                )

    def reload(self, config_path: Path, expected_sha256: str) -> None:
        """Reset both caches on the same allocated GPUs for a matched event arm."""
        content = config_path.read_bytes()
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected_sha256:
            raise ValueError("Projected launcher checksum differs from requested mode")
        updated = json.loads(content)
        if updated["runtime_files_sha256"] != self.config["runtime_files_sha256"]:
            raise ValueError("Event-mode reload cannot change harness source")
        previous = self.config
        self.close()
        archive = self.logs / ("arm-" + str(time.time_ns()))
        archive.mkdir()
        for path in self.logs.iterdir():
            if path.is_file():
                path.rename(archive / path.name)
        provenance = json.loads((archive / "provenance.json").read_text())
        provenance["launcher"] = updated
        provenance["event_mode_reload"] = {
            "previous_artifacts": str(archive),
            "previous_launcher": previous,
            "new_launcher_sha256": actual,
            "started_ns": time.time_ns(),
        }
        self.config = updated
        (self.logs / "provenance.json").write_text(json.dumps(provenance, indent=2))
        self.start("mp")
        self.start("observer")
        self.start("engine")
        provenance["event_mode_reload"]["ready_ns"] = time.time_ns()
        (self.logs / "provenance.json").write_text(json.dumps(provenance, indent=2))

    def close(self) -> None:
        """Stop all owned process groups, retaining separate logs."""
        with self.lock:
            for name in ("engine", "observer", "mp"):
                if name in self.processes:
                    self.stop(name)


def main() -> None:
    """Run the explicitly test-enabled supervisor and localhost control server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--logs", default=Path("/artifacts"), type=Path)
    args = parser.parse_args()
    if os.environ.get("LMCACHE_KV_EVENTS_E2E") != "1":
        raise RuntimeError("Test controls require LMCACHE_KV_EVENTS_E2E=1")
    config = json.loads(args.config.read_text())
    for name, expected in config["runtime_files_sha256"].items():
        actual = hashlib.sha256(
            (Path("/opt/mp-kv-events") / name).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Runtime harness checksum mismatch: {name}")
    supervisor = Supervisor(config, args.logs)
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    ).strip()
    if occupied:
        raise RuntimeError(
            "An allocated GPU already has a compute process: " + occupied
        )
    provenance = {
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_CACHE_PATH",
                "CUDA_CACHE_MAXSIZE",
                "HOME",
                "HF_HOME",
                "CPATH",
                "LIBRARY_PATH",
                "VLLM_PLUGINS",
                "LMCACHE_LOG_LEVEL",
                "FLASHINFER_USE_CUDA_NORM",
            )
        },
        "packages": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "source": json.loads(Path("/opt/mp-kv-events/source.json").read_text()),
        "launcher": config,
    }
    (args.logs / "provenance.json").write_text(json.dumps(provenance, indent=2))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            """Return health only while both serving processes remain alive."""
            healthy = all(
                name in supervisor.processes
                and supervisor.processes[name].poll() is None
                for name in ("mp", "engine")
            )
            self.send_response(200 if healthy else 503)
            self.end_headers()
            self.wfile.write(json.dumps({"healthy": healthy}).encode())

        def do_POST(self) -> None:
            """Expose bounded test-only restarts on the localhost listener."""
            if self.path == "/reload":
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                supervisor.reload(args.config, payload["launcher_sha256"])
                self.send_response(200)
                self.end_headers()
                return
            if self.path in ("/pause/engine", "/resume/engine"):
                supervisor.pause(self.path == "/pause/engine")
                self.send_response(200)
                self.end_headers()
                return
            if self.path not in ("/restart/engine", "/restart/mp"):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length)) if length else {}
            supervisor.restart(
                self.path.rsplit("/", 1)[-1],
                payload.get("wait_for_unhealthy_heartbeats", 0),
            )
            self.send_response(200)
            self.end_headers()

    def stop(signum: int, frame: object) -> None:
        supervisor.running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    server = HTTPServer(("127.0.0.1", 9900), Handler)
    server.timeout = 0.2
    try:
        supervisor.start("mp")
        supervisor.start("observer")
        supervisor.start("engine")
        while supervisor.running:
            server.handle_request()
            for name, process in supervisor.processes.items():
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited: {process.returncode}")
    finally:
        server.server_close()
        supervisor.close()


if __name__ == "__main__":
    main()
