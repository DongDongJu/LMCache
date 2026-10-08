# SPDX-License-Identifier: Apache-2.0
"""Spawn a real ``lmcache memory`` process for shared-L1 tests."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import time

# First Party
from lmcache.v1.distributed.config import L1ManagerConfig, parse_args

REGION_ID = "pool-test"


def free_port() -> int:
    """Return a TCP port that was free a moment ago on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@dataclass
class Orchestrator:
    """A running orchestrator process and where to reach it."""

    process: subprocess.Popen
    endpoint: str
    state_dir: Path

    def kill(self) -> None:
        """SIGKILL the process, as a crash would."""
        self.process.send_signal(signal.SIGKILL)
        self.process.wait(timeout=10)

    def stop(self) -> None:
        """SIGTERM the process and wait for its clean exit."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.kill()


def start_orchestrator(
    state_dir: Path,
    *,
    capacity_bytes: int,
    alignment_bytes: int = 4096,
    visibility_mode: str = "coherent",
    port: int = 0,
    region_id: str = REGION_ID,
) -> Orchestrator:
    """Start ``python -m lmcache.v1.memory_orchestrator`` and wait for it.

    Args:
        state_dir: Directory of the startup marker.
        capacity_bytes: Region capacity.
        alignment_bytes: Extent alignment.
        visibility_mode: Region visibility mode.
        port: Port to listen on; 0 picks one.
        region_id: Region identity.

    Returns:
        The running orchestrator with the endpoint read from its marker.

    Raises:
        RuntimeError: The process exited or did not come up in time.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / f"{region_id}.marker"
    started_before = marker.stat().st_mtime_ns if marker.exists() else None
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.memory_orchestrator",
            "--region-id",
            region_id,
            "--capacity-bytes",
            str(capacity_bytes),
            "--alignment",
            str(alignment_bytes),
            "--visibility-mode",
            visibility_mode,
            "--listen",
            f"127.0.0.1:{port}",
            "--state-dir",
            str(state_dir),
        ],
        env=os.environ.copy(),
    )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"orchestrator exited with {process.returncode}")
        if marker.exists() and marker.stat().st_mtime_ns != started_before:
            try:
                info = json.loads(marker.read_text())
            except json.JSONDecodeError:
                info = {}
            if info.get("pid") == process.pid and info.get("endpoint"):
                return Orchestrator(process, info["endpoint"], state_dir)
        time.sleep(0.05)
    process.kill()
    raise RuntimeError("orchestrator did not come up")


@contextmanager
def running_orchestrator(state_dir: Path, **kwargs: object) -> Iterator[Orchestrator]:
    """Run an orchestrator for the duration of a ``with`` block."""
    orchestrator = start_orchestrator(state_dir, **kwargs)  # type: ignore[arg-type]
    try:
        yield orchestrator
    finally:
        orchestrator.stop()


def shared_config(
    region_path: Path,
    endpoint: str,
    client_id: str,
    *,
    size_bytes: int,
    tag: str = "pool",
    rpc_timeout_seconds: float = 5.0,
    region_id: str = REGION_ID,
    l1_type: str = "DRAM",
) -> L1ManagerConfig:
    """Parse a shared L1 config the way ``--l1-manager`` does.

    The default maps ``region_path`` as a file (the DRAM medium); pass
    ``l1_type="DEVDAX"`` for a Device-DAX device.
    """
    return parse_l1_manager_spec(
        {
            "type": l1_type,
            "tag": tag,
            "size_gb": size_bytes / (1 << 30),
            "path": str(region_path),
            "shared": {
                "orchestrator": endpoint,
                "region_id": region_id,
                "client_id": client_id,
                "rpc_timeout_seconds": rpc_timeout_seconds,
            },
        }
    )


def parse_l1_manager_spec(spec: dict[str, object]) -> L1ManagerConfig:
    """Parse one ``--l1-manager`` JSON object through the CLI parser."""
    config = parse_args(["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)])
    return config.l1_manager_configs[0]
