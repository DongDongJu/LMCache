# SPDX-License-Identifier: Apache-2.0
"""Owned-process readiness, inode proof and optional request rendezvous."""

# Standard
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen
import json
import os
import stat
import subprocess
import time


@dataclass(frozen=True)
class FileBarrier:
    """Bounded rendezvous in a caller-created private directory.

    Args:
        directory: Fresh mode0700 directory shared by this group only.
        rank: This rank, in [0, workers).
        workers: Exact number of participating ranks.
        timeout: Maximum seconds to wait for each store/read phase.

    Files persist as evidence. A duplicate rank/phase/sequence is an error;
    an ``abort`` file makes all remaining waiters fail promptly.
    """

    directory: Path
    rank: int
    workers: int
    timeout: float

    def wait(self, phase: str, sequence: int) -> None:
        """Publish readiness and wait for every known rank before admission.

        Args:
            phase: Exactly store or read.
            sequence: Shared nonnegative logical request ID.

        Raises:
            ValueError: Identity or private-directory checks fail.
            FileExistsError: This rank already entered the same rendezvous.
            TimeoutError: A required rank did not arrive by the deadline.
            RuntimeError: The parent or another worker aborted the group.
        """
        state = self.directory.lstat()
        if (
            phase not in {"store", "read"}
            or sequence < 0
            or not 0 <= self.rank < self.workers
            or self.timeout <= 0
            or not stat.S_ISDIR(state.st_mode)
            or state.st_uid != os.getuid()
            or stat.S_IMODE(state.st_mode) & 0o077
        ):
            raise ValueError("Invalid barrier phase, rank or private directory")
        prefix = f"{sequence}-{phase}-rank"
        with (self.directory / f"{prefix}{self.rank}.json").open("x") as stream:
            json.dump(
                {
                    "rank": self.rank,
                    "pid": os.getpid(),
                    "ready_ns": time.monotonic_ns(),
                },
                stream,
            )
        deadline = time.monotonic() + self.timeout
        while True:
            if (self.directory / "abort").exists():
                raise RuntimeError("Coordinated MP group aborted")
            missing = [
                rank
                for rank in range(self.workers)
                if not (self.directory / f"{prefix}{rank}.json").is_file()
            ]
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Barrier {sequence}/{phase}: missing ranks {missing}"
                )
            time.sleep(0.001)


def server_owns_slab(
    server: subprocess.Popen[bytes], path: Path, size: int
) -> dict[str, object]:
    """Validate a newly created regular slab and identify its owning server FD.

    Args:
        server: Child server launched by this orchestrator.
        path: Expected slab in a fresh private rank directory.
        size: Exact requested byte capacity.

    Returns:
        Device/inode/size and owning server/descendant PID evidence.

    Raises:
        RuntimeError: Server exited or no owned process holds the expected inode.
        ValueError: File identity, ownership or directory isolation is wrong.
        OSError: Ownership evidence cannot be inspected.
    """
    state, parent = path.lstat(), path.parent.lstat()
    if (
        server.poll() is not None
        or not stat.S_ISREG(state.st_mode)
        or state.st_size != size
        or state.st_uid != os.getuid()
        or state.st_nlink != 1
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise ValueError("Expected fresh private server-owned regular slab")
    pending, owners = [server.pid], []
    while pending:
        pid = pending.pop()
        children = Path(f"/proc/{pid}/task/{pid}/children")
        if children.exists():
            pending.extend(int(value) for value in children.read_text().split())
        for fd in Path(f"/proc/{pid}/fd").glob("*"):
            try:
                opened = fd.stat()
            except FileNotFoundError:
                continue
            if (opened.st_dev, opened.st_ino) == (state.st_dev, state.st_ino):
                owners.append(pid)
    if not owners:
        raise RuntimeError("Cannot prove this server owns the new slab inode")
    return {
        "file": str(path),
        "device": state.st_dev,
        "inode": state.st_ino,
        "bytes": state.st_size,
        "uid": state.st_uid,
        "server_pid": server.pid,
        "open_fd_owner_pids": sorted(set(owners)),
        "private_directory": True,
    }


def wait_ready(server: subprocess.Popen[bytes], url: str, timeout: float) -> None:
    """Wait for a live owned server's health endpoint, bounded by timeout.

    Args:
        server: Launched child process.
        url: Its loopback health URL.
        timeout: Maximum startup duration in seconds.

    Raises:
        RuntimeError: Server exits before readiness.
        TimeoutError: Healthy readiness is not observed.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Owned MP server exited {server.returncode}")
        try:
            with urlopen(url + "/healthcheck", timeout=1) as response:
                if json.load(response).get("status") == "healthy":
                    return
        except (URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(0.1)
    raise TimeoutError(f"MP server readiness timed out: {url}")
