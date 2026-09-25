# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-VM failure injection: RUN_DEVDAX_QEMU_HOST_TESTS=1.

Run with --confcutdir=tests/ci on the same KVM host as run.sh. Uses the prepared
image/kernel in LMCACHE_DEVDAX_IMAGE and LMCACHE_DEVDAX_KERNEL.
"""

# Standard
from pathlib import Path
from types import ModuleType
from typing import Any
import importlib.util
import json
import os
import subprocess
import sys
import time

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".buildkite/k3_tests/devdax"
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DEVDAX_QEMU_HOST_TESTS") != "1",
    reason="opt-in QEMU host failure injection",
)


def _load_runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(CI))
    spec = importlib.util.spec_from_file_location("devdax_runner", CI / "runner.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "fault", ["payload", "missing-devices", "boot-timeout", "guest-timeout"]
)
def test_real_vm_failure_propagation(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A real guest failure must return failure, keep diagnostics and remove QEMU."""
    runner = _load_runner(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["runner.py"])
    monkeypatch.setenv("LMCACHE_DEVDAX_SUITE", "both")
    if fault == "missing-devices":
        # Physically omit emulated CXL devices from this disposable VM.
        monkeypatch.setattr(runner, "topology", lambda: [])
    if fault == "boot-timeout":
        monkeypatch.setenv("LMCACHE_DEVDAX_BOOT_TIMEOUT", "0")
    original = runner.Guest.ssh
    original_close = runner.Guest.close
    guests = []

    def close(guest: Any) -> None:
        guests.append(guest)
        original_close(guest)

    monkeypatch.setattr(runner.Guest, "close", close)

    def ssh(guest: Any, command: str, timeout: int = 60, **kwargs: object) -> object:
        guests.append(guest)
        if fault == "payload" and command.endswith("guest_test.py"):
            original(
                guest,
                "sed -i 's/raw_tensor == 0xAB/raw_tensor == 0xAA/' "
                "/root/source/tests/v1/distributed/test_devdax_l1_reconfigure_integration.py",
            )
        if fault == "guest-timeout" and command.endswith("guest-setup.py"):
            command = "sleep 30"
            timeout = 1
        return original(guest, command, timeout=timeout, **kwargs)

    monkeypatch.setattr(runner.Guest, "ssh", ssh)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code != 0
    assert guests
    output = guests[-1].output
    summary = json.loads((output / "summary.json").read_text())
    assert summary["l1"]["state"] != "passed"
    if fault == "payload":
        assert summary["l1"]["state"] == "failed"
        assert summary["l2"]["state"] == "passed"
        assert (output / "junit-l1.xml").is_file()
    else:
        assert (output / "infrastructure-error.txt").is_file()
    for guest in guests:
        assert guest.process.poll() is not None
        assert not guest.scratch.exists()
    print(f"{fault} diagnostics: {output}")


def test_cancellation_preserves_other_vm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cancelling one runner removes its VM while another real QEMU stays alive."""
    runner = _load_runner(monkeypatch)
    scratch = tmp_path / "other"
    scratch.mkdir()
    output = tmp_path / "logs"
    output.mkdir()
    image = Path(os.environ["LMCACHE_DEVDAX_IMAGE"]).resolve()
    kernel = Path(os.environ["LMCACHE_DEVDAX_KERNEL"]).resolve()
    disk = scratch / "overlay.qcow2"
    subprocess.run(
        [
            "qemu-img",
            "create",
            "-f",
            "qcow2",
            "-F",
            "qcow2",
            "-b",
            str(image),
            str(disk),
        ],
        check=True,
    )
    other = runner.Guest(scratch, output, disk, kernel)
    victim = None
    victim_child = None
    try:
        other.boot(300)
        with (tmp_path / "cancel.log").open("w") as log:
            victim = subprocess.Popen(
                [sys.executable, str(CI / "runner.py")],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                children = (
                    Path(f"/proc/{victim.pid}/task/{victim.pid}/children")
                    .read_text()
                    .split()
                )
                for pid in children:
                    try:
                        if "qemu-system" in os.readlink(f"/proc/{pid}/exe"):
                            victim_child = int(pid)
                            break
                    except FileNotFoundError:
                        pass
                if victim_child is not None:
                    break
                time.sleep(0.1)
            assert victim_child is not None
            victim.terminate()
            assert victim.wait(timeout=45) != 0
            assert not Path(f"/proc/{victim_child}").exists()
            assert other.process.poll() is None
            other.ssh("true")
    finally:
        if victim is not None and victim.poll() is None:
            victim.terminate()
            victim.wait(timeout=45)
        other.close()
