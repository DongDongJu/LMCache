# SPDX-License-Identifier: Apache-2.0
"""Provision only the disposable LMCache QEMU guest's volatile CXL devices."""

# Standard
from pathlib import Path
import json
import os
import platform
import subprocess
import sys

sys.path.insert(0, str(Path.cwd() / "tests/v1/distributed"))

# Third Party
from dax_test_utils import DeviceProvider, inspect_device


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, timeout=60)


def _nodes(value: object) -> list[dict]:
    if isinstance(value, list):
        return [node for child in value for node in _nodes(child)]
    if isinstance(value, dict):
        return [value] + [node for child in value.values() for node in _nodes(child)]
    return []


def main() -> None:
    """Create three independent RAM regions, save topology, and validate devdax."""
    if (
        Path("/sys/class/dmi/id/product_name").read_text().strip()
        != "LMCache-DevDAX-QEMU"
    ):
        raise RuntimeError(
            "provisioning is allowed only inside the disposable QEMU guest"
        )
    output = Path("artifacts/devdax-qemu")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        Path(".buildkite/k3_tests/devdax/image/manifest.json").read_text()
    )
    if platform.release() != manifest["kernel"]:
        raise RuntimeError("guest kernel differs from the pinned test kernel")
    (output / "kernel.config").write_text(_run("zcat", "/proc/config.gz"))
    _run("udevadm", "settle", "--timeout=30")
    (output / "cxl-list.json").write_text(_run("cxl", "list", "-M", "-D", "-R", "-T"))
    memdevs = [
        n["memdev"]
        for n in _nodes(json.loads(_run("cxl", "list", "-M")))
        if "memdev" in n
    ]
    decoders = [
        n["decoder"]
        for n in _nodes(json.loads(_run("cxl", "list", "-D", "-d", "root")))
        if n.get("volatile_capable")
    ]
    if len(memdevs) != 3 or not decoders:
        raise RuntimeError(
            f"expected three volatile memdevs and RAM decoder: {memdevs}, {decoders}"
        )
    Path("/sys/devices/system/memory/auto_online_blocks").write_text("offline")
    for memdev in memdevs:
        _run(
            "cxl",
            "create-region",
            "-t",
            "ram",
            "-m",
            "-d",
            decoders[0],
            "-w",
            "1",
            "-s",
            "256M",
            memdev,
        )
    _run("udevadm", "settle", "--timeout=30")
    dax_list = _run("daxctl", "list", "-R", "-D")
    (output / "daxctl-list.json").write_text(dax_list)
    (output / "cxl-list.json").write_text(_run("cxl", "list", "-M", "-D", "-R", "-T"))
    devices = [n for n in _nodes(json.loads(dax_list)) if "chardev" in n]
    if len(devices) != 3 or any(n.get("mode") != "devdax" for n in devices):
        raise RuntimeError(f"expected three independent devdax devices: {devices}")
    records = [inspect_device("/dev/" + n["chardev"]) for n in devices]
    if len({r["parent"] for r in records}) != 3:
        raise RuntimeError("DAX devices do not have independent CXL regions")
    (output / "devices.json").write_text(json.dumps(records, indent=2) + "\n")
    paths = ",".join(r["path"] for r in records)
    Path("/run/lmcache-dax-paths").write_text(paths)
    os.environ.update(
        LMCACHE_TEST_REQUIRE_REAL_DEVDAX="1",
        LMCACHE_TEST_DEVDAX_L1_PATHS=paths,
        LMCACHE_TEST_DEVDAX_MANIFEST=str(output / "devices.json"),
    )
    provider = DeviceProvider(output, "l1")
    for _ in records:
        provider.acquire(4 * provider.slot_bytes)


if __name__ == "__main__":
    main()
