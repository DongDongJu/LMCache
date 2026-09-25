# SPDX-License-Identifier: Apache-2.0
"""Run each selected suite, checking exact collection and fail-closed JUnit results."""

# Standard
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from typing import TYPE_CHECKING
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

if TYPE_CHECKING:
    # Third Party
    import pytest

HERE = Path(__file__).resolve().parent


def manifests(suite: str) -> dict[str, list[str]]:
    """Return exact node IDs for l1, l2 or both; invalid selections fail."""
    if suite not in ("l1", "l2", "both"):
        raise ValueError(f"invalid suite: {suite}")
    declared = {
        "l1": [
            "test_runtime_add_and_drain_remove_lifecycle",
            "test_kv_cache_drain_gates_device_removal",
            "test_capacity_reuse_and_batch_rollback[False]",
            "test_capacity_reuse_and_batch_rollback[True]",
        ],
        "l2": [
            "test_async_capacity_locks_and_volatile_reopen",
            "test_runtime_add_drain_migrate_and_blocked_remove",
            "test_aligned_mapping_resize",
            "test_storage_manager_dram_l1_roundtrip",
        ],
    }
    if suite == "both":
        declared["l2"].append("test_storage_manager_combined_dax_roundtrip")
    modules = {
        "l1": "tests/v1/distributed/test_devdax_l1_reconfigure_integration.py",
        "l2": "tests/v1/distributed/test_dax_l2_integration.py",
    }
    selected = {
        name: [f"{modules[name]}::{test}" for test in tests]
        for name, tests in declared.items()
        if suite in (name, "both")
    }
    return selected


def check_report(output: Path, name: str, nodes: list[str], returncode: int) -> dict:
    """Validate collection and JUnit, returning a failed result for missing reports."""
    result = dict(
        state="infrastructure failure",
        collected=0,
        executed=0,
        skipped=0,
        byte_validation="unverified",
        backing="cxl-devdax",
        returncode=returncode,
    )
    try:
        collected = json.loads((output / f"collected-{name}.json").read_text())
        cases = ET.parse(output / f"junit-{name}.xml").findall(".//testcase")
        skipped = sum(c.find("skipped") is not None for c in cases)
        failed = sum(
            c.find("failure") is not None or c.find("error") is not None for c in cases
        )
        result.update(
            collected=len(collected),
            executed=len(cases) - skipped,
            skipped=skipped,
            nodeids=collected,
        )
        if (
            sorted(collected) != sorted(nodes)
            or len(cases) != len(nodes)
            or not nodes
            or sorted(c.get("name", "") for c in cases)
            != sorted(n.rsplit("::", 1)[-1] for n in nodes)
        ):
            return result
        passed = returncode == 0 and skipped == 0 and failed == 0
        result.update(
            state="passed" if passed else "failed",
            byte_validation="passed" if passed else "failed",
        )
    except (OSError, ValueError, ET.ParseError):
        pass
    return result


def pytest_collection_finish(session: "pytest.Session") -> None:
    """Record exact node IDs for the selected suite's report."""
    Path(os.environ["LMCACHE_DEVDAX_COLLECTION"]).write_text(
        json.dumps([item.nodeid for item in session.items]) + "\n"
    )


def main() -> None:
    """Run CPU/native strict device tests serially, preserving both suite outcomes."""
    os.environ.update(
        NO_GPU_EXT="1",
        MAX_JOBS="4",
        SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0.dev0",
        LMCACHE_TRACK_USAGE="false",
    )
    for name in ("NO_NATIVE_EXT", "LMCACHE_DEVICE_BACKEND"):
        os.environ.pop(name, None)
    for args in (
        ["-m", "pip", "install", "-e", ".", "--no-deps", "--no-build-isolation"],
        ["-m", "pip", "check"],
        ["lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py"],
    ):
        subprocess.run([sys.executable, *args], check=True, timeout=900)
    sys.path.insert(0, str(Path.cwd()))

    # Third Party
    import torch

    # First Party
    import lmcache
    import lmcache.lmcache_native as native

    if not any(native.__file__.endswith(s) for s in EXTENSION_SUFFIXES):
        raise RuntimeError("real compiled lmcache_native extension required")
    assert Path(lmcache.__file__).is_relative_to("/root/source")
    assert torch.version.cuda is None and lmcache.torch_device_type == "cpu"
    versions = dict(
        kernel=platform.release(),
        python=platform.python_version(),
        torch=torch.__version__,
        lmcache=lmcache.__version__,
        native=native.__file__,
        cxl=subprocess.check_output(["cxl", "--version"], text=True).strip(),
        daxctl=subprocess.check_output(["daxctl", "--version"], text=True).strip(),
    )
    Path("artifacts/devdax-qemu/versions.json").write_text(
        json.dumps(versions, indent=2) + "\n"
    )
    selected = manifests(os.environ.get("LMCACHE_DEVDAX_SUITE", "both"))
    output = Path("artifacts/devdax-qemu").resolve()
    paths = Path("/run/lmcache-dax-paths").read_text().strip()
    env = dict(
        os.environ,
        LMCACHE_TEST_REQUIRE_REAL_DEVDAX="1",
        RUN_DEVDAX_L1_INTEGRATION="1",
        RUN_DAX_L2_INTEGRATION="1",
        LMCACHE_TEST_DEVDAX_L1_PATHS=paths,
        LMCACHE_TEST_DEVDAX_L2_PATHS=paths,
        LMCACHE_TEST_DEVDAX_MANIFEST=str(output / "devices.json"),
        PYTHONPATH=str(HERE),
        PYTEST_ADDOPTS="",
        OMP_NUM_THREADS="2",
        LMCACHE_DEVICE_BACKEND="",
        LMCACHE_TRACK_USAGE="false",
    )
    summary: dict[str, dict] = {
        name: {"state": "not selected"} for name in ("l1", "l2")
    }
    for name, nodes in selected.items():
        start = time.monotonic()
        env["LMCACHE_DEVDAX_COLLECTION"] = str(output / f"collected-{name}.json")
        with (output / f"pytest-{name}.log").open("w") as log:
            try:
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-v",
                        "-s",
                        "-p",
                        "guest_test",
                        "-o",
                        "xfail_strict=true",
                        *nodes,
                        f"--junitxml={output}/junit-{name}.xml",
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600
                    if env.get("LMCACHE_DEVDAX_QEMU_ACCEL") != "tcg"
                    else 2400,
                )
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 124
        summary[name] = check_report(output, name, nodes, code)
        summary[name]["duration_seconds"] = time.monotonic() - start
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(name, summary[name], flush=True)
    sys.exit(any(summary[name]["state"] != "passed" for name in selected))


if __name__ == "__main__":
    main()
