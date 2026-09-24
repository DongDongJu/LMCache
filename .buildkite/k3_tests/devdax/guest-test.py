# SPDX-License-Identifier: Apache-2.0
"""Run each selected suite, checking exact collection and fail-closed JUnit results."""

# Standard
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent


def manifests(suite: str) -> dict[str, list[str]]:
    """Return exact node IDs for l1, l2 or both; invalid selections fail."""
    if suite not in ("l1", "l2", "both"):
        raise ValueError(f"invalid suite: {suite}")
    declared = json.loads((HERE / "suites.json").read_text())
    selected = {
        name: declared[name] for name in ("l1", "l2") if suite in (name, "both")
    }
    if suite == "both":
        selected["l2"].append(declared["combined"])
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


def main() -> None:
    """Run CPU/native strict device tests serially, preserving both suite outcomes."""
    selected = manifests(os.environ.get("LMCACHE_DEVDAX_SUITE", "both"))
    spec = importlib.util.find_spec("lmcache.lmcache_native")
    if spec is None or not any(
        (spec.origin or "").endswith(s) for s in EXTENSION_SUFFIXES
    ):
        raise RuntimeError("real compiled lmcache_native extension required")
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
                        "pytest_report",
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
