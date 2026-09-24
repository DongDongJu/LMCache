# SPDX-License-Identifier: Apache-2.0
"""DAX routing, strict fixture, manifest and report checks without LMCache imports."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import json
import os
import subprocess
import sys

# Third Party
import pytest

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".buildkite/k3_tests/devdax"


def _module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection = _module(CI / "ci_selection.py")
guest_test = _module(CI / "guest-test.py")
provider = _module(ROOT / "tests/v1/distributed/dax_test_utils.py")


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _commit(repo: Path, name: str) -> str:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(path.read_text() + "change\n" if path.exists() else "change\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", name)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a synthetic PR branch whose base is available as origin/dev."""
    _git(tmp_path, "init", "-b", "dev")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    base = _commit(tmp_path, "initial")
    _git(tmp_path, "update-ref", "refs/remotes/origin/dev", base)
    _git(tmp_path, "checkout", "-qb", "feature")
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    "path",
    [
        "lmcache/v1/distributed/memory_manager/devdax_l1_memory_manager.py",
        "lmcache/v1/distributed/l2_adapters/dax_l2_adapter.py",
        "lmcache/v1/memory_allocators/foo.py",
        "csrc/common.cpp",
        "tests/v1/distributed/test_dax_l2_integration.py",
        ".buildkite/k3_tests/devdax/image/manifest.json",
        ".github/scripts/install_lmcache_cpu.sh",
        "requirements/common.txt",
        ".buildkite/k3_tests/common_scripts/helpers.sh",
    ],
)
def test_relevant_paths_render_cpu_job(repo: Path, path: str) -> None:
    """Dependency and installer changes reach the resolved QEMU step."""
    _commit(repo, path)
    record = selection.select({"BUILDKITE_PULL_REQUEST": "1"})
    assert record["run"] and path in record["matched_paths"]
    job = selection.pipeline(record)["steps"][0]
    assert job["env"]["LMCACHE_DEVDAX_SUITE"] == "both"
    assert job["env"]["LMCACHE_DEVDAX_QEMU_ACCEL"] == "kvm"
    assert job["agents"] == {"queue": "devdax-qemu"}
    assert "if_changed" not in job
    assert "gpu" not in json.dumps(job).lower()


@pytest.mark.parametrize(
    ("env", "run"),
    [
        ({}, False),
        ({"LMCACHE_DEVDAX_QEMU": "on"}, True),
        ({"LMCACHE_DEVDAX_QEMU": "off", "BUILDKITE_SOURCE": "schedule"}, False),
        ({"BUILDKITE_SOURCE": "schedule"}, True),
        ({"BUILDKITE_PULL_REQUEST_LABELS": "docs,force-ci"}, True),
        (
            {"LMCACHE_DEVDAX_QEMU": "off", "BUILDKITE_PULL_REQUEST_LABELS": "force-ci"},
            False,
        ),
    ],
)
def test_readme_precedence(repo: Path, env: dict[str, str], run: bool) -> None:
    """README changes honor automatic skips, explicit overrides and nightly."""
    _commit(repo, "README.md")
    record = selection.select(dict(env, BUILDKITE_PULL_REQUEST="1"))
    assert record["run"] is run
    assert bool(selection.pipeline(record)["steps"]) is run


@pytest.mark.parametrize("suite", ["l1", "l2", "both"])
@pytest.mark.parametrize("accel", ["kvm", "tcg"])
def test_resolved_controls_and_exact_manifests(suite: str, accel: str) -> None:
    """Only both includes the combined case, and controls reach uploaded env."""
    record = selection.select(
        {
            "LMCACHE_DEVDAX_QEMU": "on",
            "LMCACHE_DEVDAX_SUITE": suite,
            "LMCACHE_DEVDAX_QEMU_ACCEL": accel,
        }
    )
    job = selection.pipeline(record)["steps"][0]
    assert job["env"]["LMCACHE_DEVDAX_SUITE"] == suite
    assert job["env"]["LMCACHE_DEVDAX_QEMU_ACCEL"] == accel
    manifests = guest_test.manifests(suite)
    assert set(manifests) == ({"l1", "l2"} if suite == "both" else {suite})
    assert sum(map(len, manifests.values())) == (9 if suite == "both" else 4)
    assert any(
        "combined" in node for nodes in manifests.values() for node in nodes
    ) is (suite == "both")


@pytest.mark.parametrize(
    "key", ["LMCACHE_DEVDAX_QEMU", "LMCACHE_DEVDAX_SUITE", "LMCACHE_DEVDAX_QEMU_ACCEL"]
)
def test_invalid_controls_fail_before_off(key: str) -> None:
    """No invalid enum silently disables requested coverage."""
    with pytest.raises(ValueError, match=key):
        selection.select({"LMCACHE_DEVDAX_QEMU": "off", key: "invalid"})


def test_delete_rename_and_newline_names(repo: Path) -> None:
    """Both sides of renames and deleted paths remain visible to the selector."""
    base = _commit(repo, "csrc/deleted.cpp")
    _git(repo, "update-ref", "refs/remotes/origin/dev", base)
    _git(repo, "mv", "csrc/deleted.cpp", "unrelated.md")
    _commit(repo, "tests/v1/distributed/a\nb.py")
    record = selection.select({"BUILDKITE_PULL_REQUEST": "1"})
    assert record["run"]
    assert set(record["matched_paths"]) == {
        "csrc/deleted.cpp",
        "tests/v1/distributed/a\nb.py",
    }


def test_multi_commit_push_and_unknown_range(repo: Path) -> None:
    """A docs-only tip does not hide an earlier DAX commit in the push."""
    before = _git(repo, "rev-parse", "HEAD")
    _commit(repo, "csrc/change.cpp")
    _commit(repo, "README.md")
    assert selection.select({"LMCACHE_DEVDAX_BEFORE_SHA": before})["matched_paths"] == [
        "csrc/change.cpp"
    ]
    assert selection.select({})["run"]
    _git(repo, "update-ref", "-d", "refs/remotes/origin/dev")
    record = selection.select({"BUILDKITE_PULL_REQUEST": "1"})
    assert record["run"] and "inconclusive" in record["reason"]


def test_shallow_pr_deepens_history(
    repo: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shallow PR fetch recovers its full merge-base rather than skipping."""
    _commit(repo, "csrc/earlier.cpp")
    _commit(repo, "README.md")
    clone = tmp_path_factory.mktemp("clone")
    subprocess.run(
        ["git", "clone", "--depth=1", "--branch=feature", repo.as_uri(), str(clone)],
        check=True,
        capture_output=True,
    )
    monkeypatch.chdir(clone)
    record = selection.select({"BUILDKITE_PULL_REQUEST": "1"})
    assert record["run"] and record["matched_paths"] == ["csrc/earlier.cpp"]


def test_actual_cli_output(repo: Path) -> None:
    """The executable writes both selection evidence and the uploaded job JSON."""
    _commit(repo, "README.md")
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LMCACHE_DEVDAX", "BUILDKITE_"))
    }
    env.update(
        BUILDKITE_PULL_REQUEST="1", LMCACHE_DEVDAX_QEMU="on", LMCACHE_DEVDAX_SUITE="l2"
    )
    subprocess.run(
        [sys.executable, str(CI / "ci_selection.py")], cwd=repo, env=env, check=True
    )
    output = repo / "artifacts/devdax-qemu"
    assert (
        json.loads((output / "selection.json").read_text())["reason"] == "explicit on"
    )
    assert (
        json.loads((output / "pipeline.json").read_text())["steps"][0]["env"][
            "LMCACHE_DEVDAX_SUITE"
        ]
        == "l2"
    )


@pytest.mark.parametrize("kind", ["file", "missing", "null", "absent"])
def test_strict_provider_rejects_invalid_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Strict mode cannot replace a missing or invalid DAX device with a file."""
    path = tmp_path / "file"
    path.write_bytes(b"keep this data")
    value = {
        "file": str(path),
        "missing": str(tmp_path / "missing"),
        "null": "/dev/null",
        "absent": "",
    }[kind]
    monkeypatch.setenv("LMCACHE_TEST_REQUIRE_REAL_DEVDAX", "1")
    monkeypatch.setenv("LMCACHE_TEST_DEVDAX_L1_PATHS", value)
    with pytest.raises((ValueError, OSError)):
        provider.DeviceProvider(tmp_path, "l1")
    assert path.read_bytes() == b"keep this data"


def test_file_provider_isolation_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordinary fixtures are private files; unsupported lengths fail before mmap."""
    monkeypatch.delenv("LMCACHE_TEST_REQUIRE_REAL_DEVDAX", raising=False)
    monkeypatch.delenv("LMCACHE_TEST_DEVDAX_L1_PATHS", raising=False)
    devices = provider.DeviceProvider(tmp_path, "l1")
    with pytest.raises(ValueError, match="multiple"):
        devices.acquire(4096)
    first, second = (
        devices.acquire(devices.slot_bytes),
        devices.acquire(devices.slot_bytes),
    )
    assert first != second
    assert Path(first).read_bytes()[:4096] == bytes(range(256)) * 16


@pytest.mark.parametrize(
    "fault", ["missing", "skip", "mismatch", "empty", "wrong-collection", "exit"]
)
def test_report_fails_closed(tmp_path: Path, fault: str) -> None:
    """Missing reports, payload assertions, skips and empty suites cannot pass."""
    nodes = ["tests/example.py::test_payload"]
    collection = [] if fault in ("empty", "wrong-collection") else nodes
    (tmp_path / "collected-l1.json").write_text(json.dumps(collection))
    if fault != "missing":
        child = {
            "skip": "<skipped/>",
            "mismatch": "<failure>payload mismatch</failure>",
        }.get(fault, "")
        cases = (
            ""
            if fault == "empty"
            else f'<testcase name="test_payload">{child}</testcase>'
        )
        (tmp_path / "junit-l1.xml").write_text(
            f"<testsuites><testsuite>{cases}</testsuite></testsuites>"
        )
    result = guest_test.check_report(tmp_path, "l1", nodes, int(fault == "exit"))
    assert result["state"] != "passed"


def test_provider_manifest_alias_and_capacity_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest mismatches, aliases and short capacity fail before any mmap write."""
    record = dict(
        path="/dev/example",
        major=250,
        minor=0,
        size=1 << 20,
        alignment=2 << 20,
        mode="devdax",
        driver="device_dax",
        parent="/sys/devices/region0",
        sysfs="/sys/devices/region0/dax0.0",
    )
    monkeypatch.setattr(provider, "inspect_device", lambda path: record)
    monkeypatch.setenv("LMCACHE_TEST_REQUIRE_REAL_DEVDAX", "1")
    monkeypatch.setenv("LMCACHE_TEST_DEVDAX_L1_PATHS", "/dev/example,/dev/alias")
    with pytest.raises(ValueError, match="duplicate"):
        provider.DeviceProvider(tmp_path, "l1", count=1)
    monkeypatch.setenv("LMCACHE_TEST_DEVDAX_L1_PATHS", "/dev/example")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    monkeypatch.setenv("LMCACHE_TEST_DEVDAX_MANIFEST", str(manifest))
    with pytest.raises(ValueError, match="manifest"):
        provider.DeviceProvider(tmp_path, "l1", count=1)
    manifest.write_text(json.dumps([record]))
    devices = provider.DeviceProvider(tmp_path, "l1", count=1)
    with pytest.raises(ValueError, match="capacity"):
        devices.acquire(devices.slot_bytes)


def test_uploader_failure_and_retry(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed upload fails the bootstrap; a successful retry uploads only once."""
    # Copy the actual entrypoint and template, preserving its relative root.
    target = repo / ".buildkite/k3_tests/devdax"
    target.mkdir(parents=True)
    for name in ("upload.sh", "ci_selection.py", "pipeline.yml"):
        (target / name).write_bytes((CI / name).read_bytes())
    binary = repo / "bin"
    binary.mkdir()
    stub = binary / "buildkite-agent"
    stub.write_text("""#!/usr/bin/env python3
from pathlib import Path
import os, sys
args = sys.argv[1:]
if args[:2] == ['step', 'get']:
    sys.exit(0 if Path('uploaded').exists() else 1)
if args[:2] == ['pipeline', 'upload']:
    if os.environ.get('FAIL_UPLOAD') == '1':
        sys.exit(23)
    Path('uploaded').touch()
    with open('uploads', 'a') as log:
        log.write(Path(args[2]).read_text() + '\\n')
""")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("LMCACHE_DEVDAX_QEMU", "on")
    monkeypatch.setenv("FAIL_UPLOAD", "1")
    command = ["bash", str(target / "upload.sh")]
    assert subprocess.run(command, cwd=repo, capture_output=True).returncode == 23
    assert not (repo / "uploaded").exists()
    monkeypatch.delenv("FAIL_UPLOAD")
    subprocess.run(command, cwd=repo, check=True, capture_output=True)
    first = (repo / "uploads").read_bytes()
    subprocess.run(command, cwd=repo, check=True, capture_output=True)
    assert (repo / "uploads").read_bytes() == first


def test_bootstraps_and_yaml_keep_unit_independent() -> None:
    """DAX has one independent unit bootstrap command and no GPU pod template."""
    # Third Party
    import yaml

    unit = yaml.safe_load((CI.parent / "unit/buildkite-pipeline.yml").read_text())
    commands = [step["command"] for step in unit["steps"]]
    assert commands == [
        "bash .buildkite/k3_tests/common_scripts/upload-pipeline.sh "
        ".buildkite/k3_tests/unit/pipeline.yml",
        "bash .buildkite/k3_tests/devdax/upload.sh",
    ]
    for name in ("pipeline.yml", "buildkite-pipeline.yml"):
        assert yaml.safe_load((CI / name).read_text())["steps"]


@pytest.mark.parametrize(
    ("strict", "worker", "message"),
    [("invalid", "", "0 or 1"), ("1", "gw0", "serially")],
)
def test_strict_configuration_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strict: str,
    worker: str,
    message: str,
) -> None:
    """Invalid controls and concurrent owners fail before opening devices."""
    monkeypatch.setenv("LMCACHE_TEST_REQUIRE_REAL_DEVDAX", strict)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", worker)
    with pytest.raises(ValueError, match=message):
        provider.DeviceProvider(tmp_path, "l1")
