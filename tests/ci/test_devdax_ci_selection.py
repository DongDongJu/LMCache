# SPDX-License-Identifier: Apache-2.0
"""DAX enable/disable, strict fixture and report checks without LMCache imports."""

# Standard
from pathlib import Path
from types import ModuleType
import importlib.util
import json
import os
import subprocess

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


guest_test = _module(CI / "guest_test.py")
provider = _module(ROOT / "tests/v1/distributed/dax_test_utils.py")


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated workspace for exercising the real CI entrypoints."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def uploader(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Copy the uploader and record Buildkite calls without contacting Buildkite."""
    target = repo / ".buildkite/k3_tests/devdax"
    target.mkdir(parents=True)
    for name in ("upload.sh", "pipeline.yml"):
        (target / name).write_bytes((CI / name).read_bytes())
    binary = repo / "bin"
    binary.mkdir()
    agent = binary / "buildkite-agent"
    agent.write_text("""#!/usr/bin/env python3
from pathlib import Path
import json, os, sys
args = sys.argv[1:]
with open('agent-calls.jsonl', 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[:2] == ['step', 'get']:
    sys.exit(0 if Path('uploaded').exists() else 1)
if args[:2] == ['pipeline', 'upload']:
    if os.environ.get('FAIL_UPLOAD') == '1':
        sys.exit(23)
    Path('uploaded').touch()
    with open('uploads', 'a') as log:
        log.write(Path(args[2]).read_text() + '\\n')
""")
    agent.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.delenv("LMCACHE_DEVDAX_QEMU", raising=False)
    monkeypatch.delenv("LMCACHE_DEVDAX_ARTIFACT_DIR", raising=False)
    return target / "upload.sh"


@pytest.mark.parametrize("mode", [None, "on", "off"])
def test_enable_disable(
    uploader: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    """Default/on upload the fixed job; off explicitly reports skipped coverage."""
    if mode is not None:
        monkeypatch.setenv("LMCACHE_DEVDAX_QEMU", mode)
    subprocess.run(["bash", str(uploader)], check=True, capture_output=True)
    record = json.loads((repo / "artifacts/devdax-qemu/selection.json").read_text())
    assert record == {"LMCACHE_DEVDAX_QEMU": mode or "on", "run": mode != "off"}
    if mode == "off":
        assert not (repo / "uploads").exists()
        assert "SKIPPED" in (repo / "agent-calls.jsonl").read_text()
    else:
        job = json.loads((repo / "uploads").read_text())["steps"][0]
        assert job["agents"] == {"queue": "devdax-qemu"}
        assert job["timeout_in_minutes"] == 30
        assert "if_changed" not in job
        assert "env" not in job


@pytest.mark.parametrize("mode", ["auto", "invalid", ""])
def test_invalid_toggle(
    uploader: Path, repo: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Only on/off are accepted; invalid configuration cannot upload a job."""
    monkeypatch.setenv("LMCACHE_DEVDAX_QEMU", mode)
    result = subprocess.run(["bash", str(uploader)], capture_output=True)
    assert result.returncode == 2
    assert b"must be on or off" in result.stderr
    assert not (repo / "agent-calls.jsonl").exists()


def test_both_suites_required() -> None:
    """The manifest always includes all four L1 and five L2 cases."""
    manifests = guest_test.manifests()
    assert {name: len(nodes) for name, nodes in manifests.items()} == {"l1": 4, "l2": 5}
    assert manifests["l2"][-1].endswith("::test_storage_manager_combined_dax_roundtrip")


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
    uploader: Path, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upload failure reaches CI, and retrying a successful upload adds no duplicate."""
    monkeypatch.setenv("FAIL_UPLOAD", "1")
    command = ["bash", str(uploader)]
    assert subprocess.run(command, capture_output=True).returncode == 23
    assert not (repo / "uploaded").exists()
    monkeypatch.delenv("FAIL_UPLOAD")
    subprocess.run(command, check=True, capture_output=True)
    first = (repo / "uploads").read_bytes()
    subprocess.run(command, check=True, capture_output=True)
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


def test_container_entrypoint_uses_image_id_and_propagates_failure(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container gets current source and required devices; failures reach CI."""
    runner = repo / ".buildkite/k3_tests/devdax/run.sh"
    helpers = repo / ".buildkite/k3_tests/common_scripts/helpers.sh"
    for target, source in (
        (runner, CI / "run.sh"),
        (helpers, CI.parent / "common_scripts/helpers.sh"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    binary = repo / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text("""#!/usr/bin/env python3
import json, sys
args = sys.argv[1:]
with open('docker-calls.jsonl', 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[:2] == ['image', 'inspect']:
    print('sha256:verified' if args[3] == '{{.Id}}' else '{}')
if args[0] == 'run':
    sys.exit(23)
""")
    docker.chmod(0o755)
    stat = binary / "stat"
    stat.write_text("#!/bin/sh\necho 1234\n")
    stat.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binary}:{os.environ['PATH']}")
    monkeypatch.setenv("BUILDKITE", "true")
    monkeypatch.setenv("BUILDKITE_PULL_REQUEST", "false")
    monkeypatch.delenv("LMCACHE_DEVDAX_CONTAINER_IMAGE", raising=False)
    result = subprocess.run(["bash", str(runner)], cwd=repo, capture_output=True)
    assert result.returncode == 23, result.stderr
    calls = [
        json.loads(line)
        for line in (repo / "docker-calls.jsonl").read_text().splitlines()
    ]
    assert calls[0] == ["pull", "--", "ghcr.io/lmcache/lmcache-devdax-ci:nightly"]
    command = calls[-1]
    assert command[command.index("--") + 1] == "sha256:verified"
    assert command[command.index("--volume") + 1] == f"{repo}:{repo}"
    assert command[command.index("--device") + 1] == "/dev/kvm"
    assert "--privileged" not in command
