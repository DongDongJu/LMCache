# SPDX-License-Identifier: Apache-2.0
"""Fail-open DAX selection; writes both an audit record and a resolved pipeline."""

# Standard
from pathlib import Path
import json
import os
import subprocess

PREFIXES = (
    "lmcache/v1/memory_allocators/",
    "lmcache/v1/storage_backend/dax/",
    "lmcache/v1/distributed/",
    "lmcache/v1/multiprocess/",
    "lmcache/v1/platform/",
    "csrc/",
    "tests/v1/distributed/",
    ".buildkite/k3_tests/devdax/",
    ".buildkite/k3_tests/common_scripts/",
    "setup_extensions/",
    "requirements/",
)
FILES = {
    "lmcache/v1/memory_management.py",
    "lmcache/v1/storage_backend/plugins/dax_backend.py",
    "lmcache/__init__.py",
    "tests/v1/storage_backend/test_dax_backend.py",
    "tests/conftest.py",
    "tests/ci/test_devdax_ci_selection.py",
    "tests/ci/test_devdax_qemu_runner.py",
    ".buildkite/k3_tests/unit/buildkite-pipeline.yml",
    ".buildkite/k3_tests/unit/pipeline.yml",
    ".buildkite/k3_tests/unit/run.sh",
    ".github/scripts/install_lmcache_cpu.sh",
    ".github/workflows/build_devdax_image.yml",
    ".github/workflows/nightly_build.yml",
    "docker/Dockerfile.devdax",
    "setup.py",
    "CMakeLists.txt",
    "pyproject.toml",
    "pytest.ini",
}


def controls(env: dict[str, str]) -> dict[str, str]:
    """Validate toggle enums from env; return defaults or raise ValueError."""
    result = {}
    for key, choices in {
        "LMCACHE_DEVDAX_QEMU": ("auto", "on", "off"),
        "LMCACHE_DEVDAX_SUITE": ("both", "l1", "l2"),
        "LMCACHE_DEVDAX_QEMU_ACCEL": ("kvm", "tcg"),
    }.items():
        value = env.get(key, choices[0])
        if value not in choices:
            raise ValueError(f"{key}={value!r}; expected {choices}")
        result[key] = value
    return result


def _git(*args: str) -> str:
    return (
        subprocess.check_output(["git", *args], stderr=subprocess.PIPE, timeout=60)
        .decode()
        .strip()
    )


def changed_paths(env: dict[str, str]) -> tuple[str, str, list[str]]:
    """Return full before/after paths and SHAs, raising on ambiguous history.

    PRs use their base merge-base; pushes require an explicit webhook before SHA
    in LMCACHE_DEVDAX_BEFORE_SHA. Missing history runs coverage conservatively.
    """
    head = _git("rev-parse", "HEAD")
    if env.get("BUILDKITE_PULL_REQUEST", "false") not in ("false", "", "0"):
        branch = env.get("BUILDKITE_PULL_REQUEST_BASE_BRANCH") or "dev"
        _git("check-ref-format", f"refs/heads/{branch}")
        ref = f"refs/remotes/origin/{branch}"
        try:
            base = _git("merge-base", head, ref)
        except subprocess.CalledProcessError:
            # A failed deepening is an inconclusive diff, never a skip.
            if _git("rev-parse", "--is-shallow-repository") == "true":
                _git("fetch", "--unshallow", "--no-tags", "origin")
            _git("fetch", "--no-tags", "origin", f"+refs/heads/{branch}:{ref}")
            base = _git("merge-base", head, ref)
    else:
        before = env.get("LMCACHE_DEVDAX_BEFORE_SHA", "")
        if (
            not before
            or len(before) != 40
            or any(c not in "0123456789abcdef" for c in before)
        ):
            raise ValueError("push before SHA unavailable")
        base = _git("rev-parse", "--verify", f"{before}^{{commit}}")
        _git("merge-base", "--is-ancestor", base, head)
    # --no-renames emits both sides as delete/add, including newline filenames.
    data = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"],
        timeout=60,
    )
    return base, head, [os.fsdecode(p) for p in data.split(b"\0") if p]


def select(env: dict[str, str]) -> dict:
    """Return an auditable run/skip decision; invalid enums raise ValueError."""
    values = controls(env)
    result: dict = dict(values, base=None, head=None, matched_paths=[], run=True)
    toggle = values["LMCACHE_DEVDAX_QEMU"]
    if toggle != "auto":
        result.update(run=toggle == "on", reason=f"explicit {toggle}")
    elif env.get("BUILDKITE_SOURCE") == "schedule":
        result["reason"] = "scheduled build"
    elif "force-ci" in env.get("BUILDKITE_PULL_REQUEST_LABELS", "").split(","):
        result["reason"] = "force-ci label"
    else:
        try:
            base, head, paths = changed_paths(env)
            matches = [p for p in paths if p in FILES or p.startswith(PREFIXES)]
            result.update(
                base=base,
                head=head,
                matched_paths=matches,
                run=bool(matches),
                reason="relevant paths" if matches else "unrelated paths",
            )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            result["reason"] = f"inconclusive diff: {exc}"
    if result["head"] is None:
        try:
            result["head"] = _git("rev-parse", "HEAD")
        except (OSError, subprocess.SubprocessError):
            pass
    return result


def pipeline(record: dict) -> dict:
    """Render the CPU job using validated controls, without a second path gate."""
    if not record["run"]:
        return {"steps": []}
    # JSON is accepted by Buildkite; use one checked-in template.
    template = json.loads(Path(__file__).with_name("pipeline.yml").read_text())
    step = template["steps"][0]
    step["env"] = {k: record[k] for k in controls({})}
    if record["LMCACHE_DEVDAX_QEMU_ACCEL"] == "tcg":
        step["timeout_in_minutes"] = 120
    return template


def main() -> None:
    """Write selection.json and pipeline.json; errors fail the bootstrap."""
    record = select(dict(os.environ))
    output = Path(
        os.environ.get("LMCACHE_DEVDAX_ARTIFACT_DIR", "artifacts/devdax-qemu")
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "pipeline.json").write_text(json.dumps(pipeline(record), indent=2) + "\n")
    print(f"DevDAX {'selected' if record['run'] else 'SKIPPED'}: {record['reason']}")


if __name__ == "__main__":
    main()
