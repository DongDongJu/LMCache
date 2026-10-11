# SPDX-License-Identifier: Apache-2.0
"""CPU-only allocation ownership and portable campaign configuration checks."""

# Standard
from pathlib import Path
import json
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from benchmarks.gds_optimization.run import build_plan, parse_args
from benchmarks.gds_optimization.slab import SlabLease, pattern_bytes


def arguments(tmp_path: Path) -> list[str]:
    """Return portable CLI inputs without accessing any GPU or existing file."""
    return [
        "--source",
        str(tmp_path / "candidate"),
        "--output",
        str(tmp_path / "out"),
        "--scratch-root",
        str(tmp_path),
        "--gpu-uuid",
        "GPU-example",
        "--cufile-config",
        str(tmp_path / "cufile.json"),
    ]


def test_campaign_keeps_original_flags_separate_and_repeats(tmp_path: Path) -> None:
    """The original checkout is exercised without requiring candidate-only flags."""
    args = parse_args(
        arguments(tmp_path)
        + ["--baseline-source", str(tmp_path / "old"), "--repetitions", "3"]
    )
    plan = build_plan(args)
    assert len(plan) == 12
    assert {row["workers"] for row in plan} == {None, 0, 1, 4}
    assert len({row["name"] for row in plan}) == 12
    for repetition in range(3):
        rows = [row for row in plan if row["repetition"] == repetition]
        assert {row["workers"] for row in rows} == {None, 0, 1, 4}
        assert all(
            row["source"] == str(tmp_path / "old")
            for row in rows
            if row["workers"] is None
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--layers", "1024"],
        ["--duration", "0"],
        ["--workers", "4", "4"],
        ["--rpc-port", "25880"],
        ["--gpu-uuid", "0"],
    ],
)
def test_invalid_campaign_rejected_before_side_effects(
    tmp_path: Path, extra: list[str]
) -> None:
    """Invalid capacity, identity and timing inputs fail before allocating a slab."""
    with pytest.raises(SystemExit):
        parse_args(arguments(tmp_path) + extra)
    assert not (tmp_path / "out").exists()


def test_prepared_inode_is_nonzero_fully_verified_and_exclusively_leased(
    tmp_path: Path,
) -> None:
    """Reopening preserves the complete initialized allocation and its identity."""
    report = tmp_path / "preparation.json"
    lease = SlabLease.create(tmp_path / "scratch", 8192, report)
    identity = lease.identity()
    path = lease.path
    try:
        assert path.read_bytes() == pattern_bytes(0, 8192)
        assert all(path.read_bytes())
        result = json.loads(report.read_text())
        assert result["written_bytes"] == result["verified_bytes"] == 8192
        assert result["actual_sha256"] == result["expected_sha256"]
        with pytest.raises(BlockingIOError):
            SlabLease.open_existing(path, tmp_path, 8192)
    finally:
        lease.close()
    reopened = SlabLease.open_existing(path, tmp_path, 8192)
    try:
        assert reopened.identity() == identity
        assert path.read_bytes() == pattern_bytes(0, 8192)
    finally:
        reopened.close()


@pytest.mark.parametrize("change", ["symlink", "replacement", "marker", "outside"])
def test_reuse_rejects_unproven_allocation(tmp_path: Path, change: str) -> None:
    """A nearby file or copied preparation marker does not authorize slab reuse."""
    lease = SlabLease.create(tmp_path / "scratch", 4096, tmp_path / "prepared.json")
    path, marker = lease.path, lease.marker
    lease.close()
    root = tmp_path
    if change == "symlink":
        alias = path.with_name("alias")
        alias.symlink_to(path)
        path = alias
    elif change == "replacement":
        replacement = path.with_name("replacement")
        replacement.write_bytes(bytes(4096))
        replacement.replace(path)
    elif change == "marker":
        marker.write_text("{}")
    else:
        root = tmp_path / "other"
        root.mkdir()
    with pytest.raises(ValueError):
        SlabLease.open_existing(path, root, 4096)


def test_reuse_refuses_a_still_running_owner(tmp_path: Path) -> None:
    """The inherited lease prevents reuse after the controller closes its FD."""
    lease = SlabLease.create(tmp_path / "scratch", 4096, tmp_path / "prepared.json")
    path = lease.path
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,sys; os.fstat(int(sys.argv[1])); "
            "print('ready', flush=True); sys.stdin.read()",
            str(lease.fd),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        pass_fds=(lease.fd,),
    )
    lease.close()
    try:
        assert child.stdout is not None and child.stdout.readline() == "ready\n"
        with pytest.raises(BlockingIOError):
            SlabLease.open_existing(path, tmp_path, 4096)
    finally:
        child.communicate(timeout=5)


def test_short_preparation_has_no_reuse_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed full initialization leaves evidence but cannot be reused."""
    monkeypatch.setattr(os, "pwrite", lambda fd, data, offset: len(data) - 1)
    with pytest.raises(RuntimeError, match="Short preparation"):
        SlabLease.create(tmp_path / "scratch", 4096, tmp_path / "prepared.json")
    assert json.loads((tmp_path / "prepared.json").read_text())["status"] == "failed"
    assert not (tmp_path / "scratch/lmcache_gds_slab.benchmark.json").exists()
