# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# Third Party
import pytest
import yaml

# First Party
from benchmarks.agentic_mp_trace import replay_multi_window_raw_block as multiwin


def _manifest() -> dict:
    return {
        "traces": [
            {
                "trace_id": "tau_tool",
                "trace_path": "/traces/tau.lct",
                "storage_class": "tool_agent",
                "dataset": {"adapter": "tau_bench"},
                "trace_stats": {
                    "estimated_store_bytes": 1000,
                    "duration_seconds": 10,
                    "record_count": 3,
                    "store_count": 1,
                },
            },
            {
                "trace_id": "swe_code",
                "trace_path": "/traces/swe.lct",
                "storage_class": "coding_agent",
                "dataset": {"adapter": "swe_bench"},
                "trace_stats": {
                    "estimated_store_bytes": 2000,
                    "duration_seconds": 20,
                    "record_count": 4,
                    "store_count": 2,
                },
            },
            {
                "trace_id": "web_browser",
                "trace_path": "/traces/web.lct",
                "storage_class": "browser_agent",
                "dataset": {"adapter": "webarena"},
                "trace_stats": {
                    "estimated_store_bytes": 3000,
                    "duration_seconds": 30,
                    "record_count": 5,
                    "store_count": 3,
                },
            },
        ]
    }


def _write_manifest(tmp_path) -> str:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(_manifest()))
    return str(path)


def _args(tmp_path, *extra: str):
    manifest = _write_manifest(tmp_path)
    argv = [
        "--trace-manifest",
        manifest,
        "--device-path",
        str(tmp_path / "ng-file"),
        "--block-device-path",
        str(tmp_path / "nvme-file"),
        "--output-dir",
        str(tmp_path / "out"),
        "--device-capacity-bytes",
        "1GiB",
        "--num-windows",
        "2",
        "--num-workloads",
        "2",
        "--window-capacity-policy",
        "equal",
        "--slot-bytes",
        "16MiB",
        "--meta-total-bytes",
        "64MiB",
        "--stop-policy",
        "iterations",
        "--iterations",
        "1",
        *extra,
    ]
    return multiwin.parse_args(argv), multiwin.load_manifest(manifest)


def _plan(tmp_path, *extra: str, monkeypatch=None):
    args, manifest = _args(tmp_path, *extra)
    if monkeypatch is not None:
        monkeypatch.setattr(multiwin, "validate_ng_device_path", lambda *a, **k: None)
    return multiwin.resolve_plan(args, manifest)


def test_size_parser_accepts_units_and_rejects_invalid():
    assert multiwin.parse_size("123") == 123
    assert multiwin.parse_size("1MiB") == 1024**2
    assert multiwin.parse_size("2GiB") == 2 * 1024**3
    assert multiwin.parse_size("3TiB") == 3 * 1024**4
    assert multiwin.parse_size("auto", allow_auto=True) == "auto"
    with pytest.raises(ValueError):
        multiwin.parse_size("12 bananas")


def test_window_layout_equal_guard_start_and_alignment():
    windows = multiwin.compute_windows(
        total_capacity=1024 * 1024 * 1024,
        start_offset=128 * 1024 * 1024,
        num_windows=2,
        policy="equal",
        explicit_capacities=[],
        stride_value="auto",
        guard_bytes=4 * 1024 * 1024,
        block_align=4096,
        slot_bytes=16 * 1024 * 1024,
        meta_total_bytes=64 * 1024 * 1024,
        use_fdp=False,
        metadata_ruh_ids=[],
    )
    assert windows[0].base_offset_bytes == 128 * 1024 * 1024
    assert windows[1].base_offset_bytes > windows[0].base_offset_bytes
    assert windows[0].capacity_bytes % 4096 == 0
    assert windows[0].usable_capacity_bytes < windows[0].capacity_bytes


def test_window_layout_explicit_and_rejections():
    windows = multiwin.compute_windows(
        total_capacity=512 * 1024 * 1024,
        start_offset=0,
        num_windows=2,
        policy="explicit",
        explicit_capacities=[128 * 1024 * 1024, 64 * 1024 * 1024],
        stride_value="auto",
        guard_bytes=4096,
        block_align=4096,
        slot_bytes=16 * 1024 * 1024,
        meta_total_bytes=16 * 1024 * 1024,
        use_fdp=False,
        metadata_ruh_ids=[],
    )
    assert windows[1].base_offset_bytes == 128 * 1024 * 1024 + 4096
    with pytest.raises(ValueError, match="aligned"):
        multiwin.compute_windows(
            total_capacity=512,
            start_offset=1,
            num_windows=1,
            policy="equal",
            explicit_capacities=[],
            stride_value="auto",
            guard_bytes=0,
            block_align=4096,
            slot_bytes=4096,
            meta_total_bytes=4096,
            use_fdp=False,
            metadata_ruh_ids=[],
        )
    with pytest.raises(ValueError, match="exceeds"):
        multiwin.validate_windows(
            [
                multiwin.WindowLayout(0, 0, 8192, 4096, 4096, 1, "MW000001"),
                multiwin.WindowLayout(1, 4096, 8192, 4096, 4096, 1, "MW000002"),
            ],
            total_capacity=10_000,
            block_align=4096,
        )
    with pytest.raises(ValueError, match="overlap"):
        multiwin.validate_windows(
            [
                multiwin.WindowLayout(0, 0, 8192, 4096, 4096, 1, "MW000001"),
                multiwin.WindowLayout(1, 4096, 4096, 4096, 4096, 1, "MW000002"),
            ],
            total_capacity=20_000,
            block_align=4096,
        )


def test_usable_capacity_no_fdp_and_fdp_metadata_ruhs():
    usable, slots = multiwin.compute_usable_capacity(
        capacity_bytes=256 * 1024 * 1024,
        slot_bytes=16 * 1024 * 1024,
        metadata_reserved_bytes=64 * 1024 * 1024,
    )
    assert usable == 192 * 1024 * 1024
    assert slots == 12
    assert multiwin.metadata_reserved_bytes(
        use_fdp=True,
        metadata_ruh_ids=[3, 4],
        meta_total_bytes=64 * 1024 * 1024,
    ) == 128 * 1024 * 1024
    with pytest.raises(ValueError, match="larger"):
        multiwin.compute_usable_capacity(
            capacity_bytes=64 * 1024 * 1024,
            slot_bytes=16 * 1024 * 1024,
            metadata_reserved_bytes=64 * 1024 * 1024,
        )


def test_ruh_assignment_mixed_round_robin_and_validation(monkeypatch, tmp_path):
    plan = _plan(
        tmp_path,
        "--use-fdp",
        "true",
        "--ruh-count",
        "4",
        "--num-windows",
        "5",
        "--num-workloads",
        "2",
        monkeypatch=monkeypatch,
    )
    assert [w["fdp_data_ruh_ids"] for w in plan["windows"]] == [
        [0],
        [1],
        [2],
        [0],
        [1],
    ]
    with pytest.raises(ValueError, match="duplicates"):
        multiwin.resolve_ruh_policy(
            use_fdp=True,
            ruh_ids="1,1",
            ruh_count=None,
            ruh_start_id=0,
            metadata_ruh_ids="auto",
        )
    with pytest.raises(ValueError, match="uint16"):
        multiwin.resolve_ruh_policy(
            use_fdp=True,
            ruh_ids="70000",
            ruh_count=None,
            ruh_start_id=0,
            metadata_ruh_ids="auto",
        )


def test_per_app_ruh_assignment_auto_and_explicit(monkeypatch, tmp_path):
    auto_plan = _plan(
        tmp_path,
        "--use-fdp",
        "true",
        "--ruh-count",
        "4",
        "--ruh-assignment",
        "per_app",
        monkeypatch=monkeypatch,
    )
    assert auto_plan["windows"][0]["fdp_data_ruh_ids"] == [0]
    assert auto_plan["windows"][1]["fdp_data_ruh_ids"] == [1]

    explicit = tmp_path / "ruh_map.json"
    explicit.write_text(json.dumps({"tool_agent": [2], "coding_agent": [1]}))
    explicit_plan = _plan(
        tmp_path,
        "--use-fdp",
        "true",
        "--ruh-count",
        "4",
        "--ruh-assignment",
        "per_app",
        "--per-app-ruh-map",
        str(explicit),
        monkeypatch=monkeypatch,
    )
    assert explicit_plan["windows"][0]["fdp_data_ruh_ids"] == [2]
    assert explicit_plan["windows"][1]["fdp_data_ruh_ids"] == [1]


def test_workload_selection_fixed_random_repeat_and_too_many(tmp_path):
    fixed = multiwin.select_workloads(
        _manifest(),
        num_workloads=2,
        workload_key="storage_class",
        workload_filter=None,
        placement="fixed",
        seed=1,
    )
    assert [item.workload_key for item in fixed] == ["tool_agent", "coding_agent"]

    random_a = multiwin.select_workloads(
        _manifest(),
        num_workloads=2,
        workload_key="storage_class",
        workload_filter=None,
        placement="random",
        seed=7,
    )
    random_b = multiwin.select_workloads(
        _manifest(),
        num_workloads=2,
        workload_key="storage_class",
        workload_filter=None,
        placement="random",
        seed=7,
    )
    assert random_a == random_b
    repeated = multiwin.assign_workloads_to_windows(
        fixed,
        num_windows=5,
        placement="fixed",
        seed=1,
        allow_multiplexing=False,
    )
    assert [group[0].workload_key for group in repeated] == [
        "tool_agent",
        "coding_agent",
        "tool_agent",
        "coding_agent",
        "tool_agent",
    ]
    with pytest.raises(ValueError, match="requires --allow"):
        multiwin.assign_workloads_to_windows(
            fixed + [fixed[0]],
            num_windows=2,
            placement="fixed",
            seed=1,
            allow_multiplexing=False,
        )
    multiplexed = multiwin.assign_workloads_to_windows(
        fixed + [fixed[0]],
        num_windows=2,
        placement="fixed",
        seed=1,
        allow_multiplexing=True,
    )
    assert [[item.workload_key for item in group] for group in multiplexed] == [
        ["tool_agent", "tool_agent"],
        ["coding_agent"],
    ]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--stop-policy", "timeout"], "duration"),
        (["--stop-policy", "total_written_size"], "target"),
        (["--stop-policy", "iterations"], "iterations"),
    ],
)
def test_stop_policy_validation_errors(tmp_path, extra, message):
    args, _ = _args(tmp_path, *extra)
    if "--stop-policy" in extra:
        args.iterations = None
    with pytest.raises(ValueError, match=message):
        multiwin.validate_stop_policy(args)


def test_dry_run_writes_plan_and_commands_without_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(multiwin.subprocess, "Popen", lambda *a, **k: pytest.fail())
    rc = multiwin.main(
        [
            "--trace-manifest",
            _write_manifest(tmp_path),
            "--device-path",
            str(tmp_path / "ng-file"),
            "--block-device-path",
            str(tmp_path / "nvme-file"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device-capacity-bytes",
            "1GiB",
            "--num-windows",
            "2",
            "--num-workloads",
            "2",
            "--stop-policy",
            "iterations",
            "--iterations",
            "1",
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (tmp_path / "out/resolved_plan.json").exists()
    commands = (tmp_path / "out/commands.sh").read_text()
    assert "lmcache trace replay" in commands
    assert "--l2-adapter" in commands


def test_confirmation_requires_allow_and_exact_run(monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    assert multiwin.confirm_destructive_run(plan, yes=True, allow=False) is False
    assert multiwin.confirm_destructive_run(plan, yes=True, allow=True) is True
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert multiwin.confirm_destructive_run(plan, yes=False, allow=True) is False
    monkeypatch.setattr("builtins.input", lambda prompt: "RUN")
    assert multiwin.confirm_destructive_run(plan, yes=False, allow=True) is True


def test_adapter_json_and_meta_magic_are_correct(monkeypatch, tmp_path):
    plan = _plan(
        tmp_path,
        "--use-fdp",
        "true",
        "--ruh-count",
        "4",
        "--num-store-workers",
        "7",
        "--num-load-workers",
        "8",
        "--num-lookup-workers",
        "9",
        "--use-odirect",
        "true",
        monkeypatch=monkeypatch,
    )
    magics = [window["meta_magic"] for window in plan["windows"]]
    assert len(set(magics)) == len(magics)
    command = plan["windows"][0]["command"]
    adapter = json.loads(command[command.index("--l2-adapter") + 1])
    assert adapter["base_offset_bytes"] == plan["windows"][0]["base_offset_bytes"]
    assert adapter["capacity_bytes"] == plan["windows"][0]["capacity_bytes"]
    assert adapter["fdp_data_ruh_ids"] == [0]
    assert adapter["fdp_metadata_ruh_ids"] == [3]
    assert adapter["use_odirect"] is True
    assert adapter["num_store_workers"] == 7
    assert adapter["num_load_workers"] == 8
    assert adapter["num_lookup_workers"] == 9


def test_media_write_counter_command_fallback(monkeypatch):
    class Proc:
        returncode = 0
        stdout = '{"media_write_bytes": 12345}'
        stderr = ""

    monkeypatch.setattr(multiwin.subprocess, "run", lambda *a, **k: Proc())
    assert multiwin.capture_media_write_bytes("vendor-command") == 12345

    class FailedProc:
        returncode = 1
        stdout = ""
        stderr = "failed"

    monkeypatch.setattr(multiwin.subprocess, "run", lambda *a, **k: FailedProc())
    assert multiwin.capture_media_write_bytes("vendor-command") is None


def test_raw_block_status_accounting_is_extracted_and_aggregated(tmp_path):
    status = {
        "l2_adapters": [
            {
                "type": "RawBlockL2Adapter",
                "core": {
                    "io_accounting": {
                        "store_attempted_logical_bytes": 1000,
                        "store_committed_logical_bytes": 700,
                        "eviction_count": 3,
                        "eviction_logical_bytes": 300,
                        "data_write_payload_physical_bytes": 8192,
                        "data_write_header_physical_bytes": 4096,
                        "data_write_physical_bytes": 12288,
                        "metadata_write_physical_bytes": 4096,
                        "total_write_physical_bytes": 16384,
                    }
                },
            }
        ]
    }
    accounting = multiwin.extract_raw_block_accounting(status)
    assert accounting["store_attempted_logical_bytes"] == 1000
    assert accounting["store_committed_logical_bytes"] == 700
    assert accounting["eviction_count"] == 3
    assert accounting["eviction_logical_bytes"] == 300
    assert accounting["data_write_physical_bytes"] == 12288
    assert accounting["metadata_write_physical_bytes"] == 4096
    assert accounting["total_write_physical_bytes"] == 16384

    out_dir = tmp_path / "worker"
    out_dir.mkdir()
    (out_dir / "storage_manager_status.json").write_text(json.dumps(status))
    window = {
        "output_dir": str(out_dir),
        "workloads": [{"output_dir": str(out_dir)}],
    }
    result_accounting = multiwin.window_io_accounting(window)
    assert result_accounting is not None
    assert result_accounting["total_write_physical_bytes"] == 16384

    aggregate = multiwin.aggregate_result_io_accounting(
        [
            {"iteration": 0, "io_accounting": result_accounting},
            {"iteration": 1, "io_accounting": result_accounting},
        ]
    )
    assert aggregate["store_attempted_logical_bytes"] == 2000
    assert aggregate["eviction_count"] == 6
    assert aggregate["eviction_logical_bytes"] == 600
    assert aggregate["total_write_physical_bytes"] == 32768


def test_total_written_size_progress_prefers_lmcache_actual_written_bytes():
    progress_bytes, progress_source = multiwin.total_written_size_progress(
        [
            {
                "iteration": -1,
                "io_accounting": {
                    "store_attempted_logical_bytes": 10_000,
                    "total_write_physical_bytes": 10_000,
                },
            },
            {
                "iteration": 0,
                "io_accounting": {
                    "store_attempted_logical_bytes": 9_999,
                    "total_write_physical_bytes": 1_200,
                },
            },
        ],
        host_write_delta=5_000,
    )
    assert progress_bytes == 1_200
    assert progress_source == multiwin.END_CONDITION_ACTUAL_WRITTEN_SOURCE

    progress_bytes, progress_source = multiwin.total_written_size_progress(
        [],
        host_write_delta=5_000,
        defer_host_fallback_until_result=True,
    )
    assert progress_bytes is None
    assert progress_source == multiwin.END_CONDITION_ACTUAL_WRITTEN_SOURCE

    progress_bytes, progress_source = multiwin.total_written_size_progress(
        [{"iteration": 0, "io_accounting": None}],
        host_write_delta=5_000,
        defer_host_fallback_until_result=True,
    )
    assert progress_bytes == 5_000
    assert progress_source == multiwin.END_CONDITION_HOST_WRITE_SOURCE


def test_run_plan_total_written_size_stops_on_lmcache_actual_written_bytes(
    monkeypatch,
    tmp_path,
):
    plan = _plan(
        tmp_path,
        "--stop-policy",
        "total_written_size",
        "--target-host-write-bytes",
        "3000",
        "--iterations",
        "10",
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(multiwin, "capture_host_write_bytes", lambda _path: 0)
    monkeypatch.setattr(multiwin, "capture_media_write_bytes", lambda _command: None)
    iterations = []

    def fake_run_iteration(
        plan,
        *,
        iteration,
        timeout_seconds,
        stop_check=None,
    ):
        del plan, timeout_seconds, stop_check
        iterations.append(iteration)
        return [
            {
                "window_index": 0,
                "iteration": iteration,
                "exit_code": 0,
                "log_path": "worker.log",
                "jsonl_out": "records.jsonl",
                "failed_records": 0,
                "io_accounting": {
                    "store_attempted_logical_bytes": 10_000,
                    "total_write_physical_bytes": 2_000,
                },
                "ended_at": "now",
            }
        ]

    monkeypatch.setattr(multiwin, "run_iteration", fake_run_iteration)

    summary = multiwin.run_plan(plan, str(tmp_path / "run"))

    assert iterations == [0, 1]
    assert summary["host_write_bytes_delta"] == 0
    assert summary["target_host_write_bytes_reached"] is False
    assert summary["lmcache_store_attempted_logical_bytes"] == 20_000
    assert summary["lmcache_successful_write_physical_bytes"] == 4_000
    assert (
        summary["total_written_size_end_condition_source"]
        == multiwin.END_CONDITION_ACTUAL_WRITTEN_SOURCE
    )
    assert summary["total_written_size_end_condition_bytes"] == 4_000
    assert summary["total_written_size_target_bytes"] == 3_000
    assert summary["total_written_size_target_reached"] is True
