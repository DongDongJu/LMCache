# SPDX-License-Identifier: Apache-2.0

# Standard
import json
import os

# Third Party
import yaml

# First Party
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (
    build_l2_adapter,
    build_replay_command,
    extract_host_write_bytes,
    extract_media_write_bytes,
    expand_workers,
    main,
)


def _config(tmp_path):
    trace = tmp_path / "missing.lct"
    return {
        "device_path": "/dev/ng1n1",
        "block_device_path": "/dev/nvme1n1",
        "block_align": 4096,
        "global": {
            "replay_binary": "lmcache",
            "l2_store_policy": "skip_l1",
            "eviction_policy": "noop",
        },
        "measurement": {"enabled": False},
        "windows": {
            "start_offset_bytes": 4096 * 1024,
            "window_stride_bytes": 1024 * 1024,
            "default_capacity_bytes": 1024 * 1024,
            "auto_assign": True,
        },
        "modes": {
            "mixed": {
                "use_fdp": True,
                "default_data_ruhs": [0, 1],
                "default_metadata_ruhs": [2],
            },
            "no_fdp": {"use_fdp": False},
        },
        "workloads": [
            {
                "name": "hot",
                "class": "hot_churn",
                "trace_path": os.fspath(trace),
                "concurrency": 2,
                "slot_bytes": 4096 * 8,
                "capacity_bytes": 1024 * 1024,
                "l1_size_gb": 1,
            },
            {
                "name": "cold",
                "class": "cold_rag",
                "trace_path": os.fspath(trace),
                "concurrency": 1,
                "slot_bytes": 4096 * 16,
                "capacity_bytes": 1024 * 1024,
                "l1_size_gb": 2,
            },
        ],
    }


def test_l2_adapter_json_construction_mixed(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "mixed")[0]

    adapter = build_l2_adapter(worker, config)

    assert adapter["type"] == "raw_block"
    assert adapter["device_path"] == "/dev/ng1n1"
    assert adapter["base_offset_bytes"] == worker.base_offset_bytes
    assert adapter["capacity_bytes"] == worker.capacity_bytes
    assert adapter["meta_magic"] == "WF000001"
    assert adapter["use_fdp"] is True
    assert adapter["fdp_data_ruh_ids"] == [0, 1]
    assert adapter["fdp_metadata_ruh_ids"] == [2]


def test_l2_adapter_json_construction_no_fdp_omits_ruhs(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "no_fdp")[0]

    adapter = build_l2_adapter(worker, config)

    assert adapter["use_fdp"] is False
    assert "fdp_data_ruh_ids" not in adapter
    assert "fdp_metadata_ruh_ids" not in adapter


def test_replay_command_contains_required_flags(tmp_path):
    config = _config(tmp_path)
    worker = expand_workers(config, "mixed")[0]
    cmd = build_replay_command(
        worker,
        config,
        mode="mixed",
        run_id="waf001",
        iteration=4,
        worker_output_dir=os.fspath(tmp_path / "worker"),
        jsonl_path=os.fspath(tmp_path / "records.jsonl"),
    )

    assert cmd[:3] == ["lmcache", "trace", "replay"]
    assert "--replay-cache-salt-suffix" in cmd
    salt = cmd[cmd.index("--replay-cache-salt-suffix") + 1]
    assert salt == "waf001.mixed.hot.w0.iter_0004"
    adapter = json.loads(cmd[cmd.index("--l2-adapter") + 1])
    assert adapter["type"] == "raw_block"
    assert adapter["fdp_data_ruh_ids"] == [0, 1]
    assert "--jsonl-out" in cmd


def test_dry_run_command_output(tmp_path, capsys):
    config = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "out"
    with open(config_path, "w") as file_obj:
        yaml.safe_dump(config, file_obj)

    exit_code = main(
        [
            "--config",
            os.fspath(config_path),
            "--mode",
            "mixed",
            "--iterations",
            "1",
            "--warmup-iterations",
            "0",
            "--output-dir",
            os.fspath(output_dir),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "worker_count=3" in captured.out
    assert "--l2-adapter" in captured.out
    assert "waf001.mixed.hot.w0.iter_0000" in captured.out
    assert (output_dir / "commands.txt").exists()
    assert (output_dir / "workers.json").exists()


def test_measurement_parser_fallback_without_vendor_media_counter():
    smart = {"data_units_written": "10"}

    assert extract_host_write_bytes(smart) == 5_120_000
    assert extract_media_write_bytes(None) is None
    assert extract_media_write_bytes({"nested": {"nand_write_bytes": "1234"}}) == 1234
