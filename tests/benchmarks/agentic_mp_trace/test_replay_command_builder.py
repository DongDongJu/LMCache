# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.agentic_mp_trace.config import load_yaml_config
from benchmarks.agentic_mp_trace.replay.command_builder import (
    auto_iterations,
    build_fdp_replay_config,
    target_host_write_bytes,
)
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (
    build_l2_adapter,
    build_replay_command,
    expand_workers,
)


def _manifest(tmp_path):
    trace = tmp_path / "missing.lct"
    return {
        "traces": [
            {
                "trace_id": "tau.test",
                "trace_path": str(trace),
                "storage_class": "tool_agent",
                "dataset": {"adapter": "tau_bench"},
                "trace_stats": {"estimated_store_bytes": 1024 * 1024},
            },
            {
                "trace_id": "swe.test",
                "trace_path": str(trace),
                "storage_class": "coding_agent",
                "dataset": {"adapter": "swe_bench"},
                "trace_stats": {"estimated_store_bytes": 1024 * 1024},
            },
        ]
    }


def test_no_fdp_adapter_omits_ruhs(tmp_path):
    config = load_yaml_config("benchmarks/agentic_mp_trace/config.example.yaml")
    fdp_config = build_fdp_replay_config(
        config=config,
        manifest=_manifest(tmp_path),
        mode="no_fdp",
    )
    workers = expand_workers(fdp_config, "no_fdp")
    adapter = build_l2_adapter(workers[0], fdp_config)
    assert adapter["use_fdp"] is False
    assert "fdp_data_ruh_ids" not in adapter


def test_4ruh_fdp_separated_and_skip_l1_command(tmp_path):
    config = load_yaml_config("benchmarks/agentic_mp_trace/config.example.yaml")
    fdp_config = build_fdp_replay_config(
        config=config,
        manifest=_manifest(tmp_path),
        mode="fdp_separated",
    )
    workers = expand_workers(fdp_config, "separated")
    assert workers[0].fdp_data_ruh_ids == [0]
    assert workers[1].fdp_data_ruh_ids == [1]
    cmd = build_replay_command(
        workers[0],
        fdp_config,
        mode="separated",
        run_id="agentic001",
        iteration=0,
        worker_output_dir=str(tmp_path),
        jsonl_path=str(tmp_path / "records.jsonl"),
    )
    assert "--l2-store-policy" in cmd
    assert cmd[cmd.index("--l2-store-policy") + 1] == "skip_l1"
    assert cmd[cmd.index("--eviction-policy") + 1] == "noop"
    assert cmd[cmd.index("--l1-size-gb") + 1] == "1.0"


def test_5x_target_and_auto_iterations(tmp_path):
    config = load_yaml_config("benchmarks/agentic_mp_trace/config.example.yaml")
    config["replay"]["default_capacity_bytes"] = 1024 * 1024
    config["replay"]["trace_overrides"] = {
        "tool_agent": {"capacity_bytes": 1024 * 1024, "slot_bytes": 4096},
        "coding_agent": {"capacity_bytes": 1024 * 1024, "slot_bytes": 4096},
    }
    fdp_config = build_fdp_replay_config(
        config=config,
        manifest=_manifest(tmp_path),
        mode="fdp_mixed",
    )
    target = target_host_write_bytes(
        fdp_config=fdp_config,
        replay_cfg=config["replay"],
    )
    assert target == 10 * 1024 * 1024
    assert auto_iterations(
        requested_iterations=1,
        fdp_config=fdp_config,
        replay_cfg=config["replay"],
    ) == 5

