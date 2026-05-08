# SPDX-License-Identifier: Apache-2.0

# First Party
from benchmarks.agentic_mp_trace.config import load_yaml_config
from benchmarks.agentic_mp_trace.record_agentic_mp_traces import (
    _build_job_commands,
    _record_jobs,
)


def test_record_commands_use_existing_cli_and_raw_block(tmp_path):
    config = load_yaml_config("benchmarks/agentic_mp_trace/config.example.yaml")
    config["dataset_catalog"]["tau_bench_current"]["local_path"] = str(tmp_path)
    job = _record_jobs(config, "agentic_mix_v1")[0]
    lmcache_cmd, vllm_cmd, metadata = _build_job_commands(
        config=config,
        job=job,
        index=0,
        output_dir=str(tmp_path),
    )
    assert lmcache_cmd[:2] == ["uv", "run"]
    assert "server" in lmcache_cmd
    assert "--trace-level" in lmcache_cmd
    assert "storage" in lmcache_cmd
    assert "--l2-store-policy" in lmcache_cmd
    assert lmcache_cmd[lmcache_cmd.index("--l2-store-policy") + 1] == "skip_l1"
    assert metadata["l2_adapter"]["device_path"] == "/dev/ng1n1"
    assert metadata["l2_adapter"]["fdp_metadata_ruh_ids"] == [3]
    assert "serve" in vllm_cmd
    assert "--kv-transfer-config" in vllm_cmd

