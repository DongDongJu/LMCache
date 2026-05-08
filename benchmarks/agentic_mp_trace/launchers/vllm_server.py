# SPDX-License-Identifier: Apache-2.0

"""Build existing ``vllm serve`` commands for MP connector recording."""

# Future
from __future__ import annotations

# Standard
from typing import Any
import json

# First Party
from benchmarks.agentic_mp_trace.launchers.lmcache_server import as_command_prefix


def build_vllm_command(
    *,
    binary: Any,
    model_id: str,
    vllm_port: int,
    mp_port: int,
    max_model_len: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    host: str = "127.0.0.1",
    lmcache_mp_host: str = "tcp://localhost",
) -> list[str]:
    kv_transfer_config = {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": lmcache_mp_host,
            "lmcache.mp.port": int(mp_port),
        },
    }
    cmd = as_command_prefix(binary, "vllm")
    cmd.extend(
        [
            "serve",
            model_id,
            "--host",
            host,
            "--port",
            str(vllm_port),
            "--max-model-len",
            str(max_model_len),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--gpu-memory-utilization",
            str(gpu_memory_utilization),
            "--kv-transfer-config",
            json.dumps(kv_transfer_config, separators=(",", ":")),
        ]
    )
    return cmd

