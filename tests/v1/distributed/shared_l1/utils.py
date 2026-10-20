# SPDX-License-Identifier: Apache-2.0
"""Configurations for standalone shared-manager tests."""

# Standard
from pathlib import Path

# First Party
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    SharedL1Config,
)
from tests.v1.memory_orchestrator.harness import REGION_ID


def shared_config(
    region_path: Path,
    endpoint: str,
    client_id: str,
    *,
    size_bytes: int,
    tag: str = "pool",
    rpc_timeout_seconds: float = 5.0,
    region_id: str = REGION_ID,
    l1_type: str = "DRAM",
) -> L1ManagerConfig:
    """Build a shared-manager configuration without the serving CLI."""
    cls = {"DRAM": DRAML1ManagerConfig, "DEVDAX": DevDaxL1ManagerConfig}[l1_type]
    return cls(
        memory_config=L1MemoryManagerConfig(
            size_bytes,
            use_lazy=False,
            shm_name="",
            devdax_path=str(region_path) if l1_type == "DEVDAX" else "",
        ),
        tag=tag,
        eviction=EvictionConfig("noop"),
        shared=SharedL1Config(
            endpoint, region_id, str(region_path), client_id, rpc_timeout_seconds
        ),
    )
