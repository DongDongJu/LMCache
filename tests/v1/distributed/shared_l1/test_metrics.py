# SPDX-License-Identifier: Apache-2.0
"""Both L1 bindings export the same Prometheus gauges and unregister on close."""

# Standard
from pathlib import Path
import subprocess
import sys

# Third Party
import pytest

# First Party
from tests.v1.memory_orchestrator.harness import running_orchestrator

pytestmark = pytest.mark.no_shared_allocator


def test_shared_and_private_gauges(tmp_path: Path) -> None:
    """Use a fresh process for OTel's process-global meter-provider lifecycle."""
    with running_orchestrator(tmp_path / "state", capacity_bytes=65536) as orch:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                _CHECK_GAUGES,
                str(tmp_path / "region"),
                orch.endpoint,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr


_CHECK_GAUGES = """
from contextlib import ExitStack
from pathlib import Path
import sys
import torch
from prometheus_client import generate_latest
from prometheus_client.parser import text_string_to_metric_families
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from lmcache.v1.mp_observability.otel_init import init_otel_metrics
from tests.v1.distributed.shared_l1.utils import shared_config

def values(name):
    return {
        sample.labels['l1_tag']: sample.value
        for family in text_string_to_metric_families(generate_latest().decode())
        for sample in family.samples if sample.name == name
    }

init_otel_metrics(start_http_server=False)
with ExitStack() as stack:
    shared = SharedL1Manager(shared_config(
        Path(sys.argv[1]), sys.argv[2], 'metrics', size_bytes=65536, tag='shared'
    ))
    stack.callback(shared.close)
    private = L1Manager(L1ManagerConfig(
        L1MemoryManagerConfig(size_in_bytes=8192, align_bytes=4096,
                             use_lazy=False, shm_name=''), tag='private'
    ))
    stack.callback(private.close)
    key = ObjectKey(b'chunk', 'metrics', 0)
    layout = MemoryLayoutDesc([torch.Size([1000])], [torch.uint8])
    for manager in (private, shared):
        manager.reserve_write([key], [False], layout)
    assert values('lmcache_mp_l1_memory_usage_bytes') == {'private':4096, 'shared':4096}
    assert values('lmcache_mp_l1_staging_bytes') == {'private':1000, 'shared':1000}
    assert values('lmcache_mp_l1_usage_ratio') == {'private':0.5, 'shared':0.0625}
    for manager in (private, shared):
        manager.finish_write([key])
    assert values('lmcache_mp_l1_staging_bytes') == {'private':0, 'shared':0}
assert values('lmcache_mp_l1_memory_usage_bytes') == {}
"""
