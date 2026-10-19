# SPDX-License-Identifier: Apache-2.0
"""Filesystem plugin contracts exercised through LMCache's real controllers."""

# Standard
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
import errno
import json
import os
import time

# Third Party
from lmcache_l2_plugins.fs import FSL2Adapter, FSL2AdapterConfig
import pytest
import torch

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.data import (
    create_l1_memory_desc,
    make_aligned_tensor,
    make_memory_objects,
    make_object_keys,
    wait_eventfd,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.config import (
    parse_args_to_l2_adapters_config,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2Adapter as LegacyFSL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.fs_l2_adapter import (
    FSL2AdapterConfig as LegacyFSL2AdapterConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager

# Test helpers
from tests.v1.distributed.utils import single_row_spec

pytestmark = pytest.mark.no_shared_allocator


def _fs_spec(path: Path, adapter_type: str, **params: object) -> dict[str, object]:
    """Build either supported CLI spelling for a filesystem adapter."""
    params = {"base_path": str(path), **params}
    if adapter_type == "fs":
        return {"type": "fs", **params}
    return {
        "type": "plugin",
        "module_path": "lmcache_l2_plugins.fs",
        "class_name": "FSL2Adapter",
        "adapter_params": params,
    }


def _wait_until(condition: Callable[[], bool]) -> None:
    """Wait up to ten seconds for controller work to finish."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    pytest.fail("filesystem controller work did not finish within 10 seconds")


def test_legacy_imports_use_plugin() -> None:
    """Legacy imports retain the exact external adapter and config classes."""
    assert LegacyFSL2Adapter is FSL2Adapter
    assert LegacyFSL2AdapterConfig is FSL2AdapterConfig


@pytest.mark.parametrize("adapter_type", ["fs", "plugin"])
@pytest.mark.parametrize("pipeline", ["plain", "fp8", "fault_inject"])
def test_storage_manager_roundtrip(
    tmp_path: Path, adapter_type: str, pipeline: str
) -> None:
    """Store, detach, reattach, prefetch and delete through the public APIs."""
    spec = _fs_spec(tmp_path, adapter_type, relative_tmp_dir="tmp")
    if pipeline == "fp8":
        spec["serde"] = {"type": "fp8"}
    elif pipeline == "fault_inject":
        spec = {"type": "fault_inject", "inner": spec, "rate": 0.0}
    spec.update(affinity_tag="disk", shared=True)
    adapters = parse_args_to_l2_adapters_config(
        Namespace(l2_adapter=[json.dumps(spec)])
    )
    config = adapters.adapters[0]
    assert config.affinity_tag == "disk"
    assert config.shared
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4 << 20, use_lazy=False, shm_name=""
                ),
                tag="disk",
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=adapters,
        )
    )
    keys = [
        ObjectKey(
            model_name="test/model",
            kv_rank=0,
            chunk_hash=i.to_bytes(4, "big"),
            cache_salt="alice",
        )
        for i in range(3)
    ]
    shape = torch.Size([2, 2, 16, 16])
    layout = MemoryLayoutDesc(shapes=[shape], dtypes=[torch.float32])
    originals = [torch.linspace(-1, 1, 1024).view(shape) + i / 8 for i in range(3)]
    try:
        reserved = manager.reserve_write(keys, layout)
        assert len(reserved) == len(keys)
        for key, original in zip(keys, originals, strict=True):
            tensor = reserved[key].tensor
            assert tensor is not None
            tensor.copy_(original)
        manager.finish_write(keys)
        _wait_until(lambda: len(list(tmp_path.glob("*.data"))) == len(keys))

        descriptor, adapter = manager.l2_adapters()[0]
        assert descriptor.type_name == spec["type"]
        expected_bytes = len(keys) * (1024 if pipeline == "fp8" else 4096)
        _wait_until(lambda: adapter.get_usage().total_bytes_used == expected_bytes)
        assert adapter.get_usage().bytes_by_cache_salt == {"alice": expected_bytes}

        manager.delete_l2_adapter(descriptor.index)
        assert manager.l2_adapters() == []
        assert len(list(tmp_path.glob("*.data"))) == len(keys)
        new_id = manager.add_l2_adapter(config)
        assert new_id != descriptor.index
        manager.clear(force=True)
        assert manager.report_status()["l1_manager"]["total_object_count"] == 0

        handle = manager.submit_prefetch_task(single_row_spec(keys, layout))
        deadline = time.monotonic() + 10
        result = None
        while result is None and time.monotonic() < deadline:
            result = manager.query_prefetch_status(handle)
            time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].count_leading_ones() == len(keys)

        with manager.read_prefetched_results(keys) as loaded:
            assert loaded is not None
            for obj, original in zip(loaded, originals, strict=True):
                expected = (
                    original.to(torch.float8_e4m3fn).to(torch.float32)
                    if pipeline == "fp8"
                    else original
                )
                assert obj.tensor is not None
                torch.testing.assert_close(obj.tensor, expected, rtol=0, atol=0)
        manager.finish_read_prefetched(keys)
        manager.clear(force=True)
        _wait_until(
            lambda: manager.report_status()["l1_manager"]["memory_used_bytes"] == 0
        )
        manager.l2_adapters()[0][1].delete(keys)
        assert not list(tmp_path.glob("*.data"))
    finally:
        manager.close()


@pytest.mark.parametrize("adapter_type", ["fs", "plugin"])
@pytest.mark.parametrize("io_mode", ["buffered", "direct", "unaligned"])
def test_tensor_io_and_incomplete_reads(
    tmp_path: Path, adapter_type: str, io_mode: str
) -> None:
    """Real tensor buffers round-trip; truncated files are reported as misses."""
    if io_mode == "direct" and not hasattr(os, "O_DIRECT"):
        pytest.skip("O_DIRECT is unavailable on this OS")
    alignment = os.statvfs(tmp_path).f_bsize
    data_size = 17 if io_mode == "unaligned" else alignment
    buffer = make_aligned_tensor(2 * alignment, alignment)
    source = make_memory_objects(buffer, 1, data_size, 0, fill_offset=29)
    destination = make_memory_objects(buffer, 1, data_size, alignment)

    if io_mode == "direct":
        probe = tmp_path / "direct-io-probe"
        try:
            fd = os.open(probe, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o600)
            try:
                os.write(fd, source[0].byte_array)
            finally:
                os.close(fd)
        except OSError as error:
            if error.errno in {errno.EINVAL, errno.ENOTSUP, errno.EOPNOTSUPP}:
                pytest.skip(f"test filesystem does not support O_DIRECT: {error}")
            raise
        finally:
            probe.unlink(missing_ok=True)

    spec = _fs_spec(
        tmp_path, adapter_type, use_odirect=io_mode != "buffered", read_ahead_size=7
    )
    config = parse_args_to_l2_adapters_config(
        Namespace(l2_adapter=[json.dumps(spec)])
    ).adapters[0]
    adapter = create_l2_adapter(config, create_l1_memory_desc(buffer, alignment))
    keys = make_object_keys(1)
    try:
        task = adapter.submit_store_task(keys, source)
        assert wait_eventfd(adapter.get_store_event_fd(), timeout=5)
        assert adapter.pop_completed_store_tasks() == {task: data_size}
        task = adapter.submit_lookup_and_lock_task(keys, {})
        assert wait_eventfd(adapter.get_lookup_and_lock_event_fd(), timeout=5)
        hits = adapter.query_lookup_and_lock_result(task)
        assert hits is not None and hits.popcount() == 1
        task = adapter.submit_load_task(keys, destination)
        assert wait_eventfd(adapter.get_load_event_fd(), timeout=5)
        hits = adapter.query_load_result(task)
        assert hits is not None and hits.popcount() == 1
        assert bytes(destination[0].byte_array) == bytes(source[0].byte_array)
        adapter.submit_unlock(keys)

        next(tmp_path.glob("*.data")).write_bytes(b"partial")
        task = adapter.submit_load_task(keys, destination)
        assert wait_eventfd(adapter.get_load_event_fd(), timeout=5)
        hits = adapter.query_load_result(task)
        assert hits is not None and hits.popcount() == 0
        adapter.delete(keys)
        assert not list(tmp_path.glob("*.data"))
    finally:
        adapter.close()
