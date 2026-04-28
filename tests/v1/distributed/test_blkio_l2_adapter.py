# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for BlkioL2AdapterConfig parsing and registration.

These tests verify the config class, validation, and factory
registration — no block device or C++ extension required.
"""

# Standard
import os
import select
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.blkio_l2_adapter import (
    BlkioL2Adapter,
    BlkioL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.platform import consume_fd


class _FakeMemoryObj:
    def __init__(self, data: bytes):
        self._buf = bytearray(data)
        self.byte_array = memoryview(self._buf)

    def get_size(self):
        return len(self._buf)


class _FakeBlkioClient:
    def __init__(self, device_path: str, num_workers: int, direct_io: bool):
        del device_path, num_workers, direct_io
        self._efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._next_fid = 1
        self._data: dict[str, bytes] = {}
        self._completions: list[tuple[int, bool, str, list[bool]]] = []

    def event_fd(self):
        return self._efd

    def submit_batch_set(self, keys, memoryviews):
        fid = self._next_fid
        self._next_fid += 1
        for key, view in zip(keys, memoryviews, strict=True):
            self._data[key] = bytes(view)
        self._completions.append((fid, True, "", [True] * len(keys)))
        os.eventfd_write(self._efd, 1)
        return fid

    def submit_batch_get(self, keys, memoryviews):
        fid = self._next_fid
        self._next_fid += 1
        results = []
        for key, view in zip(keys, memoryviews, strict=True):
            payload = self._data.get(key)
            if payload is None:
                results.append(False)
                continue
            view[: len(payload)] = payload
            results.append(True)
        self._completions.append((fid, all(results), "", results))
        os.eventfd_write(self._efd, 1)
        return fid

    def drain_completions(self):
        try:
            os.eventfd_read(self._efd)
        except BlockingIOError:
            pass
        completions = self._completions
        self._completions = []
        return completions

    def close(self):
        os.close(self._efd)


def _install_fake_blkio(monkeypatch):
    module = types.ModuleType("lmcache.lmcache_blkio")
    module.LMCacheBlkioClient = _FakeBlkioClient
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_blkio", module)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="blkio-test",
        kv_rank=0,
    )


def _wait_fd(fd: int):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    assert poll.poll(5000)
    consume_fd(fd)


class TestBlkioL2AdapterConfig:
    """Config parsing and validation for the blkio L2 adapter."""

    def test_from_dict_minimal(self):
        config = BlkioL2AdapterConfig.from_dict(
            {
                "type": "blkio",
                "device_path": "/dev/nvme0n1",
            }
        )
        assert config.device_path == "/dev/nvme0n1"
        assert config.num_workers == 4
        assert config.direct_io is True

    def test_from_dict_full(self):
        config = BlkioL2AdapterConfig.from_dict(
            {
                "type": "blkio",
                "device_path": "/dev/loop0",
                "num_workers": 8,
                "direct_io": False,
            }
        )
        assert config.device_path == "/dev/loop0"
        assert config.num_workers == 8
        assert config.direct_io is False

    def test_from_dict_missing_device_path_raises(self):
        with pytest.raises(ValueError, match="device_path"):
            BlkioL2AdapterConfig.from_dict({"type": "blkio"})

    def test_from_dict_empty_device_path_raises(self):
        with pytest.raises(ValueError, match="device_path"):
            BlkioL2AdapterConfig.from_dict({"type": "blkio", "device_path": ""})

    def test_from_dict_invalid_num_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "num_workers": 0,
                }
            )

    def test_from_dict_negative_num_workers_raises(self):
        with pytest.raises(ValueError, match="num_workers"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "num_workers": -1,
                }
            )

    def test_from_dict_invalid_direct_io_raises(self):
        with pytest.raises(ValueError, match="direct_io"):
            BlkioL2AdapterConfig.from_dict(
                {
                    "type": "blkio",
                    "device_path": "/dev/loop0",
                    "direct_io": "yes",
                }
            )

    def test_registered_as_blkio(self):
        assert "blkio" in get_registered_l2_adapter_types()

    def test_help_returns_string(self):
        help_text = BlkioL2AdapterConfig.help()
        assert isinstance(help_text, str)
        assert "device_path" in help_text
        assert "num_workers" in help_text
        assert "direct_io" in help_text

    def test_constructor_defaults(self):
        config = BlkioL2AdapterConfig(device_path="/dev/sda")
        assert config.device_path == "/dev/sda"
        assert config.num_workers == 4
        assert config.direct_io is True

    def test_constructor_custom(self):
        config = BlkioL2AdapterConfig(
            device_path="/dev/nvme0n1",
            num_workers=16,
            direct_io=False,
        )
        assert config.device_path == "/dev/nvme0n1"
        assert config.num_workers == 16
        assert config.direct_io is False


def test_blkio_adapter_store_lookup_load_unlock_delete(monkeypatch):
    """Blkio MP L2 should support lookup/load, not only writes."""
    _install_fake_blkio(monkeypatch)
    adapter = BlkioL2Adapter(
        BlkioL2AdapterConfig(device_path="/tmp/fake-blkio", direct_io=False)
    )
    try:
        key = _key(1)
        src = _FakeMemoryObj(b"hello blkio")
        store_task = adapter.submit_store_task([key], [src])
        _wait_fd(adapter.get_store_event_fd())
        assert adapter.pop_completed_store_tasks()[store_task] is True

        lookup_task = adapter.submit_lookup_and_lock_task([key])
        _wait_fd(adapter.get_lookup_and_lock_event_fd())
        lookup = adapter.query_lookup_and_lock_result(lookup_task)
        assert lookup is not None
        assert lookup.test(0)

        dst = _FakeMemoryObj(b"\x00" * len(src.byte_array))
        load_task = adapter.submit_load_task([key], [dst])
        _wait_fd(adapter.get_load_event_fd())
        loaded = adapter.query_load_result(load_task)
        assert loaded is not None
        assert loaded.test(0)
        assert bytes(dst.byte_array) == b"hello blkio"

        adapter.submit_unlock([key])
        adapter.delete([key])
        lookup_task = adapter.submit_lookup_and_lock_task([key])
        _wait_fd(adapter.get_lookup_and_lock_event_fd())
        lookup = adapter.query_lookup_and_lock_result(lookup_task)
        assert lookup is not None
        assert not lookup.test(0)
    finally:
        adapter.close()


def test_blkio_adapter_factory_creates_adapter(monkeypatch):
    """The L2 adapter factory should resolve the built-in blkio type."""
    _install_fake_blkio(monkeypatch)
    adapter = create_l2_adapter_from_registry(
        BlkioL2AdapterConfig(device_path="/tmp/fake-blkio", direct_io=False)
    )
    try:
        assert isinstance(adapter, BlkioL2Adapter)
    finally:
        adapter.close()
