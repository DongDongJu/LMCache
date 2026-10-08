# SPDX-License-Identifier: Apache-2.0
"""Two server processes share one region through one orchestrator.

Each process maps the same file at its own virtual address, as two hosts map
one Device-DAX device.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import Any
import multiprocessing as mp
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from tests.v1.distributed.shared_l1.utils import running_orchestrator, shared_config

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
CAPACITY = 64 * ALIGN
LAYOUT = MemoryLayoutDesc([torch.Size([4, 512])], [torch.bfloat16])  # 4 KiB


def _key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "two-process", 0)


def _pattern(index: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(index)
    return torch.randn(4, 512, generator=generator).to(torch.bfloat16)


def _manager(region: str, endpoint: str, client_id: str) -> SharedL1Manager:
    return SharedL1Manager(
        shared_config(Path(region), endpoint, client_id, size_bytes=CAPACITY)
    )


def _writer(region: str, endpoint: str, indices: list[int], queue: mp.Queue) -> None:
    manager = _manager(region, endpoint, "writer")
    try:
        keys = [_key(i) for i in indices]
        reserved = manager.reserve_write(keys, [False] * len(keys), LAYOUT)
        addresses = []
        for index, key in zip(indices, keys, strict=True):
            error, obj = reserved[key]
            assert (
                error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
            )
            obj.tensor.copy_(_pattern(index))
            addresses.append(obj.data_ptr)
        result = manager.finish_write(keys)
        queue.put(([e.name for e in result.values()], addresses))
    finally:
        manager.close()


def _reader(region: str, endpoint: str, indices: list[int], queue: mp.Queue) -> None:
    manager = _manager(region, endpoint, "reader")
    try:
        keys = [_key(i) for i in indices]
        reserved = manager.reserve_read(keys)
        matches: list[bool | None] = []
        addresses: list[int] = []
        for index, key in zip(indices, keys, strict=True):
            error, obj = reserved[key]
            if error is not L1Error.SUCCESS or obj is None or obj.tensor is None:
                matches.append(None)
                continue
            matches.append(bool(torch.equal(obj.tensor, _pattern(index))))
            addresses.append(obj.data_ptr)
        manager.finish_read([k for k in keys if reserved[k][0] is L1Error.SUCCESS])
        queue.put((matches, addresses))
    finally:
        manager.close()


def _crashing_writer(region: str, endpoint: str, ready: mp.Queue) -> None:
    manager = _manager(region, endpoint, "writer")
    reserved = manager.reserve_write([_key(9)], [False], LAYOUT)
    ready.put(reserved[_key(9)][0].name)
    # Flush the queue's feeder thread before the hard exit below.
    ready.close()
    ready.join_thread()
    # Die mid-store: no commit, no abort, no CloseClient.
    os._exit(0)


def _run(target: Callable[..., None], *args: Any) -> Any:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue))
    process.start()
    result = queue.get(timeout=120)
    process.join(timeout=60)
    assert process.exitcode == 0
    return result


@pytest.fixture
def region(tmp_path: Path) -> str:
    path = tmp_path / "region"
    path.write_bytes(b"\0" * CAPACITY)
    return str(path)


def test_store_in_one_process_read_byte_identical_in_another(
    region: str, tmp_path: Path
) -> None:
    with running_orchestrator(tmp_path / "state", capacity_bytes=CAPACITY) as orch:
        statuses, _ = _run(_writer, region, orch.endpoint, [1, 2, 3])
        assert statuses == ["SUCCESS"] * 3
        matches, _ = _run(_reader, region, orch.endpoint, [1, 2, 3, 4])
        assert matches == [True, True, True, None]


def test_writer_killed_mid_store_leaves_a_busy_key_not_a_crash(
    region: str, tmp_path: Path
) -> None:
    with running_orchestrator(tmp_path / "state", capacity_bytes=CAPACITY) as orch:
        context = mp.get_context("spawn")
        ready = context.Queue()
        crashed = context.Process(
            target=_crashing_writer, args=(region, orch.endpoint, ready)
        )
        crashed.start()
        assert ready.get(timeout=120) == "SUCCESS"
        crashed.join(timeout=60)

        survivor = _manager(region, orch.endpoint, "reader")
        try:
            assert survivor.reserve_read([_key(9)]) == {
                _key(9): (L1Error.KEY_NOT_EXIST, None)
            }
            assert survivor.reserve_write([_key(9)], [False], LAYOUT)[_key(9)][0] is (
                L1Error.KEY_NOT_WRITABLE
            )
            # The crashed server restarts under its id: its stale write is
            # retired and the key can be stored again.
            statuses, _ = _run(_writer, region, orch.endpoint, [9])
            assert statuses == ["SUCCESS"]
            assert survivor.reserve_read([_key(9)])[_key(9)][0] is L1Error.SUCCESS
            survivor.finish_read([_key(9)])
        finally:
            survivor.close()
