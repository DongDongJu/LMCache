# SPDX-License-Identifier: Apache-2.0
"""SharedL1Manager against a real orchestrator process.

Two managers in one process stand in for two MP servers: each has its own
client identity and its own mapping of one regular file.
"""

# Standard
from collections.abc import Callable, Iterator
from pathlib import Path
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import DevDaxHotPlug, L1ManagerInterface
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from tests.v1.distributed.shared_l1.utils import (
    Orchestrator,
    free_port,
    running_orchestrator,
    shared_config,
    start_orchestrator,
)

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
EXTENTS = 8
CAPACITY = EXTENTS * ALIGN
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])  # 4000 B

ManagerFactory = Callable[..., SharedL1Manager]


def key(index: int) -> ObjectKey:
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "shared-test", 0)


@pytest.fixture
def region(tmp_path: Path) -> Path:
    path = tmp_path / "region"
    path.write_bytes(b"\0" * CAPACITY)
    return path


@pytest.fixture
def orchestrator(tmp_path: Path) -> Iterator[Orchestrator]:
    with running_orchestrator(
        tmp_path / "state", capacity_bytes=CAPACITY, alignment_bytes=ALIGN
    ) as running:
        yield running


@pytest.fixture
def make_manager(region: Path, orchestrator: Orchestrator) -> Iterator[ManagerFactory]:
    managers: list[SharedL1Manager] = []

    def create(client_id: str, **kwargs: object) -> SharedL1Manager:
        manager = SharedL1Manager(
            shared_config(
                region,
                orchestrator.endpoint,
                client_id,
                size_bytes=CAPACITY,
                **kwargs,  # type: ignore[arg-type]
            )
        )
        managers.append(manager)
        return manager

    yield create
    for manager in reversed(managers):
        manager.close()


def _store(manager: SharedL1Manager, index: int, value: float) -> None:
    error, obj = manager.reserve_write([key(index)], [False], LAYOUT)[key(index)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(value)
    assert manager.finish_write([key(index)]) == {key(index): L1Error.SUCCESS}


def test_object_stored_by_one_server_is_read_by_another(
    make_manager: ManagerFactory,
) -> None:
    writer, reader = make_manager("server-a"), make_manager("server-b")
    assert isinstance(writer, L1ManagerInterface)
    _store(writer, 1, 3.25)

    error, view = reader.reserve_read([key(1)])[key(1)]
    assert error is L1Error.SUCCESS and view is not None and view.tensor is not None
    assert view.tensor.shape == torch.Size([2, 1000])
    assert view.tensor.dtype == torch.float16
    assert torch.all(view.tensor == 3.25)
    assert view.get_l1_manager() == reader.l1_manager_id
    assert reader.unsafe_read([key(1)])[key(1)] == (L1Error.SUCCESS, view)
    assert reader.finish_read([key(1)]) == {key(1): L1Error.SUCCESS}
    assert reader.unsafe_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)


def test_one_writer_per_key_and_committed_objects_are_immutable(
    make_manager: ManagerFactory,
) -> None:
    first, second = make_manager("server-a"), make_manager("server-b")
    error, _ = first.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS
    # Another server, and the same server under the same tag, are refused.
    assert second.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )
    assert first.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )
    # An object being written is not readable anywhere.
    assert second.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert first.finish_write([key(1)]) == {key(1): L1Error.SUCCESS}
    assert second.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.KEY_NOT_WRITABLE
    )


def test_aborted_write_frees_the_key_but_not_the_extent(
    make_manager: ManagerFactory,
) -> None:
    first, second = make_manager("server-a"), make_manager("server-b")
    _, aborted = first.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert first.finish_write_and_delete([key(1)]) == {key(1): L1Error.SUCCESS}
    error, granted = second.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS and aborted is not None and granted is not None
    # Allocation is monotonic: the aborted extent is never handed out again.
    assert granted.data_ptr != aborted.data_ptr
    assert second.report_status()["shared"]["region_usage"]["consumed"] == 1


def test_full_region_reports_out_of_memory_for_the_whole_batch(
    make_manager: ManagerFactory,
) -> None:
    manager = make_manager("server-a")
    _store(manager, 0, 1.0)
    batch = [key(i) for i in range(1, EXTENTS + 1)]  # one more than the rest
    result = manager.reserve_write(batch, [False] * len(batch), LAYOUT)
    assert {error for error, _ in result.values()} == {L1Error.OUT_OF_MEMORY}
    fits = batch[:-1]
    result = manager.reserve_write(fits, [False] * len(fits), LAYOUT)
    assert {error for error, _ in result.values()} == {L1Error.SUCCESS}
    used, total = manager.get_memory_usage()
    assert total == CAPACITY


def test_finish_without_a_reservation_is_key_not_exist(
    make_manager: ManagerFactory,
) -> None:
    manager = make_manager("server-a")
    assert manager.finish_write([key(5)]) == {key(5): L1Error.KEY_NOT_EXIST}
    _store(manager, 5, 1.0)
    assert manager.finish_write([key(5)]) == {key(5): L1Error.KEY_NOT_EXIST}
    assert manager.finish_write_and_delete([key(6)]) == {key(6): L1Error.KEY_NOT_EXIST}
    assert manager.finish_read([key(6)]) == {key(6): L1Error.KEY_NOT_EXIST}


def test_read_leases_follow_read_locks(make_manager: ManagerFactory) -> None:
    writer, reader = make_manager("server-a"), make_manager("server-b")
    _store(writer, 1, 2.0)
    assert reader.reserve_read([key(1)], read_locks=3)[key(1)][0] is L1Error.SUCCESS

    def leases() -> int:
        # Usage is cached for a second; the orchestrator's count is the truth.
        return reader.report_status()["shared"]["region_usage"]["read_leases"]

    assert reader.finish_read([key(1)], read_locks=2) == {key(1): L1Error.SUCCESS}
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    assert reader.finish_read([key(1)]) == {key(1): L1Error.SUCCESS}
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    time.sleep(1.1)
    assert leases() == 0


def test_delete_eviction_and_clear_leave_shared_objects(
    make_manager: ManagerFactory,
) -> None:
    writer, reader = make_manager("server-a"), make_manager("server-b")
    _store(writer, 1, 2.0)
    assert writer.delete([key(1)], force=True) == {key(1): L1Error.KEY_IS_LOCKED}
    assert not writer.is_key_evictable(key(1))
    writer.clear(force=True)
    assert reader.reserve_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])


def test_describe_status_and_reconfiguration(
    make_manager: ManagerFactory, region: Path
) -> None:
    manager = make_manager("server-a")
    assert manager.config.shared is not None
    assert manager.get_capacity_bytes_by_backend() == {L1BackendType.DRAM: CAPACITY}
    assert manager.get_l1_memory_desc() is None
    assert manager.owns_device(str(region))
    status = manager.report_status()
    assert status["is_healthy"] and manager.memcheck()
    assert status["shared"]["client_id"] == "server-a"
    assert (status["shared"]["medium"], status["shared"]["path"]) == (
        "dram",
        str(region),
    )
    assert status["shared"]["visibility_mode"] == "coherent"
    # The orchestrator owns the region's extents, so there is nothing to hot-plug.
    assert not isinstance(manager, DevDaxHotPlug)


def test_close_retires_unfinished_writes(make_manager: ManagerFactory) -> None:
    first, second = make_manager("server-a"), make_manager("server-b")
    assert first.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS
    first.close()
    assert second.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.SUCCESS
    )


def test_restart_under_the_same_client_id_retires_the_crashed_run(
    make_manager: ManagerFactory,
) -> None:
    crashed = make_manager("server-a")
    other = make_manager("server-b")
    assert crashed.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.SUCCESS
    )
    # The "restarted" server registers a new incarnation of server-a.
    make_manager("server-a")
    assert other.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is (
        L1Error.SUCCESS
    )
    # The old incarnation is fenced: its late commit cannot land.
    assert crashed.finish_write([key(1)]) == {key(1): L1Error.KEY_IN_WRONG_STATE}
    assert not crashed.memcheck()


def test_unreachable_orchestrator_turns_into_misses_and_overflow(
    region: Path, tmp_path: Path
) -> None:
    orchestrator = start_orchestrator(
        tmp_path / "state2", capacity_bytes=CAPACITY, alignment_bytes=ALIGN
    )
    writer = SharedL1Manager(
        shared_config(
            region,
            orchestrator.endpoint,
            "server-a",
            size_bytes=CAPACITY,
            rpc_timeout_seconds=0.3,
        )
    )
    try:
        _store(writer, 1, 1.0)
        orchestrator.stop()
        result = writer.reserve_write([key(2)], [False], LAYOUT)
        assert result == {key(2): (L1Error.OUT_OF_MEMORY, None)}
        assert writer.reserve_read([key(1)]) == {key(1): (L1Error.KEY_NOT_EXIST, None)}
        assert not writer.memcheck()
    finally:
        writer.close()


def test_orchestrator_restart_fences_the_manager_until_reset(
    region: Path, tmp_path: Path
) -> None:
    port = free_port()
    state = tmp_path / "state3"
    orchestrator = start_orchestrator(
        state, capacity_bytes=CAPACITY, alignment_bytes=ALIGN, port=port
    )
    manager = SharedL1Manager(
        shared_config(region, orchestrator.endpoint, "server-a", size_bytes=CAPACITY)
    )
    restarted = None
    try:
        _store(manager, 1, 1.0)
        orchestrator.kill()
        restarted = start_orchestrator(
            state, capacity_bytes=CAPACITY, alignment_bytes=ALIGN, port=port
        )
        # Lost state: the region must not be served until an offline reset.
        assert manager.reserve_read([key(1)]) == {key(1): (L1Error.KEY_NOT_EXIST, None)}
        assert manager.reserve_write([key(2)], [False], LAYOUT)[key(2)][0] is (
            L1Error.OUT_OF_MEMORY
        )
        assert not manager.memcheck()
        assert manager.report_status()["shared"]["fenced"]
        # A new server refuses to start against a region that needs a reset.
        with pytest.raises(RuntimeError, match="offline reset"):
            SharedL1Manager(
                shared_config(
                    region, restarted.endpoint, "server-b", size_bytes=CAPACITY
                )
            )
    finally:
        manager.close()
        if restarted is not None:
            restarted.stop()


def test_region_larger_than_size_gb_is_refused(
    region: Path, orchestrator: Orchestrator
) -> None:
    with pytest.raises(ValueError, match="size_gb"):
        SharedL1Manager(
            shared_config(
                region, orchestrator.endpoint, "server-a", size_bytes=CAPACITY // 2
            )
        )
