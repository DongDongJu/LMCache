# SPDX-License-Identifier: Apache-2.0
"""SharedL1Manager against a real orchestrator process. Managers in one process
stand in for MP servers, each with its own client id and mapping of one file."""

# Standard
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.memory_orchestrator.api import (
    ReadGrantResult,
    ReadRequest,
    RegionContract,
    RegionFencedError,
    RequestRejectedError,
    WireLayout,
    WriteGrantResult,
    WriteRequest,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.distributed.shared_l1.utils import shared_config
from tests.v1.memory_orchestrator.harness import (
    Orchestrator,
    free_port,
    running_orchestrator,
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


def _open_manager(
    region: Path, endpoint: str, client_id: str, **kwargs: object
) -> SharedL1Manager:
    kwargs.setdefault("size_bytes", CAPACITY)
    return SharedL1Manager(
        shared_config(region, endpoint, client_id, **kwargs)  # type: ignore[arg-type]
    )


def _reserve_write(manager: SharedL1Manager, index: int) -> L1OperationResult:
    return manager.reserve_write([key(index)], [False], LAYOUT)[key(index)]


def _store(manager: SharedL1Manager, index: int, value: float) -> None:
    error, obj = _reserve_write(manager, index)
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(value)
    assert manager.finish_write([key(index)]) == {key(index): L1Error.SUCCESS}


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
        manager = _open_manager(region, orchestrator.endpoint, client_id, **kwargs)
        managers.append(manager)
        return manager

    yield create
    for manager in reversed(managers):
        manager.close()


def test_object_stored_by_one_server_is_read_by_another(
    make_manager: ManagerFactory,
) -> None:
    writer, reader = make_manager("server-a"), make_manager("server-b")
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
    assert _reserve_write(first, 1)[0] is L1Error.SUCCESS
    # Another server is refused while the key is being written.
    assert _reserve_write(second, 1)[0] is L1Error.KEY_NOT_WRITABLE
    # An object being written is not readable anywhere.
    assert second.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert first.finish_write([key(1)]) == {key(1): L1Error.SUCCESS}
    assert _reserve_write(second, 1)[0] is L1Error.KEY_NOT_WRITABLE


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
    assert manager.get_memory_usage()[1] == CAPACITY


def test_finish_without_a_reservation_is_key_not_exist(
    make_manager: ManagerFactory,
) -> None:
    manager = make_manager("server-a")
    assert manager.finish_write_and_delete([key(6)]) == {key(6): L1Error.KEY_NOT_EXIST}
    assert manager.finish_read([key(6)]) == {key(6): L1Error.KEY_NOT_EXIST}


def test_read_leases_follow_read_locks(make_manager: ManagerFactory) -> None:
    writer, reader = make_manager("server-a"), make_manager("server-b")
    _store(writer, 1, 2.0)
    assert reader.reserve_read([key(1)], read_locks=3)[key(1)][0] is L1Error.SUCCESS
    assert reader.finish_read([key(1)], read_locks=2) == {key(1): L1Error.SUCCESS}
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    assert reader.finish_read([key(1)]) == {key(1): L1Error.SUCCESS}
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    # The orchestrator released every lease; this first status call asks it.
    shared = reader.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["region_usage"]["read_leases"] == 0


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


def test_describe_status(make_manager: ManagerFactory, region: Path) -> None:
    manager = make_manager("server-a")
    assert manager.get_capacity_bytes_by_backend() == {L1BackendType.DRAM: CAPACITY}
    assert manager.get_l1_memory_desc() is None
    assert manager.owns_device(str(region))
    status = manager.report_status()
    assert status["is_healthy"] and manager.memcheck()
    shared = status["shared"]
    assert isinstance(shared, dict)
    assert shared["client_id"] == "server-a"
    assert shared["medium"] == "dram"
    assert shared["path"] == str(region)
    assert shared["visibility_mode"] == "coherent"


def test_close_retires_unfinished_writes(
    make_manager: ManagerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    first, second = make_manager("server-a"), make_manager("server-b")
    assert _reserve_write(first, 1)[0] is L1Error.SUCCESS
    first.close()
    assert _reserve_write(second, 1)[0] is L1Error.SUCCESS
    # No view of the unfinished write outlived close(), so the region was unmapped.
    assert "still has live views" not in caplog.text


def test_restart_under_the_same_client_id_retires_the_crashed_run(
    make_manager: ManagerFactory,
) -> None:
    crashed, other = make_manager("server-a"), make_manager("server-b")
    assert _reserve_write(crashed, 1)[0] is L1Error.SUCCESS
    assert _reserve_write(other, 1)[0] is L1Error.KEY_NOT_WRITABLE
    # The "restarted" server registers a new incarnation of server-a.
    make_manager("server-a")
    assert _reserve_write(other, 1)[0] is L1Error.SUCCESS
    # The old incarnation is fenced: its late commit cannot land.
    assert crashed.finish_write([key(1)]) == {key(1): L1Error.KEY_IN_WRONG_STATE}
    assert not crashed.memcheck()


def test_unreachable_orchestrator_turns_into_misses_and_overflow(
    make_manager: ManagerFactory, orchestrator: Orchestrator
) -> None:
    writer = make_manager("server-a", rpc_timeout_seconds=0.3)
    _store(writer, 1, 1.0)
    orchestrator.stop()
    result = writer.reserve_write([key(2)], [False], LAYOUT)
    assert result == {key(2): (L1Error.OUT_OF_MEMORY, None)}
    assert writer.reserve_read([key(1)]) == {key(1): (L1Error.KEY_NOT_EXIST, None)}
    assert not writer.memcheck()
    shared = writer.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["orchestrator_reachable"] is False


def test_orchestrator_restart_fences_the_manager_until_reset(
    region: Path, tmp_path: Path
) -> None:
    state = tmp_path / "state"
    flags = {"capacity_bytes": CAPACITY, "alignment_bytes": ALIGN, "port": free_port()}
    orchestrator = start_orchestrator(state, **flags)
    manager = restarted = None
    try:
        manager = _open_manager(region, orchestrator.endpoint, "server-a")
        _store(manager, 1, 1.0)
        orchestrator.kill()
        restarted = start_orchestrator(state, **flags)
        # Lost state: the region must not be served until an offline reset.
        assert manager.reserve_read([key(1)]) == {key(1): (L1Error.KEY_NOT_EXIST, None)}
        assert _reserve_write(manager, 2)[0] is L1Error.OUT_OF_MEMORY
        assert not manager.memcheck()
        shared = manager.report_status()["shared"]
        assert isinstance(shared, dict)
        assert shared["fenced"]
        # A new server refuses to start against a region that needs a reset.
        with pytest.raises(RuntimeError, match="offline reset"):
            _open_manager(region, restarted.endpoint, "server-b")
    finally:
        if manager is not None:
            manager.close()
        orchestrator.kill()
        if restarted is not None:
            restarted.stop()


def test_region_larger_than_size_gb_is_refused(make_manager: ManagerFactory) -> None:
    with pytest.raises(ValueError, match="size_gb"):
        make_manager("server-a", size_bytes=CAPACITY // 2)


def test_noncoherent_extents_must_not_share_cache_lines(tmp_path: Path) -> None:
    with running_orchestrator(
        tmp_path / "state",
        capacity_bytes=CAPACITY,
        alignment_bytes=64,
        visibility_mode="software_fenced",
    ) as orch:
        with pytest.raises(ValueError, match="page alignment"):
            _open_manager(tmp_path / "region", orch.endpoint, "unaligned")


def test_changed_startup_epoch_unregisters_the_failed_binding(
    make_manager: ManagerFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    describe = MemoryOrchestratorClient.describe_region
    calls = 0

    def changed_epoch(client: MemoryOrchestratorClient) -> RegionContract:
        nonlocal calls
        contract = describe(client)
        calls += 1
        return (
            replace(contract, region_epoch=contract.region_epoch ^ 1)
            if calls == 1
            else contract
        )

    with monkeypatch.context() as patch:
        patch.setattr(MemoryOrchestratorClient, "describe_region", changed_epoch)
        with pytest.raises(RegionFencedError, match="region changed"):
            make_manager("changed")
    observer = make_manager("observer")
    shared = observer.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["region_usage"]["clients"] == 1


def test_concurrent_duplicate_writers_get_one_extent(
    make_manager: ManagerFactory,
) -> None:
    managers = [make_manager(f"server-{i}") for i in range(8)]
    barrier = threading.Barrier(len(managers))

    def reserve(manager: SharedL1Manager) -> L1Error:
        barrier.wait(timeout=10)
        return _reserve_write(manager, 1)[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        errors = list(executor.map(reserve, managers))
    assert errors.count(L1Error.SUCCESS) == 1
    assert errors.count(L1Error.KEY_NOT_WRITABLE) == 7
    assert managers[0].get_memory_usage()[0] == ALIGN


def test_overlapping_reads_release_only_their_lease_count(
    make_manager: ManagerFactory,
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    _store(writer, 1, 7)
    barrier = threading.Barrier(8)

    def read(_: int) -> L1Error:
        barrier.wait(timeout=10)
        return reader.reserve_read([key(1)])[key(1)][0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(read, range(8))) == [L1Error.SUCCESS] * 8
    for _ in range(7):
        assert reader.finish_read([key(1)])[key(1)] is L1Error.SUCCESS
        assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    shared = reader.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["region_usage"]["read_leases"] == 0


def test_conflicting_write_layout_does_not_become_overflow(
    make_manager: ManagerFactory,
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    _store(writer, 1, 1)
    incompatible = MemoryLayoutDesc([torch.Size([1000, 2])], [torch.float16])
    with pytest.raises(RequestRejectedError):
        peer.reserve_write([key(1)], [False], incompatible)
    assert peer.memcheck()
    assert peer.get_staging_memory_usage() == 0


def test_failed_publish_aborts_its_write(
    make_manager: ManagerFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    assert _reserve_write(writer, 1)[0] is L1Error.SUCCESS

    def fail(self: Visibility, ranges: list[tuple[int, int]]) -> None:
        raise RuntimeError("publish failed")

    monkeypatch.setattr(Visibility, "publish", fail)
    with pytest.raises(RuntimeError, match="publish failed"):
        writer.finish_write([key(1)])
    assert writer.get_staging_memory_usage() == 0
    assert _reserve_write(peer, 1)[0] is L1Error.SUCCESS


def test_failed_acquire_keeps_the_overlapping_read(
    make_manager: ManagerFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    _store(writer, 1, 2)
    assert reader.reserve_read([key(1)])[key(1)][0] is L1Error.SUCCESS

    def fail(self: Visibility, ranges: list[tuple[int, int]]) -> None:
        raise RuntimeError("acquire failed")

    monkeypatch.setattr(Visibility, "acquire", fail)
    with pytest.raises(RuntimeError, match="acquire failed"):
        reader.reserve_read([key(1)], read_locks=2)
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    shared = reader.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["region_usage"]["read_leases"] == 0


def test_malformed_write_grant_aborts_its_token(
    make_manager: ManagerFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    reserve = MemoryOrchestratorClient.reserve_write

    def bad_grant(
        client: MemoryOrchestratorClient, entries: list[WriteRequest]
    ) -> list[WriteGrantResult]:
        grants = reserve(client, entries)
        return [replace(grants[0], handle=None)]

    with monkeypatch.context() as patch:
        patch.setattr(MemoryOrchestratorClient, "reserve_write", bad_grant)
        assert _reserve_write(writer, 1)[0] is L1Error.KEY_NOT_WRITABLE
    assert writer.get_staging_memory_usage() == 0
    assert _reserve_write(peer, 1)[0] is L1Error.SUCCESS


@pytest.mark.parametrize("fault", ["no_handle", "extent", "payload", "layout"])
def test_malformed_read_grant_preserves_the_existing_lease(
    make_manager: ManagerFactory, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    _store(writer, 1, 2)
    original = reader.reserve_read([key(1)])[key(1)]
    reserve = MemoryOrchestratorClient.reserve_read

    def bad_grant(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        grant = reserve(client, entries)[0]
        handle = grant.handle
        assert handle is not None
        if fault == "no_handle":
            grant = replace(grant, handle=None)
        elif fault == "extent":
            grant = replace(grant, handle=replace(handle, offset=CAPACITY))
        elif fault == "payload":
            grant = replace(grant, payload_bytes=CAPACITY + 1)
        else:
            grant = replace(grant, layout=WireLayout(((4, 500),), ("float16",)))
        return [grant]

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", bad_grant)
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    reader.finish_read([key(1)])
    shared = reader.report_status()["shared"]
    assert isinstance(shared, dict)
    assert shared["region_usage"]["read_leases"] == 0
