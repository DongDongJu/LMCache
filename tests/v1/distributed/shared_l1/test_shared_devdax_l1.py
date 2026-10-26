# SPDX-License-Identifier: Apache-2.0
"""A shared Device-DAX L1 is an ordinary L1Manager whose objects a memory
orchestrator owns; these tests run several such managers on one region."""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    L1BackendType,
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.shared_l1 import visibility
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.memory_orchestrator.api import (
    OrchestratorUnavailableError,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionFencedError,
    RegionUsage,
    RequestRejectedError,
    WireLayout,
    WriteGrantResult,
    WriteRequest,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.shared_l1.utils import (
    CAPACITY,
    key,
    region_file,
    shared_config,
)
from tests.v1.memory_orchestrator.harness import Orchestrator

pytestmark = pytest.mark.no_shared_allocator
Factory = Callable[..., L1Manager]
Usage = Callable[[], RegionUsage]
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])


def store(manager: L1Manager, index: int = 1, value: int = 7) -> None:
    error, obj = manager.reserve_write([key(index)], [False], LAYOUT)[key(index)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(value)
    assert manager.finish_write([key(index)])[key(index)] is L1Error.SUCCESS


def test_peer_discovers_published_object(make_manager: Factory) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    error, obj = writer.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(7)
    # Not readable anywhere until the writer publishes and commits it.
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert writer.finish_write([key(1)])[key(1)] is L1Error.SUCCESS
    error, found = reader.reserve_read([key(1)])[key(1)]
    assert error is L1Error.SUCCESS and found is not None and found.tensor is not None
    assert torch.all(found.tensor == 7)
    assert found.get_l1_manager() == reader.l1_manager_id
    assert reader.unsafe_read([key(1)])[key(1)] == (L1Error.SUCCESS, found)
    assert reader.finish_read([key(1)])[key(1)] is L1Error.SUCCESS


def test_offset_region_maps_the_same_bytes_through_another_path(
    make_manager: Factory, tmp_path: Path
) -> None:
    offset = 4096
    device = region_file(tmp_path / "device", offset + CAPACITY)
    alias = tmp_path / "alias"
    alias.symlink_to(device)
    writer = make_manager("a", path=device, offset_bytes=offset)
    reader = make_manager("b", path=alias, offset_bytes=offset)
    error, obj = writer.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    # No parent allocator: dropping the view never frees orchestrator bytes.
    assert obj.parent() is None
    obj.tensor.fill_(9)
    writer.finish_write([key(1)])
    # The extent offset is region-relative: the device offset is applied
    # once, by the mapping.
    with open(device, "rb") as backing:
        backing.seek(offset + obj.meta.address)
        data = bytearray(backing.read(obj.get_size()))
    assert torch.all(torch.frombuffer(data, dtype=torch.float16) == 9)
    found = reader.reserve_read([key(1)])[key(1)][1]
    assert found is not None and found.tensor is not None
    assert torch.all(found.tensor == 9)
    reader.finish_read([key(1)])
    assert writer.owns_range(str(alias), offset + CAPACITY - 1, 1)
    assert not writer.owns_range(str(alias), 0, offset)


def test_concurrent_writers_and_overlapping_reads(
    make_manager: Factory, usage: Usage
) -> None:
    managers = [make_manager(str(i)) for i in range(8)]
    barrier = threading.Barrier(8)

    def reserve(manager: L1Manager) -> L1OperationResult:
        barrier.wait(timeout=10)
        return manager.reserve_write([key(1)], [False], LAYOUT)[key(1)]

    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(reserve, managers))
    assert [error for error, _ in results].count(L1Error.SUCCESS) == 1
    index = next(i for i, (error, _) in enumerate(results) if error is L1Error.SUCCESS)
    writer, reader = managers[index], managers[(index + 1) % 8]
    obj = results[index][1]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(7)
    writer.finish_write([key(1)])
    assert (
        reader.reserve_write([key(1)], [False], LAYOUT)[key(1)][0]
        is L1Error.KEY_NOT_WRITABLE
    )
    with ThreadPoolExecutor(8) as workers:
        reads = list(
            workers.map(lambda _: reader.reserve_read([key(1)])[key(1)], range(8))
        )
    assert all(error is L1Error.SUCCESS for error, _ in reads)
    # Finishing one reader leaves the others protected.
    for _ in range(7):
        reader.finish_read([key(1)])
        assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    assert (usage().allocated_bytes, usage().read_leases) == (4096, 0)


def test_delete_and_clear_leave_shared_objects_in_place(
    make_manager: Factory,
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    assert writer.delete([key(1)], force=True) == {key(1): L1Error.KEY_IS_LOCKED}
    writer.clear(force=True)
    assert reader.reserve_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])


def test_aborted_write_is_writable_again_but_bytes_stay_consumed(
    make_manager: Factory, usage: Usage
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    writer.reserve_write([key(1)], [False], LAYOUT)
    assert writer.finish_write_and_delete([key(1)]) == {key(1): L1Error.SUCCESS}
    assert writer.get_staging_memory_usage() == 0
    store(peer)
    assert usage().allocated_bytes == 2 * 4096


def test_close_retires_owned_state(make_manager: Factory, usage: Usage) -> None:
    manager = make_manager("owner")
    manager.reserve_write([key(1)], [False], LAYOUT)
    store(manager, index=2)
    manager.reserve_read([key(2)], read_locks=2)
    manager.close()
    assert (usage().writing, usage().read_leases, usage().clients) == (0, 0, 1)


def test_finish_write_and_reserve_read_reports_only_what_succeeded(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager("owner")
    events: list[tuple[EventType, dict]] = []
    bus = get_event_bus()
    monkeypatch.setattr(
        bus, "publish", lambda event: events.append((event.event_type, event.metadata))
    )
    manager.reserve_write([key(1), key(2)], [False, False], LAYOUT)
    reserve_read = MemoryOrchestratorClient.reserve_read

    def lease_only_first(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        # The second key commits, but its lease is refused.
        return reserve_read(client, entries[:1]) + [
            ReadGrantResult(ReadStatus.BUSY_WRITING)
        ]

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", lease_only_first)
    ret = manager.finish_write_and_reserve_read([key(1), key(2)])
    assert ret[key(1)][0] is L1Error.SUCCESS
    assert ret[key(2)] == (L1Error.KEY_NOT_EXIST, None)
    by_type = {event_type: metadata for event_type, metadata in events}
    combined = by_type[EventType.L1_WRITE_FINISHED_AND_READ_RESERVED]
    finished = by_type[EventType.L1_WRITE_FINISHED]
    assert (combined["keys"], finished["keys"]) == ([key(1)], [key(2)])
    assert combined["shared"] and finished["shared"]
    manager.finish_read([key(1)])


def test_fenced_rpc_misses_reads_and_overflows_writes(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager("owner")
    store(manager)
    fenced = Mock(side_effect=RegionFencedError("stale"))
    with monkeypatch.context() as patch:
        for method in ("reserve_write", "reserve_read"):
            patch.setattr(MemoryOrchestratorClient, method, fenced)
        assert (
            manager.reserve_write([key(2)], [False], LAYOUT)[key(2)][0]
            is L1Error.OUT_OF_MEMORY
        )
        assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
        assert not manager.memcheck()
    # Fencing disables the region until a restart, even once RPCs work.
    assert manager.report_status()["shared"]["state"] == "disabled"
    assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST


def test_stopped_orchestrator_misses_reads_and_overflows_writes(
    make_manager: Factory, orchestrator: Orchestrator
) -> None:
    manager = make_manager("owner", rpc_timeout_seconds=0.5)
    store(manager)
    orchestrator.stop()
    assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    assert (
        manager.reserve_write([key(2)], [False], LAYOUT)[key(2)][0]
        is L1Error.OUT_OF_MEMORY
    )
    assert manager.report_status()["shared"]["state"] == "unavailable"
    assert not manager.report_status()["is_healthy"]


@pytest.mark.parametrize(
    "change,error",
    [
        ({"reset_required": True}, RuntimeError),
        ({"capacity_bytes": CAPACITY * 2}, ValueError),
        ({"visibility_mode": "software_fenced", "alignment_bytes": 64}, ValueError),
        ({"visibility_mode": "unknown"}, ValueError),
        ({"region_epoch": 0}, RegionFencedError),
    ],
)
def test_startup_contract_failure_cleans_up(
    make_manager: Factory,
    usage: Usage,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    error: type[Exception],
) -> None:
    describe = MemoryOrchestratorClient.describe_region
    calls = 0

    def changed(client: MemoryOrchestratorClient) -> RegionContract:
        nonlocal calls
        contract = describe(client)
        calls += 1
        update = (
            {"region_epoch": contract.region_epoch ^ 1}
            if "region_epoch" in change
            else change
        )
        return replace(contract, **update) if calls == 1 else contract

    monkeypatch.setattr(MemoryOrchestratorClient, "describe_region", changed)
    with pytest.raises(error):
        make_manager("bad")
    assert usage().clients == 1


def test_failed_gpu_registration_cleans_up(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: False)
    with pytest.raises(RuntimeError, match="GPU-registered"):
        make_manager("bad")
    assert usage().clients == 1


def test_region_outside_the_device_is_rejected_before_registering(
    orchestrator: Orchestrator, tmp_path: Path, usage: Usage
) -> None:
    small = region_file(tmp_path / "small", CAPACITY // 2)
    with pytest.raises(RuntimeError, match="exceeds"):
        L1Manager(shared_config(small, orchestrator.endpoint, "bad"))
    assert usage().clients == 1


def test_conflicting_layout_is_not_overflow(make_manager: Factory) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    with pytest.raises(RequestRejectedError):
        reader.reserve_write(
            [key(1)],
            [False],
            MemoryLayoutDesc([torch.Size([1000, 2])], [torch.float16]),
        )
    assert reader.get_staging_memory_usage() == 0


def test_temporary_objects_are_rejected(make_manager: Factory) -> None:
    with pytest.raises(ValueError, match="temporary"):
        make_manager("owner").reserve_write([key(1)], [True], LAYOUT)


def test_failed_publish_aborts_the_write(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    writer.reserve_write([key(1)], [False], LAYOUT)
    monkeypatch.setattr(
        Visibility, "publish", Mock(side_effect=RuntimeError("publish"))
    )
    with pytest.raises(RuntimeError, match="publish"):
        writer.finish_write([key(1)])
    assert writer.get_staging_memory_usage() == 0
    assert peer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS


def test_ambiguous_commit_is_reported_failed(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make_manager("writer")
    writer.reserve_write([key(1)], [False], LAYOUT)
    monkeypatch.setattr(
        MemoryOrchestratorClient,
        "finish_write",
        Mock(side_effect=OrchestratorUnavailableError("lost reply")),
    )
    assert writer.finish_write([key(1)]) == {key(1): L1Error.KEY_IN_WRONG_STATE}
    # The extent is never reused here: the grant is gone locally.
    assert writer.get_staging_memory_usage() == 0
    assert writer.finish_write([key(1)]) == {key(1): L1Error.KEY_NOT_EXIST}


def test_failed_acquire_preserves_existing_lease(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    original = reader.reserve_read([key(1)])[key(1)]
    monkeypatch.setattr(
        Visibility, "acquire", Mock(side_effect=RuntimeError("acquire"))
    )
    with pytest.raises(RuntimeError, match="acquire"):
        reader.reserve_read([key(1)], read_locks=2)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert usage().read_leases == 0


def test_malformed_write_grant_aborts_token(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    reserve = MemoryOrchestratorClient.reserve_write

    def misaligned(
        client: MemoryOrchestratorClient, entries: list[WriteRequest]
    ) -> list[WriteGrantResult]:
        grant = reserve(client, entries)[0]
        assert grant.handle is not None
        handle = replace(grant.handle, offset=grant.handle.offset + 1)
        return [replace(grant, handle=handle)]

    with monkeypatch.context() as patch:
        patch.setattr(MemoryOrchestratorClient, "reserve_write", misaligned)
        assert (
            writer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0]
            is L1Error.KEY_NOT_WRITABLE
        )
    assert peer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS


@pytest.mark.parametrize(
    "fault",
    ["negative", "misaligned", "outside", "short", "payload", "layout", "leases"],
)
def test_bad_read_grant_releases_only_new_lease(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    original = reader.reserve_read([key(1)])[key(1)]
    reserve = MemoryOrchestratorClient.reserve_read

    def malformed(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        grant = reserve(client, entries)[0]
        handle = grant.handle
        assert handle is not None
        if fault == "negative":
            grant = replace(grant, handle=replace(handle, offset=-4096))
        elif fault == "misaligned":
            grant = replace(grant, handle=replace(handle, offset=handle.offset + 1))
        elif fault == "outside":
            grant = replace(grant, handle=replace(handle, offset=CAPACITY))
        elif fault == "short":  # LAYOUT's payload is 4000 bytes
            grant = replace(grant, handle=replace(handle, length=2048))
        elif fault == "payload":
            grant = replace(grant, payload_bytes=CAPACITY + 1)
        elif fault == "leases":
            grant = replace(grant, leases=grant.leases * 2)
        else:
            grant = replace(grant, layout=WireLayout(((4, 500),), ("float16",)))
        return [grant]

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", malformed)
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert usage().read_leases == 0


def test_region_facts(make_manager: Factory, region: Path) -> None:
    manager = make_manager("owner")
    assert manager.get_memory_tags() == MemoryTags(
        MemorySegment.CXL, MemoryResidency.REMOTE
    )
    assert manager.get_l1_memory_desc() is None
    assert manager.memory_region_count() == 1
    assert manager.owns_range(str(region), 0, 1)
    assert not manager.owns_range(str(region), CAPACITY, 4096)
    assert manager.get_capacity_bytes_by_backend() == {L1BackendType.DEVDAX: CAPACITY}
    assert manager.get_memory_usage() == (0, CAPACITY)
    status = manager.report_status()
    assert status["is_healthy"] and status["shared"]["state"] == "available"
    assert manager.get_object_state(key(1)) is None


def test_hot_plug_is_rejected_before_mutating_state(
    make_manager: Factory, region: Path
) -> None:
    manager = make_manager("owner")
    for call in (
        manager.get_devdax_arena_statuses,
        lambda: manager.add_devdax_device(str(region), 4096),
        lambda: manager.remove_devdax_device(str(region)),
    ):
        with pytest.raises(L1ReconfigureError) as raised:
            call()
        assert raised.value.status_code == 409
    store(manager)


def test_private_devdax_l1_never_contacts_an_orchestrator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MemoryOrchestratorClient,
        "__init__",
        Mock(side_effect=AssertionError("private L1 contacted an orchestrator")),
    )
    config = shared_config(region_file(tmp_path / "dax", CAPACITY), "unused", "a")
    private = L1Manager(replace(config, shared=None))
    try:
        store(private)
        assert private.get_memory_tags().residency is MemoryResidency.LOCAL
        assert private.delete([key(1)]) == {key(1): L1Error.SUCCESS}
    finally:
        private.close()


@pytest.mark.parametrize("visibility_mode", ["software_fenced"])
def test_software_fenced_region_flushes_exact_extents(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    flushed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        visibility, "cache_flush_range", lambda ptr, size: flushed.append((ptr, size))
    )
    monkeypatch.setattr(visibility, "cache_flush_supported", lambda: True)
    writer, reader = make_manager("writer"), make_manager("reader")
    _, obj = writer.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert obj is not None
    writer.finish_write([key(1)])
    # Written back after the copies, before the commit.
    assert flushed == [(obj.data_ptr, obj.get_size())]
    found = reader.reserve_read([key(1)])[key(1)][1]
    assert found is not None
    # Invalidated before the reader's copies.
    assert flushed[1:] == [(found.data_ptr, found.get_size())]
    reader.finish_read([key(1)])


def test_stalled_orchestrator_call_does_not_hold_the_manager_lock(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager("owner")
    store(manager)
    entered, release = threading.Event(), threading.Event()
    reserve_read = MemoryOrchestratorClient.reserve_read

    def stalled(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        entered.set()
        assert release.wait(timeout=10)
        return reserve_read(client, entries)

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", stalled)
    with ThreadPoolExecutor(1) as worker:
        lookup = worker.submit(manager.reserve_read, [key(1)])
        assert entered.wait(timeout=10)
        # Operations that take L1Manager._lock complete during the RPC.
        manager.register_listener(Mock())
        assert manager.report_status()["shared"]["state"] == "available"
        assert manager.get_staging_memory_usage() == 0
        release.set()
        assert lookup.result(timeout=10)[key(1)][0] is L1Error.SUCCESS
    manager.finish_read([key(1)])
