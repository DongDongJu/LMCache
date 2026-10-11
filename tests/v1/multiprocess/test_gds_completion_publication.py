# SPDX-License-Identifier: Apache-2.0
"""Deferred GDS failures must reach the real MP publication/response boundary."""

# Standard
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import gds_context, gpu_ops
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_context import GDSContext
from lmcache.v1.gpu_connector.gds_workers import GDSWorkerPool
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObjMetadata
from lmcache.v1.multiprocess import object_group_transfer
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as module_impl
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
)
from lmcache.v1.platform.base.cache_context import BaseCacheContext


@pytest.mark.parametrize("method", ["store", "retrieve"])
@pytest.mark.parametrize(
    "completed_bytes",
    [-5, 8, None],
    ids=["deferred-error", "short-io", "partial-submit"],
)
def test_deferred_io_failure_prevents_publication_or_retrieve_hit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    method: str,
    completed_bytes: int | None,
) -> None:
    """Exercise actual GDS submission/result checking through the MP handlers.

    Only the storage/device boundaries are mocked. The object-group transfer
    function and GDSContext validation execute normally, including submission
    retention until the fake issuing stream completes.
    """
    trace: list[str] = []
    pending: list[Submission] = []
    stream = SimpleNamespace(raw_handle=17)
    size = 16 if completed_bytes is not None else (16 << 20) + 16
    buffer = torch.empty(size, dtype=torch.uint8)
    backend = MagicMock(spec=GDSBackend)
    backend.name = "test"
    handle = MagicMock(spec=GDSHandle)
    handle.path = "mock-slab"
    backend.open_slab.return_value = handle

    def submit(
        base: int, size: int, offset: int, buffer_offset: int, raw_stream: int
    ) -> Submission:
        del base, raw_stream
        if completed_bytes is None and pending:
            trace.append("submission-error")
            raise OSError("submission failed after enqueue")
        trace.append("submitted")
        result = Submission(size, offset, buffer_offset)
        pending.append(result)
        return result

    def synchronize(device: object, issuing_stream: object) -> None:
        del device, issuing_stream
        trace.append("io-drained")
        for submission in pending:
            submission.result.value = -5 if completed_bytes is None else completed_bytes

    handle.read_async.side_effect = submit
    handle.write_async.side_effect = submit
    monkeypatch.setattr(gds_context.platform_stream, "current_stream", lambda _: stream)
    monkeypatch.setattr(gds_context.platform_stream, "stream_handle", lambda *_: 17)
    monkeypatch.setattr(gds_context.platform_stream, "synchronize_stream", synchronize)
    monkeypatch.setattr(
        gds_context.platform_stream, "synchronize_device", lambda _: None
    )
    context = GDSContext(backend=backend)
    context.initialize(GdsL1Config("mock-slab", (size + 4095) // 4096 * 4096))
    context.register_gpu_buffer(buffer)
    monkeypatch.setattr(gpu_ops, "get_l1_gds_context", lambda _: context)
    monkeypatch.setattr(object_group_transfer, "get_l1_gds_context", lambda _: context)

    obj = GDSMemoryObject(
        MemoryObjMetadata(
            torch.Size([size]), torch.uint8, 0, (size + 4095) // 4096 * 4096, 0
        )
    )
    obj.set_l1_manager(9)
    cache = MagicMock()
    cache.device = torch.device("cpu")
    cache.stream = stream
    cache.lmcache_tokens_per_chunk = 256
    cache.max_batch_size = 4
    cache.calculate_num_blocks.side_effect = lambda tokens, group: int(tokens > 0)
    cache.get_slots_per_chunk_in_sw.return_value = 256
    cache.get_temp_object_group_buffer.return_value = buffer
    cache.get_temp_kernel_group_buffer.return_value = buffer
    cache.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=1,
        num_kernel_groups=1,
        object_groups=[SimpleNamespace(kernel_group_indices=[0])],
        get_subchunk_sw_size_tokens=lambda _: 256,
        get_attn_desc=lambda: SimpleNamespace(
            is_full_attention=lambda _: True,
            num_chunks_in_sw=[-1],
            group_kinds=("attention",),
        ),
    )
    events = MagicMock()
    events.record_event.side_effect = lambda *_: trace.append("ipc-recorded")

    def export_event(*_: object) -> bytes:
        trace.append("ipc-exported")
        return b"done"

    events.export_event.side_effect = export_event
    server_context = MagicMock()
    server_context.chunk_size = 256
    server_context.null_block_id = 0
    server_context.resolve_obj_keys.return_value = [["object-key"]]
    server_context.get_read_owners.return_value = {"object-key": 9}
    server_context.event_bus.has_subscribers.return_value = False
    storage = server_context.storage_manager
    storage.reserve_write.return_value = {"object-key": obj}
    storage.unsafe_read.return_value = (["object-key"], [obj])
    storage.prepare_read_completion.return_value = [(9, ["object-key"])]
    monkeypatch.setattr(module_impl, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(module_impl, "torch_dev", MagicMock())
    monkeypatch.setattr(module_impl, "get_layout_desc", MagicMock())
    monkeypatch.setattr(
        module_impl, "downsample_and_stage_block_ids", lambda _, ids: ids
    )
    callback = MagicMock(side_effect=lambda _, kind, payload: trace.append(kind))
    monkeypatch.setattr(module_impl, "submit_callback_to_stream", callback)
    kernel = MagicMock()
    monkeypatch.setattr(
        object_group_transfer.device_ops, "multi_layer_block_kv_transfer", kernel
    )
    module = LMCacheDrivenTransferModule(server_context)
    entry = ContextEntry(
        cast(BaseCacheContext, cache), "model", 1, event_backend=events
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda _: entry)
    key = IPCCacheServerKey(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=tuple(range(256)),
        start=0,
        end=256,
        request_id="request",
    )
    try:
        if method == "store":
            event, success, mask = module.store_with_chunk_mask(
                key, 1, [[1]], b"producer"
            )
            assert mask == [False]
            assert "finish_write_by_owner" not in [
                c.args[1] for c in callback.call_args_list
            ]
            storage.prepare_write_completion.assert_not_called()
            storage.finish_write_by_owner.assert_not_called()
            event_name = "MP_STORE_END"
            count_name = "stored_count"
        else:
            event, success = module.retrieve(key, 1, [[1]], b"producer")
            kernel.assert_not_called()
            assert "finish_read_by_owner" in [
                c.args[1] for c in callback.call_args_list
            ]
            event_name = "MP_RETRIEVE_END"
            count_name = "retrieved_count"
        assert event == b"done"
        assert not success
        assert trace.index("submitted") < trace.index("io-drained")
        assert trace.index("io-drained") < trace.index("ipc-recorded")
        assert trace.index("ipc-recorded") < trace.index("ipc-exported")
        if method == "retrieve":
            assert trace.index("io-drained") < trace.index("finish_read_by_owner")
        if completed_bytes is None:
            assert trace.index("submission-error") < trace.index("io-drained")
            failure = next(
                record
                for record in caplog.records
                if record.name == module_impl.__name__ and record.exc_info is not None
            )
            assert failure.exc_info is not None
            assert isinstance(failure.exc_info[1], OSError)
            assert str(failure.exc_info[1]) == "submission failed after enqueue"
        end = next(
            c.args[1]
            for c in server_context.event_bus.publish_on_stream.call_args_list
            if c.args[1].event_type.name == event_name
        )
        assert end.metadata[count_name] == 0
        assert end.metadata["num_tokens"] == 0
    finally:
        context.close()


@pytest.fixture
def worker_transfer(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Use the real scheduler/pool, mocking only device waits and native I/O."""
    trace: list[str] = []
    cache = MagicMock(spec=BaseCacheContext)
    cache.device, cache.stream = torch.device("cuda", 0), object()
    cache.lmcache_tokens_per_chunk, cache.max_batch_size = 1, 2
    cache.calculate_num_blocks.side_effect = lambda tokens, group: tokens
    cache.get_slots_per_chunk_in_sw.return_value = 1
    cache.kv_layer_groups_manager = SimpleNamespace(
        object_groups=[SimpleNamespace(kernel_group_indices=[0])],
        get_subchunk_sw_size_tokens=lambda _: 1,
        get_attn_desc=lambda: SimpleNamespace(is_full_attention=lambda _: True),
    )
    buffers = [torch.zeros(4, dtype=torch.uint8) for _ in range(2)]
    streams = [object(), object()]
    cache.get_temp_object_group_buffer.side_effect = lambda slot, group: buffers[slot]
    cache.get_temp_kernel_group_buffer.side_effect = lambda slot, group: buffers[slot]
    cache.get_gds_transfer_stream.side_effect = streams.__getitem__
    cache.record_gds_transfer_ready.return_value.synchronize.side_effect = lambda: (
        trace.append("ready")
    )
    runtime, kernel, legacy = MagicMock(), MagicMock(), MagicMock()
    kernel.side_effect = lambda *_: trace.append("kernel")
    monkeypatch.setattr(object_group_transfer, "torch_dev", runtime)
    monkeypatch.setattr(
        object_group_transfer.device_ops, "multi_layer_block_kv_transfer", kernel
    )
    monkeypatch.setattr(
        object_group_transfer.platform_stream,
        "synchronize_stream",
        lambda *_: trace.append("sync"),
    )
    monkeypatch.setattr(object_group_transfer, "lmcache_memcpy_async_h2d", legacy)
    pool = GDSWorkerPool(2, 8)
    owners = {key: MagicMock(io_workers=2, max_inflight_bytes=8) for key in (1, 2)}
    for owner in owners.values():
        owner.submit_io.side_effect = pool.submit
        owner.transfer_sync.side_effect = lambda *_: trace.append("io")
    monkeypatch.setattr(object_group_transfer, "get_l1_gds_context", owners.__getitem__)

    def run(
        objects: Sequence[GDSMemoryObject],
        direction: object_group_transfer.lmcache_native.TransferDirection,
        ids: list[int] | None = None,
    ) -> None:
        host_ids = list(range(len(objects))) if ids is None else ids
        object_group_transfer.transfer_kv_per_object_group(
            cache,
            [torch.tensor(host_ids)],
            objects,
            0,
            1,
            0,
            direction,
            transfer_key="ownership-test",
            block_ids_host=[host_ids],
        )

    objects = [
        GDSMemoryObject(
            MemoryObjMetadata(torch.Size([4]), torch.uint8, index * 4096, 4096, 0)
        )
        for index in range(4)
    ]
    for index, obj in enumerate(objects):
        obj.set_l1_manager(1 + index % 2)
    try:
        yield SimpleNamespace(
            run=run,
            objects=objects,
            owners=owners,
            streams=streams,
            trace=trace,
            kernel=kernel,
            legacy=legacy,
            pool=pool,
        )
    finally:
        pool.close()


def test_worker_write_completes_producer_and_gather_before_io(
    worker_transfer: SimpleNamespace,
) -> None:
    """An actual worker cannot write before its producer and gather waits finish."""
    case = worker_transfer
    case.run(
        case.objects[:1], object_group_transfer.lmcache_native.TransferDirection.D2H
    )
    assert case.trace == ["ready", "kernel", "sync", "io", "sync"]


def test_worker_read_retains_slots_until_scatter_finishes(
    worker_transfer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocked scatter completion prevents admission/reuse of both occupied slots."""
    case = worker_transfer
    entered = [threading.Event(), threading.Event()]
    release = threading.Event()

    def synchronize(device: object, stream: object) -> None:
        entered[case.streams.index(stream)].set()
        assert release.wait(5)

    monkeypatch.setattr(
        object_group_transfer.platform_stream, "synchronize_stream", synchronize
    )
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            case.run,
            case.objects,
            object_group_transfer.lmcache_native.TransferDirection.H2D,
        )
        try:
            assert all(event.wait(5) for event in entered)
            assert (
                sum(owner.transfer_sync.call_count for owner in case.owners.values())
                == 2
            )
            assert case.pool.snapshot()["inflight_operations"] == 2
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=5)
    assert case.pool.snapshot()["inflight_operations"] == 0


def test_worker_failure_drains_another_owner(
    worker_transfer: SimpleNamespace,
) -> None:
    """One owner's error cannot return while another accepted operation is live."""
    case = worker_transfer
    started, failed, release = threading.Event(), threading.Event(), threading.Event()

    def fail(*_: object) -> None:
        assert started.wait(5)
        failed.set()
        raise OSError("short read")

    def hold(*_: object) -> None:
        started.set()
        assert release.wait(5)

    case.owners[1].transfer_sync.side_effect = fail
    case.owners[2].transfer_sync.side_effect = hold
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            case.run,
            case.objects[:2],
            object_group_transfer.lmcache_native.TransferDirection.H2D,
        )
        try:
            assert failed.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
        finally:
            release.set()
        with pytest.raises(OSError, match="short read"):
            future.result(timeout=5)
    assert case.pool.snapshot()["inflight_operations"] == 0
    assert case.kernel.call_count == 1


def test_worker_aliases_keep_ordered_fallback(worker_transfer: SimpleNamespace) -> None:
    """Repeated destination blocks must retain the serial read path's ordering."""
    case = worker_transfer
    case.run(
        case.objects[:2],
        object_group_transfer.lmcache_native.TransferDirection.H2D,
        [1, 1],
    )
    assert [call.args[0] for call in case.legacy.call_args_list] == case.objects[:2]
    for owner in case.owners.values():
        owner.submit_io.assert_not_called()
