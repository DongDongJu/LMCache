# SPDX-License-Identifier: Apache-2.0
"""Deferred GDS failures must reach the real MP publication/response boundary."""

# Standard
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import gds_context, gpu_ops
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_context import GDSContext
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
