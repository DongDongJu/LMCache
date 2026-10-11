# SPDX-License-Identifier: Apache-2.0
"""Registered-range and exact-completion contracts for worker GDS I/O."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector import gds_context
from lmcache.v1.gpu_connector.gds_backends.base import GDSBackend, GDSHandle, Submission
from lmcache.v1.gpu_connector.gds_context import GDSContext, SlabDirection


@pytest.fixture
def registered_context(monkeypatch):
    """Register a real CPU allocation against a mocked native storage boundary."""
    backend = Mock(spec=GDSBackend)
    backend.name = "test"
    handle = Mock(spec=GDSHandle)
    handle.path = "/test/slab"
    handle.supports_sync_io = True
    handle.read_sync.side_effect = lambda base, size, file, offset: size
    handle.write_sync.side_effect = lambda base, size, file, offset: size
    backend.open_slab.return_value = handle
    platform = gds_context.platform_stream
    monkeypatch.setattr(platform, "current_stream", lambda device: object())
    monkeypatch.setattr(platform, "stream_handle", lambda device, stream: 1)
    monkeypatch.setattr(platform, "synchronize_device", lambda device: None)
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 64 << 20, io_workers=4))
    buffer = torch.empty(32 << 20, dtype=torch.uint8)
    context.register_gpu_buffer(buffer)
    yield context, handle, buffer
    context.close()


@pytest.mark.parametrize("direction", list(SlabDirection))
def test_registered_boundary_preserves_nonzero_offsets_and_tail(
    registered_context, direction
) -> None:
    context, handle, buffer = registered_context
    start = (16 << 20) - 4096
    obj = SimpleNamespace(get_size=lambda: 8193, slab_offset=12288)
    context.transfer_sync(obj, buffer[start:], direction)
    operation = (
        handle.read_sync if direction is SlabDirection.READ else handle.write_sync
    )
    assert [call.args for call in operation.call_args_list] == [
        (buffer.data_ptr(), 4096, 12288, start),
        (buffer.data_ptr() + (16 << 20), 4097, 16384, 0),
    ]


@pytest.mark.parametrize("direction", list(SlabDirection))
@pytest.mark.parametrize("completed", [-5030, -1, 0, 4095, 4097])
def test_short_negative_or_excess_completion_fails_closed(
    registered_context, direction, completed
) -> None:
    context, handle, buffer = registered_context
    operation = (
        handle.read_sync if direction is SlabDirection.READ else handle.write_sync
    )
    operation.side_effect = [completed]
    obj = SimpleNamespace(get_size=lambda: 8192, slab_offset=0)
    with pytest.raises(RuntimeError, match=f"completed {completed}"):
        context.transfer_sync(obj, buffer[(16 << 20) - 4096 :], direction)
    assert operation.call_count == 1


def test_immediate_error_does_not_issue_remaining_regions(registered_context) -> None:
    context, handle, buffer = registered_context
    handle.write_sync.side_effect = OSError("submission failed")
    obj = SimpleNamespace(get_size=lambda: 32 << 20, slab_offset=0)
    with pytest.raises(OSError, match="submission failed"):
        context.transfer_sync(obj, buffer, SlabDirection.WRITE)
    assert handle.write_sync.call_count == 1


@pytest.mark.parametrize(
    "offset,size", [(-1, 4096), (64 << 20, 1), (0, 0), (0, 33 << 20)]
)
def test_invalid_range_rejected_before_io(registered_context, offset, size) -> None:
    context, handle, buffer = registered_context
    obj = SimpleNamespace(get_size=lambda: size, slab_offset=offset)
    with pytest.raises(ValueError, match="exceeds"):
        context.transfer_sync(obj, buffer, SlabDirection.READ)
    handle.read_sync.assert_not_called()


def test_unavailable_capability_keeps_worker_path_disabled() -> None:
    backend = Mock(spec=GDSBackend)
    backend.name = "other"
    handle = Mock(spec=GDSHandle)
    handle.path = "/test/slab"
    handle.supports_sync_io = False
    backend.open_slab.return_value = handle
    context = GDSContext(backend)
    context.initialize(GdsL1Config("/test", 4096, io_workers=4))
    try:
        assert context.io_workers == 0
        with pytest.raises(RuntimeError, match="disabled"):
            context.submit_io(lambda: None, 4096)
    finally:
        context.close()


@pytest.mark.parametrize("completed", [4096, 2048, -1, -5001])
def test_async_results_inspected_only_after_stream_completion(
    registered_context, monkeypatch, completed
) -> None:
    context, handle, buffer = registered_context
    submission = Submission(4096, 8192, 0)
    handle.read_async.return_value = submission
    obj = SimpleNamespace(get_size=lambda: 4096, slab_offset=8192)
    context.transfer_async(obj, buffer, SlabDirection.READ)
    # Only the completion signal makes the deferred result available.
    monkeypatch.setattr(
        gds_context.platform_stream,
        "synchronize_stream",
        lambda device, stream: setattr(submission.result, "value", completed),
    )
    if completed == 4096:
        context.synchronize_and_check(buffer.device, object())
    else:
        with pytest.raises(RuntimeError, match=f"completed {completed}"):
            context.synchronize_and_check(buffer.device, object())
    # Failed completed submissions are retired once and cannot poison another request.
    context.synchronize_and_check(buffer.device, object())


def test_staging_allocation_budget_uses_total_bytes(
    registered_context, monkeypatch
) -> None:
    context, _, _ = registered_context
    monkeypatch.setattr(gds_context, "get_gds_context", lambda: context)
    gds_context.validate_gds_staging_size(128 << 20)
    with pytest.raises(ValueError, match="staging needs"):
        gds_context.validate_gds_staging_size((128 << 20) + 1)


def test_worker_start_failure_releases_opened_slab(monkeypatch) -> None:
    backend = Mock(spec=GDSBackend)
    backend.name = "test"
    handle = Mock(spec=GDSHandle)
    handle.supports_sync_io = True
    backend.open_slab.return_value = handle
    monkeypatch.setattr(
        gds_context, "GDSWorkerPool", Mock(side_effect=RuntimeError("thread start"))
    )
    context = GDSContext(backend)
    with pytest.raises(RuntimeError, match="thread start"):
        context.initialize(GdsL1Config("/test", 4096, io_workers=1))
    handle.close.assert_called_once()
    backend.close_driver.assert_called_once()
    assert not context.initialized
