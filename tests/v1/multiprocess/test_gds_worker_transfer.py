# SPDX-License-Identifier: Apache-2.0
"""Exercise public GDS transfers with queued CPU stand-ins for GPU operations."""

# Standard
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
import sys
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gds_context import SlabDirection
from lmcache.v1.gpu_connector.gds_workers import GDSWorkerPool
from lmcache.v1.memory_management import GDSMemoryObject, MemoryObj
from lmcache.v1.platform.base.cache_context import BaseCacheContext
import lmcache.v1.multiprocess.object_group_transfer as transfer


class QueuedStream:
    """Queue work until this particular stream is synchronized."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cuda_stream = 0
        self.pending: list[Callable[[], None]] = []
        self.lock = threading.Lock()

    def enqueue(self, operation: Callable[[], None]) -> None:
        """Append simulated GPU work without completing it."""
        with self.lock:
            self.pending.append(operation)

    def synchronize(self) -> None:
        """Complete queued work; concurrent event waiters see completed producers."""
        with self.lock:
            while self.pending:
                self.pending.pop(0)()


class Event:
    """A recorded producer point, never an unrecorded future promise."""

    def __init__(self) -> None:
        self.stream: QueuedStream | None = None
        self.generation = 0

    def record(self, stream: QueuedStream) -> None:
        """Record the producer stream that must finish before dependent I/O."""
        self.stream = stream
        self.generation += 1

    def synchronize(self) -> None:
        """Wait for the recorded producer work without synchronizing other streams."""
        assert self.stream is not None, "wait on unrecorded event"
        self.stream.synchronize()


class Runtime:
    """Maintain independently selected fake devices and streams on each worker."""

    Event = Event

    def __init__(self) -> None:
        self.local = threading.local()

    def Stream(self, device: torch.device) -> QueuedStream:
        """Create a context-owned CPU stream for the selected fake device."""
        return QueuedStream(str(device))

    @contextmanager
    def device(self, device: torch.device) -> Iterator[None]:
        """Select and restore this thread's simulated CUDA device."""
        previous = getattr(self.local, "device", None)
        self.local.device = device
        try:
            yield
        finally:
            self.local.device = previous

    @contextmanager
    def stream(self, stream: QueuedStream) -> Iterator[None]:
        """Select and restore this thread's simulated CUDA stream."""
        previous = getattr(self.local, "stream", None)
        self.local.stream = stream
        try:
            yield
        finally:
            self.local.stream = previous


class Owner:
    """Use real bounded CPU admission while replacing only the blocking I/O."""

    def __init__(self, harness: "Harness") -> None:
        self.harness = harness
        self.io_workers = 2
        self.max_inflight_bytes = 8
        self.pool = GDSWorkerPool(self.io_workers, self.max_inflight_bytes)
        self.before_io: Callable[[MagicMock, torch.Tensor], None] | None = None
        self.calls: list[int] = []
        self.validation_error: Exception | None = None

    def submit_io(self, operation: Callable[[], None], nbytes: int) -> Future[None]:
        """Submit work through the production bounded pool."""
        return self.pool.submit(operation, nbytes)

    def transfer_sync(
        self, memory: MagicMock, buffer: torch.Tensor, direction: SlabDirection
    ) -> None:
        """Complete a fake slab I/O, checking the worker device and gather order."""
        assert self.harness.runtime.local.device == self.harness.cache.device
        assert self.harness.scope_active
        self.calls.append(memory.index)
        if self.before_io is not None:
            self.before_io(memory, buffer)
        if direction is SlabDirection.READ:
            buffer.copy_(memory.payload)
        else:
            ids = self.harness.ids[memory.index * 4 : (memory.index + 1) * 4]
            expected = self.harness.engine[ids]
            assert torch.equal(buffer, expected), "write preceded completed gather"
            memory.payload.copy_(buffer)

    def synchronize_and_check(self, device: torch.device, stream: QueuedStream) -> None:
        """Model the existing-path completion/error boundary before consumption."""
        assert device == self.harness.cache.device
        stream.synchronize()
        if self.validation_error is not None:
            raise self.validation_error


class Harness:
    """Small nonzero payloads expose ordering, block mapping and slot reuse."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.runtime = Runtime()
        self.caller = QueuedStream("caller")
        self.streams = [QueuedStream("slot0"), QueuedStream("slot1")]
        self.buffers = [torch.full((4,), 231, dtype=torch.uint8) for _ in range(2)]
        self.engine = torch.full((24,), 197, dtype=torch.uint8)
        self.ids = torch.arange(23, -1, -1, dtype=torch.long)
        self.buffer_by_ptr = {buffer.data_ptr(): buffer for buffer in self.buffers}
        self.scope_active = False
        self.ready_event = Event()
        self.before_scatter: Callable[[list[int]], None] | None = None
        self.legacy_calls: list[int] = []
        self.owners: dict[int, Owner] = {}
        attention = SimpleNamespace(
            is_full_attention=lambda group: True, num_chunks_in_sw=[-1]
        )
        self.cache = SimpleNamespace(
            device=torch.device("cuda", 1),
            stream=self.caller,
            max_batch_size=2,
            lmcache_tokens_per_chunk=4,
            kv_layer_groups_manager=SimpleNamespace(
                object_groups=[SimpleNamespace(kernel_group_indices=[0])],
                get_subchunk_sw_size_tokens=lambda group: 4,
                get_attn_desc=lambda: attention,
            ),
            gds_transfer_scope=self.scope,
            record_gds_transfer_ready=self.record_ready,
            get_gds_transfer_stream=lambda slot: self.streams[slot],
            get_temp_object_group_buffer=lambda slot, group: self.buffers[slot],
            get_temp_kernel_group_buffer=lambda slot, group: self.buffers[slot],
            calculate_num_blocks=lambda tokens, group: tokens,
            get_kernel_group_kv_pointers=lambda group: [self.engine.data_ptr()],
            get_slots_per_chunk_in_sw=lambda group: 4,
            get_shape_desc=lambda group: (),
            get_engine_kv_format=lambda group: 0,
        )
        monkeypatch.setattr(transfer, "torch_dev", self.runtime)
        monkeypatch.setattr(transfer, "get_l1_gds_context", self.get_owner)
        monkeypatch.setattr(
            transfer.platform_stream,
            "synchronize_stream",
            lambda device, stream: stream.synchronize(),
        )
        monkeypatch.setattr(
            transfer,
            "device_ops",
            SimpleNamespace(multi_layer_block_kv_transfer=self.kernel),
        )
        monkeypatch.setattr(transfer, "lmcache_memcpy_async_h2d", self.legacy_read)
        monkeypatch.setattr(transfer, "lmcache_memcpy_async_d2h", self.legacy_write)

    def record_ready(self) -> Event:
        """Record one reusable event while the transfer owns the context lease."""
        assert self.scope_active
        self.ready_event.record(self.caller)
        return self.ready_event

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Keep the simulated context lease alive through every worker's drain."""
        assert not self.scope_active
        self.scope_active = True
        try:
            yield
        finally:
            assert all(not stream.pending for stream in self.streams)
            self.scope_active = False

    def get_owner(self, owner: int) -> Owner:
        """Resolve each object to its own slab context."""
        if owner not in self.owners:
            self.owners[owner] = Owner(self)
        return self.owners[owner]

    def memory(self, index: int, owner: int = 1, gds: bool = True) -> MagicMock:
        """Return a nonzero object with explicit original index and slab owner."""
        memory = MagicMock(spec=GDSMemoryObject if gds else MemoryObj)
        memory.index = index
        memory.get_l1_manager.return_value = owner
        memory.get_size.return_value = 4
        memory.payload = torch.arange(index * 4 + 1, index * 4 + 5, dtype=torch.uint8)
        return memory

    def kernel(
        self,
        pointers: list[int],
        temporaries: list[int],
        ids: torch.Tensor,
        device: torch.device,
        direction: object,
        shape: object,
        chunk_size: int,
        layout: object,
        skip: int,
    ) -> None:
        """Queue gather/scatter; reading staging bytes only at stream completion."""
        stream = self.runtime.local.stream
        assert device == self.cache.device

        def complete() -> None:
            for index, pointer in enumerate(temporaries):
                selected = ids[index * 4 : (index + 1) * 4]
                buffer = self.buffer_by_ptr[pointer]
                if direction == transfer.lmcache_native.TransferDirection.H2D:
                    if self.before_scatter is not None:
                        self.before_scatter(selected.tolist())
                    self.engine[selected[skip:]] = buffer[skip:]
                else:
                    buffer.copy_(self.engine[selected])

        stream.enqueue(complete)

    def legacy_read(self, memory: MagicMock, buffer: torch.Tensor) -> None:
        """Queue the original path's copy for disabled or mixed-object cases."""
        self.legacy_calls.append(memory.index)
        self.caller.enqueue(lambda: buffer.copy_(memory.payload))

    def legacy_write(self, buffer: torch.Tensor, memory: MagicMock) -> None:
        """Queue a store after the original path's gather."""
        self.legacy_calls.append(memory.index)
        self.caller.enqueue(lambda: memory.payload.copy_(buffer))

    def run(
        self,
        objects: Sequence[MagicMock | None],
        direction: transfer.lmcache_native.TransferDirection,
        skip: int = 0,
    ) -> None:
        """Invoke the public transfer entry point in its normal caller context."""
        with self.runtime.device(self.cache.device), self.runtime.stream(self.caller):
            transfer.transfer_kv_per_object_group(
                cast(BaseCacheContext, self.cache),
                [self.ids],
                list(objects),
                object_group_id=0,
                batch_size=1,
                skip_first_n_tokens=skip,
                direction=direction,
                transfer_key="test-owner-ordering",
                block_ids_host=[self.ids.tolist()],
            )

    def close(self) -> None:
        """Join every real CPU worker before the test drops its buffers."""
        for owner in self.owners.values():
            owner.pool.close()


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """Provide CPU-only stream and storage stand-ins with deterministic cleanup."""
    harness = Harness(monkeypatch)
    yield harness
    harness.close()


def test_read_slots_are_not_reused_before_scatter(harness: Harness) -> None:
    """Two blocked scatters retain their slots; six objects still arrive intact."""
    objects = [harness.memory(index) for index in range(6)]
    started = [threading.Event(), threading.Event()]
    release = threading.Event()

    def scatter(ids: list[int]) -> None:
        index = (23 - ids[0]) // 4
        if index < 2:
            started[index].set()
            assert release.wait(5)

    harness.before_scatter = scatter
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            harness.run, objects, transfer.lmcache_native.TransferDirection.H2D
        )
        try:
            assert all(event.wait(5) for event in started)
            assert set(harness.get_owner(1).calls) == {0, 1}
            assert harness.get_owner(1).pool.snapshot()["inflight_operations"] == 2
            assert not future.done()
        finally:
            release.set()
        future.result(timeout=5)
    for index, memory in enumerate(objects):
        assert torch.equal(
            harness.engine[harness.ids[index * 4 : (index + 1) * 4]], memory.payload
        )


def test_write_waits_for_producer_and_gather(harness: Harness) -> None:
    """A queued producer and queued gathers must finish before blocking writes."""
    expected = torch.arange(11, 35, dtype=torch.uint8)
    harness.caller.enqueue(lambda: harness.engine.copy_(expected))
    objects = [harness.memory(index) for index in range(3)]
    harness.run(objects, transfer.lmcache_native.TransferDirection.D2H)
    for index, memory in enumerate(objects):
        assert torch.equal(
            memory.payload, expected[harness.ids[index * 4 : (index + 1) * 4]]
        )


def test_failure_drains_other_owner_before_return(harness: Harness) -> None:
    """A failed owner cannot let the handler release another owner's live work."""
    objects = [harness.memory(0, 1), harness.memory(1, 2)]
    other_started, failed, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def fail(memory: MagicMock, buffer: torch.Tensor) -> None:
        assert other_started.wait(5)
        buffer[:2].copy_(memory.payload[:2])
        failed.set()
        raise RuntimeError("short read")

    def blocked(memory: MagicMock, buffer: torch.Tensor) -> None:
        other_started.set()
        assert release.wait(5)

    harness.get_owner(1).before_io = fail
    harness.get_owner(2).before_io = blocked
    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(
            harness.run, objects, transfer.lmcache_native.TransferDirection.H2D
        )
        try:
            assert failed.wait(5)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.05)
            assert harness.scope_active
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="short read"):
            future.result(timeout=5)
    assert torch.all(harness.engine[harness.ids[:4]] == 197)
    assert torch.equal(harness.engine[harness.ids[4:8]], objects[1].payload)
    assert not harness.scope_active


def test_read_preserves_skipped_prefix_and_object_mapping(harness: Harness) -> None:
    """Whole and partial prefix skips protect engine blocks without shifting objects."""
    objects = [harness.memory(index) for index in range(3)]
    harness.run(objects, transfer.lmcache_native.TransferDirection.H2D, skip=6)
    assert set(harness.get_owner(1).calls) == {1, 2}
    assert torch.all(harness.engine[harness.ids[:6]] == 197)
    assert torch.equal(harness.engine[harness.ids[6:8]], objects[1].payload[2:])
    assert torch.equal(harness.engine[harness.ids[8:12]], objects[2].payload)


@pytest.mark.parametrize(
    "skip_kind", ["logical_prefix", "sliding_window", "non_gds_prefix"]
)
def test_skipped_owner_is_not_resolved(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, skip_kind: str
) -> None:
    """An unused old L1 owner cannot reject a valid retained-suffix retrieve."""
    owner = harness.get_owner(1)
    monkeypatch.setattr(transfer, "get_l1_gds_context", lambda key: {1: owner}[key])
    objects = [
        harness.memory(0, owner=999, gds=skip_kind != "non_gds_prefix"),
        harness.memory(1),
    ]
    skip = 4
    if skip_kind == "sliding_window":
        skip = 0
        harness.cache.kv_layer_groups_manager.get_attn_desc = lambda: SimpleNamespace(
            is_full_attention=lambda group: False, num_chunks_in_sw=[1]
        )
    harness.run(objects, transfer.lmcache_native.TransferDirection.H2D, skip=skip)
    assert owner.calls == [1]
    assert torch.all(harness.engine[harness.ids[:4]] == 197)
    assert torch.equal(harness.engine[harness.ids[4:8]], objects[1].payload)


def test_two_device_contexts_share_pool_without_crossing_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shared bounded pool selects each job's device and independent staging."""
    contexts = [Harness(monkeypatch), Harness(monkeypatch)]
    runtime = Runtime()
    pool = GDSWorkerPool(2, 8)
    rendezvous = threading.Barrier(2)
    seen_devices: set[int] = set()
    for index, context in enumerate(contexts):
        context.runtime = runtime
        context.cache.device = torch.device("cuda", index)
        context.cache.max_batch_size = 1

    def io(memory: MagicMock, buffer: torch.Tensor, direction: SlabDirection) -> None:
        assert runtime.local.device.index == memory.target_device
        assert direction is SlabDirection.READ
        if memory.index == 0:
            rendezvous.wait(timeout=5)
        seen_devices.add(runtime.local.device.index)
        buffer.copy_(memory.payload)

    def kernel(
        pointers: list[int],
        temporaries: list[int],
        ids: torch.Tensor,
        device: torch.device,
        direction: object,
        shape: object,
        chunk_size: int,
        layout: object,
        skip: int,
    ) -> None:
        contexts[cast(int, device.index)].kernel(
            pointers,
            temporaries,
            ids,
            device,
            direction,
            shape,
            chunk_size,
            layout,
            skip,
        )

    owner = SimpleNamespace(
        io_workers=2,
        max_inflight_bytes=8,
        submit_io=pool.submit,
        transfer_sync=io,
    )
    monkeypatch.setattr(transfer, "torch_dev", runtime)
    monkeypatch.setattr(transfer, "get_l1_gds_context", lambda key: owner)
    monkeypatch.setattr(
        transfer, "device_ops", SimpleNamespace(multi_layer_block_kv_transfer=kernel)
    )
    batches = []
    for index, context in enumerate(contexts):
        objects = [context.memory(i) for i in range(2)]
        for memory in objects:
            memory.target_device = index
            memory.payload += index * 100
        batches.append(objects)
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            futures = [
                callers.submit(
                    context.run, objects, transfer.lmcache_native.TransferDirection.H2D
                )
                for context, objects in zip(contexts, batches, strict=True)
            ]
            for future in futures:
                future.result(timeout=5)
        assert seen_devices == {0, 1}
        assert pool.snapshot()["peak_inflight_operations"] == 2
        for context, objects in zip(contexts, batches, strict=True):
            for index, memory in enumerate(objects):
                assert torch.equal(
                    context.engine[context.ids[index * 4 : (index + 1) * 4]],
                    memory.payload,
                )
    finally:
        pool.close()
        for context in contexts:
            context.close()


def test_none_store_keeps_original_block_positions(harness: Harness) -> None:
    """A missing store object must not shift the next object's gather mapping."""
    harness.engine.copy_(torch.arange(11, 35, dtype=torch.uint8))
    objects = [harness.memory(0), None, harness.memory(2)]
    harness.run(objects, transfer.lmcache_native.TransferDirection.D2H)
    assert objects[2] is not None
    assert torch.equal(objects[2].payload, harness.engine[harness.ids[8:12]])


def test_none_read_fails_and_drains(harness: Harness) -> None:
    """Missing read data propagates failure with no outstanding worker work."""
    with pytest.raises(ValueError, match="Missing GDS object"):
        harness.run(
            [harness.memory(0), None], transfer.lmcache_native.TransferDirection.H2D
        )
    harness.close()
    assert harness.get_owner(1).pool.snapshot()["inflight_operations"] == 0


@pytest.mark.parametrize("mode", ["disabled", "mixed"])
def test_unsupported_worker_path_keeps_original_transfer(
    harness: Harness, mode: str
) -> None:
    """Disabled GDS or mixed non-GDS objects retain the existing data path."""
    objects = [harness.memory(0), harness.memory(1, gds=mode != "mixed")]
    owner = harness.get_owner(1)
    if mode == "disabled":
        owner.io_workers = 0
    harness.run(objects, transfer.lmcache_native.TransferDirection.H2D)
    harness.caller.synchronize()
    assert owner.calls == []
    assert harness.legacy_calls == [0, 1]
    for index, memory in enumerate(objects):
        assert torch.equal(
            harness.engine[harness.ids[index * 4 : (index + 1) * 4]], memory.payload
        )


def test_disabled_path_validates_read_before_scatter(harness: Harness) -> None:
    """A deferred error on the fallback path cannot scatter its partial payload."""
    owner = harness.get_owner(1)
    owner.io_workers = 0
    owner.validation_error = RuntimeError("deferred short read")
    with pytest.raises(RuntimeError, match="deferred short read"):
        harness.run([harness.memory(0)], transfer.lmcache_native.TransferDirection.H2D)
    assert torch.all(harness.engine == 197)


def test_repeated_retrieve_destinations_use_ordered_fallback(harness: Harness) -> None:
    """Chunks targeting the same GPU blocks preserve serial last-write behavior."""
    harness.ids[4:8].copy_(harness.ids[:4])
    objects = [harness.memory(0), harness.memory(1)]
    harness.run(objects, transfer.lmcache_native.TransferDirection.H2D)
    harness.caller.synchronize()
    assert harness.get_owner(1).calls == []
    assert harness.legacy_calls == [0, 1]
    assert torch.equal(harness.engine[harness.ids[:4]], objects[1].payload)


def test_real_cache_context_close_waits_for_transfer_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public context close cannot unmap KV or staging during a leased transfer."""
    # First Party
    import lmcache.v1.platform.devices.cuda.cache_context as cache_module

    runtime = Runtime()
    caller_stream = QueuedStream("real-context-caller")
    caller_stream.cuda_stream = 0
    monkeypatch.setattr(runtime, "Stream", lambda device: caller_stream)
    fake_cupy = ModuleType("cupy")
    monkeypatch.setattr(
        fake_cupy,
        "cuda",
        SimpleNamespace(
            ExternalStream=lambda handle, device: SimpleNamespace(
                launch_host_func=lambda callback, data: None
            )
        ),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(cache_module, "torch_dev", runtime)
    monkeypatch.setattr(cache_module.torch.cuda, "Event", Event)
    monkeypatch.setattr(cache_module, "register_gds_gpu_buffer", lambda buffer: None)
    deregistered = threading.Event()
    monkeypatch.setattr(
        cache_module, "deregister_gds_gpu_buffer", lambda buffer: deregistered.set()
    )
    wrapper = MagicMock()
    wrapper.to_tensor.return_value = torch.zeros((2, 4, 2, 1, 2), dtype=torch.float16)
    context = cache_module.GPUCacheContext(
        [wrapper], lmcache_tokens_per_chunk=4, layout_hints={"kv_layout": "NHD"}
    )
    with context.gds_transfer_scope():
        first = context.record_gds_transfer_ready()
        first.synchronize()
        assert first.generation == 1
    with context.gds_transfer_scope():
        second = context.record_gds_transfer_ready()
        assert second is first
        assert second.generation == 2
        second.synchronize()
    leased, release = threading.Event(), threading.Event()

    def hold_lease() -> None:
        with context.gds_transfer_scope():
            leased.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        worker = executor.submit(hold_lease)
        assert leased.wait(5)
        closer = executor.submit(context.close)
        try:
            with pytest.raises(TimeoutError):
                closer.result(timeout=0.05)
            assert not deregistered.is_set()
            wrapper.close.assert_not_called()
        finally:
            release.set()
        worker.result(timeout=5)
        closer.result(timeout=5)
    assert deregistered.is_set()
    wrapper.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        with context.gds_transfer_scope():
            pytest.fail("closed context admitted another transfer")
