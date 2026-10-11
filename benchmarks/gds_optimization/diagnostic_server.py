# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only public-call spans and checked cuFile completion observations.

Never enabled during steady performance trials. GPU completion is polled from
an ordinary observer thread; this wrapper adds events but no stream barriers.
"""

# Standard
from collections.abc import Callable
from concurrent.futures import Future
from typing import cast
import atexit
import functools
import json
import os
import runpy
import threading
import time

# Third Party
import torch

# First Party
from lmcache import device_ops
from lmcache.logging import init_logger
from lmcache.v1.distributed.config import GdsL1Config
from lmcache.v1.gpu_connector.gds_backends.base import Submission
from lmcache.v1.gpu_connector.gds_backends.cufile import AsyncHandle
from lmcache.v1.gpu_connector.gds_context import GDSContext, SlabDirection
from lmcache.v1.memory_management import GDSMemoryObject
from lmcache.v1.multiprocess import object_group_transfer
from lmcache.v1.platform.base.event_ipc import DefaultEventIPCBackend, EventIPCBackend
from lmcache.v1.platform.devices.cuda.cache_context import GPUCacheContext
from lmcache.v1.platform.devices.cuda.timeline_semaphore_event_ipc import (
    TimelineSemaphoreEventIPCBackend,
)

logger = init_logger(__name__)

output = open(os.environ["GDS_DIAGNOSTIC_EVENTS"], "x", buffering=1)
lock = threading.Lock()
pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event, dict[str, object]]] = []
stop = threading.Event()
inflight = {"operations": 0, "bytes": 0}


def install_wrapper(owner: object, name: str, wrapper: Callable[..., object]) -> None:
    """Replace an observed public method only inside this diagnostic process.

    Args:
        owner: Imported class or module holding the measured method.
        name: Public method name being wrapped.
        wrapper: Forwarding instrumentation retaining the original semantics.
    """
    setattr(owner, name, wrapper)


def emit(kind: str, **values: object) -> None:
    """Append a diagnostic record with its emitting process/thread and host time.

    Args:
        kind: Stable event category.
        values: JSON-serializable fields describing the observed operation.
    """
    with lock:
        output.write(
            json.dumps(
                {
                    "kind": kind,
                    "pid": os.getpid(),
                    "thread": threading.get_ident(),
                    "observed_ns": time.monotonic_ns(),
                    **values,
                }
            )
            + "\n"
        )


def enqueue_completion(
    kind: str, start: torch.cuda.Event, end: torch.cuda.Event, **values: object
) -> None:
    """Retain issuing-stream events and arguments for ordinary-thread polling.

    Args:
        kind: Completion category.
        start: Already-recorded timing event preceding the operation.
        end: Already-recorded event following the operation on the same stream.
        values: Event fields, optionally retaining a deferred Submission.
    """
    with lock:
        pending.append((kind, start, end, values))


def observe() -> None:
    """Poll recorded CUDA events without inserting barriers into the issuing stream.

    Deferred byte counts are inspected only after their end event completes.
    Errors become diagnostic records and never manufacture successful I/O.
    """
    torch.cuda.set_device(0)
    while not stop.is_set():
        with lock:
            work = list(pending)
        for item in work:
            kind, start, end, values = item
            try:
                if not end.query():
                    continue
                values = dict(values)
                submission = cast(Submission | None, values.pop("submission", None))
                if submission is not None:
                    values["completed_bytes"] = submission.bytes_done
                    values["exact_completion"] = (
                        submission.bytes_done == values["requested_bytes"]
                    )
                values["cuda_interval_us"] = start.elapsed_time(end) * 1000
                emit(kind, **values)
            except Exception as error:
                logger.exception("Diagnostic completion observation failed")
                emit("observer_error", event_kind=kind, error=repr(error))
            with lock:
                pending.remove(item)
        stop.wait(0.001)


def wrap_handle(name: str) -> None:
    """Observe an available public cuFile method without changing its result.

    Args:
        name: Synchronous or stream-ordered read/write method name.
    """
    original = getattr(AsyncHandle, name, None)
    if original is None:
        return

    @functools.wraps(original)
    def wrapped(
        self: AsyncHandle,
        buf_base: int,
        size: int,
        file_offset: int,
        buf_offset: int,
        *rest: int,
    ) -> Submission | int:
        """Retain async result storage through an observed issuing-stream event."""
        started = time.monotonic_ns()
        if name.endswith("async"):
            stream = torch.cuda.ExternalStream(rest[0])
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record(stream)
        with torch.cuda.nvtx.range("diagnostic_" + name):
            result = original(self, buf_base, size, file_offset, buf_offset, *rest)
        values: dict[str, object] = {
            "start_ns": started,
            "submit_return_ns": time.monotonic_ns(),
            "requested_bytes": size,
            "file_offset": file_offset,
            "buffer_offset": buf_offset,
            "direction": name.split("_")[0],
        }
        if name.endswith("async"):
            end.record(stream)
            enqueue_completion(
                "cuFile_" + name, begin, end, submission=result, **values
            )
        else:
            emit(
                "cuFile_" + name,
                completed_bytes=result,
                exact_completion=result == size,
                **values,
            )
        return result

    install_wrapper(AsyncHandle, name, wrapped)


for method in ("read_async", "write_async", "read_sync", "write_sync"):
    wrap_handle(method)


def wrap_transfer(name: str) -> None:
    """Time a public logical-object transfer call, excluding initialization.

    Args:
        name: GDSContext transfer method to observe when present.
    """
    original = getattr(GDSContext, name, None)
    if original is None:
        return

    @functools.wraps(original)
    def wrapped(
        self: GDSContext,
        memory_obj: GDSMemoryObject,
        gpu_buffer: torch.Tensor,
        direction: SlabDirection,
    ) -> None:
        """Observe the existing wrapper boundary and preserve completion semantics."""
        started = time.monotonic_ns()
        with torch.cuda.nvtx.range("diagnostic_" + name):
            original(self, memory_obj, gpu_buffer, direction)
        emit(
            name,
            start_ns=started,
            end_ns=time.monotonic_ns(),
            direction=direction.name,
            logical_bytes=memory_obj.get_size(),
            checked_completion=name == "transfer_sync",
        )

    install_wrapper(GDSContext, name, wrapped)


wrap_transfer("transfer_sync")
wrap_transfer("transfer_async")

initialize_context = GDSContext.initialize


def observed_initialize(self: GDSContext, config: GdsL1Config) -> None:
    """Record context initialization separately from steady transfer timings.

    Args:
        self: Context receiving the original call.
        config: Original slab and worker configuration.
    """
    started = time.monotonic_ns()
    with torch.cuda.nvtx.range("diagnostic_context_initialize"):
        initialize_context(self, config)
    emit(
        "context_initialize",
        start_ns=started,
        end_ns=time.monotonic_ns(),
        io_workers=getattr(config, "io_workers", 0),
        max_inflight_bytes=getattr(config, "max_inflight_bytes", None),
    )


install_wrapper(GDSContext, "initialize", observed_initialize)

register_buffer = GDSContext.register_gpu_buffer


def observed_registration(self: GDSContext, buffer: torch.Tensor) -> None:
    """Record each registration call without allocating another staging buffer.

    Args:
        self: Context receiving the original call.
        buffer: Original registered staging tensor.
    """
    started = time.monotonic_ns()
    register_buffer(self, buffer)
    emit(
        "registered_staging",
        start_ns=started,
        end_ns=time.monotonic_ns(),
        base=int(buffer.data_ptr()),
        nbytes=buffer.numel() * buffer.element_size(),
        device=str(buffer.device),
    )


install_wrapper(GDSContext, "register_gpu_buffer", observed_registration)

kernel = cast(Callable[..., object], device_ops.multi_layer_block_kv_transfer)


def timed_kernel(*args: object, **kwargs: object) -> object:
    """Observe a gather/scatter call and later its CUDA event interval.

    Args:
        args: Unmodified positional kernel arguments.
        kwargs: Unmodified keyword kernel arguments.

    Returns:
        The original kernel call result, without synchronizing its stream.
    """
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    started = time.monotonic_ns()
    begin.record()
    with torch.cuda.nvtx.range("diagnostic_gather_scatter"):
        result = kernel(*args, **kwargs)
    end.record()
    enqueue_completion(
        "gather_scatter",
        begin,
        end,
        start_ns=started,
        submit_return_ns=time.monotonic_ns(),
        direction=str(args[4]),
    )
    return result


install_wrapper(device_ops, "multi_layer_block_kv_transfer", timed_kernel)
submit = getattr(GDSContext, "submit_io", None)
if submit is not None:
    submit_io = cast(
        Callable[[GDSContext, Callable[[], None], int], Future[None]], submit
    )

    def timed_submit(
        self: GDSContext, operation: Callable[[], None], nbytes: int
    ) -> Future[None]:
        """Observe admission and completion while retaining the existing future."""
        admitted = time.monotonic_ns()
        token = f"{threading.get_ident()}-{admitted}"

        def work() -> None:
            """Measure slot lifetime including final GPU stream completion."""
            started = time.monotonic_ns()
            try:
                return operation()
            finally:
                emit(
                    "slot_work",
                    token=token,
                    start_ns=started,
                    end_ns=time.monotonic_ns(),
                    admitted_ns=admitted,
                    logical_bytes=nbytes,
                )

        future = submit_io(self, work, nbytes)
        returned = time.monotonic_ns()
        with lock:
            inflight["operations"] += 1
            inflight["bytes"] += nbytes
            state = dict(inflight)
        emit(
            "slot_admitted",
            token=token,
            start_ns=admitted,
            end_ns=returned,
            logical_bytes=nbytes,
            inflight=state,
        )

        def finished(done: Future[None]) -> None:
            """Record callback observation and authoritative pool high water."""
            with lock:
                inflight["operations"] -= 1
                inflight["bytes"] -= nbytes
                state = dict(inflight)
            emit(
                "slot_retired",
                token=token,
                logical_bytes=nbytes,
                cancelled=done.cancelled(),
                inflight=state,
                worker_stats=self.worker_stats(),
            )

        future.add_done_callback(finished)
        return future

    install_wrapper(GDSContext, "submit_io", timed_submit)

wait_slots = getattr(object_group_transfer, "wait", None)
if wait_slots is not None:
    wait_for_slots = cast(Callable[..., object], wait_slots)

    def observed_wait(*args: object, **kwargs: object) -> object:
        """Time the existing bounded future wait without changing its result."""
        started = time.monotonic_ns()
        with torch.cuda.nvtx.range("diagnostic_slot_wait"):
            result = wait_for_slots(*args, **kwargs)
        emit("slot_wait", start_ns=started, end_ns=time.monotonic_ns())
        return result

    install_wrapper(object_group_transfer, "wait", observed_wait)


generations: dict[int, int] = {}


def observe_ipc(backend: type[EventIPCBackend]) -> None:
    """Record engine-event record/export order and event-generation identity.

    Args:
        backend: Existing public event IPC implementation to observe.
    """
    record = backend.record_event
    export = backend.export_event

    def recorded(self: EventIPCBackend, event: object, stream: object) -> None:
        """Observe after the intended event generation has been recorded."""
        started = time.monotonic_ns()
        record(self, event, stream)
        ended = time.monotonic_ns()
        key = id(event)
        with lock:
            generation = generations.get(key, 0) + 1
            generations[key] = generation
        emit(
            "engine_ready_record",
            start_ns=started,
            end_ns=ended,
            event_identity=key,
            generation=generation,
            backend=backend.__name__,
        )

    def exported(self: EventIPCBackend, event: object, device: object) -> bytes:
        """Return the original IPC handle and record its latest event generation."""
        result = export(self, event, device)
        with lock:
            generation = generations.get(id(event))
        emit(
            "engine_ready_export",
            event_identity=id(event),
            generation=generation,
            backend=backend.__name__,
        )
        return result

    install_wrapper(backend, "record_event", recorded)
    install_wrapper(backend, "export_event", exported)


observe_ipc(DefaultEventIPCBackend)
observe_ipc(TimelineSemaphoreEventIPCBackend)


ready_input = getattr(GPUCacheContext, "record_gds_transfer_ready", None)
if ready_input is not None:
    record_input = cast(Callable[[GPUCacheContext], torch.cuda.Event], ready_input)

    def observed_input_ready(self: GPUCacheContext) -> torch.cuda.Event:
        """Observe reuse of the context-owned event after its next record."""
        event = record_input(self)
        key = id(event)
        with lock:
            generation = generations.get(key, 0) + 1
            generations[key] = generation
        emit("gds_input_ready_record", event_identity=key, generation=generation)
        return event

    install_wrapper(GPUCacheContext, "record_gds_transfer_ready", observed_input_ready)


observer = threading.Thread(target=observe, name="gds-diagnostic-observer", daemon=True)
observer.start()
emit(
    "diagnostic_start",
    scope="instrumented server only; includes warmup; never pooled with steady rates",
)


def finish() -> None:
    """Bound observer shutdown and report any events not observed before exit."""
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with lock:
            unresolved = len(pending)
        if not unresolved:
            break
        time.sleep(0.005)
    stop.set()
    observer.join(timeout=3)
    with lock:
        unresolved = len(pending)
    emit("diagnostic_stop", unresolved_events=unresolved)


atexit.register(finish)
runpy.run_module("lmcache.cli.main", run_name="__main__")
