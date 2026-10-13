# SPDX-License-Identifier: Apache-2.0
"""Opt-in fault controls for real MP operations in the KV-event E2E fixture.

Load with the existing --server-module extension. No event is fabricated and
only public operation entry points are wrapped. The module refuses to load
unless LMCACHE_KV_EVENTS_E2E=1. Its CLI uses the normal request transport:

    python mp_test_hooks.py --url tcp://127.0.0.1:5555 '{"action":"status"}'
"""

# Standard
from dataclasses import asdict
from functools import wraps
from pathlib import Path
import argparse
import json
import os
import threading
import time
import tracemalloc

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey, MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVEventBatch,
    KVEventProfile,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleBuildContext,
    ExtServerModuleComponents,
    server_module_handler,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory


class Control(msgspec.Struct):
    """JSON control schema; rank is the logical worker rank, keys are real L1 IDs."""

    action: str
    rank: int = -1
    keys: tuple[EncodedObjectKey, ...] = ()
    model_name: str = ""
    world_size: int = 2
    cursor: int = 0


class TestControls:
    """Control fault injection without changing inference or heartbeat clients."""

    def __init__(
        self,
        context: MPCacheServerContext,
        transfer: LMCacheDrivenTransferModule,
        events: KVEventModule,
        read_trace_path: Path | None = None,
    ) -> None:
        if os.environ.get("LMCACHE_KV_EVENTS_E2E") != "1":
            raise RuntimeError("MP test controls require LMCACHE_KV_EVENTS_E2E=1")
        self._ctx = context
        self._transfer = transfer
        self._events = events
        self._store = transfer.store_with_chunk_mask
        self._reserve = context.storage_manager.reserve_write
        self._retrieve = transfer.retrieve
        self._unsafe_read = context.storage_manager.unsafe_read
        self._read_context = threading.local()
        self._read_trace_path = read_trace_path
        if read_trace_path is not None:
            read_trace_path.touch(exist_ok=True)
        self._read_trace_lock = threading.Lock()
        self._subscribe = events.subscribe_kv_events
        self._lock = threading.Lock()
        self._gate = threading.Event()
        self._gate.set()
        self._delay_rank = -1
        self._fail_rank = -1
        self._held = 0
        self._failures = 0
        self._disconnected = False
        self._closed = False
        self._memory_baseline: tracemalloc.Snapshot | None = None

        @wraps(self._store)
        def store(
            key: IPCCacheServerKey,
            instance_id: int,
            gpu_block_ids: list[list[int]],
            event_ipc_handle: bytes,
        ) -> tuple[bytes, bool, list[bool]]:
            with self._lock:
                delayed = key.worker_id == self._delay_rank
                if delayed:
                    self._held += 1
            if delayed:
                try:
                    if not self._gate.wait(timeout=60):
                        raise RuntimeError("test rank delay exceeded 60 seconds")
                finally:
                    with self._lock:
                        self._held -= 1
            return self._store(key, instance_id, gpu_block_ids, event_ipc_handle)

        @wraps(self._reserve)
        def reserve(
            keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
        ) -> dict[ObjectKey, MemoryObj]:
            with self._lock:
                fail = any((key.kv_rank & 0xFF) == self._fail_rank for key in keys)
                if fail:
                    self._fail_rank = -1
                    self._failures += 1
            if fail:
                raise RuntimeError("test-injected KV rank reservation failure")
            return self._reserve(keys, layout_desc)

        @wraps(self._retrieve)
        def retrieve(
            key: IPCCacheServerKey,
            instance_id: int,
            gpu_block_ids: list[list[int]],
            event_ipc_handle: bytes,
            skip_first_n_tokens: int = 0,
        ) -> tuple[bytes, bool]:
            self._read_context.request = {
                "request_id": key.request_id,
                "instance_id": instance_id,
                "worker_id": key.worker_id,
                "model_name": key.model_name,
                "world_size": key.world_size,
                "token_ids": key.token_ids,
                "start": key.start,
                "end": key.end,
                "skip_first_n_tokens": skip_first_n_tokens,
            }
            try:
                return self._retrieve(
                    key,
                    instance_id,
                    gpu_block_ids,
                    event_ipc_handle,
                    skip_first_n_tokens,
                )
            finally:
                del self._read_context.request

        @wraps(self._unsafe_read)
        def unsafe_read(
            keys: list[ObjectKey],
            l1_owners: dict[ObjectKey, int] | None = None,
        ) -> tuple[list[ObjectKey], list[MemoryObj]]:
            found, objects = self._unsafe_read(keys, l1_owners=l1_owners)
            request = getattr(self._read_context, "request", None)
            if self._read_trace_path is not None and request is not None:
                record = {
                    "event": "l1_read",
                    "time_ns": time.time_ns(),
                    **request,
                    "requested_keys": [
                        asdict(key.to_encoded_object_key()) for key in keys
                    ],
                    "found_keys": [
                        asdict(key.to_encoded_object_key()) for key in found
                    ],
                    "owner_ids": [obj.get_l1_manager() for obj in objects],
                    "bytes": sum(obj.get_size() for obj in objects),
                }
                with self._read_trace_lock:
                    with self._read_trace_path.open("a") as stream:
                        stream.write(json.dumps(record) + "\n")
            return found, objects

        @wraps(self._subscribe)
        def subscribe(
            instance_id: int,
            model_name: str,
            cursor: int,
            max_events: int,
            world_size: int,
            profile: KVEventProfile,
        ) -> MessagingStream[KVEventBatch]:
            with self._lock:
                disconnected = self._disconnected
            if disconnected:

                def reject(closed: threading.Event) -> KVEventBatch:
                    raise StopIteration

                return MessagingStream(reject, lambda: None)
            return self._subscribe(
                instance_id, model_name, cursor, max_events, world_size, profile
            )

        # Existing module discovery reads the instance's public methods. Their
        # wraps annotations preserve the negotiated RPC schema.
        transfer.store_with_chunk_mask = store  # type: ignore[method-assign]
        context.storage_manager.reserve_write = reserve  # type: ignore[method-assign]
        transfer.retrieve = retrieve  # type: ignore[method-assign]
        context.storage_manager.unsafe_read = unsafe_read  # type: ignore[method-assign]
        events.subscribe_kv_events = subscribe  # type: ignore[method-assign]

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared server context."""
        return self._ctx

    def report_status(self) -> dict:
        """Return current fault controls and public native-event diagnostics."""
        with self._lock:
            state = {
                "delay_rank": self._delay_rank,
                "held_stores": self._held,
                "fail_rank": self._fail_rank,
                "reservation_failures": self._failures,
                "disconnected": self._disconnected,
            }
        return {"kv_events_e2e": state, **self._events.report_status()}

    @server_module_handler("kv_events_e2e.control")
    def control(self, payload: bytes) -> bytes:
        """Apply a JSON Control and return status or real deletion counts.

        Actions: status, delay_rank, release_rank, fail_rank, evict, disconnect,
        reconnect. evict accepts EncodedObjectKeys captured from real L1 records;
        locked objects are skipped, never force-freed during a device transfer.
        """
        command = msgspec.json.decode(payload, type=Control)
        if command.action in ("delay_rank", "fail_rank"):
            if command.rank not in (0, 1):
                raise ValueError("test rank must be 0 or 1")
            with self._lock:
                if command.action == "delay_rank":
                    self._delay_rank = command.rank
                    self._gate.clear()
                else:
                    self._fail_rank = command.rank
        elif command.action == "release_rank":
            with self._lock:
                self._delay_rank = -1
                self._gate.set()
        elif command.action == "evict":
            deleted, skipped = self._ctx.storage_manager.delete_l1_keys(
                [key.to_object_key() for key in command.keys]
            )
            return msgspec.json.encode({"deleted": deleted, "skipped": skipped})
        elif command.action == "memory_start":
            if tracemalloc.is_tracing():
                raise ValueError("memory tracing is already running")
            tracemalloc.start(8)
            self._memory_baseline = tracemalloc.take_snapshot()
            return msgspec.json.encode({"tracing": True})
        elif command.action == "memory":
            if self._memory_baseline is None:
                raise ValueError("memory_start must precede memory")
            filters = [
                tracemalloc.Filter(
                    True, "*/multiprocess/modules/kv_events.py", all_frames=True
                ),
                tracemalloc.Filter(
                    True, "*/multiprocess/custom_types.py", all_frames=True
                ),
            ]
            snapshot = tracemalloc.take_snapshot().filter_traces(filters)
            baseline = self._memory_baseline.filter_traces(filters)
            current, peak = tracemalloc.get_traced_memory()
            return msgspec.json.encode(
                {
                    "tracing": True,
                    "native_event_traced_bytes": sum(
                        stat.size for stat in snapshot.statistics("filename")
                    ),
                    "native_event_delta_bytes": sum(
                        stat.size_diff
                        for stat in snapshot.compare_to(baseline, "filename")
                    ),
                    "global_traced_current_bytes": current,
                    "global_traced_peak_bytes": peak,
                    "process_rss_bytes": int(
                        Path("/proc/self/statm").read_text().split()[1]
                    )
                    * os.sysconf("SC_PAGE_SIZE"),
                    "counts": self._events.report_status()["kv_events"],
                    "scope": (
                        "Python allocations with kv_events.py/custom_types.py stack "
                        "frames; excludes externally allocated shared payloads and "
                        "native allocator memory; global peak includes all traced "
                        "Python allocations"
                    ),
                }
            )
        elif command.action == "memory_stop":
            if self._memory_baseline is not None:
                tracemalloc.stop()
                self._memory_baseline = None
            return msgspec.json.encode({"tracing": False})
        elif command.action == "events":
            return msgspec.json.encode(
                self._events.read_kv_events(
                    command.model_name, command.cursor, 1024, command.world_size
                )
            )
        elif command.action == "disconnect":
            with self._lock:
                self._disconnected = True
            self._events.close()
        elif command.action == "reconnect":
            with self._lock:
                self._disconnected = False
        elif command.action != "status":
            raise ValueError(f"unknown test action: {command.action}")
        return msgspec.json.encode(self.report_status())

    def close(self) -> None:
        """Release delayed stores and restore the original public methods."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._gate.set()
        self._transfer.store_with_chunk_mask = self._store  # type: ignore[method-assign]
        self._ctx.storage_manager.reserve_write = self._reserve  # type: ignore[method-assign]
        self._transfer.retrieve = self._retrieve  # type: ignore[method-assign]
        self._ctx.storage_manager.unsafe_read = self._unsafe_read  # type: ignore[method-assign]
        self._events.subscribe_kv_events = self._subscribe  # type: ignore[method-assign]
        if self._memory_baseline is not None:
            tracemalloc.stop()
            self._memory_baseline = None


def build_server_modules(
    context: ExtServerModuleBuildContext,
) -> ExtServerModuleComponents:
    """Install the guarded fixture through the normal extension-module seam."""
    transfer = next(
        module
        for module in context.modules
        if isinstance(module, LMCacheDrivenTransferModule)
    )
    events = next(
        module for module in context.modules if isinstance(module, KVEventModule)
    )
    return ExtServerModuleComponents(
        modules=(
            TestControls(
                context.server_context,
                transfer,
                events,
                Path("/artifacts/mp-reads.jsonl"),
            ),
        )
    )


def main() -> None:
    """Send one control operation through the selected local MP transport."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("command", help="JSON Control payload")
    args = parser.parse_args()
    msgspec.json.decode(args.command, type=Control)
    client = RequestClientFactory.create(args.url)
    try:
        result = client.server_module_call(
            ServerModuleCallRequest("kv_events_e2e.control", args.command.encode())
        ).result(timeout=10)
        if not result.success:
            raise RuntimeError(result.error)
        print(json.dumps(msgspec.json.decode(result.payload)))
    finally:
        client.close()


if __name__ == "__main__":
    main()
