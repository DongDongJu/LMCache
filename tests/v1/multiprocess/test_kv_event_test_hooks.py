# SPDX-License-Identifier: Apache-2.0
"""The opt-in E2E controls affect actual operations and preserve normal clients."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import cast
import json
import time

# Third Party
import msgspec
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVEventProfile
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from tests.e2e.mp_kv_events.mp_test_hooks import TestControls as FaultControls


def test_hooks_require_explicit_fixture_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LMCACHE_KV_EVENTS_E2E", raising=False)
    with pytest.raises(RuntimeError, match="LMCACHE_KV_EVENTS_E2E=1"):
        FaultControls(
            cast(MPCacheServerContext, None),
            cast(LMCacheDrivenTransferModule, None),
            cast(KVEventModule, None),
        )


def test_controls_gate_real_calls_fail_once_and_close_only_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_KV_EVENTS_E2E", "1")
    called: list[str] = []

    def store(*args: object) -> tuple[bytes, bool, list[bool]]:
        called.append("store")
        return b"", True, [True]

    def reserve(*args: object) -> dict:
        called.append("reserve")
        return {}

    def delete(keys: list[ObjectKey]) -> tuple[int, int]:
        called.append("delete")
        return len(keys), 0

    transfer = SimpleNamespace(
        store_with_chunk_mask=store, retrieve=lambda *args: (b"", True)
    )
    storage = SimpleNamespace(
        reserve_write=reserve,
        delete_l1_keys=delete,
        unsafe_read=lambda *args, **kwargs: ([], []),
    )
    events = SimpleNamespace(
        subscribe_kv_events=lambda *args: called.append("subscribe"),
        close=lambda: called.append("close_streams"),
        report_status=lambda: {"kv_events": {"subscribers": 0}},
    )
    ctx = SimpleNamespace(storage_manager=storage)
    hooks = FaultControls(
        cast(MPCacheServerContext, ctx),
        cast(LMCacheDrivenTransferModule, transfer),
        cast(KVEventModule, events),
    )
    key = IPCCacheServerKey("model", 2, 1, (1,), 0, 1, "req")
    obj = ObjectKey(b"key", "model", ObjectKey.ComputeKVRank(2, 1, 2, 1))
    layout = MemoryLayoutDesc([], [])
    try:
        hooks.control(b'{"action":"delay_rank","rank":1}')
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(
                transfer.store_with_chunk_mask, key, 1, [[1]], b"event"
            )
            deadline = time.monotonic() + 2
            while hooks.report_status()["kv_events_e2e"]["held_stores"] == 0:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            assert "store" not in called
            hooks.control(b'{"action":"release_rank"}')
            assert waiting.result(timeout=2)[1]
        hooks.control(b'{"action":"fail_rank","rank":1}')
        with pytest.raises(RuntimeError, match="reservation failure"):
            storage.reserve_write([obj], layout)
        assert storage.reserve_write([obj], layout) == {}
        assert hooks.report_status()["kv_events_e2e"]["reservation_failures"] == 1
        hooks.control(b'{"action":"disconnect"}')
        assert "close_streams" in called
        stream = events.subscribe_kv_events(
            1, "model", 0, 10, 2, KVEventProfile("test", 1)
        )
        with pytest.raises(StopIteration):
            next(stream)
        stream.close()
        hooks.control(b'{"action":"reconnect"}')
        events.subscribe_kv_events(1, "model", 0, 10, 2, KVEventProfile("test", 1))
        assert "subscribe" in called
        result = hooks.control(
            msgspec.json.encode(
                {"action": "evict", "keys": [obj.to_encoded_object_key()]}
            )
        )
        assert msgspec.json.decode(result) == {"deleted": 1, "skipped": 0}
        with pytest.raises(ValueError, match="unknown"):
            hooks.control(b'{"action":"fabricate"}')
        with pytest.raises(ValueError, match="memory_start"):
            hooks.control(b'{"action":"memory"}')
        hooks.control(b'{"action":"memory_start"}')
        memory = msgspec.json.decode(hooks.control(b'{"action":"memory"}'))
        assert memory["native_event_traced_bytes"] >= 0
        assert memory["process_rss_bytes"] > 0
        assert memory["counts"]["subscribers"] == 0
        assert msgspec.json.decode(hooks.control(b'{"action":"memory_stop"}')) == {
            "tracing": False
        }
    finally:
        hooks.close()
        hooks.close()


@pytest.mark.parametrize("found_count", [0, 1, 2])
def test_read_trace_preserves_actual_found_objects_and_request_context(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, found_count: int
) -> None:
    """The trace reports a partial read as partial and never invents H2D success."""
    monkeypatch.setenv("LMCACHE_KV_EVENTS_E2E", "1")
    keys = [
        ObjectKey(bytes([number]), "model", ObjectKey.ComputeKVRank(2, 1, 2, 1))
        for number in (1, 2)
    ]
    objects = [
        SimpleNamespace(get_l1_manager=lambda: 9, get_size=lambda: 64)
        for _ in keys[:found_count]
    ]
    storage = SimpleNamespace(
        reserve_write=lambda *args: {},
        unsafe_read=lambda *args, **kwargs: (keys[:found_count], objects),
    )

    def retrieve(*args: object) -> tuple[bytes, bool]:
        found, returned = storage.unsafe_read(keys, l1_owners={key: 9 for key in keys})
        assert found == keys[:found_count] and returned == objects
        return b"completion-event", found_count == 2

    transfer = SimpleNamespace(
        store_with_chunk_mask=lambda *args: (b"", True, []), retrieve=retrieve
    )
    events = SimpleNamespace(
        subscribe_kv_events=lambda *args: None, report_status=lambda: {"kv_events": {}}
    )
    context = SimpleNamespace(storage_manager=storage)
    trace = tmp_path / "reads.jsonl"
    hooks = FaultControls(
        cast(MPCacheServerContext, context),
        cast(LMCacheDrivenTransferModule, transfer),
        cast(KVEventModule, events),
        trace,
    )
    key = IPCCacheServerKey("model", 2, 1, (1, 2, 3, 4), 0, 4, "engine-request")
    try:
        assert transfer.retrieve(key, 42, [[1]], b"producer", 2) == (
            b"completion-event",
            found_count == 2,
        )
        record = json.loads(trace.read_text())
        assert record["request_id"] == "engine-request"
        assert record["worker_id"] == 1 and record["instance_id"] == 42
        assert record["token_ids"] == [1, 2, 3, 4]
        assert record["skip_first_n_tokens"] == 2
        assert len(record["requested_keys"]) == 2
        assert len(record["found_keys"]) == found_count
        assert record["owner_ids"] == [9] * found_count
        assert record["bytes"] == 64 * found_count
        storage.unsafe_read(keys)
        assert len(trace.read_text().splitlines()) == 1
    finally:
        hooks.close()
