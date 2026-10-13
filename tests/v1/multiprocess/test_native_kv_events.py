# SPDX-License-Identifier: Apache-2.0
"""Native binding/residency contracts, using the real EventBus and wire codecs."""

# Standard
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
from typing import Literal, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc, ObjectKey
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.custom_types import (
    EventBinding,
    IPCCacheServerKey,
    KVEventBatch,
    KVEventProfile,
    KVEventRecord,
    NativeBlockHash,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.engine_context import (
    LayoutDescRegistry,
    MPCacheServerContext,
)
from lmcache.v1.multiprocess.modules.kv_events import KVEventModule
from lmcache.v1.multiprocess.transport.grpc_impl.method_registry import (
    get_method_codec_registry,
)

Cache = tuple[KVEventModule, EventBus]


@pytest.fixture
def cache() -> Iterator[Cache]:
    """Provide registered TP2 dense two-group geometry and a real event bus."""
    bus = EventBus(EventBusConfig(enabled=True))
    registry = LayoutDescRegistry()
    layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
    for world in (1, 2):
        registry.register("model", world, layout, AttnWindowDesc([-1, -1]))
    ctx = SimpleNamespace(
        event_bus=bus,
        chunk_size=4,
        layout_desc_registry=registry,
        validate_native_event_storage=lambda: None,
    )
    module = KVEventModule(cast(MPCacheServerContext, ctx), kv_event_log_size=16)
    yield module, bus
    module.close()
    bus.stop()


def binding(offset: int = 0) -> EventBinding:
    """Return two native blocks for one four-token LMCache chunk."""
    return EventBinding(
        "sglang:sha256:signed64:v1",
        (offset + 10, offset + 11),
        None if offset == 0 else offset + 9,
        tuple(range(offset, offset + 4)),
        2,
    )


def store_key(world: int = 2, offset: int = 0) -> IPCCacheServerKey:
    """Create metadata with exact request tokens and an explicit profile."""
    descriptor = binding(offset)
    return IPCCacheServerKey(
        "model",
        world,
        0,
        tuple(range(offset + 4)),
        offset,
        offset + 4,
        "request",
        event_profile=KVEventProfile(descriptor.profile_id, 2),
        event_bindings=(descriptor,),
    )


def object_keys(world: int = 2, chunk: bytes = b"chunk") -> list[ObjectKey]:
    """Return every rank/group required by registered geometry."""
    return [
        ObjectKey(
            chunk, "model", ObjectKey.ComputeKVRank(world, rank, world, rank), group
        )
        for rank in range(world)
        for group in range(2)
    ]


def publish(
    bus: EventBus, kind: EventType, keys: list[ObjectKey], owner: str = "cpu0"
) -> None:
    """Flush one storage transition through the public EventBus API."""
    bus.publish(Event(kind, metadata={"keys": keys, "l1_tag": owner}))
    bus.stop()


def finish(bus: EventBus, transfer: str = "store") -> None:
    """Flush the marker produced after actual host write completion."""
    bus.publish(
        Event(
            EventType.MP_TOKENS,
            session_id="request",
            metadata={
                "chunk_hashes": [],
                "token_chunks": [],
                "token_offsets": [],
                "parent_hashes": [],
                "native_store_finished": ("request", transfer),
            },
        )
    )
    bus.stop()


def events(module: KVEventModule, world: int = 2) -> list[KVEventRecord]:
    """Read public retained transitions."""
    return module.read_kv_events("model", 0, 100, world).events


@pytest.mark.parametrize("binding_first", [True, False])
def test_complete_ranks_groups_and_both_join_orders(
    cache: Cache, binding_first: bool
) -> None:
    module, bus = cache
    key, keys = store_key(), object_keys()
    if binding_first:
        module.bind_store(key, [keys[0]], "store")
    publish(bus, EventType.L1_WRITE_FINISHED, keys[:-1])
    assert events(module) == []
    publish(bus, EventType.L1_WRITE_FINISHED_AND_READ_RESERVED, keys[-1:])
    if not binding_first:
        assert events(module) == []
        module.bind_store(key, [keys[0]], "store")
        bus.stop()
    assert [e.kind for e in events(module)] == ["stored"]
    assert events(module)[0].block_hashes == (10, 11)
    assert events(module)[0].block_size == 2
    finish(bus)
    publish(bus, EventType.L1_WRITE_FINISHED, keys)
    module.bind_store(key, [keys[0]], "duplicate")
    finish(bus, "duplicate")
    assert len(events(module)) == 1
    publish(bus, EventType.L1_KEYS_EVICTED, keys[-1:])
    assert [e.kind for e in events(module)] == ["stored", "removed"]
    publish(bus, EventType.L1_WRITE_FINISHED, keys[-1:])
    assert [e.kind for e in events(module)] == ["stored", "removed", "stored"]


def test_multiple_l1_copies_and_exact_removal(cache: Cache) -> None:
    module, bus = cache
    keys = object_keys()
    module.bind_store(store_key(), [keys[0]], "store")
    publish(bus, EventType.L1_WRITE_FINISHED, keys)
    publish(bus, EventType.L1_WRITE_FINISHED, keys, "cpu1")
    finish(bus)
    publish(bus, EventType.L1_KEYS_EVICTED, keys)
    assert len(events(module)) == 1
    publish(bus, EventType.L1_KEYS_EVICTED, keys[:1], "cpu1")
    assert events(module)[-1].kind == "removed"
    assert events(module)[-1].block_hashes == events(module)[0].block_hashes
    publish(bus, EventType.L1_KEYS_EVICTED, keys[1:], "cpu1")
    assert module.report_status()["kv_events"]["bindings"] == 0
    assert module.report_status()["kv_events"]["live_objects"] == 0


def test_failed_store_cleans_pending_without_announcement(cache: Cache) -> None:
    module, bus = cache
    for i in range(200):
        module.bind_store(store_key(), [object_keys(chunk=str(i).encode())[0]], str(i))
        finish(bus, str(i))
    assert events(module) == []
    status = module.report_status()["kv_events"]
    assert status["bindings"] == status["pending_stores"] == 0


def test_live_binding_survives_request_and_other_metadata_pressure(
    cache: Cache,
) -> None:
    module, bus = cache
    keys = object_keys()
    module.bind_store(store_key(), [keys[0]], "store")
    publish(bus, EventType.L1_WRITE_FINISHED, keys)
    finish(bus)
    original = events(module)[0]
    for i in range(100):
        module.bind_store(store_key(), [object_keys(chunk=str(i).encode())[0]], str(i))
        finish(bus, str(i))
    publish(bus, EventType.L1_KEYS_EVICTED, keys)
    assert events(module)[-1].block_hashes == original.block_hashes
    assert original.token_ids == (0, 1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        original.block_size = 4  # type: ignore[misc]


def test_binding_conflicts_and_namespace_isolation(cache: Cache) -> None:
    module, bus = cache
    keys = object_keys()
    key = store_key()
    module.bind_store(key, keys[:1], "store")
    with pytest.raises(ValueError, match="conflicting"):
        module.bind_store(
            replace(key, event_bindings=(replace(binding(), block_hashes=(99, 100)),)),
            keys[:1],
            "bad",
        )
    with pytest.raises(ValueError, match="profile.*mismatch"):
        module.register_profile("model", 2, KVEventProfile("other", 2))
    with pytest.raises(ValueError, match="cache_salt"):
        module.bind_store(replace(key, cache_salt="tenant"), keys[:1], "salt")
    # Same chunk in another TP context must not satisfy this context.
    publish(bus, EventType.L1_WRITE_FINISHED, object_keys(world=1))
    assert events(module) == []
    finish(bus)


@pytest.mark.parametrize(
    "world_size,block_size,attn,error",
    [
        (2, 3, AttnWindowDesc([-1]), "multiple of native block size"),
        (4, 2, AttnWindowDesc([-1]), "TP=1 or TP=2"),
        (2, 2, AttnWindowDesc([2]), "dense full attention"),
        (
            2,
            2,
            AttnWindowDesc([-1], group_kinds=("recurrent",)),
            "recurrent or auxiliary",
        ),
    ],
)
def test_native_registration_rejects_unsupported_geometry(
    cache: Cache, world_size: int, block_size: int, attn: AttnWindowDesc, error: str
) -> None:
    """Unsupported registration cannot create an event context or publication."""
    module, _ = cache
    with pytest.raises(ValueError, match=error):
        module.register_profile(
            "model", world_size, KVEventProfile("invalid", block_size), attn_desc=attn
        )
    assert module.report_status()["kv_events"]["bindings"] == 0
    assert events(module) == []


def test_parent_survives_eviction_and_replay_owns_old_descriptor(cache: Cache) -> None:
    module, bus = cache
    parent, child = object_keys(), object_keys(chunk=b"child")
    module.bind_store(store_key(), parent[:1], "parent")
    module.bind_store(store_key(offset=4), child[:1], "child")
    publish(bus, EventType.L1_WRITE_FINISHED, parent + child)
    finish(bus, "parent")
    finish(bus, "child")
    publish(bus, EventType.L1_KEYS_EVICTED, parent)
    assert events(module)[1].parent_block_hash == 13
    assert events(module)[1].block_hashes == (14, 15)


def test_event_bus_loss_invalidates_projection(cache: Cache) -> None:
    module, bus = cache
    keys = object_keys()
    module.bind_store(store_key(), keys[:1], "store")
    publish(bus, EventType.L1_WRITE_FINISHED, keys)
    finish(bus)
    # EventBus queue capacity is public configuration; a dropped removal must
    # invalidate owners, not merely append a subscriber loss record.
    for _ in range(10001):
        bus.publish(
            Event(EventType.L1_KEYS_EVICTED, metadata={"keys": [], "l1_tag": "cpu0"})
        )
    bus.stop()
    assert module.read_kv_events("model", 0, 100, 2).lost
    assert module.report_status()["kv_events"]["live_objects"] == 0
    module.bind_store(store_key(), keys[:1], "fresh")
    publish(bus, EventType.L1_WRITE_FINISHED, keys[:1])
    assert events(module) == []
    publish(bus, EventType.L1_WRITE_FINISHED, keys[1:])
    assert [e.kind for e in events(module)] == ["stored"]


@pytest.mark.parametrize(
    "native_hashes,parent",
    [
        ((b"a", b"b"), None),
        ((-(2**63), 2**63 - 1), -1),
        ((2**64 - 1, 2**63), 2**64 - 2),
    ],
)
def test_zmq_and_grpc_native_wire_round_trip(
    native_hashes: tuple[NativeBlockHash, ...], parent: NativeBlockHash | None
) -> None:
    descriptor = EventBinding("wire:v1", native_hashes, parent, (1, 2, 3, 4), 2)
    profile = KVEventProfile("wire:v1", 2)
    key = replace(store_key(), event_profile=profile, event_bindings=(descriptor,))
    assert get_customized_decoder(IPCCacheServerKey).decode(
        get_customized_encoder(IPCCacheServerKey).encode(key)
    ).event_bindings == (descriptor,)
    registry = get_method_codec_registry()
    store = next(
        codec
        for name, codec in registry.by_full_name.items()
        if name.endswith(".Store")
    )
    args = (key, 42, [[1]], b"event")
    encoded = store.request_encoder(args, {})
    decoded = store.request_decoder(encoded)
    assert decoded[0].event_bindings == (descriptor,)
    assert decoded[0].event_profile == profile
    subscribe = next(
        codec
        for name, codec in registry.by_full_name.items()
        if name.endswith(".SubscribeKvEvents")
    )
    record = KVEventRecord(
        1,
        "stored",
        "CPU",
        "model",
        native_hashes,
        parent,
        (1, 2, 3, 4),
        2,
        profile.profile_id,
        2,
    )
    batch = KVEventBatch(True, 1, 1, False, [record])
    assert subscribe.response_decoder(subscribe.response_encoder(batch)) == batch
    assert (
        get_customized_decoder(KVEventBatch).decode(
            get_customized_encoder(KVEventBatch).encode(batch)
        )
        == batch
    )


def test_real_l1_commit_clear_and_temporary_objects(
    cache: Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Storage operations, rather than transfer success, prove CPU readiness."""
    # First Party
    from lmcache.v1.distributed import l1_manager as l1_module
    from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
    from lmcache.v1.distributed.error import L1Error

    module, bus = cache
    monkeypatch.setattr(l1_module, "get_event_bus", lambda: bus)
    manager = l1_module.L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024, use_lazy=False
            ),
            tag="cpu-real",
        )
    )
    keys = object_keys()
    layout = MemoryLayoutDesc([torch.Size([4])], [torch.float32])
    try:
        module.bind_store(store_key(), keys[:1], "real")
        reserved = manager.reserve_write(keys, [False] * len(keys), layout)
        assert all(result[0] == L1Error.SUCCESS for result in reserved.values())
        bus.stop()
        assert events(module) == []
        manager.finish_write(keys[:-1])
        bus.stop()
        assert events(module) == []
        manager.finish_write(keys[-1:])
        bus.stop()
        assert [event.kind for event in events(module)] == ["stored"]
        finish(bus, "real")
        manager.clear()
        bus.stop()
        assert [event.kind for event in events(module)] == ["stored", "removed"]
        assert module.report_status()["kv_events"]["bindings"] == 0
        temporary = object_keys(chunk=b"temporary")
        module.bind_store(store_key(), temporary[:1], "temporary")
        manager.reserve_write(temporary, [True] * len(temporary), layout)
        manager.finish_write(temporary)
        finish(bus, "temporary")
        assert len(events(module)) == 2
        assert module.report_status()["kv_events"]["bindings"] == 0
    finally:
        manager.close()


@pytest.mark.parametrize("use_l2", [False, True])
def test_private_cpu_storage_profile_validation(use_l2: bool) -> None:
    """Event-enabled registration rejects an actual configured L2 adapter."""
    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
    from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig

    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024, use_lazy=False, shm_name=""
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
        l2_adapter_config=L2AdaptersConfig(
            [MockL2AdapterConfig(max_size_gb=0.001, mock_bandwidth_gb=1.0)]
            if use_l2
            else []
        ),
    )
    context = MPCacheServerContext(config, chunk_size=4)
    try:
        if use_l2:
            with pytest.raises(ValueError, match="without L2"):
                context.validate_native_event_storage()
        else:
            context.validate_native_event_storage()
    finally:
        context.close()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_transport_subscriber_overflow_and_bus_loss(
    cache: Cache, transport: Literal["zmq", "grpc"]
) -> None:
    """Both real transports withdraw CPU state after queue and server loss."""
    # Standard
    import socket
    import time

    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.kv_event_subscriber import (
        KVEventReset,
        MPKVEventSubscriber,
    )
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache.v1.multiprocess.transport.server_factory import create_request_server

    module, bus = cache
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = create_request_server(
        [module], MPServerConfig(transport=transport, host="127.0.0.1", port=port)
    )
    server.start()
    scheme = "grpc" if transport == "grpc" else "tcp"
    client = RequestClientFactory.create(f"{scheme}://127.0.0.1:{port}")
    subscriber = MPKVEventSubscriber(
        client,
        7,
        "model",
        world_size=2,
        profile=KVEventProfile(binding().profile_id, 2),
        max_pending_events=4,
    )
    cpu: set[NativeBlockHash] = set()

    def wait_counter(name: str, minimum: int) -> None:
        deadline = time.monotonic() + 5
        while subscriber.report_status()[name] < minimum:
            if time.monotonic() >= deadline:
                pytest.fail(f"subscriber did not reach {name}={minimum}")
            time.sleep(0.01)

    def drain() -> None:
        for update in subscriber.drain():
            if isinstance(update, KVEventReset):
                assert update.error is None
                cpu.clear()
            else:
                for event in update.events:
                    if event.kind == "stored":
                        cpu.update(event.block_hashes)
                    else:
                        cpu.difference_update(event.block_hashes)

    try:
        subscriber.start(timeout=5)
        keys = object_keys()
        module.bind_store(store_key(), keys[:1], "initial")
        publish(bus, EventType.L1_WRITE_FINISHED, keys)
        finish(bus, "initial")
        wait_counter("received_events", 1)
        drain()
        assert cpu == {10, 11}
        for index in range(7):
            publish(
                bus,
                EventType.L1_KEYS_EVICTED
                if index % 2 == 0
                else EventType.L1_WRITE_FINISHED,
                keys[-1:],
            )
            wait_counter("received_events", index + 2)
        assert subscriber.report_status()["resets"] > 0
        drain()
        assert cpu == set()
        publish(bus, EventType.L1_WRITE_FINISHED, keys[-1:])
        wait_counter("received_events", 9)
        drain()
        assert cpu == {10, 11}
        resets = subscriber.report_status()["resets"]
        for _ in range(10001):
            bus.publish(Event(EventType.MP_TOKENS, metadata={}))
        bus.stop()
        wait_counter("resets", resets + 1)
        drain()
        assert cpu == set()
        assert module.report_status()["kv_events"]["live_objects"] == 0
        module.bind_store(store_key(), keys[:1], "fresh")
        publish(bus, EventType.L1_WRITE_FINISHED, keys[:1])
        assert events(module) == []
        publish(bus, EventType.L1_WRITE_FINISHED, keys[1:])
        wait_counter("received_events", 10)
        drain()
        assert cpu == {10, 11}
    finally:
        subscriber.close()
        client.close()
        server.close()


def test_native_markers_preserve_coordinator_token_event_contract(cache: Cache) -> None:
    """Existing coordinator/trace consumers accept native lifecycle markers."""
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.v1.mp_coordinator.cache_events import (
        CacheEventSink,
        CacheEventSubscriber,
    )

    module, bus = cache
    legacy = CacheEventSubscriber(MagicMock(spec=CacheEventSink), "server", 1)
    legacy.register(bus)
    module.bind_store(store_key(), object_keys()[:1], "store")
    finish(bus)
    assert bus.subscriber_exception_counts() == {}


def test_registration_rejects_nonowner_and_noop_profile_mismatch(
    cache: Cache, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every worker validates the native profile before inference readiness."""
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.multiprocess.modules import (
        lmcache_driven_transfer as transfer_module,
    )

    events_module, _ = cache
    context = cast(SimpleNamespace, events_module.context)
    context.storage_manager = MagicMock()
    context.separate_object_groups = False
    context.full_sw_kv = False
    cache_context = MagicMock(num_layers=2, device=torch.device("cpu"))
    cache_context.kv_layer_groups_manager.num_object_groups = 2
    cache_context.kv_layer_groups_manager.get_attn_desc.return_value = AttnWindowDesc(
        [-1, -1]
    )
    monkeypatch.setattr(
        transfer_module, "create_cache_context", lambda *args, **kwargs: cache_context
    )
    monkeypatch.setattr(
        transfer_module,
        "get_layout_desc",
        lambda *args, **kwargs: MemoryLayoutDesc([], []),
    )
    monkeypatch.setattr(
        transfer_module, "get_event_ipc_backend", lambda device: MagicMock()
    )
    monkeypatch.setattr(transfer_module, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(transfer_module, "torch_dev", MagicMock())
    transfer = transfer_module.LMCacheDrivenTransferModule(events_module.context)
    profile = KVEventProfile("registered:v1", 2)
    try:
        transfer.register_kv_cache(1, [], "model", 2, EngineType.VLLM, {}, [], profile)
        assert transfer.tracked_instance_count() == 1
        for instance_id in (2, 1):
            with pytest.raises(ValueError, match="profile.*mismatch"):
                transfer.register_kv_cache(
                    instance_id,
                    [],
                    "model",
                    2,
                    EngineType.VLLM,
                    {},
                    [],
                    KVEventProfile("wrong", 2),
                )
        assert transfer.tracked_instance_count() == 1
        transfer.register_kv_cache(2, [], "model", 2, EngineType.VLLM, {}, [], profile)
        assert transfer.tracked_instance_count() == 2
    finally:
        transfer.close()


def test_event_bus_drop_prevents_false_store_before_loss_barrier(cache: Cache) -> None:
    """A dropped shard removal must not produce even a transient Stored."""
    existing, _ = cache
    bus = EventBus(EventBusConfig(max_queue_size=1))
    ctx = SimpleNamespace(
        event_bus=bus,
        chunk_size=4,
        layout_desc_registry=existing.context.layout_desc_registry,
        validate_native_event_storage=lambda: None,
    )
    module = KVEventModule(cast(MPCacheServerContext, ctx), kv_event_log_size=16)
    keys = object_keys()
    observed: list[KVEventRecord] = []
    try:
        module.bind_store(store_key(), keys[:1], "partial")
        bus.stop()
        publish(bus, EventType.L1_WRITE_FINISHED, keys[:2])
        bus.subscribe(
            EventType.L1_WRITE_FINISHED, lambda event: observed.extend(events(module))
        )
        bus.publish(
            Event(
                EventType.L1_WRITE_FINISHED,
                metadata={"keys": keys[2:], "l1_tag": "cpu0"},
            )
        )
        bus.publish(
            Event(
                EventType.L1_KEYS_EVICTED, metadata={"keys": keys[:2], "l1_tag": "cpu0"}
            )
        )
        bus.stop()
        assert observed == []
        assert module.read_kv_events("model", 0, 100, 2).lost
    finally:
        module.close()
        bus.stop()


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_mp_restart_before_registration_reconnects_same_source(
    cache: Cache, transport: Literal["zmq", "grpc"]
) -> None:
    """A replacement MP waits for heartbeat registration without terminal fallback."""
    # Standard
    import socket
    import time

    # First Party
    from lmcache.v1.multiprocess.config import MPServerConfig
    from lmcache.v1.multiprocess.kv_event_subscriber import (
        KVEventReset,
        MPKVEventSubscriber,
    )
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
    from lmcache.v1.multiprocess.transport.server_factory import create_request_server

    module, bus = cache
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = MPServerConfig(transport=transport, host="127.0.0.1", port=port)
    server = create_request_server([module], config)
    server.start()
    scheme = "grpc" if transport == "grpc" else "tcp"
    client = RequestClientFactory.create(f"{scheme}://127.0.0.1:{port}")
    subscriber = MPKVEventSubscriber(
        client,
        7,
        "model",
        world_size=2,
        profile=KVEventProfile(binding().profile_id, 2),
    )
    try:
        subscriber.start(timeout=5)
        module.bind_store(store_key(), object_keys()[:1], "store")
        publish(bus, EventType.L1_WRITE_FINISHED, object_keys())
        deadline = time.monotonic() + 5
        while subscriber.report_status()["received_events"] < 1:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        old_incarnation = module.read_kv_events("model", 0, 10, 2).incarnation
        subscriber.drain()
        server.close()
        module.close()
        registry = LayoutDescRegistry()
        bus = EventBus(EventBusConfig())
        context = SimpleNamespace(
            event_bus=bus,
            chunk_size=4,
            layout_desc_registry=registry,
            validate_native_event_storage=lambda: None,
        )
        module = KVEventModule(
            cast(MPCacheServerContext, context), kv_event_log_size=16
        )
        server = create_request_server([module], config)
        server.start()
        deadline = time.monotonic() + 5
        while subscriber.report_status()["resets"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        resets = subscriber.drain()
        assert all(
            isinstance(update, KVEventReset) and update.error is None
            for update in resets
        )
        registry.register(
            "model", 2, MemoryLayoutDesc([], []), AttnWindowDesc([-1, -1])
        )
        module.bind_store(store_key(), object_keys()[:1], "new")
        publish(bus, EventType.L1_WRITE_FINISHED, object_keys())
        deadline = time.monotonic() + 5
        while subscriber.report_status()["received_events"] < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        updates = subscriber.drain()
        assert any(
            isinstance(update, KVEventReset) and update.reason == "server_restart"
            for update in updates
        )
        batches = [update for update in updates if isinstance(update, KVEventBatch)]
        assert batches[-1].incarnation != old_incarnation
        assert batches[-1].events[0].seq == 1
    finally:
        subscriber.close()
        client.close()
        server.close()
        module.close()
        bus.stop()


def test_private_devdax_is_not_rejected_by_native_event_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate topology only; no Device-DAX device or transfer is exercised."""
    # Standard
    from unittest.mock import MagicMock

    # First Party
    from lmcache.v1.distributed.config import (
        EvictionConfig,
        L1ManagerConfig,
        L1MemoryManagerConfig,
        StorageManagerConfig,
    )
    from lmcache.v1.multiprocess import engine_context

    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1024 * 1024,
                use_lazy=False,
                shm_name="",
                devdax_path="/dev/dax0.0",
            )
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    monkeypatch.setattr(engine_context, "StorageManager", MagicMock())
    context = MPCacheServerContext(config, chunk_size=4)
    try:
        context.validate_native_event_storage()
    finally:
        context.close()
