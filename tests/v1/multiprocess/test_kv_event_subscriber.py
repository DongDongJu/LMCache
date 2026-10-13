# SPDX-License-Identifier: Apache-2.0
"""Stream ordering and resource ownership without an inference engine."""

# Standard
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast
import queue
import subprocess
import sys
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    KVEventBatch,
    KVEventProfile,
    KVEventRecord,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.kv_event_subscriber import (
    KVEventReset,
    MPKVEventSubscriber,
)
from lmcache.v1.multiprocess.transport.base import RequestClient

PROFILE = KVEventProfile("test-native-v1", 2)


@dataclass
class Delivery:
    value: KVEventBatch | Exception | None
    consumed: threading.Event


class Source:
    """Cancellable transport boundary with deterministic delivery barriers."""

    def __init__(self) -> None:
        self.inbox: queue.Queue[Delivery] = queue.Queue()
        self.previous: Delivery | None = None
        self.closed = threading.Event()
        self.stream = MessagingStream(self.read, self.cancel)

    def read(self, closed: threading.Event) -> KVEventBatch:
        if self.previous is not None:
            self.previous.consumed.set()
        delivery = self.inbox.get(timeout=5)
        self.previous = delivery
        if closed.is_set():
            raise StopIteration
        if isinstance(delivery.value, Exception):
            raise delivery.value
        assert delivery.value is not None
        return delivery.value

    def cancel(self) -> None:
        if self.previous is not None:
            self.previous.consumed.set()
        self.closed.set()
        self.inbox.put(Delivery(None, threading.Event()))

    def send(self, value: KVEventBatch | Exception) -> None:
        delivery = Delivery(value, threading.Event())
        self.inbox.put(delivery)
        assert delivery.consumed.wait(5), "receiver did not consume delivery"


class Client:
    """Expose subscriptions separately from the borrowed request lifetime."""

    def __init__(self) -> None:
        self.connections: queue.Queue[Source] = queue.Queue()
        self.arguments: list[tuple[object, ...]] = []
        self.closed = False

    def subscribe_kv_events(self, *args: object) -> MessagingStream[KVEventBatch]:
        self.arguments.append(args)
        source = Source()
        self.connections.put(source)
        return source.stream

    def close(self) -> None:
        self.closed = True


def batch(
    seq: int,
    kind: str = "stored",
    *,
    incarnation: int = 1,
    cursor: int | None = None,
    lost: bool = False,
) -> KVEventBatch:
    return KVEventBatch(
        True,
        incarnation,
        seq if cursor is None else cursor,
        lost,
        [
            KVEventRecord(
                seq=seq,
                kind=kind,
                medium="CPU",
                model_name="model",
                block_hashes=(b"native-block",),
                parent_block_hash=None,
                token_ids=(1, 2) if kind == "stored" else (),
                block_size=2,
                profile_id=PROFILE.profile_id,
                world_size=2,
            )
        ],
    )


@pytest.fixture
def connected() -> Iterator[tuple[MPKVEventSubscriber, Client, Source]]:
    client = Client()
    subscriber = MPKVEventSubscriber(
        cast(RequestClient, client),
        42,
        "model",
        world_size=2,
        profile=PROFILE,
        max_pending_events=4,
    )
    subscriber.start()
    source = client.connections.get(timeout=5)
    yield subscriber, client, source
    subscriber.close()
    assert not client.closed


def test_import_does_not_load_engines() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from lmcache.v1.multiprocess.kv_event_subscriber "
            "import MPKVEventSubscriber; "
            "assert not any(n.split('.')[0] in ('vllm', 'sglang') "
            "for n in sys.modules)",
        ],
        check=True,
        timeout=30,
    )


def test_order_and_filtered_cursor_gaps(connected: tuple) -> None:
    subscriber, client, source = connected
    source.send(batch(1, cursor=3))
    source.send(batch(5, "removed", cursor=6))
    source.send(batch(8))
    updates = subscriber.drain()
    assert [r.kind for b in updates for r in b.events] == [
        "stored",
        "removed",
        "stored",
    ]
    assert client.arguments == [(42, "model", 0, 3, 2, PROFILE)]
    assert subscriber.drain() == []


def test_incarnation_precedes_deduplication(connected: tuple) -> None:
    subscriber, _, source = connected
    source.send(batch(100))
    subscriber.drain()
    source.send(batch(1, incarnation=2))
    updates = subscriber.drain()
    assert updates[0] == KVEventReset("server_restart")
    assert updates[1].events[0].seq == 1


def test_duplicate_batches_and_explicit_loss(connected: tuple) -> None:
    subscriber, _, source = connected
    source.send(batch(1))
    source.send(batch(1))
    assert len(subscriber.drain()) == 1
    source.send(batch(7, "removed", lost=True))
    updates = subscriber.drain()
    assert updates[0] == KVEventReset("events_lost")
    assert updates[1].events[0].kind == "removed"


def test_overflow_preserves_reliable_invalidation(connected: tuple) -> None:
    subscriber, _, source = connected
    source.send(batch(1))
    subscriber.drain()
    for seq in range(2, 7):
        source.send(batch(seq, "removed" if seq == 2 else "stored"))
    updates = subscriber.drain()
    assert updates[0] == KVEventReset("queue_overflow")
    assert updates[1].events[0].seq == 6
    assert subscriber.report_status()["buffer_depth"] == 0


def test_reconnect_invalidates_then_replays_from_zero(connected: tuple) -> None:
    subscriber, client, source = connected
    source.send(batch(20))
    subscriber.drain()
    source.send(ConnectionError("transport interrupted"))
    replacement = client.connections.get(timeout=5)
    replacement.send(batch(1))
    updates = subscriber.drain()
    assert updates[0] == KVEventReset("stream_closed")
    assert updates[1].events[0].seq == 1
    assert client.arguments[1][2] == 0
    assert source.closed.is_set()
    assert not client.closed


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("incompatible profile"),
        KVEventBatch(False, 1, 0, False, []),
        batch(2, cursor=1),
    ],
)
def test_terminal_protocol_failure_is_visible(
    connected: tuple, failure: object
) -> None:
    subscriber, client, source = connected
    source.send(batch(1))
    subscriber.drain()
    source.send(failure)
    assert source.closed.wait(5)
    updates = subscriber.drain()
    assert len(updates) == 1
    assert updates[0].reason == "protocol_error"
    assert updates[0].error
    assert len(client.arguments) == 1


def test_cursor_regression_is_terminal(connected: tuple) -> None:
    subscriber, _, source = connected
    source.send(batch(10))
    subscriber.drain()
    source.send(batch(9))
    assert "regressed" in subscriber.drain()[0].error


def test_close_cancels_owned_thread_only(connected: tuple) -> None:
    subscriber, client, source = connected
    subscriber.start()
    subscriber.close()
    subscriber.close()
    assert source.closed.is_set()
    assert len(client.arguments) == 1
    assert not client.closed
    with pytest.raises(RuntimeError, match="closed"):
        subscriber.start()


def test_reconnect_backoff_and_cancellation(connected: tuple) -> None:
    subscriber, client, source = connected
    start = time.monotonic()
    source.send(ConnectionError("first"))
    client.connections.get(timeout=5).send(ConnectionError("second"))
    client.connections.get(timeout=5).send(ConnectionError("third"))
    assert time.monotonic() - start >= 0.25
    subscriber.close()
    assert len(client.arguments) == 3


def test_startup_waits_for_negotiated_status(connected: tuple) -> None:
    subscriber, _, source = connected
    source.send(KVEventBatch(True, 1, 0, False, []))
    subscriber.start(timeout=0.1)
    assert subscriber.drain() == []


def test_startup_negotiation_error_is_not_deferred_to_inference(
    connected: tuple,
) -> None:
    subscriber, _, source = connected
    source.send(ValueError("incompatible native profile"))
    with pytest.raises(RuntimeError, match="incompatible native profile"):
        subscriber.start(timeout=0.1)


def test_startup_timeout_cancels_owned_stream(connected: tuple) -> None:
    subscriber, client, source = connected
    with pytest.raises(LMCacheTimeoutError, match="negotiation timed out"):
        subscriber.start(timeout=0.05)
    assert source.closed.is_set()
    assert not client.closed
