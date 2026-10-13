# SPDX-License-Identifier: Apache-2.0
"""Common stream extraction preserves order and borrowed-client lifetime."""

# Standard
from typing import cast
import queue
import subprocess
import sys
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.custom_types import KVEventBatch, KVEventRecord
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.kv_event_subscriber import MPKVEventSubscriber
from lmcache.v1.multiprocess.transport.base import RequestClient


class Source:
    """Cancellable transport boundary for deterministic receiver tests."""

    def __init__(self) -> None:
        self.responses: queue.Queue[KVEventBatch | Exception | None] = queue.Queue()
        self.closed = threading.Event()
        self.stream = MessagingStream(self.read, self.close)

    def read(self, stopped: threading.Event) -> KVEventBatch:
        """Return one queued batch or transport error."""
        response = self.responses.get(timeout=5)
        if stopped.is_set() or response is None:
            raise StopIteration
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        """Wake a pending read and record stream cancellation."""
        self.closed.set()
        self.responses.put(None)


class Client:
    """Keep request-client ownership distinct from each event subscription."""

    def __init__(self) -> None:
        self.connections: queue.Queue[Source] = queue.Queue()
        self.calls: list[tuple[object, ...]] = []
        self.closed = False

    def subscribe_kv_events(self, *args: object) -> MessagingStream[KVEventBatch]:
        """Create a fresh stream without altering the request client."""
        self.calls.append(args)
        source = Source()
        self.connections.put(source)
        return source.stream

    def close(self) -> None:
        """Record request-client closure, which reception must never request."""
        self.closed = True


def test_extraction_preserves_order_reconnect_and_borrowed_lifetime() -> None:
    client = Client()
    output: queue.Queue[KVEventBatch | str] = queue.Queue()

    def accept(batch: KVEventBatch) -> bool:
        output.put(batch)
        return True

    subscriber = MPKVEventSubscriber(
        cast(RequestClient, client),
        7,
        "model",
        accept,
        lambda: output.put("reset"),
        lambda error: output.put(error),
    )
    subscriber.start()
    subscriber.start()
    source = client.connections.get(timeout=5)
    try:
        for sequence, kind in enumerate(("stored", "removed", "stored"), start=1):
            record = KVEventRecord(sequence, kind, "CPU", "model", [b"hash"], None, [1])
            batch = KVEventBatch(True, 1, sequence, False, [record])
            source.responses.put(batch)
            assert output.get(timeout=5) == batch
        source.responses.put(ConnectionError("disconnect"))
        assert output.get(timeout=5) == "reset"
        replacement = client.connections.get(timeout=5)
        assert source.closed.wait(5)
        assert client.calls == [(7, "model", 0, 1024), (7, "model", 0, 1024)]
        subscriber.close()
        subscriber.close()
        assert replacement.closed.is_set()
        assert not client.closed
        with pytest.raises(RuntimeError, match="closed"):
            subscriber.start()
    finally:
        subscriber.close()


def test_terminal_error_is_delegated_to_existing_engine_policy() -> None:
    client = Client()
    errors: queue.Queue[str] = queue.Queue()
    subscriber = MPKVEventSubscriber(
        cast(RequestClient, client),
        7,
        "model",
        lambda batch: True,
        lambda: None,
        errors.put,
    )
    subscriber.start()
    source = client.connections.get(timeout=5)
    try:
        source.responses.put(ValueError("incompatible server"))
        assert errors.get(timeout=5) == "incompatible server"
        assert source.closed.wait(5)
        assert len(client.calls) == 1
        assert not client.closed
    finally:
        subscriber.close()


def test_common_receiver_imports_without_inference_engines() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import lmcache.v1.multiprocess.kv_event_subscriber; "
            "assert not any(n.split('.')[0] in ('vllm','sglang') for n in sys.modules)",
        ],
        check=True,
        timeout=30,
    )
