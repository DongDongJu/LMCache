# SPDX-License-Identifier: Apache-2.0
"""Real native publisher/observer transport tests; no inference or GPU required."""

# Standard
import atexit
import threading
import time

# Third Party
from observe import NativeSubscriber, _Reconnect
import msgspec
import pytest
import zmq


def test_observer_rejects_handshake_without_replay():
    with zmq.Context() as context:
        with pytest.raises(ValueError, match="requires"):
            NativeSubscriber(context, "inproc://unused", handshake=True)


def test_native_replay_dedup_and_new_publisher_epoch():
    native = pytest.importorskip("sglang.srt.disaggregation.kv_events")
    context = zmq.Context()
    records = []
    errors = []
    stop = threading.Event()
    reconnect = threading.Event()
    subscribers = []

    def publisher(endpoint, replay):
        result = native.ZmqEventPublisher(
            0,
            endpoint=endpoint,
            bind=True,
            replay_endpoint=replay,
            replay_handshake=True,
            topic="native-events",
        )
        atexit.unregister(result.shutdown)
        return result

    current = publisher("tcp://127.0.0.1:0", "tcp://127.0.0.1:0")
    endpoint = current._local_pub_endpoint
    replay = current._local_replay_endpoint

    def emit(ts):
        batch = native.KVEventBatch(
            ts=ts,
            events=[
                native.BlockRemoved(
                    block_hashes=[123],
                    medium=native.StorageMedium.CPU,
                )
            ],
        )
        current.publish(batch)
        current._event_queue.join()
        return msgspec.msgpack.encode(batch)

    emit(1.0)  # Retained before either subscriber exists.

    class ReconnectingSubscriber(NativeSubscriber):
        def _connections(self):
            if reconnect.is_set():
                reconnect.clear()
                raise _Reconnect
            super()._connections()

    def consume():
        subscriber = ReconnectingSubscriber(context, endpoint, replay, True)
        subscribers.append(subscriber)
        try:
            records.extend(subscriber.records(lambda: not stop.is_set()))
        except Exception as exc:
            errors.append(exc)
        finally:
            subscriber.close()

    thread = threading.Thread(target=consume)
    thread.start()

    def wait_count(count):
        deadline = time.monotonic() + 8
        while len(records) < count and not errors and time.monotonic() < deadline:
            stop.wait(0.01)
        assert not errors
        assert len(records) == count

    try:
        wait_count(1)
        emit(2.0)
        wait_count(2)
        assert [r["sequence"] for r in records] == [0, 1]
        assert [r["delivery"] for r in records] == ["replay", "live"]
        assert [r["publisher_epoch"] for r in records] == [1, 1]
        original_incarnation = records[0]["publisher_incarnation"]
        reconnect.set()
        deadline = time.monotonic() + 5
        while subscribers[0].connection < 2 and time.monotonic() < deadline:
            stop.wait(0.01)
        assert subscribers[0].connection >= 2
        emit(2.5)
        wait_count(3)
        assert records[-1]["sequence"] == 2
        assert records[-1]["publisher_incarnation"] == original_incarnation
        current.shutdown()
        current = publisher(endpoint, replay)
        emit(3.0)
        wait_count(4)
        assert records[-1]["sequence"] == 0
        assert records[-1]["publisher_epoch"] == 2
        assert records[-1]["connection"] > records[0]["connection"]
        assert records[-1]["delivery"] == "replay"
        assert records[-1]["publisher_incarnation"] != original_incarnation
        emit(4.0)
        wait_count(5)
        assert [r["sequence"] for r in records] == [0, 1, 2, 0, 1]
        # Accepted raw native bytes retain CPU scoping in both delivery paths.
        assert all(r["payload"][1][0]["medium"] == "CPU_PINNED" for r in records)
    finally:
        stop.set()
        thread.join(3)
        current.shutdown()
        context.term()
    assert not thread.is_alive()
    assert not errors
