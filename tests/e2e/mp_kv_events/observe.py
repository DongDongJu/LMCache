# SPDX-License-Identifier: Apache-2.0
"""Capture native publisher frames independently of LMCache binding construction."""

# Standard
import argparse
import base64
import json
import signal
import time
import uuid

# Third Party
from zmq.utils.monitor import recv_monitor_message
import msgspec
import zmq

END_SEQ = (-1).to_bytes(8, "big", signed=True)
READY_SEQ = (-2).to_bytes(8, "big", signed=True)
READY_PREFIX = b"\xffkv-replay-ready:"


class _Reconnect(Exception):
    """The native publisher connection ended during a replay."""


class NativeSubscriber:
    """Merge native replay/live frames before recording, with bounded catchup.

    Native connection changes trigger catchup, while the handshake's explicit
    publisher incarnation determines whether to reset the dedup watermark.
    The observer therefore survives engine restarts and ordinary reconnects.
    """

    def __init__(self, context, endpoint, replay_endpoint=None, handshake=False):
        if handshake and not replay_endpoint:
            raise ValueError("--replay-handshake requires --replay-endpoint")
        self.context = context
        self.endpoint = endpoint
        self.replay_endpoint = replay_endpoint
        self.handshake = handshake
        self.connection = 0
        self.epoch = 0
        self.next_sequence = 0
        self.incarnation = None
        self._open()

    def _open(self):
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVHWM, 100_000)
        self.monitor = self.socket.get_monitor_socket(
            events=zmq.EVENT_CONNECTED | zmq.EVENT_DISCONNECTED
        )
        self.connected = False
        self.socket.connect(self.endpoint)

    def close(self):
        self.socket.disable_monitor()
        self.monitor.close(linger=0)
        self.socket.close(linger=0)

    def _connections(self):
        while self.monitor.poll(0):
            event = recv_monitor_message(self.monitor)["event"]
            if event == zmq.EVENT_DISCONNECTED:
                raise _Reconnect
            if event == zmq.EVENT_CONNECTED:
                self.connection += 1
                self.connected = True
                if not self.handshake:
                    self.epoch += 1
                    self.next_sequence = 0

    def _record(self, frames, delivery):
        if len(frames) != 3 or len(frames[1]) != 8:
            raise RuntimeError("Malformed native event framing")
        sequence = int.from_bytes(frames[1], "big")
        if sequence < self.next_sequence:
            return None
        if sequence != self.next_sequence:
            raise RuntimeError(
                f"Native sequence gap: {self.next_sequence} -> {sequence}"
            )
        self.next_sequence += 1
        return {
            "received_ns": time.time_ns(),
            "sequence": sequence,
            "connection": self.connection,
            "publisher_epoch": self.epoch,
            "publisher_incarnation": self.incarnation.hex()
            if self.incarnation
            else None,
            "delivery": delivery,
            "frames_base64": [base64.b64encode(frame).decode() for frame in frames],
            "payload": msgspec.to_builtins(msgspec.msgpack.decode(frames[-1])),
        }

    def _replay(self, start, running):
        nonce = READY_PREFIX + uuid.uuid4().bytes if self.handshake else None
        dealer = self.context.socket(zmq.DEALER)
        dealer.setsockopt(zmq.LINGER, 0)
        dealer.connect(self.replay_endpoint)
        deadline = time.monotonic() + 5.0
        expected = start
        challenged = not self.handshake
        subscribed = False
        try:
            dealer.send_multipart(
                [b"", start.to_bytes(8, "big")] + ([nonce] if nonce else [])
            )
            while running():
                self._connections()
                if time.monotonic() >= deadline:
                    raise RuntimeError("Native replay timed out")
                if not dealer.poll(100):
                    continue
                frames = dealer.recv_multipart()
                if len(frames) != 4 or frames[0] != b"" or len(frames[2]) != 8:
                    raise RuntimeError("Malformed native replay framing")
                if not challenged:
                    if frames[:3] != [b"", nonce, READY_SEQ] or len(frames[3]) != 16:
                        raise RuntimeError("Invalid native replay readiness challenge")
                    if self.incarnation != frames[3]:
                        self.incarnation = frames[3]
                        self.epoch += 1
                        self.next_sequence = 0
                    self.socket.setsockopt(zmq.SUBSCRIBE, nonce)
                    challenged = subscribed = True
                    continue
                if frames[2] == END_SEQ:
                    if frames[1] or frames[3]:
                        raise RuntimeError("Malformed native replay END")
                    return expected
                sequence = int.from_bytes(frames[2], "big")
                if sequence != expected:
                    raise RuntimeError(
                        f"Native replay truncated: {expected} -> {sequence}"
                    )
                expected += 1
                record = self._record(frames[1:], "replay")
                if record is not None:
                    yield record
        finally:
            if subscribed:
                self.socket.setsockopt(zmq.UNSUBSCRIBE, nonce)
            dealer.close(linger=0)

    def records(self, running):
        needs_replay = bool(self.replay_endpoint)
        while running():
            try:
                self._connections()
                if not self.connected:
                    self.monitor.poll(100)
                    continue
                if needs_replay:
                    yield from self._replay(0, running)
                    needs_replay = False
                if not self.socket.poll(100):
                    continue
                # Check disconnect before reading potentially queued old-source data.
                self._connections()
                frames = self.socket.recv_multipart()
                if not self.replay_endpoint:
                    # Preserve the original unsequenced observer mode for vLLM.
                    self.next_sequence = int.from_bytes(frames[1], "big")
                sequence = int.from_bytes(frames[1], "big")
                if sequence > self.next_sequence and self.replay_endpoint:
                    end = yield from self._replay(self.next_sequence, running)
                    if end is not None and end <= sequence:
                        raise RuntimeError("Native replay did not repair the live gap")
                record = self._record(frames, "live")
                if record is not None:
                    yield record
            except _Reconnect:
                # Recreate SUB too, discarding old-connection queued messages.
                self.close()
                self._open()
                needs_replay = bool(self.replay_endpoint)


def observe(endpoint, output, replay_endpoint=None, replay_handshake=False):
    """Save timestamped native messages until SIGTERM or SIGINT."""
    running = True

    def stop(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with zmq.Context() as context:
        subscriber = NativeSubscriber(
            context, endpoint, replay_endpoint, replay_handshake
        )
        try:
            with open(output, "a", buffering=1) as stream:
                for record in subscriber.records(lambda: running):
                    stream.write(json.dumps(record) + "\n")
        finally:
            subscriber.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-endpoint")
    parser.add_argument("--replay-handshake", action="store_true")
    args = parser.parse_args()
    observe(args.endpoint, args.output, args.replay_endpoint, args.replay_handshake)
