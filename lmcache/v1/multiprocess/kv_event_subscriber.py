# SPDX-License-Identifier: Apache-2.0
"""Common receiver lifecycle for the MP KV-event stream."""

# Standard
from collections.abc import Callable
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.custom_types import KVEventBatch
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


class MPKVEventSubscriber:
    """Borrow a request client and own only the event stream and receiver thread.

    This extraction preserves existing callback delivery on the receiver thread.
    Callbacks own event translation, publication buffering, and source policy.
    They must synchronize their own state. Registration and heartbeat remain
    with the engine that owns ``req_client``.

    Args:
        req_client: Borrowed MP request client, never closed by this object.
        instance_id: Elected worker instance ID.
        model_name: Model namespace to subscribe to.
        on_batch: Apply a batch; return False to stop reception.
        on_disconnect: Withdraw stale state before reconnecting from cursor zero.
        on_error: Handle a terminal protocol error under the engine's policy.
        batch_size: Maximum event records requested in each response.
    """

    def __init__(
        self,
        req_client: RequestClient,
        instance_id: int,
        model_name: str,
        on_batch: Callable[[KVEventBatch], bool],
        on_disconnect: Callable[[], None],
        on_error: Callable[[str], None],
        *,
        batch_size: int = 1024,
    ) -> None:
        self._client = req_client
        self._instance_id = instance_id
        self._model_name = model_name
        self._on_batch = on_batch
        self._on_disconnect = on_disconnect
        self._on_error = on_error
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream: MessagingStream[KVEventBatch] | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the receiver once; raise RuntimeError if already closed."""
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._receive, daemon=True, name="lmcache-kv-events"
                )
                self._thread.start()

    def close(self) -> None:
        """Cancel and join the owned receiver without closing the borrowed client."""
        self._stop.set()
        with self._lock:
            stream, thread = self._stream, self._thread
        if stream is not None:
            stream.close()
        if thread is not None:
            thread.join()

    def _receive(self) -> None:
        """Reconnect the same push source and preserve callback order."""
        while not self._stop.is_set():
            stream = None
            try:
                stream = self._client.subscribe_kv_events(
                    self._instance_id, self._model_name, 0, self._batch_size
                )
                with self._lock:
                    self._stream = stream
                if self._stop.is_set():
                    return
                for batch in stream:
                    if self._stop.is_set() or not self._on_batch(batch):
                        return
                raise ConnectionError("KV event subscription ended")
            except ConnectionError:
                if not self._stop.is_set():
                    self._on_disconnect()
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("MP KV event subscription failed")
                    self._on_error(str(exc))
                return
            finally:
                if stream is not None:
                    stream.close()
                with self._lock:
                    self._stream = None
