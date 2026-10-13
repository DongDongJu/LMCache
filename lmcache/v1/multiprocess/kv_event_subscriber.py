# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral reception of the MP native-binding event stream."""

# Standard
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_KIND_REMOVED,
    KV_EVENT_KIND_STORED,
    KV_EVENT_MEDIUM_CPU,
    KVEventBatch,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.custom_types import KVEventProfile
    from lmcache.v1.multiprocess.futures import MessagingStream
    from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)


@dataclass(frozen=True)
class KVEventReset:
    """Withdraw the engine's CPU ledger before applying subsequent batches.

    A non-None ``error`` is terminal: the engine must report the failure and
    must not switch to a different source. GPU advertisements are unaffected.
    """

    reason: str
    error: str | None = None


class MPKVEventSubscriber:
    """Borrow a request client and own only one stream and receiver thread.

    Registration, heartbeat, election, and native publication belong to the
    engine. Call ``start`` after registration, then ``drain`` on the engine's
    publication thread. Every reconnect withdraws CPU advertisements and
    replays from cursor zero. Warm entries outside retained history may remain
    undiscovered; this class never reconstructs them from request history.

    Args:
        req_client: Borrowed, already connected MP client.
        instance_id: Elected subscriber's registered worker identity.
        model_name: Registered cache namespace.
        world_size: Number of KV ranks owned by the private MP.
        profile: Immutable native identity profile negotiated with MP.
        max_pending_events: Maximum buffered records, batches, and resets.

    Raises:
        ValueError: If the buffer bound cannot hold a reset and a batch.
    """

    def __init__(
        self,
        req_client: "RequestClient",
        instance_id: int,
        model_name: str,
        *,
        world_size: int,
        profile: "KVEventProfile",
        max_pending_events: int = 65536,
    ) -> None:
        if max_pending_events < 2:
            raise ValueError("max_pending_events must be at least two")
        self._client = req_client
        self._instance_id = instance_id
        self._model_name = model_name
        self._world_size = world_size
        self._profile = profile
        self._capacity = max_pending_events
        self._batch_size = min(1024, max_pending_events - 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._terminal_error: str | None = None
        self._thread: threading.Thread | None = None
        self._stream: MessagingStream[KVEventBatch] | None = None
        self._pending: list[KVEventBatch | KVEventReset] = []
        self._pending_size = 0
        self._incarnation: int | None = None
        self._cursor = 0
        self._received = 0
        self._drained = 0
        self._resets = 0

    def start(self, *, timeout: float | None = None) -> None:
        """Start reception once; repeated calls before close are harmless.

        Args:
            timeout: When provided, wait for an enabled, compatible first batch
                before engine readiness. None starts reception asynchronously.

        Raises:
            RuntimeError: If closed, or negotiation fails during a bounded start.
            LMCacheTimeoutError: If negotiation does not complete before timeout.
        """
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber is closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._receive, name="lmcache-kv-events", daemon=True
                )
                self._thread.start()
        if timeout is not None:
            if not self._ready.wait(timeout):
                self.close()
                raise LMCacheTimeoutError("MP KV event profile negotiation timed out")
            if self._terminal_error is not None:
                raise RuntimeError(self._terminal_error)
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber closed during negotiation")

    def drain(self) -> list[KVEventBatch | KVEventReset]:
        """Return pending batches and invalidations in source order.

        A reset always precedes data recovered after loss. The caller owns its
        native CPU ledger and must process resets even when no data follows.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            self._pending_size = 0
            self._drained += sum(
                len(item.events) for item in pending if isinstance(item, KVEventBatch)
            )
            if pending:
                logger.debug(
                    "MP_KV_SUBSCRIBER drained=%d received=%d buffer_depth=0",
                    self._drained,
                    self._received,
                )
            return pending

    def close(self) -> None:
        """Idempotently cancel and join reception, keeping the client alive."""
        self._stop.set()
        self._ready.set()
        with self._lock:
            stream, thread = self._stream, self._thread
        if stream is not None:
            stream.close()
        if thread is not None:
            thread.join()

    def report_status(self) -> dict[str, int]:
        """Return bounded-buffer and reception counters without native IDs."""
        with self._lock:
            return {
                "received_events": self._received,
                "drained_events": self._drained,
                "buffer_depth": self._pending_size,
                "resets": self._resets,
            }

    def _enqueue(self, item: KVEventBatch | KVEventReset) -> None:
        """Keep an invalidation path even when the data buffer overflows."""
        size = max(1, len(item.events)) if isinstance(item, KVEventBatch) else 1
        with self._lock:
            if self._pending_size + size > self._capacity:
                self._pending = [KVEventReset("queue_overflow")]
                self._pending_size = 1
                self._resets += 1
            self._pending.append(item)
            self._pending_size += size
            if isinstance(item, KVEventReset):
                self._resets += 1
                logger.debug("MP_KV_SUBSCRIBER reset=%s", item.reason)
            else:
                self._received += len(item.events)
            logger.debug(
                "MP_KV_SUBSCRIBER buffer_depth=%d received=%d resets=%d",
                self._pending_size,
                self._received,
                self._resets,
            )

    def _accept(self, batch: KVEventBatch) -> None:
        """Validate source continuity; numeric gaps are valid filtered streams."""
        if not batch.enabled:
            raise ValueError("MP native KV event stream is disabled")
        if len(batch.events) > self._batch_size:
            raise ValueError("MP event batch exceeds the requested limit")
        if batch.next_cursor < 0:
            raise ValueError("MP event cursor must be nonnegative")
        # An incarnation can reuse sequence numbers from a previous server.
        if self._incarnation != batch.incarnation:
            if self._incarnation is not None:
                self._enqueue(KVEventReset("server_restart"))
            self._incarnation = batch.incarnation
            self._cursor = 0
        if batch.lost:
            self._enqueue(KVEventReset("events_lost"))
            self._cursor = 0
        if batch.next_cursor < self._cursor:
            raise ValueError("MP event cursor regressed without a loss marker")
        previous = 0
        for record in batch.events:
            if record.seq <= previous or record.seq > batch.next_cursor:
                raise ValueError("MP event records are not ordered within the cursor")
            if (
                record.profile_id != self._profile.profile_id
                or record.block_size != self._profile.block_size
                or record.model_name != self._model_name
                or record.world_size != self._world_size
                or record.cache_salt
                or record.medium != KV_EVENT_MEDIUM_CPU
                or record.kind not in (KV_EVENT_KIND_STORED, KV_EVENT_KIND_REMOVED)
            ):
                raise ValueError(
                    "MP event does not match the subscribed native profile"
                )
            previous = record.seq
        records = [record for record in batch.events if record.seq > self._cursor]
        self._cursor = batch.next_cursor
        logger.debug(
            "MP_KV_SUBSCRIBER received_ns=%d incarnation=%d cursor=%d events=%d",
            time.time_ns(),
            batch.incarnation,
            batch.next_cursor,
            len(records),
        )
        if records:
            self._enqueue(replace(batch, events=records))

    def _receive(self) -> None:
        """Restore the same stream with bounded retry, or expose terminal failure."""
        delay = 0.1
        while not self._stop.is_set():
            stream = None
            try:
                stream = self._client.subscribe_kv_events(
                    self._instance_id,
                    self._model_name,
                    0,
                    self._batch_size,
                    self._world_size,
                    self._profile,
                )
                with self._lock:
                    self._stream = stream
                if self._stop.is_set():
                    return
                self._cursor = 0
                for batch in stream:
                    if self._stop.is_set():
                        return
                    self._accept(batch)
                    self._ready.set()
                    delay = 0.1
                raise ConnectionError("MP KV event subscription ended")
            except (ConnectionError, TimeoutError) as exc:
                if not self._stop.is_set():
                    logger.warning("Reconnecting MP KV event stream: %s", exc)
                    self._enqueue(KVEventReset("stream_closed"))
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("MP KV event stream failed")
                    self._enqueue(KVEventReset("protocol_error", str(exc)))
                    self._terminal_error = str(exc)
                    self._ready.set()
                return
            finally:
                if stream is not None:
                    stream.close()
                with self._lock:
                    self._stream = None
            if self._stop.wait(delay):
                return
            delay = min(delay * 2, 5.0)
