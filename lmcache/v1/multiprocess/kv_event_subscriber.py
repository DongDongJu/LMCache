# SPDX-License-Identifier: Apache-2.0
"""Engine-neutral reception of the MP CPU KV-event stream."""

# Standard
from collections.abc import Callable
from dataclasses import dataclass, replace
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEvent, CacheRemoveEvent, CacheStoreEvent
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.custom_types import (
    KV_EVENT_KIND_REMOVED, KV_EVENT_KIND_STORED, KV_EVENT_MEDIUM_CPU,
    KVEventBatch, KVEventRecord, KVEventProfile, NativeBlockHash,
)
from lmcache.v1.multiprocess.futures import MessagingStream
from lmcache.v1.multiprocess.transport.base import RequestClient

logger = init_logger(__name__)

_BATCH_SIZE = 1024
_RECONNECT_DELAY = 1.0


@dataclass(frozen=True)
class KVEventStreamClosed:
    """Invalidate the consumer's CPU placements after a stream failure.

    ``error=None`` signals a transient disconnect followed by replay from zero.
    A non-None error is terminal; the consumer owns the failure policy.
    """

    error: str | None = None


def apply_kv_event_record(
    record: KVEventRecord, announced: set[NativeBlockHash]
) -> CacheEvent | None:
    """Convert a validated record on the engine's publication thread.

    Args:
        record: Native CPU record validated by the subscriber.
        announced: Caller-owned ledger, updated in place.

    Returns:
        A cache event, or None for an already-applied transition.

    Raises:
        ValueError: A STORE overlaps only part of an advertised chunk.
    """
    if record.kind == KV_EVENT_KIND_REMOVED:
        hashes = [h for h in record.block_hashes if h in announced]
        announced.difference_update(hashes)
        return CacheRemoveEvent(hashes, "CPU") if hashes else None
    overlap = announced.intersection(record.block_hashes)
    if len(overlap) == len(record.block_hashes):
        return None
    if overlap:
        raise ValueError("Native STORE partially overlaps an advertised chunk")
    announced.update(record.block_hashes)
    return CacheStoreEvent(
        block_hashes=list(record.block_hashes),
        parent_block_hash=record.parent_block_hash,
        token_ids=list(record.token_ids),
        block_size=record.block_size,
        lora_id=None,
        medium="CPU",
        lora_name=None,
    )


class KVEventLedger:
    """Track chunk contributions to native CPU placements on the engine thread.

    One ledger belongs to one subscribed model/profile. A chunk is identified by
    its ordered native hashes; sibling chunks can share constituent blocks.
    Only the first contribution publishes a block and the last removal withdraws
    it. Replayed transitions for an already-known chunk are idempotent.
    """

    def __init__(self) -> None:
        self._chunks: set[tuple[NativeBlockHash, ...]] = set()
        self._block_refcounts: dict[NativeBlockHash, int] = {}
        self._descriptors: dict[
            NativeBlockHash, tuple[NativeBlockHash | None, tuple[int, ...], int]
        ] = {}

    def apply(self, record: KVEventRecord) -> list[CacheEvent]:
        """Apply a validated chunk transition on the engine's publication thread.

        Args:
            record: Complete native chunk descriptor validated by the subscriber.

        Returns:
            CPU placement changes, or an empty list for a duplicate or an unknown
            removal. Stored runs contain only newly advertised blocks, with their
            corresponding original tokens and preceding native parent hash.

        Raises:
            ValueError: A STORE is incomplete, repeats a native identity, or
                conflicts with a live block's parent/tokens. Validation finishes
                before any contribution changes.
        """
        hashes = record.block_hashes
        if record.kind == KV_EVENT_KIND_REMOVED:
            if hashes not in self._chunks:
                return []
            self._chunks.remove(hashes)
            removed = []
            for block_hash in hashes:
                count = self._block_refcounts[block_hash] - 1
                if count:
                    self._block_refcounts[block_hash] = count
                else:
                    del self._block_refcounts[block_hash]
                    del self._descriptors[block_hash]
                    removed.append(block_hash)
            return (
                [CacheRemoveEvent(removed, "CPU")]
                if removed
                else []
            )
        if (
            record.block_size <= 0 or not hashes
            or len(set(hashes)) != len(hashes)
            or len(record.token_ids) != len(hashes) * record.block_size
        ):
            raise ValueError("native STORE does not describe complete distinct blocks")
        descriptors = {}
        for index, block_hash in enumerate(hashes):
            descriptor = (
                hashes[index - 1] if index else record.parent_block_hash,
                record.token_ids[index * record.block_size : (index + 1) * record.block_size],
                record.block_size,
            )
            existing = self._descriptors.get(block_hash)
            if existing is not None and existing != descriptor:
                raise ValueError("conflicting native block parent or tokens")
            descriptors[block_hash] = descriptor
        if hashes in self._chunks:
            return []
        self._descriptors.update(descriptors)
        self._chunks.add(hashes)
        ranges: list[tuple[int, int]] = []
        for index, block_hash in enumerate(hashes):
            count = self._block_refcounts.get(block_hash, 0)
            self._block_refcounts[block_hash] = count + 1
            if count:
                continue
            if ranges and ranges[-1][1] == index:
                ranges[-1] = (ranges[-1][0], index + 1)
            else:
                ranges.append((index, index + 1))
        return [
            CacheStoreEvent(
                block_hashes=list(hashes[start:end]),
                parent_block_hash=hashes[start - 1]
                if start
                else record.parent_block_hash,
                token_ids=list(
                    record.token_ids[
                        start * record.block_size : end * record.block_size
                    ]
                ),
                block_size=record.block_size,
                lora_id=None,
                medium="CPU",
                lora_name=None,
            )
            for start, end in ranges
        ]

    def reset(self) -> list[CacheEvent]:
        """Forget chunk ownership and withdraw each advertised CPU block once.

        Returns:
            One CPU removal event, or an empty list when no blocks were advertised.
            Subsequent replay starts with an empty ledger.
        """
        events: list[CacheEvent] = (
            [CacheRemoveEvent(list(self._block_refcounts), "CPU")]
            if self._block_refcounts
            else []
        )
        self._chunks.clear()
        self._block_refcounts.clear()
        self._descriptors.clear()
        return events


class MPKVEventSubscriber:
    """Borrow a request client and own one event stream and receiver thread.

    The owning adapter controls heartbeat, client lifetime, and event
    conversion. Batches and stream failures retain their arrival order. The
    consumer must drain updates on notification; no second consumer may drain
    concurrently. Disabled batches and terminal failures stop reception.

    Args:
        req_client: Borrowed client, kept alive until after ``close`` returns.
        instance_id: Owner's identity used for subscription liveness.
        model_name: Model namespace used to filter the server's event log.
        world_size: Complete KV rank count registered with the server.
        profile: Native identity and block-size contract checked for every record.
        on_update: Receiver-thread callback invoked outside the subscriber lock.
            It must drain updates before returning and must not call ``close``.
            Stream-closure updates must be handled without raising.
            This preserves asynchronous adapter metrics and failure handling.
    """

    def __init__(
        self,
        req_client: RequestClient,
        instance_id: int,
        model_name: str,
        on_update: Callable[["MPKVEventSubscriber"], None],
        *,
        world_size: int,
        profile: KVEventProfile,
    ) -> None:
        self._client = req_client
        self._instance_id = instance_id
        self._model_name = model_name
        self._world_size = world_size
        self._profile = profile
        self._cursor = 0
        self._incarnation: int | None = None
        self._ready = threading.Event()
        self._terminal_error: str | None = None
        self._on_update = on_update
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream: MessagingStream[KVEventBatch] | None = None
        self._pending: list[KVEventBatch | KVEventStreamClosed] = []
        self._thread = threading.Thread(
            target=self._receive, name="lmcache-kv-events", daemon=True
        )

    def start(self, *, timeout: float | None = None) -> None:
        """Start reception once; repeated calls do not restart a stopped stream.

        Args:
            timeout: If provided, wait for the first valid native batch. Scheduler
                construction starts asynchronously before worker registration.

        Raises:
            RuntimeError: If closed or native stream validation fails.
            LMCacheTimeoutError: If bounded native negotiation does not finish.
        """
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber is closed")
            if self._thread.ident is None:
                self._thread.start()
        if timeout is not None:
            if not self._ready.wait(timeout):
                self.close()
                raise LMCacheTimeoutError("MP KV event profile negotiation timed out")
            if self._terminal_error is not None:
                raise RuntimeError(self._terminal_error)
            if self._stop.is_set():
                raise RuntimeError("KV event subscriber closed during negotiation")

    def drain(self) -> list[KVEventBatch | KVEventStreamClosed]:
        """Remove and return pending batches and stream failures in source order.

        Call from ``on_update``. The consumer interprets batch incarnation/loss
        fields and applies its event-conversion and failure policies.

        Returns:
            Updates received since the previous drain, or an empty list.
        """
        with self._lock:
            pending, self._pending = self._pending, []
            return pending

    def close(self) -> None:
        """Cancel and join reception without closing the borrowed request client.

        Idempotent, including before ``start``. Call from the owning engine
        thread without holding locks needed by ``on_update``. An in-flight
        subscription call must return before its stream can be cancelled.
        """
        with self._lock:
            self._stop.set()
            self._ready.set()
            stream = self._stream
            started = self._thread.ident is not None
        if stream is not None:
            stream.close()
        if started:
            self._thread.join()

    def _publish(self, update: KVEventBatch | KVEventStreamClosed) -> None:
        with self._lock:
            if self._stop.is_set():
                return
            self._pending.append(update)
        self._on_update(self)

    def _validate_batch(self, batch: KVEventBatch) -> KVEventBatch:
        """Validate source continuity; numeric gaps are valid filtered streams."""
        if not batch.enabled:
            raise ValueError("MP native KV event stream is disabled")
        if len(batch.events) > _BATCH_SIZE:
            raise ValueError("MP event batch exceeds the requested limit")
        if batch.next_cursor < 0:
            raise ValueError("MP event cursor must be nonnegative")
        # An incarnation can reuse sequence numbers from a previous server.
        if self._incarnation != batch.incarnation:
            self._incarnation = batch.incarnation
            self._cursor = 0
        if batch.lost:
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
        return replace(batch, events=records)

    def _receive(self) -> None:
        while not self._stop.is_set():
            stream = None
            try:
                stream = self._client.subscribe_kv_events(
                    self._instance_id, self._model_name, 0, _BATCH_SIZE,
                    self._world_size, self._profile,
                )
                with self._lock:
                    if self._stop.is_set():
                        return
                    self._stream = stream
                self._cursor = 0
                for batch in stream:
                    if self._stop.is_set():
                        return
                    self._publish(self._validate_batch(batch))
                    self._ready.set()
                raise ConnectionError("KV event subscription ended")
            except ConnectionError as exc:
                if not self._stop.is_set():
                    logger.warning("Reconnecting KV event subscription: %s", exc)
                    self._publish(KVEventStreamClosed())
            except Exception as exc:
                if not self._stop.is_set():
                    logger.exception("KV event subscription failed")
                    self._terminal_error = str(exc)
                    self._ready.set()
                    self._publish(KVEventStreamClosed(error=str(exc)))
                return
            finally:
                if stream is not None:
                    stream.close()
                with self._lock:
                    self._stream = None
            if self._stop.wait(_RECONNECT_DELAY):
                return
