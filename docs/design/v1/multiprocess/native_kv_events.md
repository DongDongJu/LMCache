# Native CPU KV events

The `kv_event_native_bindings_v1` capability requires engine-prepared content
identities instead of reconstructed storage hashes. Enabling events
requires this capability and a compatible immutable `KVEventProfile`; it never
selects local store announcements or hash reconstruction on failure.

`IPCCacheServerKey` preserves packed `token_bytes` and carries optional
`event_profile` and `event_bindings` metadata through STORE. A binding contains immutable native hashes,
parent, original token IDs, native block size and profile ID. Its tokens cover
exactly one complete LMCache chunk, including all C/B native subblocks. The
profile ID names the engine identity semantics, algorithm and seed contract.
MessagePack and gRPC preserve bytes and integer identities without conversions
between signed and unsigned representations. gRPC wraps each native identity in
a MessagePack scalar; its legacy LM chunk hash fields are reserved.

## Registration and commitment

Every worker supplies the optional profile in `register_kv_cache`, including
non-subscriber TP ranks and heartbeat re-registration. The actual candidate
layout is validated before changing shared registration state. The scheduler
subscriber also supplies model name, KV world size and profile in
`subscribe_kv_events`. Subscriptions reaching a replacement MP before its worker
registrations receive a transient error and retry the same source. The registered
layout supplies every required rank and object group. The supported profile is
dense full attention, TP1/TP2 and C divisible by B;
salted keys, L2, GDS and incompatible profiles fail explicitly. Private
CPU-addressable Device-DAX uses the same L1 commitment notifications; the DRAM
acceptance matrix does not establish Device-DAX hardware validation. Engine
connectors reject unsupported model and topology configurations at startup.
Native bindings use existing vLLM connector hooks; request-free publication also
needs vLLM's idle connector drain. Requests with
salted, LoRA, multimodal, embedding or session namespaces retain ordinary caching
without native event bindings. An absent hash source is distinct from a supported
request's empty hash list, which still must cover every advertised chunk.
SGLang's companion integration rejects unsupported namespaces before admission.

The event module keys bindings by `(model, world_size, cache_salt, chunk_hash)`.
`bind_store` validates conflicts synchronously. The existing EventBus joins those
bindings with `L1_WRITE_FINISHED` and `L1_WRITE_FINISHED_AND_READ_RESERVED` records.
A set of unique configured L1 tags identifies readable copies of each ObjectKey.
All required rank/group objects must have a committed copy before one Stored is
emitted. Losing the last copy of any required object emits one Removed with the
exact earlier native identities. Duplicate notifications and redundant copies do
not change this transition rule.

Bindings are retained by actual live objects and in-flight transfer IDs, not
request lifetime or a fixed LRU. A completion callback first calls
`StorageManager.finish_write` with the reserved per-ObjectKey owner IDs, then
enqueues the pending-binding cleanup marker on the
same EventBus as the resulting L1 notifications. GPU completion and
`MP_STORE_END` alone cannot prove that the host completion dispatcher has
finished. Failed writes release pending metadata without publishing a store;
partial committed chunks retain their binding while any required object lives.

## Replay and loss

Records own immutable descriptors, so cleanup or a later re-store cannot change
retained history. Records have monotonically increasing sequence IDs and a server
incarnation. Numeric gaps in a model-filtered stream are valid; `next_cursor`
tracks the global log. Log truncation reports explicit loss.

The module uses `CommittedKVResidency` for rank/group/copy completeness. On
EventBus loss it clears that projection, pending bindings, live bindings and
replay history, then advances the replay floor. The merged loss barrier discards
queued updates through an observed empty boundary, including concurrent refill
during loss notification. A later binding cannot resurrect pre-loss owners;
newly observed complete writes are required. This is conservative recovery, not
an authoritative snapshot. Warm objects whose records have fallen out of replay
may remain undiscovered until re-established with sufficient fresh evidence.
STORE binding rejects active recovery, including an unobserved drop count,
with a retryable `ConnectionError` before adding pending metadata or reserving
write owners. Only the event consumer ends the discard barrier; an RPC thread
cannot infer that boundary from queue depth while a popped event is still in flight.

The shared subscriber borrows the client and owns its stream/thread. Its callback
drains raw updates immediately, with no second raw consumer. The vLLM scheduler
owns a native contribution ledger and bounded converted-event queue. The ledger
validates parent/token descriptors before mutation, announces shared native
blocks once, and removes them only after the final chunk contribution is gone.
Reset/replay and queue overflow emit CPU-scoped removals; GPU placements are
unaffected. SGLang queues bounded raw updates for its scheduler-owned ledger.

ZMQ uses its existing `ZMQ_ERROR_OPERATION` and `ZmqErrorResponse` envelope for
validation/transient handler failures. All three gRPC modes use
`INVALID_ARGUMENT`/`UNAVAILABLE`. Clients receive the canonical `MPRequestError`
or `MPServerUnavailableError`, so incompatible registration fails explicitly
while registration recovery can retry. Invalid handler output remains an
application failure. Mixed client/server wire versions are unsupported; payload
arity is exact and missing fields are not filled by a compatibility shim.
