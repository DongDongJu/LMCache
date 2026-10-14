# Native CPU KV events

The `kv_event_native_bindings_v1` capability requires engine-prepared content
identities. It replaces the old chunk-hash stream capability. Enabling events
requires this capability and a compatible immutable `KVEventProfile`; it never
selects local store announcements or hash reconstruction on failure.

`IPCCacheServerKey` carries optional `event_profile` and `event_bindings` metadata
through the existing STORE RPC. A binding contains immutable native hashes,
parent, original token IDs, native block size and profile ID. Its tokens cover
exactly one complete LMCache chunk, including all C/B native subblocks. The
profile ID names the engine identity semantics, algorithm and seed contract.
MessagePack and gRPC preserve bytes and integer identities without conversions
between signed and unsigned representations. gRPC wraps each native identity in
a MessagePack scalar; its legacy LM chunk hash fields are reserved.

## Registration and commitment

Every worker supplies the optional profile in `register_kv_cache`, including
non-subscriber TP ranks and heartbeat re-registration. The actual candidate
layout is validated before changing shared registration state. The elected
subscriber also supplies model name, KV world size and profile in
`subscribe_kv_events`. Subscriptions reaching a replacement MP before its worker
registrations receive a transient error and retry the same source. The registered
layout supplies every required rank and object group. Dense full attention,
TP1/TP2 and C divisible by B are validated;
salted keys, L2, GDS and incompatible profiles fail explicitly. Private
CPU-addressable Device-DAX uses the same L1 commitment notifications; the DRAM
acceptance matrix does not establish Device-DAX hardware validation. Engine
connectors reject unsupported model, topology and request namespaces before
STORE. vLLM uses the companion class-level `validate_kv_event_request` hook in
frontend admission, gated on explicitly enabled KV events. MultiConnector
forwards it to each configured child; no worker connector is constructed there.
Unsupported requests raise `VLLMValidationError` before EngineCore enqueue.

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
`finish_write_by_owner`, then enqueues the pending-binding cleanup marker on the
same EventBus as the resulting L1 notifications. GPU completion and
`MP_STORE_END` alone cannot prove that the host completion dispatcher has
finished. Failed writes release pending metadata without publishing a store;
partial committed chunks retain their binding while any required object lives.

## Replay and loss

Records own immutable descriptors, so cleanup or a later re-store cannot change
retained history. Records have monotonically increasing sequence IDs and a server
incarnation. Numeric gaps in a model-filtered stream are valid; `next_cursor`
tracks the global log. Log truncation reports explicit loss.

EventBus drop notification occurs after its preceding queue has drained. The
module ignores projection updates once it observes an unacknowledged drop, then
clears its rank/copy projection, pending bindings, live bindings and replay log,
then advances the replay floor. A later binding cannot resurrect pre-loss owners;
newly observed complete writes are required. This is conservative recovery, not
an authoritative snapshot. Warm objects whose records have fallen out of replay
may remain undiscovered until re-established with sufficient fresh evidence.

The common subscriber borrows the request client and owns only its stream/thread.
It invalidates CPU advertisements on reconnect/loss and replays the retained log.
Engine publication retains its existing ownership, tracks advertised CPU IDs and
emits CPU-scoped removals; GPU placements are unaffected by CPU invalidation.
