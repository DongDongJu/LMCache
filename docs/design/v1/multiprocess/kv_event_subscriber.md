# MP scheduler KV event subscription

When vLLM KV events are enabled, `LMCacheMPSchedulerAdapter` owns one
`MPKVEventSubscriber` for the replica's private MP server. `LMCacheMPConnector.take_events()`
drains the scheduler directly and converts its CPU events to vLLM events.
Workers have no event subscription, event buffer, local store fallback, or
worker-metadata event carry. Lazy-offload store completion metadata is unchanged.

Records carry the native block hashes, original tokens and profile negotiated
with the MP server. Committed residency remains the server module's responsibility.
`KVEventLedger` counts chunk contributions to shared native blocks and validates
conflicting parent/token descriptors before changing state. Publication while
idle additionally requires vLLM to invoke `take_events()` while idle.

## Ownership and delivery

| Owner | State and responsibility |
| --- | --- |
| `MPKVEventSubscriber` | Borrowed client, owned stream/thread, native profile/cursor validation, stop signal, pending raw updates |
| `LMCacheMPSchedulerAdapter` | Client lifetime, heartbeat, incarnation, native contribution ledger, bounded ordered event buffer, metrics |
| `LMCacheMPConnector` | Direct scheduler drain and vLLM event wrapping |
| vLLM | Drain opportunities and router-facing event publisher |

1. Scheduler construction creates every subscriber and starts heartbeats and
   reception. No worker inference, lookup, or first drain is required.
2. Each subscription uses `(instance_id, model_name, 0, 1024, world_size, profile)`. The scheduler
   generates its own UUID, masked to a nonnegative 63-bit integer to fit the
   existing signed-int64 RPC field. Every server heartbeat uses that identity.
   Two schedulers in one process have separate identities; worker registration
   and worker shutdown do not own the scheduler subscription.
3. A receiver validates the native profile and cursor, filters duplicate records,
   queues the batch and invokes the adapter callback.
   The callback drains and converts it under the scheduler's event lock.
4. `get_kv_events()` swaps out the converted buffer under that same lock.
   `take_events()` wraps the result without a worker output hop or TP merge.

The supported native profile requires one private server per replica. Its
incarnation and CPU contributions are independent of every other engine's state.
Ordering is preserved within the stream and each batch. Losing one of two chunks
sharing native prefix blocks removes only blocks whose final contribution is gone.

The notification callback is the sole raw-update consumer. It runs outside the
subscriber lock, drains before returning, and must never call `close()` on its
own subscriber. The receiver waits for conversion, so it cannot build a second
raw backlog while conversion is blocked. Buffer and generated metrics update
without a model step; drain metrics update when the scheduler consumes events.
The existing metric names and label keys remain; `worker_id` now contains the
scheduler subscription identity rather than an elected worker rank.

## Lifecycle and failures

`MPKVEventSubscriber` remains engine-neutral and does not own its request client.
`start()` starts once; terminal receivers are not restarted. An optional timeout
lets a registered caller wait for initial native validation; vLLM starts
asynchronously because its workers may still be registering. `drain()` returns
pending raw batches and stream failures in order. `close()` cancels and joins
reception and is idempotent, including before startup. An in-flight subscription
call must return before it can be cancelled; no connection timeout is added.

Scheduler shutdown closes and joins subscribers without holding the event
lock, then stops heartbeats, then closes clients. An in-flight callback can
finish before its borrowed client closes. With events disabled there are no
subscriptions; heartbeat still starts on the first lookup with no instance ID.

| Input | Subscriber action | Scheduler action |
| --- | --- | --- |
| Enabled batch | Validate profile/cursor and deliver unseen records | Interpret incarnation/loss and apply CPU contributions |
| `enabled=False` or invalid native descriptor | Deliver terminal closure and stop | Withdraw CPU placements; no local store fallback |
| `ConnectionError` or stream end | Deliver closure, wait one interruptible second, resubscribe from zero | Withdraw that server's placements before replay |
| Other subscription/receive error | Log, deliver terminal closure, stop | Withdraw that server's placements; no alternate event source |
| Owner closes subscriber | Cancel/join with no synthetic failure | Stop heartbeat, then close client |

Model-filtered sequence gaps are allowed. The first batch may be trimmed;
subsequent loss or incarnation changes invalidate known CPU hashes. Replaying
from zero retains the existing bounded-history limit. Invalidation emits CPU
`BlockRemoved`, never `AllBlocksCleared`, so GPU placements remain intact.
The converted vLLM queue is bounded to 65,536 events. Overflow discards queued
history, withdraws the union of pending removals and live CPU contributions, and
retains the incoming valid transition.
An incoming removal is retained even after reset, so previously published blocks
cannot remain stale. SGLang uses a bounded raw-update queue and drains its ledger
from the publisher thread. Overflow recovery is conservative rather than a full
residency snapshot.

## Validation

`tests/v1/multiprocess/test_kv_event_subscriber.py` covers the engine-neutral
receiver, cancellation, callback ownership, reconnect delay, and borrowed client
lifetime and native validation. `tests/v1/multiprocess/test_native_kv_event_ledger.py`
covers shared prefixes, atomic descriptor rejection and CPU-only reset/replay.
`tests/v1/test_vllm_mp_adapter.py` covers scheduler startup and identity,
ordered conversion and metrics, bounded-buffer resets, and failures without local
store fallback. `tests/v1/test_mp_connector_kv_roles.py` checks direct and real
`MultiConnector` drains without worker execution and preserves lazy-offload
completion behavior. Server completeness and binding lifetime have separate
native-event and worker-liveness regressions.
