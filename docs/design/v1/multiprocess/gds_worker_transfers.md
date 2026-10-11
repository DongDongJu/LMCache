# Bounded GDS worker transfers

The opt-in LMCache-driven CUDA path uses a small persistent pool of ordinary
threads calling synchronous cuFile I/O. Initial measurements found that cuFile
stream-async reads were much slower than synchronous reads on a host-staged
compatibility path; Python wrapper overhead was small. This change tests that
specific alternative without replacing the storage-manager or backend framework.
Native direct-DMA performance requires separate measurement.

## Ownership and bounds

`GDSContext` owns one `GDSWorkerPool` for its slab. `io_workers=0` retains the
stream-ordered path. A handle advertises the optional `supports_sync_io` capability;
other backends need no new implementation. The pool admits at most `io_workers`
operations and `max_inflight_bytes` useful bytes across submitting GPU contexts.
An operation comprises one object plus its gather/scatter and sequential region
I/Os; these counters are not hardware queue-depth measurements.

The CUDA cache context already allocates four staging entries, each sized from
the sum of its object-group layouts. Worker transfers reuse those entries and
allocate no additional staging. The handler holds a context lease until every
submitted task has completed, so unregister/reaping cannot release mappings while
a worker can still access them. Closing the slab drains the pool before native
buffer, file-handle or driver cleanup. Every worker selects the task's CUDA device.
Before allocating staging, its actual size must fit `max_inflight_bytes` when
workers are enabled. This allocation bound applies separately to each registered
CUDA cache context, while the active-byte admission bound is process-wide.

## Slot dependencies

One recorded input event follows the original context stream's producer wait and
block-ID staging. Each worker waits for that event before using the engine data.
The event is reused only while holding the context lease, after every waiter
from the previous generation has completed; no engine consumes this input event.

```
retrieve: free -> input ready -> read -> exact byte validation
          -> scatter -> scatter complete -> free
store:    free -> input ready -> gather -> gather complete
          -> write -> exact byte validation -> free
```

Each slot has a persistent stream. The original object index still determines
block-ID slicing, sliding-window drops and prefix protection; only the staging
slot index changes. Each kernel handles one object, preserving the store layout
constraint. The issuing ordinary worker waits only for its own stream. No CUDA
API runs in a host callback, and no per-transfer device synchronization is added.
Retrieval requires host block IDs proving unique destinations in each retained
kernel-group range. Repeated destination IDs, or missing host IDs, use the
checked stream-ordered fallback so concurrent scatter cannot change aliasing
semantics. Skipped prefix objects do not participate in owner or budget checks.

The existing MP blocking affinity handler waits for the bounded set of futures.
The transport event loop does not wait. Once all workers succeed, existing code
records and exports its normal IPC event and schedules owner-aware publication.
The consumer therefore never receives an unrecorded event as a future promise.
This deliberately trades earlier RPC acknowledgement for an exact success result.

## Completion and failure

Synchronous cuFile returns signed byte counts. Each segment must return its exact
length; negative, zero, short or excess counts fail the operation. No retry or
extra read across an object boundary is introduced. Reads are checked before
scatter; stores are checked before publication. Submissions retain the registered
base pointer plus region-relative offset. Splitting follows the existing 16 MiB
registration geometry, which is not a universal cuFile registration limit.
Unaligned lengths/offsets are passed through without claiming an aligned fast
path; stream flags remain `0x7`, and physical/logical sizes are unchanged.

The stream-ordered fallback also checks deferred `Submission` results after an
issuing-stream completion and before scatter/publication. Completed checkpoint
batches cannot silently discard errors. The wait runs on the affinity handler,
never on the transport event loop or a CUDA callback.

On submission or worker failure, the handler stops submitting, cancels pending
work where possible, and drains running work. Each worker drains its own kernel
work in `finally`. Existing store logic commits no reserved objects on failure;
retrieve returns failure and releases retained owner locks through its existing
completion path. A successful owner cannot turn another owner's failure into
overall success. Failed reservations retain the existing TTL behavior.

Engine-driven transfers and legacy in-process/L2 GDS paths do not opt into this
scheduler. No fsync or new durability guarantee is added.
