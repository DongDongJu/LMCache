# Shared Device-DAX L1

Several MP servers share one Device-DAX region as an L1. Sharing is an option
of the existing Device-DAX L1, not another L1 implementation: the server keeps
its ordinary `L1Manager`, and a memory orchestrator
([memory_orchestrator](../../memory_orchestrator/memory_orchestrator.md))
owns allocation and object readiness for the region. User configuration,
limits and failure behavior are in `docs/source/mp/shared_l1.rst`.

## Ownership

| Component | Shared-mode responsibility |
| --- | --- |
| `StorageManager` | Builds ordinary `L1Manager`s; owner routing and write order unchanged |
| `L1Manager` | Public L1 API, owner ID, listeners, events, metrics; routes object operations |
| `DevDaxL1MemoryManager` | Maps and GPU-registers the region at the configured offset; views of granted extents |
| `OrchestratedObjectState` | Orchestrator client, this server's write grants and read leases, publish/acquire fences |
| Memory orchestrator | Sole authority for allocation and which objects exist and are readable |

`OrchestratedObjectState.connect()` runs the startup sequence once and releases
everything it acquired if a step fails:

```text
DescribeRegion -> check contract (reset, capacity <= size_gb, alignment)
  -> map + GPU-register region -> RegisterClient -> epoch unchanged
```

## Routing and locking

`l1_mgr_synchronized` sends a shared L1's synchronized methods to
`L1Manager._shared_<name>` before `_lock` is taken: those calls wait on the
orchestrator, which never happens under `_lock` or the allocator lock. The
`_shared_*` methods call `OrchestratedObjectState` and then notify listeners
and publish events themselves. `OrchestratedObjectState` keeps its own lock for
its grant and lease tables and never holds it across an RPC. The mode is fixed
per `L1Manager` by its config; private and shared keys never mix in one
instance, so the private `_objects`, `_staging`, TTL and free paths stay empty.

| Operation | Shared behavior |
| --- | --- |
| `reserve_write` | ReserveWrite; a full region or unreachable orchestrator is `OUT_OF_MEMORY` (overflow), a taken key `KEY_NOT_WRITABLE`, a conflicting layout raises |
| `finish_write` | publish fence over the extents, then FinishWrite; a refused or lost commit is `KEY_IN_WRONG_STATE` |
| `finish_write_and_delete` | AbortWrite; a failed store calls it per captured owner through `abort_write_by_owner` on the copy stream, after its partial copies drain |
| `finish_write_and_reserve_read` | commit, then lease; reports only the steps that succeeded |
| `reserve_read` | ReserveRead, then acquire fence; malformed grants are released and miss |
| `unsafe_read` | views of keys this server holds leases on (the retrieve path) |
| `finish_read` | FinishRead of `read_locks` leases per key |
| `delete` | DeleteObjects: unreadable for every server; `KEY_IS_LOCKED` while still written or when the orchestrator cannot be used; nothing freed even if forced |
| `clear` | DeletePublished: the objects this server's `client_id` wrote; other servers' objects stay |
| `is_key_evictable`, `touch_keys` | never evictable; listeners only |
| usage, status, memcheck | this server's grants and leases; capacity is the region's |
| `close` | CloseClient retires this incarnation's grants and leases, then the region is unregistered and unmapped; callers drain copies first |
| hot-plug | 409 before any state changes |

Leases on one key all cover the same committed object, which stays immutable
until it is deleted, and its bytes are never reused, so releasing any
`read_locks` of them is exact; a lease taken before a delete stays readable. A failed
acquire releases only the leases that call took.

## Visibility

`Visibility("software_fenced")` writes back and invalidates the CPU cache lines
of every extent with CLFLUSHOPT + SFENCE (x86-64 CPUs with CLFLUSHOPT only).
The writer publishes after its D2H copies complete and before FinishWrite; the
reader acquires after ReserveRead and before its H2D copies. `coherent`
performs no cache maintenance. Metadata completion alone never implies payload
completion.

## Integration

The prefetch controller checks a shared object's layout against the reader's
after recording the locks, so a mismatch releases all of them. Shared events carry `shared: True` and no `meta`; cache
events skip them, since the region holds the object rather than the server.
