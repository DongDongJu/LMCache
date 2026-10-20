# Shared L1

## Scope

Several MP servers share one memory region as an L1: a chunk stored by one
server is retrieved by every other server from its own mapping of the region,
without a network copy.

`SharedL1Manager` is the first remote binding of `L1ManagerInterface`
([`l1_manager_interface.md`](../l1_manager_interface.md)). It maps the
region through a *medium* on its host and moves KV bytes between that mapping
and the GPU: `DevDaxMedium` for a Device-DAX/CXL device several hosts attach,
`FileMedium` for a tmpfs or hugetlbfs file several servers on one host share.
Every extent, commit, read lease and release comes from the memory
orchestrator, `lmcache memory`
([`memory_orchestrator.md`](../../memory_orchestrator/memory_orchestrator.md));
the binding never allocates, evicts or deletes on its own.

Limits of this milestone: one shared L1 per server, TP1/2/4/8, `lmcache_driven`
transfer, `noop` eviction. Not here: eviction or deletion of shared objects,
hot-plug of the region, fleet cache events for shared placements.

## Architecture

```text
+----------------------------------+        +----------------------------------+
| Host A                           |        | Host B                           |
| engine workers (TP1/2/4/8)                |        | engine workers (TP1/2/4/8)                |
|     | MP API, KV cache over IPC  |        |     | MP API, KV cache over IPC  |
|     v                            |        |     v                            |
| lmcache server (MP)              |        | lmcache server (MP)              |
|   L1 #1: shared L1 (DAX medium)  |        |   L1 #1: shared L1 (DAX medium)  |
|   L1 #2: DRAM L1 (overflow)      |        |   L1 #2: DRAM L1 (overflow)      |
+-----+---------------------+------+        +------+---------------------+-----+
      |                     |                      |                     |
      |                     |  gRPC: metadata only |                     |
      |                     v                      v                     |
      |       +--------------------------------------------------+       |
      |       | lmcache memory (memory orchestrator)             |       |
      |       | one process per region; keys, extents, epoch,    |       |
      |       | write tokens, read leases; never sees KV bytes   |       |
      |       +--------------------------------------------------+       |
      |                                                                  |
      | KV bytes: D2H on store, H2D on retrieve, through                 |
      | each host's own mapping (never the orchestrator)                 |
      v                                                                  v
+------------------------------------------------------------------------------+
| Shared region, mapped on each host through a medium:                         |
| DevDaxMedium (/dev/daxX.Y, CXL across hosts, software_fenced) or             |
| FileMedium (tmpfs/hugetlbfs file, several servers on one host, coherent)     |
+------------------------------------------------------------------------------+
```

Control goes through the orchestrator; KV bytes never do. The MP server owns
its mapping and DMA registration, copy scheduling, the publish and acquire
fences and cancellation; the embedded L1s keep their allocator, index, locks
and eviction; the MP coordinator stays an optional hint directory. The
operator establishes that every server's path names the same physical region.

## Flows

`reserve_write` is a `ReserveWrite`; `finish_write` publishes the written
extents and sends `FinishWrite`; `reserve_read` is a `ReserveRead` followed by
an acquire fence; `finish_read` a `FinishRead`.

- Duplicate store: one writer gets `WRITE_GRANTED`, the other `BUSY_WRITING`
  and skips; a later store of a committed key gets `EXISTS_VALID` and copies
  nothing. One resident copy per key.
- Lookup is the reservation: the prefetch lock pass asks local L1s first and
  the shared L1 only for the keys they missed, so a DRAM hit costs no round
  trip. `MISS` and `BUSY_WRITING` are ordinary misses.
- Writes follow `--l1-manager` order; list the shared L1 first so stores land
  where every server can read them. Only `OUT_OF_MEMORY` (region full, or
  orchestrator unreachable) moves a key to the next L1.
- A store that fails part-way aborts its reservations on the copy stream
  (`StorageManager.abort_write_by_owner`); here that is an `AbortWrite`,
  so another writer can retry without waiting for the orchestrator write TTL.

| Failure | Behaviour |
| --- | --- |
| Store fails on the server | Reservations aborted after the partial copies drain; keys writable again |
| Server dies mid-store | Keys stay `WRITING` until the write TTL expires or the client re-registers; extents remain consumed |
| Orchestrator unreachable | Reads miss, writes overflow to the next L1, status says unreachable; no local fallback authority |
| Orchestrator restarted | Every shared L1 is fenced until its server restarts after an offline reset; private L1s keep serving |
| Layout, size or visibility mismatch | `RegisterClient` fails; the server does not start |

## Visibility and media

With `visibility_mode = software_fenced` on non-coherent shared memory, the server
fences both sides of every copy; the orchestrator trusts the fences.

- Publish: after the D2H copy, `clflushopt` the range and `sfence`, then
  `FinishWrite`. Intel DDIO can leave DMA-written lines in the writer's LLC,
  so this is not optional.
- Acquire: after `READ_GRANTED`, `clflushopt` the range and `sfence` before
  the H2D copy. `lfence` would not order `clflushopt` against the copy's
  doorbell write.
- Both run in the native `lmcache_native.cache_flush_range` (x86-64,
  `clflush` + `mfence` fallback), GIL released, up to eight threads per batch.
- `coherent` skips both: one host, or a coherent fabric. The orchestrator
  advertises the mode and refuses a client with a different one.

A medium (`media.py`) maps the first `capacity_bytes` of its path with the
`open_devdax_mapping` helper the private Device-DAX allocator uses, registers
the mapping for GPU DMA (`direct-pinned`, else `pageable-staging`) and builds
bounds-checked memory-object views at granted offsets. `DevDaxMedium`
requires a character device and checks its sysfs size; `FileMedium` creates
or extends the file to the capacity so any server can start first. The L1
type picks the medium: `DEVDAX` maps a device, `DRAM` maps a file; GDS cannot
be shared.

## Guards

Before the shared L1 serves anything: `DescribeRegion` reachable and not
`RESET_REQUIRED`, region fits `size_gb`, `RegisterClient` accepted;
`--supported-transfer-mode lmcache_driven`; no CacheBlend, P2P or
runtime plugins;
at most one shared L1 per server and no L2 `affinity_tag` on it; `noop`
eviction; `lmcache trace replay` refuses it; its placements carry
`shared: true` and stay out of fleet cache events; it does not implement
`DevDaxHotPlug`, so the arena reconfiguration API answers 409. Configuration
and the operator view are in the user guide, `docs/source/mp/shared_l1.rst`.

## Code map and tests

`manager.py` (`SharedL1Manager`), `media.py` (`RegionMedium`, `DevDaxMedium`,
`FileMedium`), `visibility.py` (`software_fenced`, `coherent`), `wire.py`
(keys and layouts on the wire), `csrc/lmcache_native/cache_flush.{h,cpp}`.
Wiring: `config.py` (`shared` on `L1ManagerConfig`), `storage_manager.py`
(`_create_l1_manager`), `prefetch_controller.py` (lock pass
order), `multiprocess/server.py` (`prepare_shared_l1`),
`lmcache_driven_transfer.py` (owner-routed abort after failed copies), `mp_coordinator/cache_events.py`,
`trace/replay_command.py`.

Tests under `tests/v1/distributed/shared_l1/`: `test_manager.py` (against a
real orchestrator process: cross-server reads, one writer per key, abort,
overflow, leases, refused delete and evict, crash-restart retirement,
unreachable and restarted orchestrator), `test_storage_manager_binding.py`
(also against one), `test_media.py`, `test_config.py` and
`test_server_guards.py` (the guards), `test_visibility.py`;
`tests/v1/distributed/test_l1_interface.py` runs the interface contract
against both bindings, `test_l1_lookup_order.py` the local-first lookup, and
`tests/v1/multiprocess/test_shared_l1_servers.py` (GPU) two MP servers on one
host over a tmpfs region.

## TP ownership and P/D handoff

The existing ObjectKey encodes world size, rank and object group. Each worker
stores its own keys; scheduler lookup expands every required rank/group. The
existing grouped prefix policy retains only a prefix all rows can serve, with
sliding-window rules for hybrid state. A late rank remains a miss. Shared L1
checks peer shapes/dtypes before lookup retains an object; incompatible writes
are rejected instead of overflowing into a private pool.

Read reservations accumulate independent leases for overlapping lookups. Owner
metadata selects the L1 for release after each worker's H2D copy. A failed
visibility acquisition releases only that call's lease count. Failed publication
aborts the write before another writer retries, without reclaiming its extent.

The MP P/D example proxy waits for distinct store notifications from every TP
rank. GPU completion can precede the dispatcher thread's FinishWrite RPC, so
shared-L1 deployments also set `--decoder-cache-url`. The proxy polls the
existing warm-prefetch API until `ready_chunks` covers the complete prompt
chunks. This count uses the same grouped fold as serving lookup and includes
hybrid windows; the probe retains no read leases. Decode then performs its own
lookup and takes its own leases. Shared extents are immutable and never evicted,
so readiness cannot be lost to eviction between the probe and decode. Fencing
and transport failures retain the usual serving failure behavior.

The readiness check times out instead of forwarding a request whose prefix
exists only in the prefiller's private overflow L1. It adds no synchronization
to the device store path and requires no serving-engine or MP wire-schema change.
Peers must use the same model revision/namespace, engine layout, TP size, chunk
size, dtype and hash algorithm. Cross-TP resharding is outside this binding.

## Follow-ups

Reclamation and eviction (the orchestrator's `RequestEvict`; the binding frees
views); other media (cuFile for a GDS slab, Maru's resource manager);
visibility without flushes (uncacheable mapping plus DDIO off); hot-plug of
the region (capacity change in the orchestrator, remap in the media).
