# DAX L2 Adapter Design

This document describes the built-in `dax` L2 adapter for LMCache
multiprocess mode and how it shares implementation with the non-MP DAX
storage backend.

## Goals

- Reuse one synchronous DAX core for MP and non-MP DAX storage.
- Keep the MP controller flow unchanged: adapters still use the normal
  submit, event-fd, and query-result contract.
- Keep the adapter facade stable while the runtime DAX device pool changes.
- Keep DAX volatile-only. Keys are indexed in process memory and are not
  recovered from device bytes after restart.

## Components

`lmcache/v1/storage_backend/dax/core.py` defines `DaxCore[KeyT]`. The core owns
the mapped DAX arena, fixed-size slot allocation, in-memory index, LRU order,
in-flight writes, external lock refcounts, active read borrow counts, close
coordination, and direct `ctypes.memmove` copies.

`lmcache/v1/storage_backend/plugins/dax_backend.py` is the non-MP wrapper. It
keeps existing non-MP behavior such as the local CPU backend requirement, TP=1
validation, optional async put, and the staging-slab batched restore path.

`lmcache/v1/distributed/l2_adapters/dax_l2_adapter.py` is the MP adapter. It
self-registers adapter type `dax`, owns separate event notifiers and worker
pools for store, lookup, and load operations, and uses one or more
`DaxCore[ObjectKey]` instances behind a stable facade.

`lmcache/v1/multiprocess/http_apis/reconfigure_api.py` exposes runtime
reconfiguration endpoints:

- `GET /reconfigure/dax/status`
- `POST /reconfigure/dax/add`
- `POST /reconfigure/dax/remove`
- `POST /reconfigure/dax/resize`

The HTTP layer routes `backend`, `operation`, and adapter-specific JSON payloads
into the generic L2 adapter reconfiguration API on `StorageManager`.
`StorageManager` only routes `operation` plus payload to a reconfigurable
adapter; DAX path, mode, and migration semantics stay inside `DaxL2Adapter`.
The same interface is intended for future adapters such as P2P, so the HTTP
layer does not inspect private adapter lists or DAX core state directly.

## Slot State

Each committed key points to one fixed-size slot. A slot is reusable only when:

- The key has been removed from the index.
- No external lock is held for the key.
- No store is in flight for the key.
- No active read has borrowed the slot.

Delete operations remove unlocked keys from the index immediately. If a read
has borrowed the slot, the slot is marked pending-free and recycled when the
borrow count reaches zero.

## MP Flow

Store:

1. `StoreController` calls `submit_store_task(keys, objects)`.
2. The adapter chooses an active DAX device. Existing keys prefer their current
   mapped device; new keys use the active device with the lowest slot usage.
3. A store worker copies each object into a DAX slot through `DaxCore.put_many`.
4. The adapter records task-level success as `all(per_key_results)`.
5. The store event fd is signaled and store listeners are notified for the keys
   that were actually accepted by the core.

Lookup and load:

1. `PrefetchController` calls `submit_lookup_and_lock_task(keys)`.
2. The adapter checks `key -> device` mappings first, then scans readable
   devices if needed.
3. The adapter calls `DaxCore.exists_many(keys, lock=True)` and returns a full
   bitmap, including holes.
4. Load workers call `DaxCore.load_many_into(keys, objects)` on the device that
   currently owns each key.
5. `submit_unlock(keys)` releases the external lock refcounts on every DAX
   core. This is deliberate because migration can update `key -> device`
   mappings between lookup and unlock.

## Runtime Hotplug

The DAX facade keeps the event fds and worker pools stable. Runtime hotplug only
mutates the device pool behind the facade, so `StoreController`,
`PrefetchController`, and the vLLM MP connector do not need ZMQ protocol changes
or poll-set re-registration.

Add:

1. Validate `hotplug_enabled`, path, and size.
2. Map a new `DaxCore[ObjectKey]`.
3. Append a `DaxDeviceEntry(state="active")`.
4. Return per-device status. Existing KV entries stay on their current devices.

Remove with migration:

1. Mark the source device `draining` so new stores do not choose it.
2. Reject the operation if externally locked or borrowed slots would be deleted.
3. Snapshot source keys and reserve source reads.
4. Copy each reserved payload from the source DAX pointer into another active
   DAX core with `put_reserved_from_ptr`.
5. Update `key -> device` mappings, delete the source entries, then close the
   source core.

Resize:

- Grow remaps the same core to a larger size after active reads and writes
  drain. No KV payload movement is needed.
- Shrink first proves that every live slot fits below the new slot count. If
  not, the out-of-range keys must migrate to another active device or the
  request fails. Shrink never silently evicts data.

## Restart Behavior

The adapter stores keys and metadata only in memory. Closing the adapter and
opening a new adapter against the same DAX device starts with an empty index.
Old bytes may remain on the device, but they are unreachable because PR1 does
not define any on-device metadata, scan, checkpoint, or recovery format.

## Capacity And Eviction

Usage is slot-based, not payload-byte-based. `get_usage()` reports occupied
slot capacity because the DAX arena is exhausted by slot count. The eviction
controller calls `delete(keys)`, which skips externally locked keys and
reclaims slots after active read borrows drain.

Runtime capacity is the sum of active, draining, migrating, resizing, and
removing device capacities. Closed, removed, and failed devices are excluded.

## Health After A Controlled Remove

A remove (`evict` or `migrate`) closes the source core and keeps its entry as
a `state="removed"` tombstone so device indexes stay stable and the history
is visible in status. The tombstone's own `is_healthy` is `False` (its core is
closed), but the adapter-level `is_healthy` in `report_status()` aggregates
only devices that still serve I/O: `closed` and `removed` entries are skipped.
Without that exclusion one successful remove would leave the adapter, the
storage manager, and the engine `/status` unhealthy forever. A `failed` device
is a real fault and still makes the adapter unhealthy. External controllers
(e.g. the MP Memory Coordinator) must likewise treat a `removed` entry as a
tombstone, never as an owned or attachable device.

## Physical Inspection And Presence Watcher

`lmcache/v1/distributed/l2_adapters/dax_physical.py` adds a read-only view of
what the kernel currently exposes for a DAX path, and an optional presence
watcher that scans one directory for device nodes.

### Contract

- `probe_device(path)` classifies a path as `devdax`, `system-ram`,
  `unbound`, `not-a-device`, `absent`, or `unknown` (`DaxPhysicalMode`) and
  reports `major:minor`, the kernel dax name, the bound driver, sysfs `size`
  and `align`, and a `detail` string for the inconclusive modes. Identity is
  `st_rdev`, never the file name: per-pod names such as `dax0.1` do not encode
  the kernel device, and minor numbers do not encode the `.Y` suffix.
- `scan_directory(directory)` probes every non-directory entry, sorted by
  name; a missing directory yields `[]`.
- `DaxDeviceWatcher(directory, interval_seconds)` runs `scan_directory` on a
  daemon thread (`dax-l2-watch`) and publishes an immutable
  `DaxWatcherSnapshot`; `snapshot()` never blocks on a scan, and a failing scan
  keeps the previous snapshot. An empty directory creates a disabled watcher
  with no thread, so the adapter always owns a watcher instead of an
  optional one.
- `DaxL2AdapterConfig.watch_directory` / `watch_interval_seconds` enable the
  watcher. `DaxDeviceEntry.physical` stores the probe taken at attach time;
  `hotplug_status()` reports it per device and adds a `watcher` block
  (`{"enabled": false}` when disabled, otherwise `DaxWatcherSnapshot.as_dict()`).

### Why read-only

The probe is `stat(2)` plus plain reads under `/sys/dev/char/<M>:<m>`. It
never `open()`s the candidate: opening an unbound dax node forks `modprobe`
on the host and can stall for seconds, and an `open()` would be a side effect
on a device this server may not own. sysfs is mounted read-only in every
container runtime and the attributes used (`subsystem`, `driver`, `size`,
`align`) are world-readable, so the probe works unprivileged. `fstat` reports
`st_size == 0` for a Device-DAX node and the kernel accepts `mmap` lengths
beyond `size` (faulting with `SIGBUS` on first touch), so the sysfs `size` is
the only in-process source for the add gate.

### Why no auto-attach

Presence proves usability, never ownership. The same `major:minor` can be
`mknod`ed into any number of directories, a deallocated donor device stays
present on its host, and a receiver device may be visible before the
allocator grants it. Under the coordinator's ownership rule a device is
managed iff the outside allocator lists its exact path under exactly the one
`worker_ip` this instance registered, and only the MP Memory Coordinator can
check that. The MP server therefore only reports `watcher.present_devices`;
the coordinator decides and issues `/reconfigure/dax/add`. A watcher that
attached on its own would re-map a donor device within one interval of the
coordinator's evict, bypass capacity publication (only
`StorageManager.reconfigure_l2_adapter` publishes `SM_CAPACITY_CHANGED`), and
need a size it cannot learn safely.

### Lock discipline

- `probe_device` runs **outside** `_device_lock`: in `__init__` all configured
  devices are probed first, then the lock is taken to map them; in
  `hotplug_add_device` the probe and the gate run before the lock. sysfs I/O
  under the device lock would stall every DAX store/lookup/load worker.
- `_device_status_locked` refreshes `entry.physical` only from the watcher's
  in-memory snapshot (`DaxWatcherSnapshot.find`), which is a pure lookup.
  `find` matches the exact `os.path.join(watch_directory, name)` string, so
  only devices attached directly under `watch_directory` (spelled that way)
  are refreshed; others keep their attach-time probe. A snapshot state is
  adopted only when its `probed_at` is at least as fresh as the state the
  entry holds, so a scan that started before the attach-time probe cannot
  overwrite it (and `probed_at` never goes backwards).
- The watcher has its own small lock guarding one attribute; the scan thread
  never takes `_device_lock`, and the adapter takes the watcher lock only
  while already holding `_device_lock`, so lock order is
  `_device_lock -> watcher._lock` and there is no inverse path.
- `close()` stops the watcher (bounded join) before draining the executors.

### Add gate

`hotplug_add_device` fails closed only on a positive bad signal: `system-ram`,
`unbound`, or `devdax` with `0 < size < requested`. `not-a-device`, `absent`
and `unknown` proceed exactly as before, so regular-file arenas keep working
and an absent path still fails in `mmap`. Error payloads stay sanitised
(`{"error": message}`, no path); the server log line carries the path.
Configured devices at startup are probed and reported but not gated, so
existing deployments see no behaviour change beyond the new status fields.
The gate is evaluated before the already-active (same path, same size)
check: a healthy same-size re-add stays idempotent (`200` with the existing
entry), but a re-add whose probe now reports `system-ram` or `unbound` is
rejected with `400` and the existing entry is left untouched. That is the
honest answer (the mapping is dead once the device is rebound) and does not
affect the coordinator, which only plans an add when no live entry exists.

## Current Limits

- Runtime hotplug does not perform kernel-level CXL or DAX reconfiguration.
- No per-TP partitioning.
- No restart recovery.
- Only single-buffer objects are supported.
