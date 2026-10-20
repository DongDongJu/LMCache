# Shared-region data path

`RegionMedium` owns one mapping, its GPU registration and tensor views.
DRAM uses a regular file on tmpfs/hugetlbfs; DEVDAX requires a character device
with readable sysfs capacity. The caller supplies the path, backend, capacity
and alignment. Mapping uses the same helper as the private Device-DAX allocator.

Views validate offset, alignment, extent length and payload size. They have no
parent allocator: dropping a view cannot free shared bytes. GPU registration
must succeed; a failure closes acquired resources. After all copies drain,
`close()` unregisters and unmaps; outstanding views defer unmapping.

For noncoherent memory, `Visibility("software_fenced")` writes back and
invalidates cache lines using CLFLUSHOPT + SFENCE, or CLFLUSH + MFENCE.
The writer must publish after D2H and before exposing metadata. The reader must
acquire before H2D. Native flushes release the GIL and a batch uses up to eight
threads. `coherent` performs no cache maintenance.

```text
D2H completion -> publish fence -> metadata commit
read grant -> acquire fence -> H2D -> release read
```

The operator establishes that peer paths map the same physical region.
Allocation, write publication and read leases belong to the memory orchestrator;
these primitives do not own them. The manager binding and serving/CLI integration
are the next steps in the series.

Tests cover view aliasing, extent bounds, backing types, failed registration
cleanup, and native/cache-visibility operations. The private Device-DAX allocator
continues using the same mapping behavior after the helper extraction.

## Manager lifecycle

`SharedL1Manager` implements L1ManagerInterface using SharedL1Config and the
orchestrator's canonical ObjectKey, layouts and tokens. DRAM/DEVDAX configuration
subclasses select the medium. The binding is available as a Python component;
CLI/factory and serving integration follow in the next patch.

ReserveWrite returns private write tokens/views. After D2H completes, FinishWrite
publishes visibility and commits metadata. Failed publication aborts its tokens.
ReserveRead acquires independent leases and fences before H2D; overlapping reads
retain separate counts. Failed acquire releases only that call's leases. Malformed
extents, payloads and layouts release known tokens without disturbing existing
reads. CloseClient retires this incarnation after copies have drained.

Startup verifies capacity, page alignment and the epoch across mapping and
registration. Epoch fencing disables the binding. Unavailable metadata makes
reads miss and writes return OUT_OF_MEMORY; failed-close leases remain until
re-registration and writes also expire under the orchestrator TTL. Extents are
never reclaimed, and shared objects cannot be evicted or forcibly deleted.

Status and common L1 gauges report region-wide allocation/capacity and local
staging payload bytes. RPCs and visibility operations run outside metadata locks.
Device-DAX hot-plug is a separate capability; the shared binding does not expose
it, and the storage-manager API rejects unsupported reconfiguration.

Shared placement events are excluded from the per-server fleet directory, since
several bindings refer to the same region-owned copy.
