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
