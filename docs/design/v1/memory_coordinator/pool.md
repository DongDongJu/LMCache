# Memory Coordinator reservation core

This first upstream slice extracts the allocation and object-lifetime state
machine from the standalone shared Device-DAX prototype. It is an in-process
metadata library with no serving integration. It does not map or register memory.
A future service must add durable state and exclusive writer fencing before
using this core to authorize shared DAX access.

## Active role

| Field | Contract in this PR |
| --- | --- |
| Component and capability | `MemoryPool`: allocate extents and record object/reservation ownership |
| Authority mode and domain | Authority for metadata within one pool object; no authority over a physical device |
| Managed resource | Aligned byte extents within one fixed-capacity logical region |
| Tier and scope | Foundation for shared L1; one pool in one process |
| Request path | Direct synchronous library calls; no deployed request path |
| Inputs and freshness | Constructor geometry and current typed requests; no external observations |
| Proposals and decisions | Grants are authoritative only within this pool's lifetime |
| Authoritative state | Allocation cursor, generations, object states, write tokens and read tokens |
| Derived state | Token-free snapshots, rebuilt from the current state under the same lock |
| Effects and executor | Metadata changes only; no payload, device, GPU, quota or fleet effects |
| Identity | Operator region ID and layout ID, fresh pool epoch, unique reservation tokens |
| Locality and reachability | Local process only; region ID is not a device path, node ID or endpoint |
| Consistency | One lock serializes operations; failed batch validation changes no state |
| Durability and recovery | Ephemeral; reconstruction starts a new epoch; never recover live payload from this state |
| Leadership and fencing | No cross-process leader; inherited objects reject use after fork; close fences subsequent operations |
| Public contracts | Typed key/layout/handle/grant models and `MemoryPool` methods |
| Failure and readiness | Invalid geometry, layout, token, handle or epoch fails explicitly; no service readiness surface |
| Non-responsibilities | Persistence, transport, topology validation, DAX visibility, eviction, capacity rebalancing and MP Coordinator views |

| Fact/effect | Observation | Decision/state authority and public read | Executor | Serialization |
| --- | --- | --- | --- | --- |
| Extent allocation | Validated layout and current cursor | `MemoryPool`; `snapshot()` | Same pool | Pool lock |
| Object visibility and pins | Current state plus reservation token/handle | `MemoryPool`; `snapshot()` | Same pool | Pool lock |
| Physical memory identity/visibility | Outside this PR | Deployment and future DAX adapter | Outside this PR | Must be established before integration |

## Resource lifecycle and invariants

```text
ABSENT --reserve_writes--> WRITING --finish_writes--> VALID
                            |
                       abort_writes
                            v
                          ABSENT (allocated bytes remain consumed)

VALID --reserve_reads--> independent read tokens --finish/abort_reads--> released
```

Offsets are aligned, non-overlapping, and bounded by capacity. `used_bytes`
means the monotonic allocation cursor, including alignment gaps and aborted
extents; it is not resident payload bytes. Capacity is immutable. An absent-key
write batch either fits completely or changes nothing. Duplicate keys within a
batch are rejected; a later writer for an existing key receives no grant.
Only `VALID` objects can be read, and committed objects remain immutable.
Each read has its own token; releasing one cannot release another.

All finish/abort batches validate every token and exact handle before changing
state. Snapshots expose no reservation tokens and grant/layout values cannot
mutate authoritative state through aliases. Generations increase across aborted
writes. Extents are never reused, even after cancellation or reader release;
there is no deletion, eviction, expiry or automatic reclamation in this slice.

Epoch validation distinguishes pool incarnations. Local callers bind to one pool
object; future transports must check the latched epoch for every request.
Constructing a second pool for the same region creates an independent metadata
universe and must never be used to control the same live physical memory.
Shutdown and recreation require quiescing every user before any physical range
can be reused. A process-ID check prevents inherited authority after fork.

The core cannot prove payload completion. Before a future caller publishes a
write, it must finish D2H and the DAX publish barrier; before consuming a granted
read it must complete the acquire barrier, and release only after H2D completion.
No physical sharing or performance claim follows from these metadata tests.

## Extraction and compatibility

| Dimension | Current local prototype | First upstream PR | Later migration |
| --- | --- | --- | --- |
| Capability/authority | Durable standalone shared-pool authority | Same metadata transitions, ephemeral library only | Add durable ownership before exposing a service |
| Deployment/policy | HTTP service, fixed monotonic allocation | In-process library, same allocation policy | Single fenced service; no MP controller move |
| Resource/identity | One DAX region, layout and durable epoch | One logical region, layout and fresh epoch | Persist epoch and validate physical incarnation |
| APIs/config/state | HTTP, CLI, durable JSON and MP configuration | Core types/methods only | Add versioned transport and stored schema separately |
| Failure/rollout | Service fencing and coordinated reset | Exceptions and process-lifetime fence | Recovery and readiness before activation |

Existing upstream L1/L2 allocators, MP Coordinator APIs, CLI and configuration
remain unchanged. Object keys retain `EncodedObjectKey` semantics, including
object-group and cache-salt isolation. The full local prototype remains a
reference for later slices; its service is not enabled by importing this module.

## Verification

Public-interface tests cover allocation bounds/alignment, capacity atomicity,
concurrent duplicate writers and disjoint allocations, illegal token/handle
transitions, partial reads, independent pins, non-reuse, snapshot/layout
isolation, canonical keys, fresh epochs, close and fork fencing. These tests
require neither CUDA nor a DAX device. Durable recovery, competing processes,
lost RPC replies, real visibility and transfer lifetime belong to the later
slices that introduce those contracts.
