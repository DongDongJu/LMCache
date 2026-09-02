# MP Memory Coordinator

A standalone process that rebalances LMCache-visible Device-DAX capacity
between MP servers. It reads fleet membership and occupancy from the existing
MP Coordinator, picks one LOW donor and one HIGH receiver after three stable
samples, and moves **one** Memory-Coordinator-managed allocation through a
crash-safe saga:

    donor drain -> donor evict -> outside deallocate -> outside allocate -> receiver add

Code: `lmcache/v1/mp_memory_coordinator/`. CLI: `lmcache mp-memory-coordinator
--config FILE`. Manifests and E2E: `tests/e2e/mp_memory_coordinator/`.
Testing against a real outside service and real MP servers:
[testing.md](testing.md).

Under the development topology this changes which pre-existing server-local
Device-DAX device is active; it does not move a physical device or migrate
live KV data.

## Boundaries

| Rule | Where enforced |
| --- | --- |
| Talks to the MP Coordinator over HTTP only (`GET /instances`, `GET /instances/usage`) | `clients/mp_coordinator_client.py` |
| Never imports `lmcache.v1.mp_coordinator` (context, managers, registries, EventGate, persistence, schemas) | `tests/v1/mp_memory_coordinator/test_architecture_boundary.py` (AST) |
| Own configuration, journal, inventory, Lease, PVC, health, lifecycle | `config.py`, `persistence/`, `leader.py`, `app.py`, manifests |
| Outside API frozen: exact methods, paths, fields, literal values | `models.py` (`extra="forbid"` requests), `clients/memory_allocation_client.py`, `test_clients.py` |
| Never retries an outside POST; at most one move at a time; mutation off by default | `recovery.py`, `controller.py`, `config.py` |
| Never moves a shared DAX, a bootstrap (index 0) device, an unmanaged path, or a device of a multi-adapter MP | `policy.py` |

## Modules

```
config.py        strict YAML config (unknown keys / wrong types rejected), actuation_enabled=false
models.py        local wire DTOs (coordinator, MP server, frozen outside API) + durable records
clients/         one reusable httpx client per remote; bounded GET retries; POST exactly once
policy.py        sandwich read, LOW/HIGH history, ranking, preflight rules, donor device choice
recovery.py      the pure decision function: (MoveRecord, Evidence) -> one next safe action
controller.py    the cycle: observe -> propose -> persist -> one effect -> persist
persistence/     atomic, checksummed, versioned journal (tmp + fsync + rename + dir fsync)
adoption.py      explicit one-time allowlist adoption; never discovers devices
discovery.py     per-cycle inventory derivation from outside status (the default)
attachment.py    per-cycle attach plan for present, outside-assigned, unattached devices
leader.py        Kubernetes Lease elector (resourceVersion CAS) or static single-process leader
app.py           /healthz /readyz /status /journal /metrics; graceful stop
```

## Observation and eligibility

Every accepted sample comes from a sandwich read: an instance is accepted only
when `registration_time`, `ip`, `http_port` and `metadata.worker_ip` match in
both `/instances` reads, its usage row is `registered`, has
`declared_capacity`, and carries a non-null private `l2/dax` ratio. Worker
IPs must be present and unique. `registration_time` is a registration epoch:
any change invalidates the sample and the history.

`metadata.worker_ip` is registered by the MP server from
`LMCACHE_WORKER_NODE_IP` (injected from `status.hostIP` by the operator's
DaemonSet). It is the outside allocator's `target_node` and is never the
direct MP address (`advertise_ip`, the Pod IP).

History is keyed by the full identity `(instance_id, epoch, endpoint,
worker_ip)`; a move needs `stable_samples` consecutive same-level samples.
Receivers rank by descending ratio, donors by ascending ratio, ties by
`instance_id`. Before proposing, both participants are preflighted live:

* `GET /status`: root `is_healthy`, `storage_manager.is_healthy`, exactly one
  DAX L2 adapter, healthy, `closing=false`, `hotplug_enabled=true`;
* `GET /reconfigure/dax/status`: `enabled`, `num_adapters=1`, adapter index 0,
  `status/add/remove` supported, every non-tombstone device healthy,
  non-closing, active. `state="removed"` tombstones are ignored (the DAX
  adapter's aggregate health ignores them too -- see `l2_adapters/dax.md`).

Live pressure uses the DAX totals; the coordinator's LOW/HIGH must still
hold live. The donor device is the least-used active managed device at index
`> 0` whose map is whole GiB and equals the allocation size; removing it must
leave `min_devices_per_instance` active devices and a projected donor ratio
`used / (capacity - slot_capacity) <= projected_donor_max_ratio`.

Map size and slot capacity are distinct: the outside size and the DAX add
use the map size; capacity deltas use `max_slots * slot_bytes`.

## The saga and its durability

A `MoveRecord` carries the states

    SELECTED -> DONOR_DRAINING -> DONOR_REMOVED -> DEALLOCATING -> DEALLOCATED
             -> ALLOCATING -> ALLOCATED -> COMPLETE   (ROLLING_BACK | BLOCKED)

plus an **effect ledger**: one `EffectRecord` per side effect with
`intent_at`, `before_paths` (outside path set of the target node captured
before the POST), `dispatched`, `attempts`, `response`, `error`,
`confirmed`. Each cycle the controller gathers `Evidence` (sandwich read,
both DAX statuses, outside status, usage capacities, leadership) and asks
`recovery.decide()` for exactly one action:

| Decision | Effect |
| --- | --- |
| `Hold` | wait; nothing persisted |
| `Persist` | state change or confirmation from status; no side effect |
| `DoEffect` | persist intent -> re-check leadership + sandwich identity -> persist `dispatched` -> one POST -> persist result |
| `Block` | terminal `BLOCKED`; no further mutation |
| `Finish` | `COMPLETE` with `SUCCEEDED` or `ROLLED_BACK`; inventory and cooldowns updated |

Because `decide()` reads only the journal and current status, a restart
simply re-enters the loop -- recovery *is* the normal path. The persistence
bar for every effect: intent durable (fsync + atomic rename + directory
fsync) before the POST; result durable before the next effect.

Outside effects are issued at most once. An outside effect that is
`dispatched` with neither a response nor an explicit failure after a restart
has an unknown outcome and the move enters `BLOCKED`: released/granted sizes
cannot be proven and a re-issued POST could consume a second device. A
connect failure (request provably never sent) is not dispatched and may be
retried within `get_retry_attempts`. DAX effects are re-driven from
`/reconfigure/dax/status`, which is authoritative and idempotent-safe.

Before every POST: current leadership (`ensure_leader` renews immediately),
reachable MP Coordinator, unchanged sandwich identity of both instances, and
the expected prior effect still visible.

### Rollback rules

| Known failure | Result |
| --- | --- |
| Receiver vanishes / preconditions fail before outside deallocation | evict if draining, confirm terminal, re-add the still-outside-owned old path (`DONOR_EVICT` -> `DONOR_READD`), `ROLLED_BACK`; else `BLOCKED` |
| Drain does not reach zero busy references within `drain_timeout_seconds` | `BLOCKED`, no outside call (there is no undrain API) |
| Deallocation explicitly refused, path still listed under the donor | re-add old path, `ROLLED_BACK` |
| Allocation explicitly refused, no new path under the receiver | allocate the same GiB back to the donor, attach the returned path (`RESTORE_DONOR_ALLOCATE` -> `RESTORE_DONOR_ADD`), `ROLLED_BACK` |
| Allocation returned a wrong size / invalid path but exactly one new path appeared | release that proven path (`RELEASE_RECEIVER`), restore donor, `ROLLED_BACK` |
| Receiver add fails `dax_add_max_attempts` times | release the receiver path, restore donor, `ROLLED_BACK` |
| Outside POST may have committed but the response/effect is unprovable | `BLOCKED`; no retry, no later mutation |

Returned paths are validated as absolute, normalized (no `..`), under
`allowed_device_path_prefix`, absent from the persisted before-set, the
unique new path of the target node, and listed under that node only.

## Journal

`journal.json` = `{"schema_version", "checksum": "sha256:...", "payload"}`
holding inventory (`ManagedAllocation`), cooldowns, the single active move,
bounded history, counters, and the `initialized` marker. Loading fails
closed on a corrupt, truncated, checksum-invalid, or unknown-version file:
the process stays alive (`/healthz` 503), unready, and mutates nothing.

## Owning a device: discovery and adoption

The coordinator only ever moves a device that is in its managed inventory,
because the donor step ends in `POST /deallocations` and must never hand
back memory the coordinator does not own. There are two ways in.

**Discovery (every cycle).** `discovery.py` re-derives ownership from the
outside Memory Allocation service, which is already the single writer for
managed nodes and paths. A live device is adopted when its DAX index is
`> 0`, its state is `active` and it is healthy and not closing, its path
starts with `allowed_device_path_prefix`, outside status lists that exact
path under exactly the one worker IP the instance registered, its map size
is a positive whole number of GiB, and no inventory entry already claims
it. Because it re-runs every cycle, a path that changes with its Pod name
is re-derived rather than going stale, and no allowlist has to be
maintained. Discovery is purely additive: it never removes an entry, and it
is skipped entirely (leaving the inventory untouched) when the outside
status read fails. Every live device it does *not* adopt is reported with
a reason in `/status.last_cycle.discovery`, so an empty inventory is
always explained.

**Allowlist adoption** remains available. `lmcache mp-memory-coordinator
--config C --adopt allowlist.yaml` (or `adoption_file` when the journal is
uninitialized) adopts an entry only when its path is active at DAX index
`> 0` on the instance registered for that worker IP, appears under the same
worker in outside status, matches the approved map size, and is not owned.
Note that `--adopt` adopts and exits without starting the control loop, and
that `adoption_file` is read only while the journal carries no
`initialized` marker.

### Attach orchestration

Discovery only sees devices an MP server has already mapped. An MP server
whose DAX adapter runs a presence watcher (`watch_directory` in its
`--l2-adapter` config; see `docs/design/v1/distributed/l2_adapters/dax.md`)
additionally reports every path in its watched directory under
`/reconfigure/dax/status ... status.watcher.present_devices`, each with a
read-only physical inspection (`mode`, sysfs `size`, driver). The server
**never** attaches one on its own: presence is not ownership -- FREE runtime
devices are visible on every worker and a deallocated donor path stays
present after its removal -- and only the coordinator can prove ownership.

`attachment.py` therefore applies the *same* rule discovery uses. Each cycle
with no active move, for every accepted instance whose watcher is enabled,
a present device is planned for `POST /reconfigure/dax/add` (size = the
whole device from sysfs) when:

* its physical `mode` is `devdax` (`system-ram`, `unbound`, `absent`,
  `not-a-device`, `unknown` are never attached);
* its path starts with `allowed_device_path_prefix`;
* no non-tombstone adapter entry already has the path;
* the adapter accepts hotplug (`hotplug_enabled`);
* outside status lists that exact path under exactly the one worker IP the
  instance registered;
* its sysfs size is a positive whole number of GiB; and
* no add of it failed within the last `cooldown_seconds` (the same knob as
  the post-move participant cooldown).

The step runs after discovery and before ranking, and **only when no move
is active**: a move's receiver add and a donor's post-evict window must never
be raced by a second writer of the same adapter, and the move already
re-attaches its own paths. One outside read per cycle serves both discovery
and attach orchestration so they never disagree about ownership. With
`actuation_enabled: false` (or after a stop request) the plan is reported
as `would_attach` and no POST is made; with actuation, leadership is
re-renewed once (planned adds are withheld and reported as `skipped_pass:
not leader` if that fails) and each planned add is issued at most once per
cycle. A failure (transport error, non-2xx, or a non-`active` entry in the
response) is logged and remembered so the path is retried only after
`cooldown_seconds`.

**An attach cycle never proposes.** The sandwich read predates the add, so
the adapter's capacity it recorded is stale once an add was issued (or may
be, after an ambiguous failure). A `MoveRecord` built from that snapshot
would carry the pre-attach capacity and the move would then wait for
capacity convergence forever. Whenever an add was attempted -- `attached` or `failed` non-empty
-- the cycle reconciles the inventory, reports the decision `attach issued;
re-observing next cycle`, and returns before ranking; the next cycle reads a
fresh sandwich and proposes from post-attach capacities.

No journal record is written for an attach and nothing about it is
persisted: a same-size re-add is idempotent on the MP server (`200` with the
existing entry), and the next cycle's discovery adopts the now-live device
through the unchanged ownership rule, so a crash between the add and the
adoption loses nothing. The success count (`counters.attached` in `/status`,
`lmcache_memcoord_devices_attached_total`) is in-memory and restarts from
zero on purpose: attaches are idempotent and need no durable count. Every
present device that is *not* planned is reported with its reason in
`/status.last_cycle.attachments.skipped`.

## Leadership and deployment

A `coordination.k8s.io/v1` Lease (pre-created by the manifests) is
coordination, not a fencing token: production safety is `replicas: 1`,
`strategy: Recreate`, and a `ReadWriteOncePod` PVC. Conflict, timeout, or
holder-identity loss is an immediate loss of permission to POST; the elector
re-renews immediately before every mutating call. A non-leader never writes
the journal; a newly elected leader reloads it.

`/readyz` is true only when the process is leader, the MP Coordinator was
reached on the last cycle, the inventory is reconciled, and no move is
`BLOCKED`. Stopping starts no new move and persists the current state.

## E2E

`tests/e2e/mp_memory_coordinator/` runs the real coordinator against two
detached test services -- the scenario server (fake MP Coordinator + fake
donor/receiver MP with the complete golden schemas) and the strict mock
Memory Allocation service -- as separate processes locally or as Pods in
kind. Both services expose production routes on their public ports and
test controls on separate admin ports; the coordinator cannot tell they are
fakes. Assertions correlate endpoint-local audits with the journal by
request id, device path, and confirmed phase.
