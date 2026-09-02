# MP Memory Coordinator

A standalone process that rebalances LMCache-visible Device-DAX capacity
between MP servers. It reads fleet membership and occupancy from the existing
MP Coordinator and, for a HIGH receiver that stayed HIGH for three stable
samples, runs **one** crash-safe saga at a time. It first tries to **grow**
the receiver from the outside pool without any donor:

    outside allocate -> receiver add                          (GROW)

and only when the allocator explicitly refuses does it pick one LOW donor
and **move** one Memory-Coordinator-managed allocation:

    donor drain -> donor evict -> outside deallocate -> outside allocate -> receiver add   (MOVE)

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
| Never re-issues an outside POST that may have been delivered; at most one saga (GROW or MOVE) at a time; mutation off by default | `recovery.py`, `controller.py`, `config.py` |
| Grows before it moves; a refused grow has zero side effects | `policy.py` (`evaluate_grow`), `recovery.py` (`NOT_SERVED`), `controller.py` |
| Never moves a shared DAX, a bootstrap (index 0) device, an unmanaged path, or a device of a multi-adapter MP | `policy.py` |

## Modules

```
config.py        strict YAML config (unknown keys / wrong types rejected), actuation_enabled=false
models.py        local wire DTOs (coordinator, MP server, frozen outside API) + durable records
clients/         one reusable httpx client per remote; bounded GET retries; POST exactly once
policy.py        sandwich read, LOW/HIGH history, ranking, preflight rules, grow size, donor device choice
recovery.py      the pure decision function: (MoveRecord, Evidence) -> one next safe action
controller.py    the cycle: observe -> propose -> persist -> one effect -> persist
persistence/     atomic, checksummed, versioned journal (tmp + fsync + rename + dir fsync)
adoption.py      explicit one-time allowlist adoption (optional; discovery covers assigned devices)
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

### Grow before move

The frozen outside API cannot be asked for free capacity: the only way to
learn whether the pool can serve a receiver is to POST an allocation. So
with no saga active the cycle proposes, for each stable-HIGH receiver best
first, a **GROW** before any donor search:

* receiver-only preflight (healthy engine and storage manager, exactly one
  hot-pluggable DAX adapter, every live device active), the live ratio still
  HIGH, no cooldown, and no active **grow backoff** for the receiver's
  worker;
* the request size is derived, never configured: (a) the most common
  size-consistent managed allocation size on the receiver's worker, else
  (b) the most common across the whole inventory, else (c) the receiver's
  index-0 bootstrap device map size rounded down to whole GiB; nothing
  derivable rejects the receiver with `grow_size_undeterminable` and no
  POST is made.

Only when no receiver can grow (every one is in grow backoff or fails a
receiver-only check) does the donor/receiver pair evaluation of today's MOVE
run -- best effort: with no eligible donor there is simply no proposal.

A **grow backoff** (`JournalDocument.grow_backoffs`, `worker_ip -> until`,
duration `max(cooldown_seconds, 2 * poll_interval_seconds)`) is written
only when the allocator explicitly refused a GROW; it is consulted only by
the GROW pass, never by candidate ranking, so the same receiver is a MOVE
candidate in the very next cycle. The floor of two idle polls is what makes
that fallback structural rather than a race: the `NOT_SERVED` finish is
followed by one `poll_interval_seconds` of sleep and then the reads of the
next cycle, and a backoff that had already expired by then would make that
cycle propose the same refused GROW again, forever, never reaching the
donor search (with `cooldown_seconds <= poll_interval_seconds` that is
exactly what an unfloored backoff did). It is distinct from `cooldowns`,
which a completed saga (SUCCEEDED or ROLLED_BACK) writes for its
participants -- the receiver alone for a GROW -- and which blocks both
kinds. A dry run (`actuation_enabled: false`) logs the GROW proposal
(`kind: "grow"`) every cycle and never probes, so the MOVE alternative only
becomes visible after a real refusal. Note that an unhealthy allocator
(5xx, 429) also counts as a refusal: it triggers the MOVE fallback, whose
own outside POSTs then hold or block on the same allocator.

A consequence worth stating plainly: with an exhausted pool every MOVE is
preceded by one refused allocation POST (the probe) and a `NOT_SERVED`
history entry, and starts two cycles later than it did before GROW existed.
The MOVE saga's own effect sequence is unchanged; anything that asserts on
raw audit order, save counts, or history indices has to account for the
probe (the test suites do so explicitly).

## The saga and its durability

A `MoveRecord` carries a `kind` (`move`, the default an older journal loads
as, or `grow`) and the states

    MOVE: SELECTED -> DONOR_DRAINING -> DONOR_REMOVED -> DEALLOCATING -> DEALLOCATED
                   -> ALLOCATING -> ALLOCATED -> COMPLETE   (ROLLING_BACK | BLOCKED)
    GROW: SELECTED -> ALLOCATING -> ALLOCATED -> COMPLETE   (ROLLING_BACK/RELEASE_RECEIVER | BLOCKED)

A GROW record has no donor: `donor` is the sentinel `NO_DONOR`, every
donor-side field is empty, and `old_map_size_bytes` / `allocation_size_gib`
carry the requested size so the receiver add and the inventory entry read
them unchanged (`MoveRecord.has_donor` is the one question consumers ask).
Both kinds share an **effect ledger**: one `EffectRecord` per side effect
with `intent_at`, `before_paths` (outside path set of the target node
captured before the POST), `dispatched`, `attempts`, `response`, `error`,
`failure` (`none` | `explicit` non-2xx | `contract` 2xx violation) and
`confirmed`. Each cycle the controller gathers `Evidence` (sandwich read,
the participants' DAX statuses, outside status, usage capacities,
leadership) and asks `recovery.decide()` for exactly one action:

| Decision | Effect |
| --- | --- |
| `Hold` | wait; nothing persisted |
| `Persist` | state change, confirmation from status, or a receiver rebind (GROW); no side effect |
| `DoEffect` | persist intent -> re-check leadership + sandwich identity -> persist `dispatched` -> one POST -> persist result |
| `Block` | terminal `BLOCKED`; no further mutation |
| `Finish` | `COMPLETE` with `SUCCEEDED`, `ROLLED_BACK`, or `NOT_SERVED` (GROW only: no inventory or cooldown change, a grow backoff recorded); inventory, cooldowns and backoffs updated in one save |

Because `decide()` reads only the journal and current status, a restart
simply re-enters the loop -- recovery *is* the normal path. The persistence
bar for every effect: intent durable (fsync + atomic rename + directory
fsync) before the POST; result durable before the next effect.

Outside effects reach the service at most once. An outside effect that is
`dispatched` with neither a response nor an explicit failure after a restart
has an unknown outcome and the move enters `BLOCKED`: released/granted sizes
cannot be proven and a re-issued POST could consume a second device. A
connect failure (`ClientConnectionError`: the connection was never
established, so the request provably delivered nothing) is not dispatched
and the same request id, with the same `before_paths`, is re-issued within
`get_retry_attempts`, after which the saga blocks. So the precise invariant
is: per request id, at most `get_retry_attempts` POSTs are *issued*, of
which at most one may have *reached* the service, and one that may have is
never followed by another. DAX effects are re-driven from
`/reconfigure/dax/status`, which is authoritative and idempotent-safe.

Before every POST: current leadership (`ensure_leader` renews immediately),
reachable MP Coordinator, unchanged sandwich identity of both instances (the
receiver alone for a GROW), and the expected prior effect still visible.

### GROW invariants

* **At most one allocation POST that may reach the allocator, no
  recursion.** One ledger-agnostic issuer (`_issue_grow_allocation`) is
  called from `SELECTED` (never with a ledger: `_perform` persists
  `ALLOCATING` together with the ledger) and from `ALLOCATING` only when the
  persisted intent is provably undispatched -- a crash or lost gate before
  the POST, or a connect failure that delivered nothing; the controller
  then reuses the existing ledger, request id and `before_paths`, which the
  `NOT_SERVED` proof below depends on. Up to `get_retry_attempts` POSTs may
  thus be issued for one request id (each a re-issue after a provably
  undelivered one), the allocator sees the id at most once, and after any
  POST that may have been delivered (answered, contract violation, lost
  mid-flight, or a crash after `dispatched`) no further one is issued: the
  saga classifies or blocks. A receiver lost before any effect finishes
  `ROLLED_BACK`.
* **Explicit refusal, nothing assigned -> `NOT_SERVED`.** In `ALLOCATING`,
  `failure == explicit` and no new path under the receiver versus
  `before_paths` is the only route to `NOT_SERVED`. A 2xx contract violation
  with no visible path blocks (a 2xx is never taken as "nothing happened");
  a failure or invalid path with exactly one new path releases that path;
  a dispatched POST with no recorded outcome blocks, exactly as for a MOVE.
  Residual window (inherited from the MOVE rule): `before_paths` is the
  receiver's outside path set read in the cycle that issued the POST, so a
  path that an external actor assigns to the receiver's worker between that
  read and the next cycle's read looks like the single new path and is
  released. Only a refusal can be misread this way, the window is one poll
  interval, and the coordinator never releases a path that is attached.
* **A lost receiver never holds forever.** After the allocation (in
  `ALLOCATING` once answered, `ALLOCATED`, and `RELEASE_RECEIVER`), when the
  receiver identity no longer matches: if exactly one accepted instance
  registers the receiver's worker, the record is rebound to it (the
  allocation is node-level and the add is idempotent, so the add's
  confirmation is dropped and the add is re-driven from the new identity's
  status: a restarted MP server that lost the hot-added path receives it
  again; its attempt budget is kept). A rebind is a persisted change and so
  restarts the grace below; a saga therefore rebinds at most
  `GROW_MAX_RECEIVER_REBINDS` (3) times (`MoveRecord.receiver_rebinds`),
  and the next loss with a replacement blocks, naming what is known about
  the path's attachment -- a receiver in a restart loop cannot keep the
  saga alive forever. Otherwise the saga holds for
  `drain_timeout_seconds` since its last persisted change and then decides
  from the receiver's endpoint first: a readable status is authoritative
  (path absent -> provably unattached, path live -> `BLOCKED` as attached
  under an identity the sandwich no longer accepts); when it is unreadable,
  raw membership decides (no instance at all registered on that worker,
  accepted or rejected -> provably unattached; anything registered there ->
  `BLOCKED`, `attachment unknown`). A sandwich-rejected receiver whose path
  is live is therefore never mistaken for vanished-and-unattached. "Provably
  unattached" enters the release step, but the release POST is gated on a
  matching receiver identity and a readable status, so it is issued only
  once an instance is back on that worker (rebound, or the same identity
  accepted again); with nobody coming back the release step blocks after a
  second `drain_timeout_seconds`. The bound on the whole vanished-receiver
  path is therefore `2 * drain_timeout_seconds` from the last persisted
  change, of which there are at most `GROW_MAX_RECEIVER_REBINDS`, and its
  outcome without a returning receiver is always `BLOCKED` with the path
  still assigned (never a release underneath a possibly attached mapping).
* **A cycle that attached proposes nothing** (not even a GROW): every
  proposal's capacity baseline comes from a sandwich taken after the last
  adapter write (see "Attach orchestration").
* **Bounded convergence wait.** A GROW whose new path is confirmed active on
  the receiver and listed by the allocator under it finishes `SUCCEEDED`
  with a WARNING once `capacity_convergence_timeout_seconds` elapses without
  the usage view converging: nothing unproven remains. Past that timeout the
  allocator's view decides, and every branch is bounded: listed elsewhere or
  not at all contradicts the proven allocation and blocks; an unreadable
  allocator is given a further `drain_timeout_seconds` (the allocation
  cannot be re-verified) and then blocks, with the path still active on the
  receiver for the operator to reconcile. Convergence finishes the saga at
  any time. A MOVE keeps waiting unconditionally, unchanged.
* **What stays unbounded, exactly as for a MOVE.** A receiver whose identity
  the sandwich still accepts but whose DAX status is unreadable holds
  (before or after the add), and so do the pre-dispatch holds on an
  unreadable allocator status. Those are the MOVE rules, inherited
  unchanged; every wait a GROW adds -- after the allocation, on the receiver
  identity or on the allocator's view -- is bounded and ends `BLOCKED` or
  finished, never green forever.
* **Cycle hardening.** `run_once` records any exception of the cycle body
  in `report.error`, logs the traceback, and still publishes the report, so
  `/readyz` turns 503 (`last cycle failed`) instead of staying ready on a
  stale successful report.

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
| GROW allocation explicitly refused, no new path under the receiver | `NOT_SERVED`: nothing changed, no cooldown, grow backoff for the worker; the next cycle may MOVE |
| GROW allocation answered 2xx but violated the contract, no new path | `BLOCKED` (the effect is unprovable; the pool is not assumed exhausted) |
| GROW invalid/failed allocation with one visible path, or receiver add fails `dax_add_max_attempts` times | release the receiver path (`RELEASE_RECEIVER`), `ROLLED_BACK`; no donor to restore |
| GROW receiver vanished after the allocation | rebind to the re-registered instance on that worker and re-drive the add (at most `GROW_MAX_RECEIVER_REBINDS` times per saga, then `BLOCKED`); else after `drain_timeout_seconds`: provably unattached (readable status without the path, or no instance registered on the worker) enters the release step, which releases only once an instance is back on that worker and blocks after a second `drain_timeout_seconds` otherwise; attached or unknown -> `BLOCKED` |
| GROW add confirmed, usage view not converged within `capacity_convergence_timeout_seconds` | allocator lists the path under the receiver -> `SUCCEEDED` with a warning; lists it elsewhere or not at all -> `BLOCKED`; unreadable for a further `drain_timeout_seconds` -> `BLOCKED` (path active on the receiver, allocation to be re-verified by hand) |

Returned paths are validated as absolute, normalized (no `..`), under
`allowed_device_path_prefix`, absent from the persisted before-set, the
unique new path of the target node, and listed under that node only.

## Journal

`journal.json` = `{"schema_version", "checksum": "sha256:...", "payload"}`
holding inventory (`ManagedAllocation`), cooldowns, grow backoffs, the
single active saga, bounded history, counters, and the `initialized`
marker. Loading fails closed on a corrupt, truncated, checksum-invalid, or
unknown-version file: the process stays alive (`/healthz` 503), unready,
and mutates nothing.

Compatibility is forward only: every field GROW added (`kind`,
`EffectRecord.failure`, `MoveRecord.receiver_rebinds`, `grow_backoffs`,
`counters.not_served`, `counters.grown`) is defaulted, so a journal written
before GROW existed
loads and behaves as a MOVE, and `schema_version` stays 1. A journal
written by this build (any save, even with no GROW ever run) does not load
in the previous build: every save dumps the added keys, and the previous
build's models forbid unknown keys, so its load fails closed
(`JournalCorruptError`, unready and inert). Downgrading requires clearing
the journal first (with no saga active), or upgrading again.

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
is active**: a saga's receiver add and a donor's post-evict window must never
be raced by a second writer of the same adapter, and the saga already
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
would carry the pre-attach capacity and the saga would then wait for
capacity convergence forever (a MOVE) or until the bounded GROW wait
expires. Whenever an add was attempted -- `attached` or `failed` non-empty
-- the cycle reconciles the inventory, reports the decision `attach issued;
re-observing next cycle`, and returns before ranking; the next cycle reads a
fresh sandwich and proposes (a GROW or a MOVE) from post-attach capacities.

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

`/readyz` is true only when the process is leader, the last cycle did not
fail, the MP Coordinator was reached on that cycle, the inventory is
reconciled, and no saga is `BLOCKED`. Stopping starts no new saga and
persists the current state.

## E2E

`tests/e2e/mp_memory_coordinator/` runs the real coordinator against two
detached test services -- the scenario server (fake MP Coordinator + fake
donor/receiver MP with the complete golden schemas) and the strict mock
Memory Allocation service -- as separate processes locally or as Pods in
kind. Both services expose production routes on their public ports and
test controls on separate admin ports; the coordinator cannot tell they are
fakes. Assertions correlate endpoint-local audits with the journal by
request id, device path, and confirmed phase. The mock allocator carries a
global **pool budget** the harness sets to the fixture's initially assigned
total at every reset, so the receiver's GROW probe is refused
(`NOT_SERVED`, no mutation) and every move scenario keeps its exact
sequence; grow scenarios raise the budget.
