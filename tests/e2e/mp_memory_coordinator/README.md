# MP Memory Coordinator E2E

End-to-end verification of the standalone MP Memory Coordinator
(`lmcache/v1/mp_memory_coordinator/`, design in
`docs/design/v1/mp_memory_coordinator/overall.md`) against two detached,
deterministic test services:

| Service | Package | Production-facing ports | Admin port |
|---|---|---|---|
| Scenario server (fake MP Coordinator + fake donor MP + fake receiver MP) | `scenario_server/` | 9300, 8081(+8181), 8082(+8182) | 9091 |
| Strict mock Memory Allocation service (frozen `/api/v2/apps/lmcache` API) | `mock_memory_allocation_service/` | 8080 | 9090 |

Both load the same immutable fixture `fixtures/two_server_local_dax.yaml`
and share nothing else. `fixtures/golden/` holds the complete current JSON
schemas captured from the real MP Coordinator and MP server (Phase 0); the
scenario server reproduces them and the coordinator's DTOs are tested
against them. Exception: the presence-watcher additions in the DAX status
goldens (`watcher`, per-device `physical`) were written by hand to the
watcher contract, not captured; re-capture them from a real MP server with
`watch_directory` set once one is available.

## Run

```bash
# unit + contract tests of the coordinator
uv run pytest -q tests/v1/mp_memory_coordinator

# the two test services on their own
uv run pytest -q tests/e2e/mp_memory_coordinator/test_mock_memory_allocation_service.py
uv run pytest -q tests/e2e/mp_memory_coordinator/test_scenario_server.py

# E2E, local mode: services and the real coordinator as subprocesses
uv run pytest -q tests/e2e/mp_memory_coordinator
# or
make e2e-mp-memory-coordinator-local

# E2E in kind (builds the three images, creates the cluster, runs -m kind, deletes)
make e2e-mp-memory-coordinator-kind
```

`MEMCOORD_E2E_ARTIFACTS=<dir>` collects, for every failed test, both
services' audits and state, the outside status, the journal, and the
coordinator log.

## Developing a real outside Memory Allocation service

The frozen contract is `PLAN.md` section 2; three executable references pin it:

1. `test_clients.py` (`tests/v1/mp_memory_coordinator/`): the exact bodies the
   coordinator sends and every field/echo/size it validates on responses.
2. `mock_memory_allocation_service/`: a strict reference implementation of the
   public routes (its `README.md` lists the codes and rejection rules).
3. `test_outside_api_conformance.py`: runs against **any** implementation by
   URL and needs nothing but `requests`:

   ```bash
   make test-outside-api-conformance OUTSIDE_API_URL=http://127.0.0.1:18080
   # or
   uv run pytest -q tests/e2e/mp_memory_coordinator/test_outside_api_conformance.py \
       -m outside_api --outside-api-url http://127.0.0.1:18080
   ```

   It checks the bare status shape, a deallocate -> allocate round trip with the
   golden key sets and exact echo/size validation (restoring the assignment),
   and rejection-without-mutation of malformed requests and unknown nodes.

Then point a coordinator at the service (`memory_allocation_url`) with
`actuation_enabled: false` for a dry run, and finally run the E2E suite with
the real service in place of the mock.

## What the happy path proves

After reset, the coordinator adopts the donor's runtime path through the real
`--adopt` command, observes three stable samples with zero mutation, then
performs exactly: donor drain -> donor evict -> outside deallocate -> outside
allocate -> receiver add, with the exact frozen request bodies, every echo and
size validated, `COMPLETE/SUCCEEDED` only after the new path is active and
capacity converged (donor 128 -> 64 GiB, receiver 64 -> 128 GiB), the final
assigned runtime GiB back at 64, immutable path-to-worker bindings, and a
cooldown that prevents a second move.

## Failure matrix

`test_failures.py` covers the CI matrix of PLAN.md section 8: eligibility
faults, snapshot race, adapter gate, ownership gate, live mismatch, busy
drain with evict 409s, drain deadline (BLOCKED, no outside POST), removed
tombstone, delayed capacity, allocation failure, no receiver-local match,
response contract violations, wrong size, invalid returned path, transient
and persistent attach failure, deallocation/allocation commit-then-drop
(BLOCKED, no blind retry), coordinator outage before and during a move,
coordinator restart (history reset), MP re-registration, journal damage,
crash recovery at every effect, and cooldown across restart.
`test_leadership.py` proves ordinary Lease handoff with a single writer.

Tests marked `local_only` need process control and are skipped in cluster
mode; everything else also runs against the kind topology
(`-m kind --kube-context kind-lmcache-memcoord-e2e`). A requested but
unreachable cluster is a failure, never a skip.

## Scenario server admin API

The scenario server (`scenario_server/`) exposes six listeners in one process
(coordinator 9300, donor MP 8081/+8181, receiver MP 8082/+8182, admin 9091).
Run it with `python -m tests.e2e.mp_memory_coordinator.scenario_server --fixture
fixtures/two_server_local_dax.yaml`; see its module docstrings for the DAX
lifecycle it models.

### Admin routes

```
GET    /__test/health                      -> {"status":"ok","seq":N}
POST   /__test/reset                       -> reload fixture, clear faults/barriers/audit; returns state
                                              (allocator: optional {"pool_budget_gib": N}; the harness passes the
                                              fixture's initially assigned 64 GiB so the coordinator's grow-before-move
                                              probe is refused and every move scenario keeps its exact sequence)
POST   /__test/pool_budget                 -> allocator only: {"pool_budget_gib": N|null}; grow scenarios raise it
GET    /__test/state                       -> instances (identity, ports, usage, capacities, devices), faults, barriers
GET    /__test/audit?after_seq=N           -> {"records":[{seq,kind,service,method,path,body,status_code,response,mutation,timestamp}]}
POST   /__test/faults                      -> FaultSpec patch (only given keys change); returns active faults
DELETE /__test/faults                      -> reset faults to defaults
POST   /__test/usage                       -> {"instance_id","used_bytes"}
POST   /__test/devices                     -> {"instance_id","device_path", any of locked_key_count, borrowed_slot_count,
                                              active_read_count, active_write_count, inflight_*_tasks, used_bytes}
POST   /__test/present_devices             -> {"instance_id","device_path", optional mode ("devdax" default), size_bytes, align_bytes}:
                                              the fake presence watcher reports the path present without attaching it
POST   /__test/instances/{id}/reregister   -> {"bump":"registration_time"|"endpoint"|"both"}
POST   /__test/barriers                    -> {"instance_id","operation":"drain"|"evict"|"add","when":"before"|"after","name"}
POST   /__test/barriers/{name}/release
```

FaultSpec:

```json
{"coordinator": {"unavailable": false, "undeclared_capacity": [], "null_ratio": [],
                 "shared_dax": [], "unregistered": [], "worker_ip_override": {"mp-receiver": null},
                 "identity_flip": {"instance_id": "mp-donor", "field": "registration_time", "every_n_reads": 2},
                 "delayed_capacity_seconds": 0.0},
 "mp": {"mp-donor": {"status_unavailable": false, "adapters": 1, "unhealthy": false, "closing": false,
                     "hotplug_disabled": false, "evict_409_count": 0, "add_fail_count": 0,
                     "add_always_fail": false, "remove_route_failure": false}}}
```

Notes:

- `identity_flip` reports the altered field on every Nth `/instances` read
  (registration_time + 1.0 or http_port + 1), so a sandwich read mismatches.
- `reregister` with `endpoint` toggles the advertised `http_port` between the
  primary and alternate listener; both serve the same app so the MP stays
  reachable. `registration_time` assigns a strictly newer epoch.
- `delayed_capacity_seconds` keeps publishing the pre-change capacity in
  `/instances/usage` for that long after an add/evict.
- Barriers are one-shot: the next matching mutation pauses (before touching
  state or after it, before the response) until released. Every mutation
  attempt after body validation counts, including ones that end in 404/409.
- Audit `seq` starts at 1 after reset and is strictly increasing. Admin-driven
  changes (usage, devices, reregister) add `mutation` records under the
  instance's service with the admin path.

## Manifests

`manifests/base` is the production shape: one Deployment (replicas 1,
Recreate), dedicated ServiceAccount + namespace-scoped Lease Role, a
pre-created Lease, a ReadWriteOncePod PVC, a ConfigMap with
`actuation_enabled: false`, liveness/readiness probes, and no dependency on
the MP Coordinator Deployment. `overlays/kind` adds the two test services
and E2E timings; `overlays/two-replicas` is a test-only Lease-handoff
overlay on shared hostPath storage; `overlays/hardware` is the template for
the real two-worker Device-DAX gate (`make e2e-mp-memory-coordinator-hardware`).

## What it does not prove

No Device-DAX node is created, no memory moves between servers, no host
access is revoked, and no live KV data migrates. The mock allocator models a
logical FREE/ASSIGNED map of pre-existing server-local devices only.
