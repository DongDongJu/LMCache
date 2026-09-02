# Mock Memory Allocation service

Development-only, strict mock of the *outside* Memory Allocation service that the
MP Memory Coordinator talks to (PLAN.md Section 2 / Phase 1A). It is test code:
it is never installed as LMCache production code, never imports
`lmcache.v1.mp_memory_coordinator`, and no module under `lmcache/` may import it.

Its purpose is to prove API serialization, logical DAX reservation, orchestration
order, and recovery behaviour against the frozen API, with fixed per-server paths
and exact request/response field spellings.

## Run

```bash
uv run python -m \
  tests.e2e.mp_memory_coordinator.mock_memory_allocation_service \
  --fixture tests/e2e/mp_memory_coordinator/fixtures/two_server_local_dax.yaml \
  --public-host 127.0.0.1 --public-port 18080 \
  --admin-host 127.0.0.1 --admin-port 19090
```

Optional `--state-file PATH` persists the full state (device states and seen
request IDs) atomically after every mutation; on restart an existing file is
loaded instead of the fixture, and `POST /__test/reset` reloads the fixture.

With Docker Compose (ports mapped to `127.0.0.1:18080` / `127.0.0.1:19090`):

```bash
docker compose -f tests/e2e/mp_memory_coordinator/dev/docker-compose.yaml up --build
```

Tests:

```bash
uv run pytest -q tests/e2e/mp_memory_coordinator/test_mock_memory_allocation_service.py
```

## Public listener (frozen API only)

```bash
curl -s http://127.0.0.1:18080/api/v2/apps/lmcache | jq

curl -s -X POST http://127.0.0.1:18080/api/v2/apps/lmcache/deallocations \
  -H 'content-type: application/json' \
  -d '{"request_id":"shrink-0001","target_node":"192.0.2.40",
       "device_path":"/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"}'

curl -s -X POST http://127.0.0.1:18080/api/v2/apps/lmcache/allocations \
  -H 'content-type: application/json' \
  -d '{"request_id":"grow-0001","target_node":"192.0.2.41","request_size_gib":64,
       "mode":"devdax","purpose":"lmcache-dax","access":"exclusive"}'
```

* Status is the bare `{target_node: [assigned runtime paths]}` mapping for every
  configured node (empty arrays included); bootstrap devices never appear.
* Requests are strict (`extra="forbid"`, exact types): a missing, renamed, or extra
  field is a 422 and mutates nothing. The allocation request field is
  `request_size_gib`; the response field is `requested_size_gib`.
* Deallocation flips one assigned runtime device on `target_node` to free.
  Allocation selects the lexicographically first free runtime device on
  `target_node` whose size equals `request_size_gib` exactly. Paths are never
  generated, renamed, combined, or moved between nodes.
* Request IDs are remembered; a repeated `request_id` is rejected with 409 rather
  than replayed. The mock is deliberately **not** idempotent: a blind retry of a
  POST may fail or consume another free device, which is exactly why the
  production client must never retry a POST automatically.
* Error bodies are `{"error": "<message>"}` (422 uses FastAPI's default
  `{"detail": [...]}`). Codes: 404 unknown node/path, 409 wrong owner / already
  free / duplicate request ID / no matching free device, 403 bootstrap path.
  These codes and shapes are development behaviour only.
* `/`, `/docs`, `/openapi.json`, `/redoc`, and every `/__test/*` path are 404.

## Admin listener (test-only)

| Route | Purpose |
|-------|---------|
| `GET /__test/health` | `{"status":"ok","fixture":..., "seq": <last audit seq>}` |
| `POST /__test/reset` | Reload fixture, clear faults/barriers/audit, rewrite state file; returns state. Optional body `{"pool_budget_gib": N}` applies a pool budget (default: unlimited) |
| `POST /__test/pool_budget` | `{"pool_budget_gib": N}` or `{"pool_budget_gib": null}` (unlimited): the global assigned runtime GiB an allocation may reach; a request that would exceed it is refused 409 before any fault or barrier and mutates nothing. A deallocation frees budget |
| `GET /__test/state` | Nodes, devices (role/state), per-node and global GiB accounting, `pool_budget_gib`, seen request IDs, faults, barriers |
| `GET /__test/audit?after_seq=N` | Ordered `request` / `response` / `mutation` records with strictly increasing `seq` |
| `POST /__test/faults` | Install a `FaultSpec` (replaces one with the same operation+mode) |
| `DELETE /__test/faults` | Clear all faults |
| `POST /__test/barriers` | `{"operation","when":"before"|"after","name"}`; the next matching request blocks there |
| `POST /__test/barriers/{name}/release` | Release a barrier (204); barriers are one-shot |

The admin app does not mount the public routes. Both listeners share one event
loop, which barriers rely on.

## Fault modes

`POST /__test/faults` takes `{"operation": "deallocate"|"allocate", "mode": ...,
"count": 1, ...}`. A fault is consumed by the next `count` requests of that
operation that pass schema validation.

| Mode | Effect |
|------|--------|
| `fail_before_mutation` | Respond `status_code` (default 500) with an error; no mutation |
| `commit_then_drop` | Commit and audit the mutation, wait `delay_seconds`, then abort after the response headers: no valid `DONE` body reaches the client |
| `delay` | Sleep `delay_seconds`, then respond normally |
| `wrong_echo` | Mutate; respond with `echo_field` replaced by `"wrong-" + original` |
| `missing_field` | Mutate; respond without `missing_field_name` |
| `wrong_size` | Mutate; every size field set to `size_gib_override` |
| `invalid_path` | Mutate; `device_path` replaced by `path_override` |
| `insufficient_capacity` | Respond 409; no mutation even if a device is free |

The E2E harness resets the mock with `pool_budget_gib` equal to the fixture's
initially assigned total (64 GiB): the coordinator tries to *grow* the HIGH
receiver before moving a donor device, and with an exhausted pool that probe
is refused (`NOT_SERVED`) so every move scenario keeps its exact
drain/evict/deallocate/allocate sequence. Grow scenarios raise the budget.

## What it does not simulate

No Device-DAX node is created, no memory moves between servers, no host access is
revoked, no udev/IOMMU behaviour is validated, and no real mmap is torn down. It
models neither a switch region nor a shared pool: a runtime path stays local to
the worker the fixture declares it on. Ordering (deallocate before allocate) and
equal released/requested/granted sizes are enforced by the coordinator, not here.
