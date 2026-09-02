# MP Memory Coordinator on Kubernetes

This example deploys the standalone MP Memory Coordinator with Kustomize. It
does not deploy the MP Coordinator, MP servers, Device-DAX devices, or the
outside Memory Allocation service; those must already exist and be reachable
from the coordinator Pod.

The deployment starts in observation mode (`actuation_enabled: false`). Keep it
there until the adopted inventory and dry-run proposals match the intended
donor and receiver.

## Prerequisites

- A Kubernetes cluster with a default StorageClass that supports
  `ReadWriteOncePod`, plus `kubectl` with Kustomize support.
- A pinned LMCache image containing the `lmcache mp-memory-coordinator`
  command, available to the cluster.
- A reachable MP Coordinator and at least two registered MP servers. Each
  server must report L2 usage, register a unique `metadata.worker_ip`, and
  expose exactly one healthy, private, hotplug-enabled Device-DAX adapter.
  Operator-managed engines must set `spec.coordinator.l2EventReporting: true`.
- A reachable outside Memory Allocation service implementing exactly:

  ```text
  GET  /api/v2/apps/lmcache
  POST /api/v2/apps/lmcache/deallocations
  POST /api/v2/apps/lmcache/allocations
  ```

- At least one runtime Device-DAX allocation the coordinator may manage. The
  path must be active at adapter index greater than zero and appear under the
  same worker IP in the outside status response. Bootstrap devices at index
  zero are never movable. The coordinator picks this up automatically every
  cycle; no allowlist is required.

The coordinator Pod does not need a GPU, privileged mode, or a Device-DAX
mount. It does need network access to the Kubernetes API, the MP Coordinator,
the outside allocator, and every registered MP server HTTP endpoint.

## 1. Validate the outside allocator

From the repository root, run the conformance test against a development
instance. The test deallocates and restores one assigned runtime device, so do
not point it at an allocation that may not be temporarily changed.

```bash
export OUTSIDE_API_URL=http://memory-allocation.example:8080
make test-outside-api-conformance OUTSIDE_API_URL="$OUTSIDE_API_URL"
curl -fsS "$OUTSIDE_API_URL/api/v2/apps/lmcache" | jq
```

## 2. Configure the example

Edit `kubernetes/config/mp-memory-coordinator.yaml`:

- Set `mp_coordinator_url` to the MP Coordinator base URL. An
  operator-managed `LMCacheCoordinator` named `my-coordinator` in this
  namespace normally uses `http://my-coordinator.lmcache-system.svc:9300`.
- Set `memory_allocation_url` to the allocator **base URL only**, for example
  `http://memory-allocation-service.lmcache-system.svc:8080`. Do not append an
  API path: the client calls `<base>/api/v2/apps/lmcache` for status and
  `<base>/api/v2/apps/lmcache/allocations` or
  `<base>/api/v2/apps/lmcache/deallocations` for mutations.
- Keep `actuation_enabled: false` for the initial rollout.
- `adoption_file` can stay empty unless an operator must approve every path
  by hand.
- Confirm that the thresholds, Device-DAX path prefix, and cooldown match the
  fleet.

### Which devices the coordinator will manage

Each cycle adopts a live device only while the outside allocator lists its
exact path under the same worker IP the MP instance registered. Nothing has to be transcribed, and a path that changes
with its Pod name is re-derived rather than going stale. Confirm what was
picked up, and why anything was declined, in `/status`:

```bash
curl -s http://localhost:9400/status | jq '{inventory, discovery: .last_cycle.discovery}'
```

To approve a path explicitly instead, fill in
`kubernetes/config/adoption.yaml` with the exact managed allocation, then point
`adoption_file` at `/etc/lmcache/adoption.yaml`:

```yaml
allocations:
  - worker_ip: 192.168.0.40
    device_path: /dev/dax-cxl/NAMESPACE_POD_NAME/dax0.1
    allocation_size_gib: 64
    device_map_size_bytes: 68719476736
```

That allowlist is applied only while the persistent journal is uninitialized,
and a restart never replaces the adopted inventory. `lmcache
mp-memory-coordinator --config C --adopt allowlist.yaml` adopts explicitly at
any time, but adopts and exits without starting the control loop -- stop the
coordinator before running it, since the journal has no cross-process lock.

Set the `images` override in `kubernetes/kustomization.yaml` to a pinned tag or
digest containing this feature. The checked-in `REPLACE_WITH_RELEASE_TAG`
deliberately prevents an unreviewed image from starting. To build the
standalone image from this checkout:

```bash
export IMAGE=registry.example.com/lmcache/standalone:mp-memory-coordinator
docker build --target lmcache-final -f docker/Dockerfile.standalone \
  -t "$IMAGE" .
docker push "$IMAGE"
```

For that image, replace the `images` block with:

```yaml
images:
  - name: lmcache/standalone
    newName: registry.example.com/lmcache/standalone
    newTag: mp-memory-coordinator
```

Use `digest: sha256:...` instead of `newTag` when deploying by digest.

The manifest overrides the image's normal entrypoint with
`lmcache mp-memory-coordinator`; the coordinator does not start vLLM.

Optionally validate the strict YAML configuration before deploying:

```bash
docker run --rm --entrypoint lmcache \
  -v "$PWD/examples/mp_memory_coordinator/kubernetes/config:/etc/lmcache:ro" \
  "$IMAGE" mp-memory-coordinator \
  --config /etc/lmcache/mp-memory-coordinator.yaml --check
```

## 3. Render and deploy

Inspect the generated resources, then apply them:

```bash
kubectl kustomize examples/mp_memory_coordinator/kubernetes
kubectl apply -k examples/mp_memory_coordinator/kubernetes
kubectl -n lmcache-system rollout status \
  deployment/lmcache-mp-memory-coordinator --timeout=5m
```

The Kustomization creates namespace `lmcache-system` and generates the
configuration ConfigMap from `kubernetes/config/`. The deployment deliberately
uses one replica, `Recreate` strategy, a pre-created Kubernetes Lease, and a
dedicated `ReadWriteOncePod` PVC for the crash-safe journal.

If the rollout does not become ready, inspect the reason instead of deleting
the journal:

```bash
kubectl -n lmcache-system get pods,pvc,lease
kubectl -n lmcache-system logs deployment/lmcache-mp-memory-coordinator
```

## 4. Verify observation mode

Forward the read-only HTTP service in a separate terminal:

```bash
kubectl -n lmcache-system port-forward \
  service/lmcache-mp-memory-coordinator 9400:9400
```

Then verify health, leadership, adoption, and the last observation cycle:

```bash
curl -fsS http://127.0.0.1:9400/healthz | jq
curl -sS http://127.0.0.1:9400/readyz | jq
curl -fsS http://127.0.0.1:9400/status | jq \
  '{leader, initialized, actuation_enabled, inventory, active_move, counters, last_cycle}'
kubectl -n lmcache-system get lease lmcache-mp-memory-coordinator \
  -o jsonpath='{.spec.holderIdentity}{"\n"}'
curl -fsS http://127.0.0.1:9400/metrics | grep '^lmcache_memcoord_'
```

Require `leader: true`, `initialized: true`, `actuation_enabled: false`, the
exact approved inventory, a reachable MP Coordinator, and no journal error.
Readiness can remain `503` until Lease acquisition, adoption, and the first
inventory reconciliation finish.

Under an eligible LOW/HIGH workload, wait for `stable_samples` accepted
samples and check the dry-run result:

```bash
curl -fsS http://127.0.0.1:9400/status | jq \
  '.last_cycle | {proposal, rejections, error}'
```

The expected donor, receiver, and device should appear in `proposal`, with an
`actuation_disabled` rejection. Confirm in the allocator logs that no mutating
POST was sent. Observe this mode for the intended canary window before
enabling changes.

## 5. Enable and verify one canary move

Change only `actuation_enabled` to `true` in
`kubernetes/config/mp-memory-coordinator.yaml`, then render, review, and apply
the Kustomization again. The generated ConfigMap name changes and triggers a
`Recreate` rollout; the PVC preserves the existing journal and adoption.
Do not apply this or any other configuration update while `/status` reports an
active or `BLOCKED` move.

```bash
kubectl kustomize examples/mp_memory_coordinator/kubernetes
kubectl apply -k examples/mp_memory_coordinator/kubernetes
kubectl -n lmcache-system rollout status \
  deployment/lmcache-mp-memory-coordinator --timeout=5m
```

Watch `/journal`. A successful move records effects in this order:

```text
donor_drain -> donor_evict -> deallocate -> allocate -> receiver_add
```

It finishes in `history[-1]` with `state: COMPLETE` and
`outcome: SUCCEEDED`. Also verify the outside status at
`/api/v2/apps/lmcache`, both MP servers' `/reconfigure/dax/status`, and the MP
Coordinator's `/instances/usage` capacity totals.

## Safety constraints

- Keep exactly one replica, `Recreate` strategy, the pre-created Lease, and
  the `ReadWriteOncePod` journal PVC. The Lease alone is not a fencing token.
- Never delete or replace the PVC or `journal.json`, downgrade the image, or
  detach the journal while a move is active or `BLOCKED`.
- Setting `actuation_enabled: false` prevents new moves; it does not cancel or
  pause recovery of a move already stored in the journal.
- A `BLOCKED` move makes `/readyz` return `503`. Preserve all evidence and
  reconcile the allocator and Device-DAX state manually before taking further
  action.
