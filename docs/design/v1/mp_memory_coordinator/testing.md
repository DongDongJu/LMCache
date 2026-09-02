# Testing the MP Memory Coordinator against a real outside API

This guide covers the *real-service* test path: your own implementation of
the frozen Memory Allocation API (developed in any repository), real LMCache
MP servers with Device-DAX, the real MP Coordinator, and the MP Memory
Coordinator from this repository (`lmcache/v1/mp_memory_coordinator/`, design
in [overall.md](overall.md)). Load is generated with `lmcache bench server`;
DAX fill patterns can be captured and replayed offline with
`lmcache trace replay`.

The fake-based CI E2E in `tests/e2e/mp_memory_coordinator/` is independent
of this path; only the URL-based conformance test is reused here.

Everything below runs from the repository root with the project
environment active:

```bash
source .venv/bin/activate   # or prefix commands with .venv/bin/
export LMCACHE_TRACK_USAGE=false
```

---

## 0. Prerequisites and topology

| Item | Donor worker (`W1`) | Receiver worker (`W2`) |
|---|---|---|
| Host IP (`metadata.worker_ip`, the outside `target_node`) | `W1_IP` | `W2_IP` |
| Bootstrap Device-DAX, always active in LMCache, never in outside status | `/dev/dax-cxl/.../dax0.0` | `/dev/dax-cxl/.../dax0.0` |
| Runtime candidate(s), pre-provisioned, same size | `dax0.1` (assigned), `dax0.2` (free) | `dax0.1`, `dax0.2` (free) |
| MP server ZMQ / HTTP | `tcp://W1_IP:5555` / `http://W1_IP:9000` | `tcp://W2_IP:5555` / `http://W2_IP:9000` |

- Every runtime path is a pre-existing character device bound to exactly one
  worker; the outside service only flips a logical FREE/ASSIGNED state and
  returns *pre-existing* paths. Nothing here creates, renames, or rebinds a
  device, and LMCache never runs `ndctl`/`daxctl` or writes sysfs.
- `MOVE_SIZE_GIB` must equal one complete candidate device (64 below) and
  keep the donor's projected ratio `used / (capacity - 64 GiB) <= 0.70`.
- The outside service must be the **single writer** for these paths.

Shell variables used below:

```bash
export OUTSIDE_API_URL=http://<your-service-host>:8080
export COORD_URL=http://<coordinator-host>:9300
export W1_IP=192.168.0.40  W2_IP=192.168.0.41
export DAX_ROOT=/dev/dax-cxl/NAMESPACE_POD_NAME      # fixed parent visible on both hosts
```

---

## 1. Conformance-test your outside API first

The frozen contract is `PLAN.md` section 2. Before touching LMCache, prove
your implementation against it by URL (plain `requests`, no LMCache code):

```bash
make test-outside-api-conformance OUTSIDE_API_URL=$OUTSIDE_API_URL
# equivalently
pytest -q tests/e2e/mp_memory_coordinator/test_outside_api_conformance.py \
    -m outside_api --outside-api-url $OUTSIDE_API_URL
```

It checks: bare `{target_node: [device_path, ...]}` status with no wrapper;
one deallocate -> allocate round trip with the exact key sets, echoes and
`released == requested == granted` (the assignment is restored afterwards);
`request_size_gib` in requests vs `requested_size_gib` in responses;
rejection **without mutation** of missing/renamed/extra fields, wrong
literals, and unknown nodes. Run it against a dev instance: the round trip
mutates one assigned path.

Expected initial status for the topology above:

```bash
curl -s $OUTSIDE_API_URL/api/v2/apps/lmcache | jq
# {"192.168.0.40": ["/dev/dax-cxl/.../dax0.1"], "192.168.0.41": []}
```

Contract reminders that implementations most often get wrong:

- Allocation must pick an already-free device **on `target_node`** whose
  size equals `request_size_gib` exactly, never another node or size.
- A repeated `request_id` must not be silently replayed; the coordinator
  never retries a POST, so idempotency is not required, but replay would
  hide a double effect.
- `mode`, `purpose`, `access` are the literals `devdax`, `lmcache-dax`,
  `exclusive`.

---

## 2. Bring up the MP Coordinator and two DAX-backed MP servers

### 2.1 MP Coordinator (unchanged component)

```bash
lmcache coordinator --host 0.0.0.0 --port 9300 \
    --checkpoint-path /var/lib/lmcache-coordinator/checkpoint
```

### 2.2 Donor MP server on `W1` (bootstrap + one runtime device, hotplug on)

`LMCACHE_WORKER_NODE_IP` is the host IP the outside service knows the worker
by (on Kubernetes the operator injects it from `status.hostIP`). It is
registered as `metadata.worker_ip` and is **never** the advertised MP
address.

```bash
LMCACHE_WORKER_NODE_IP=$W1_IP \
lmcache server \
    --host 0.0.0.0 --port 5555 --http-host 0.0.0.0 --http-port 9000 \
    --chunk-size 256 --l1-size-gb 8 \
    --supported-transfer-mode engine_driven \
    --coordinator-url $COORD_URL \
    --coordinator-advertise-ip $W1_IP \
    --coordinator-event-reporting \
    --l2-adapter '{
      "type": "dax",
      "devices": [
        {"device_path": "'$DAX_ROOT'/dax0.0", "max_dax_size_gb": 64},
        {"device_path": "'$DAX_ROOT'/dax0.1", "max_dax_size_gb": 64}
      ],
      "slot_bytes": 1048576,
      "hotplug_enabled": true,
      "num_store_workers": 1, "num_lookup_workers": 1, "num_load_workers": 4,
      "eviction": {"eviction_policy": "LRU", "trigger_watermark": 0.95, "eviction_ratio": 0.05}
    }'
```

### 2.3 Receiver MP server on `W2` (bootstrap only)

Same command with `LMCACHE_WORKER_NODE_IP=$W2_IP`,
`--coordinator-advertise-ip $W2_IP`, and only the `dax0.0` entry in
`devices`.

`slot_bytes` must hold one full chunk of the KV layout you will generate in
step 4 (`chunk_size x layers x kv x heads x head_dim x dtype bytes`); with
1 MiB slots and whole-GiB devices, slot capacity equals map size, which is
what the coordinator's capacity-delta check expects.

Keep the eviction `trigger_watermark` **above** the coordinator's
`high_ratio` (0.75), otherwise local eviction prevents the receiver from ever
reading HIGH.

### 2.4 Verify membership, metadata, capacity, and DAX state

```bash
curl -s $COORD_URL/instances | jq '.instances[] | {instance_id, ip, http_port, registration_time, metadata}'
lmcache query coordinator --url $COORD_URL --api usage      # l2/dax rows: donor 128 GiB, receiver 64 GiB
curl -s http://$W1_IP:9000/reconfigure/dax/status | jq '.adapters[0].status.devices[] | {index, device_path, state, max_dax_size_bytes, live_slot_count}'
curl -s http://$W2_IP:9000/status | jq '{is_healthy, sm: .storage_manager.is_healthy, adapters: [.storage_manager.l2_adapters[] | {type, is_healthy, closing, hotplug_enabled}]}'
```

Stop here if: `metadata.worker_ip` is missing or duplicated, either server
shows zero or two DAX adapters, `hotplug_enabled` is false, or the outside
status disagrees with the DAX state (e.g. a "free" candidate is active in
LMCache).

---

## 3. Start the Memory Coordinator in observation mode and adopt

`memcoord.yaml`:

```yaml
mp_coordinator_url: http://<coordinator-host>:9300
memory_allocation_url: http://<your-service-host>:8080
poll_interval_seconds: 10
stable_samples: 3
high_ratio: 0.75
low_ratio: 0.40
minimum_ratio_gap: 0.25
projected_donor_max_ratio: 0.70
cooldown_seconds: 300
allowed_device_path_prefix: /dev/dax-cxl/
drain_timeout_seconds: 300
state_directory: /var/lib/lmcache-memory-coordinator
actuation_enabled: false          # dry run first
http_port: 9400
leader_election: none             # kubernetes + a pre-created Lease on a cluster
```

Adopt the donor's runtime device explicitly (never discovered); it must be
active at DAX index > 0, listed under `W1_IP` in outside status, and match
the approved size:

```bash
cat > adopt.yaml <<EOF
allocations:
  - worker_ip: $W1_IP
    device_path: $DAX_ROOT/dax0.1
    allocation_size_gib: 64
    device_map_size_bytes: 68719476736
EOF
lmcache mp-memory-coordinator --config memcoord.yaml --check
lmcache mp-memory-coordinator --config memcoord.yaml --adopt adopt.yaml   # prints "adopted ..."
lmcache mp-memory-coordinator --config memcoord.yaml &                    # observation only
curl -s localhost:9400/readyz; curl -s localhost:9400/status | jq '{inventory, last_cycle: .last_cycle.rejections}'
```

With both instances LOW/NORMAL you will see `history_not_stable` and then
no proposal; that is the expected idle state.

---

### 3.1 Watching attach orchestration

If the MP servers run the DAX presence watcher (add `"watch_directory":
"$DAX_ROOT"` to their `--l2-adapter` config), every path in that directory
appears in their DAX status, and the coordinator attaches a present device
only when the outside service lists it under the same worker. To see it work
without a move: with the coordinator in observation mode, assign a free
candidate to `W1` through your outside service and confirm the dry run
reports it; then enable actuation and confirm the single add and the
adoption that follows:

```bash
curl -s http://$W1_IP:9000/reconfigure/dax/status | jq '.adapters[0].status.watcher'   # present_devices lists dax0.2, mode devdax
curl -s -X POST $OUTSIDE_API_URL/api/v2/apps/lmcache/allocations -H 'content-type: application/json'   -d '{"request_id":"grow-w1-1","target_node":"'$W1_IP'","request_size_gib":64,"mode":"devdax","purpose":"lmcache-dax","access":"exclusive"}'
curl -s localhost:9400/status | jq '.last_cycle.attachments'    # dry run: would_attach [".../dax0.2"], no add issued
# actuation on: attached [".../dax0.2"] on one cycle, then discovery adopts it
curl -s localhost:9400/status | jq '{attachments: .last_cycle.attachments, attached: .counters.attached, discovery: .last_cycle.discovery, inventory: [.inventory[].device_path]}'
curl -s http://$W1_IP:9000/reconfigure/dax/status | jq '.adapters[0].status.devices[] | {device_path, state, physical: .physical.mode}'
```

`attachments.skipped` explains every present device that was not attached
(`already attached`, `hotplug disabled`, `mode is system-ram`, `outside
status lists the path under [] ...`, `recent attach failure`). The cycle that
issued the add reports `decision: "attach issued; re-observing next cycle"`
and proposes nothing. `counters.attached` is in-memory (not in `/journal`)
and restarts from zero. Nothing is attached while a move is active, and a
device the outside service does not list under that worker is never
attached however visible it is.

---

## 4. Make the receiver HIGH with `lmcache bench server`

`lmcache bench server` drives a running MP server over ZMQ: a cold pass
`STORE`s synthetic KV for every sequence in `[--start, --end)`, a warm pass
`RETRIEVE`s it and verifies checksums. With `--mode cpu` no GPU is needed
(the server must have been started with
`--supported-transfer-mode engine_driven`, as above).

Bytes stored per sequence = `num_tokens x per-token KV bytes` from
`--kvcache-shape-spec`; size the run so the **receiver** stores more than
`high_ratio x 64 GiB = 48 GiB` while the donor stays below
`low_ratio x 128 GiB = 51 GiB`. Example (adjust `--end` to your layout):

```bash
# Receiver: fill until > 48 GiB is resident in its DAX tier.
lmcache bench server --rpc-url tcp://$W2_IP:5555 --url http://$W2_IP:9000 \
    --mode cpu --num-tokens 8192 --start 0 --end 800 --interval 0

# Donor: a small warm set only (stays LOW).
lmcache bench server --rpc-url tcp://$W1_IP:5555 --url http://$W1_IP:9000 \
    --mode cpu --num-tokens 8192 --start 0 --end 60 --interval 0
```

Watch the pressure the coordinator sees, then the coordinator's own view:

```bash
lmcache query coordinator --url $COORD_URL --api usage
curl -s localhost:9400/status | jq '{history, proposal: .last_cycle.proposal, rejections: .last_cycle.rejections}'
```

After three consecutive accepted samples with receiver `>= 0.75` and donor
`<= 0.40`, `/status.last_cycle.proposal` names `mp-donor -> mp-receiver`,
`device_path .../dax0.1`, `allocation_size_gib 64`, and the rejection list
contains only `actuation_disabled`. Confirm the outside targets are the
**worker IPs** (`donor_worker_ip`, `receiver_worker_ip`), not Pod/MP
addresses. Zero mutating calls have been made: check your service's logs
and both `/reconfigure/dax/status` outputs are unchanged.

---

## 5. Enable actuation and verify the exact move

```bash
sed -i 's/actuation_enabled: false/actuation_enabled: true/' memcoord.yaml
kill %1; lmcache mp-memory-coordinator --config memcoord.yaml &        # journal on disk survives
watch -n2 'curl -s localhost:9400/journal | jq "{state: .active_move.state, effects: (.active_move.effects // {} | keys), last: (.history[-1] // {} | {state, outcome})}"'
```

The effect ledger must appear in exactly this order and end in
`COMPLETE / SUCCEEDED`:

```
donor_drain -> donor_evict -> deallocate -> allocate -> receiver_add
```

Verify each contract point:

```bash
J=$(curl -s localhost:9400/journal)
echo "$J" | jq '.history[-1] | {old_path, new_path, released_size_gib, granted_size_gib, deallocation_request_id, allocation_request_id}'
curl -s $OUTSIDE_API_URL/api/v2/apps/lmcache | jq            # donor: [], receiver: [".../dax0.1"] on W2
curl -s http://$W1_IP:9000/reconfigure/dax/status | jq '.adapters[0].status.devices[] | {device_path, state}'   # dax0.1 = "removed" tombstone
curl -s http://$W2_IP:9000/reconfigure/dax/status | jq '.adapters[0].status.devices[] | {device_path, state}'   # W2 dax0.1 = "active"
lmcache query coordinator --url $COORD_URL --api usage   # donor 128 -> 64 GiB, receiver 64 -> 128 GiB
```

In your outside service's request log check the two POST bodies verbatim:

```json
{"request_id": "<move>-deallocate", "target_node": "192.168.0.40", "device_path": ".../dax0.1"}
{"request_id": "<move>-allocate", "target_node": "192.168.0.41", "request_size_gib": 64,
 "mode": "devdax", "purpose": "lmcache-dax", "access": "exclusive"}
```

and that the responses echoed `request_id`/`target_node` exactly, the
deallocation returned `released_size_gib: 64`, and the allocation returned
a **pre-declared receiver-local** path with `requested_size_gib ==
granted_size_gib == 64`. Total assigned runtime GiB is back at 64.

Then prove the receiver really serves the new capacity: run the bench again
against `W2` with a larger `--end` than before; the cold pass must store
beyond the old 64 GiB without local eviction and the warm pass must verify
checksums cleanly. A second move must **not** start within
`cooldown_seconds` (`/status.last_cycle.rejections` shows `cooldown`).

If the journal shows `BLOCKED`, stop: read `block_reason`, keep the journal
and device state, reconcile the outside service by hand, and only then clear
the state directory. Never delete the journal during an active move.

---

## 6. Capture and replay the DAX workload offline (`lmcache trace replay`)

To reproduce a fill pattern without hardware (for debugging capacity math,
slot sizing, or eviction interplay), record the storage-level trace on an MP
server while the bench runs, then replay it against a fresh
`StorageManager` configured with a DAX adapter over plain files:

```bash
# On the receiver, restart the server with tracing for the bench run:
lmcache server ... --trace-level storage --trace-output /tmp/receiver-fill.lct
lmcache bench server --rpc-url tcp://$W2_IP:5555 --url http://$W2_IP:9000 --mode cpu --start 0 --end 800 --interval 0

# Anywhere (no GPU, no DAX hardware): file-backed "devices" of the same size.
truncate -s 64G /tmp/dax0.0; truncate -s 64G /tmp/dax0.1
lmcache trace info /tmp/receiver-fill.lct
lmcache trace replay /tmp/receiver-fill.lct --l1-size-gb 8 --eviction-policy LRU \
    --l2-adapter '{"type": "dax",
      "devices": [{"device_path": "/tmp/dax0.0", "max_dax_size_gb": 64},
                  {"device_path": "/tmp/dax0.1", "max_dax_size_gb": 64}],
      "slot_bytes": 1048576, "hotplug_enabled": true}'
```

`replay` honors the recorded inter-call timings and reports per-op counts,
so you can confirm how many GiB the receiver run actually pins into the DAX
tier before enabling actuation on hardware.

---

## 7. Cleanup (restore the baseline exactly)

With the Memory Coordinator stopped (or scaled to zero):

```bash
# 1. detach the moved device from the receiver
curl -s -X POST http://$W2_IP:9000/reconfigure/dax/remove -H 'content-type: application/json' \
  -d '{"adapter_index":0,"device_path":"'$DAX_ROOT'/dax0.1","mode":"drain","force":false}'
curl -s -X POST http://$W2_IP:9000/reconfigure/dax/remove -H 'content-type: application/json' \
  -d '{"adapter_index":0,"device_path":"'$DAX_ROOT'/dax0.1","mode":"evict","force":false}'
# 2. give it back to the pool, allocate the same size to the donor, attach the returned path
curl -s -X POST $OUTSIDE_API_URL/api/v2/apps/lmcache/deallocations -H 'content-type: application/json' \
  -d '{"request_id":"cleanup-shrink-1","target_node":"'$W2_IP'","device_path":"'$DAX_ROOT'/dax0.1"}'
curl -s -X POST $OUTSIDE_API_URL/api/v2/apps/lmcache/allocations -H 'content-type: application/json' \
  -d '{"request_id":"cleanup-grow-1","target_node":"'$W1_IP'","request_size_gib":64,"mode":"devdax","purpose":"lmcache-dax","access":"exclusive"}'
curl -s -X POST http://$W1_IP:9000/reconfigure/dax/add -H 'content-type: application/json' \
  -d '{"adapter_index":0,"device_path":"<returned path>","size":"64GiB"}'
# 3. re-adopt and confirm the initial outside status / usage view
rm -rf /var/lib/lmcache-memory-coordinator/journal.json
lmcache mp-memory-coordinator --config memcoord.yaml --adopt adopt.yaml
```

---

## Rejection and failure reference

| `/status.last_cycle.rejections[].reason` | Meaning / fix |
|---|---|
| `missing_worker_ip`, `duplicate_worker_ip` | `LMCACHE_WORKER_NODE_IP` unset or the same on two servers |
| `undeclared_capacity`, `null_usage_ratio` | `--coordinator-event-reporting` off, or capacity not yet declared after a coordinator restart |
| `identity_changed_between_reads` | an MP server re-registered mid-sample; history restarts |
| `history_not_stable` | fewer than three consecutive same-level samples |
| `preflight_failed` | unhealthy/closing adapter, hotplug disabled, zero or two DAX adapters, non-active device |
| `live_ratio_mismatch` | coordinator usage says HIGH/LOW but `/reconfigure/dax/status` totals disagree |
| `no_managed_runtime_device` | nothing managed on the donor, or only the bootstrap device is active; `/status.last_cycle.discovery.skipped` names the reason per live device |
| `projected_donor_ratio` | removing the device would push the donor above 0.70 |
| `cooldown` | a move completed within `cooldown_seconds` |
| `actuation_disabled` | dry run; the proposal is logged only |

Journal `BLOCKED` reasons are always terminal and never retried: drain
deadline (no undrain API), an outside POST whose outcome could not be
proven, a returned path that is not the unique new receiver-local path, or
an outside status that contradicts a DONE response.
