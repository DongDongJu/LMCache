# MP native KV-event handoff

**Measured matrix and cleanup verified for the declared profile.** All required live rows are accounted for: 78 PASS and 6 documented idle/pause limitations. The final harness/evidence delivery is the commit containing this report.

The implementation carries immutable engine-native bindings through STORE, tracks committed CPU residency across required ranks/groups/L1 owners, and drains the shared subscriber through the existing engine publishers. Capability: `kv_event_native_bindings_v1`.

| Component | Dependency order |
| --- | --- |
| LMCache | upstream `b9a5ce02` → unchanged #5524 `08f0d18d` → extraction `647f11dd` → MP contract `ac6b4a1d` → integrations `1364a64c` → harness/evidence: commit containing this report |
| vLLM | base `db9527a4` → admission bridge `84b4f71f` |
| SGLang | base `43e4e847` → native bridge `14f0725d` → launcher compatibility `a3ee7a8f` → replay readiness `bf0a5c6f` |
| Dynamo | base `2093662a` → optional native replay `67a38a7f` for SGLang; tested vLLM image uses the base runtime |

Local companion metadata is verified: DongDongJu `<commisori28@gmail.com>` author/committer, Dongjoo Seo `<dongjoo.seo1@samsung.com>` co-author, and both DCO sign-offs in that order. Exact revisions, trees, patch/image hashes, per-scenario evidence and source-row indices are in [acceptance.json](acceptance.json).

| Profile | Evidence state |
| --- | --- |
| vLLM TP1/ZMQ/C256, B16 | Required scenarios verified; ordinary idle and paused-child windows are documented LIMITATION with active convergence PASS |
| vLLM TP2/gRPC/C16, B16 | 18 final rows PASS, including two-worker rank faults |
| SGLang TP1/ZMQ/C256, B16 | 26 PASS; 2 paused-child LIMITATION rows with active convergence PASS; normal idle PASS |
| SGLang TP2/gRPC/C16, B16 | All 18 final rows PASS |
| Optional llm-d, real NIXL P/D, minikube | NOT RUN |

The compact [vLLM](vllm-t03-trace.json) and [SGLang](sglang-t03-compact-trace.json) T03 traces correlate router selection, GPU prefix, exact MP L1 keys, successful H2D/engine consumption, and output. Their referenced raw-source hashes were rechecked. [Warm-replay audit](/mnt/nvme1/dongjoo-mp-kv-events/sglang-final-warm-replay-audit.json) proves final SG sequence-zero replay, new publisher identity, GPU hit 0 / CPU hit 512 tokens, and a 512-token/6,291,456-byte actual load on both owners. The old-image startup/replay failures remain preserved in the evidence index.

## Scoped overhead

Each pair uses the same source/image/GPU and 60 identical prompt/output/token rows, with 30 cold+30 repeat requests at concurrency 1 and fixed diagnostics. These are observed measurements without a general-performance or significance claim.

| Metric | vLLM enabled vs disabled | SGLang enabled vs disabled |
| --- | --- | --- |
| Request throughput | +0.60% | -4.72% |
| Cold TTFT p50 | +3.60% | +1.52% |
| Cold TTFT p95 | +4.24% | +1.15% |
| Repeat TTFT p50 | +2.45% | +1.98% |
| Repeat TTFT p95 | -1.37% | +6.40% |

Both disabled arms have zero MP event subscribers/native GPU or CPU events. Enabled queue peak was 2 and drained to 0, with one subscriber per private MP. SG metadata tracing recorded 43,184 attributed Python bytes live and 36,680 after clear plus one barrier binding; this excludes prior metadata and native/shared allocations. CPU/RSS are owned-process aggregates, not request-level or unique-page attribution. Event-lag evidence is separate: final SG idle publication was observed in 16–17 ms and routing in 631–648 ms after physical eviction, without inference.

Raw evidence archive locations are the absolute `/mnt/nvme1/dongjoo-mp-kv-events/` and `/mnt/nvme1/dongjoo/mp-kv-events/` paths recorded in the JSON. Large raw logs stay in those archives; this bundle includes both compact traces.

## Reproduce the profiles

From the LMCache checkout, follow the tracked [harness README](../README.md) for clean-host prerequisites, companion patches, the Rust 1.96.1 runtime wheel, image builds/imports, and optional thin-runtime mounts. Use a new run root so saved evidence is not overwritten:

```bash
export KV_PYTHON=python3.12
export KV_NODE=control-190
export KV_REPRO=/mnt/nvme1/mp-native-kv-events-reproduce
export KV_STATE="$KV_REPRO/state"
export LMCACHE_E2E_WORK_ROOT="$KV_REPRO/work"
mkdir -p "$KV_STATE" "$KV_REPRO/evidence"
```

Build each engine into `--state "$KV_STATE"` using the README and run engines sequentially. Preserve the build reports with their pinned image/native/runtime-volume identities. The following commands reproduce the profiles; captured measured-run commands and artifacts are indexed in acceptance.json.

### vllm-tp1

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --tp 1 --artifacts "$KV_REPRO/evidence/vllm-tp1-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite core --artifacts "$KV_REPRO/evidence/vllm-tp1-core"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite capacity --artifacts "$KV_REPRO/evidence/vllm-tp1-capacity"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite recovery --artifacts "$KV_REPRO/evidence/vllm-tp1-recovery"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/vllm-tp1-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### vllm-overflow

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --tp 1 --artifacts "$KV_REPRO/evidence/vllm-overflow-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 8 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite overflow --artifacts "$KV_REPRO/evidence/vllm-overflow-overflow"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/vllm-overflow-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### vllm-tp2

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --tp 2 --artifacts "$KV_REPRO/evidence/vllm-tp2-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --router dynamo --tp 2 --transport grpc --chunk-size 16 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite core --artifacts "$KV_REPRO/evidence/vllm-tp2-core"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite capacity --artifacts "$KV_REPRO/evidence/vllm-tp2-capacity"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite tp --artifacts "$KV_REPRO/evidence/vllm-tp2-tp"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite recovery --cases T08 --artifacts "$KV_REPRO/evidence/vllm-tp2-recovery"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/vllm-tp2-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### vllm-overhead

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine vllm --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite overhead --artifacts "$KV_REPRO/evidence/vllm-overhead-enabled"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/vllm-overhead-enabled-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py reconfigure-events --state "$KV_STATE" --events-disabled --artifacts "$KV_REPRO/evidence/vllm-overhead-mode-change"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine vllm --suite overhead --artifacts "$KV_REPRO/evidence/vllm-overhead-disabled"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/vllm-overhead-disabled-collection"
$KV_PYTHON tests/e2e/mp_kv_events/overhead.py "$KV_REPRO/evidence/vllm-overhead-enabled/overhead.json" "$KV_REPRO/evidence/vllm-overhead-disabled/overhead.json" --output "$KV_REPRO/evidence/vllm-overhead-comparison.json"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### sglang-tp1

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --tp 1 --artifacts "$KV_REPRO/evidence/sglang-tp1-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite core --artifacts "$KV_REPRO/evidence/sglang-tp1-core"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite capacity --artifacts "$KV_REPRO/evidence/sglang-tp1-capacity"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite recovery --artifacts "$KV_REPRO/evidence/sglang-tp1-recovery"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/sglang-tp1-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### sglang-overflow

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --tp 1 --artifacts "$KV_REPRO/evidence/sglang-overflow-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 8 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite overflow --artifacts "$KV_REPRO/evidence/sglang-overflow-overflow"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/sglang-overflow-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### sglang-tp2

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py preflight --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --tp 2 --artifacts "$KV_REPRO/evidence/sglang-tp2-preflight"
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --router dynamo --tp 2 --transport grpc --chunk-size 16 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite core --artifacts "$KV_REPRO/evidence/sglang-tp2-core"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite capacity --artifacts "$KV_REPRO/evidence/sglang-tp2-capacity"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite tp --artifacts "$KV_REPRO/evidence/sglang-tp2-tp"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite recovery --cases T08 --artifacts "$KV_REPRO/evidence/sglang-tp2-recovery"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/sglang-tp2-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

### sglang-overhead

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py deploy --state "$KV_STATE" --cluster k3s --node "$KV_NODE" --engine sglang --router dynamo --tp 1 --transport zmq --chunk-size 256 --event-log-size 32768 --model-cache "$KV_REPRO/models"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite overhead --artifacts "$KV_REPRO/evidence/sglang-overhead-enabled"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/sglang-overhead-enabled-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py reconfigure-events --state "$KV_STATE" --events-disabled --artifacts "$KV_REPRO/evidence/sglang-overhead-mode-change"
$KV_PYTHON tests/e2e/mp_kv_events/run.py test --state "$KV_STATE" --engine sglang --suite overhead --artifacts "$KV_REPRO/evidence/sglang-overhead-disabled"
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/sglang-overhead-disabled-collection"
$KV_PYTHON tests/e2e/mp_kv_events/overhead.py "$KV_REPRO/evidence/sglang-overhead-enabled/overhead.json" "$KV_REPRO/evidence/sglang-overhead-disabled/overhead.json" --output "$KV_REPRO/evidence/sglang-overhead-comparison.json"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
```

## Limits and cleanup

- Scope: dense text-only full attention; private CPU/L1; LMCache-driven transfer; TP1/TP2 with DP=PP=1; C divisible by native B. Private CPU-addressable Device-DAX follows the contract but hardware was not exercised.
- vLLM idle publication needs an engine step. SGLang normal idle passed; paused scheduler/model children cannot receive/publish until resumed. Active post-resume convergence is separate.
- After MP replacement, vLLM requires an observed unhealthy→healthy heartbeat edge and restored registrations. SGLang requires engine restart and discovery refresh before inference. Both supported procedures have final TP1 evidence.
- MP retained replay has no snapshot and can leave older warm entries undiscovered. Native replay is separately bounded and fails closed on gaps/truncation/incarnation changes; an undetectable final PUB-frame loss remains an existing transport limit.
- The tested thin SG runtime requires the exact read-only `flashinfer_cubin` and `nvidia` NVMe directories and their manifest. It is not a standalone image. The new Dynamo wheel is ABI3/manylinux_2_39 and targets the tested Ubuntu 24.04/glibc 2.39 runtime.
- Persistent CUDA caches and separate setup warmup address preserved first-use PTX JIT timeout diagnostics; setup timing is excluded from measured inference.
- Strict docs retain 31 identical baseline diagnostics, including an existing `code-block` ERROR; zero new diagnostics, not a clean strict build.

After final collection, use the saved-state cleanup commands below. Plugin cleanup is conditional on this harness having installed the recorded plugin. Preserve models/evidence and unrelated workloads. The immutable [cleanup receipt](cleanup.json) records PASS: no task workloads, owned worker image references, GPU compute applications, or private build daemon remain. Unrelated identity verification is scoped from the saved restoration baseline; initial preflight lacked those identifiers. Potentially shared dependency images and earlier shared-Docker orphan build-cache layers were retained rather than broadly pruned. Models, raw artifacts, compiler caches, wheels and runtime directories are preserved.

```bash
$KV_PYTHON tests/e2e/mp_kv_events/run.py collect --state "$KV_STATE" --artifacts "$KV_REPRO/evidence/final-collection"
$KV_PYTHON tests/e2e/mp_kv_events/run.py cleanup --state "$KV_STATE" --namespace kv-events-e2e
$KV_PYTHON tests/e2e/mp_kv_events/bootstrap.py cleanup --state "$KV_REPRO/bootstrap.json"  # only if this reproduction installed that recorded plugin
```

The report records completed validation and cleanup. Inspect the commit containing this report for the final harness/evidence changes and DCO trailers; its own SHA is intentionally omitted to avoid circular provenance. The eight preceding task commits have a bundled metadata audit. Final source hooks and the packaging-helper roundtrip/type checks passed; receipts are indexed in acceptance.json. The containing commit also checks the report files. Preserved failed diagnostics remain distinct from their successful corrections.
