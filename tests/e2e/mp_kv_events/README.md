# MP native KV-event acceptance harness

This harness runs two serving replicas on one selected Kubernetes node. Each
GPU-requesting container owns an engine and a private MP process, with a shared
`/dev/shm`. Dynamo chooses the serving replica. The observer reads native engine
ZMQ publication, and `mp_trace` records actual MP EventBus operations through the
existing server-module API. Neither observer changes cache state or placements.

See the recorded [acceptance matrix](evidence/acceptance.json),
[evidence guide](evidence/README.md), and
[vLLM mixed-prefix trace](evidence/vllm-t03-trace.json) for the measured results,
their exact source pins, and the separately reported limitations.

`versions.lock.yaml` is JSON-compatible YAML. It pins source revisions and image
digests; `build` additionally records every source input's SHA256 and the image
ID. vLLM 0.31.0 retains both the native block identity helpers and the engine
startup API used by Dynamo 1.5.1. The newer 0.31.1 development wheel removed that
startup API and is not selected. SGLang requires the accompanying committed
native-event bridge in the sibling `sglang` checkout. Its companion patch also
restores the model-configuration accessor used by Dynamo 1.5.1, delegating to
SGLang's existing cached resolver. vLLM requires the committed
KV-event request-admission hook in the sibling `vllm` checkout; unsupported
namespaces receive a request-level validation error before EngineCore enqueue.
Both companion patches and their source/hash pins are tracked here.
SGLang's restart profile also requires the tracked Dynamo native replay companion.
Its existing listener replays the existing native publisher's retained batches;
the opt-in nonce handshake proves the live subscription is installed before
replay begins. Sequence deduplication precedes the existing event normalizer.

## Prepare a clean host

The selected profile is Linux amd64, Python 3.12, an NVIDIA driver supporting
CUDA 13, and Blackwell GPUs (the native build selects SM120). Install the NVIDIA
Container Toolkit before creating k3s. Follow the vendor's
[APT repository setup](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then install the tested toolkit packages explicitly:

```bash
sudo apt-get install nvidia-container-toolkit=1.18.1-1 \
  nvidia-container-toolkit-base=1.18.1-1 \
  libnvidia-container-tools=1.18.1-1 libnvidia-container1=1.18.1-1
```

On a host without an existing Kubernetes cluster, download and verify both pinned
k3s artifacts before installing. These commands do not belong on an existing
cluster. K3s [detects NVIDIA runtimes at startup](https://docs.k3s.io/advanced#nvidia-container-runtime-support);
keep its ordinary containerd default and select `runtimeClassName: nvidia` only
for the GPU pods.

```bash
mkdir -p artifacts/mp-kv-events/bootstrap
curl -fL 'https://github.com/k3s-io/k3s/releases/download/v1.36.2%2Bk3s1/k3s' \
  -o artifacts/mp-kv-events/bootstrap/k3s
curl -fL 'https://raw.githubusercontent.com/k3s-io/k3s/v1.36.2+k3s1/install.sh' \
  -o artifacts/mp-kv-events/bootstrap/install.sh
python3 - <<'VERIFY'
from pathlib import Path
import hashlib, json
lock = json.loads(Path('tests/e2e/mp_kv_events/versions.lock.yaml').read_text())
for name, key in [('k3s', 'k3s_amd64_sha256'), ('install.sh', 'installer_sha256')]:
    path = Path('artifacts/mp-kv-events/bootstrap') / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lock['cluster'][key]
VERIFY
sudo install -m 0755 artifacts/mp-kv-events/bootstrap/k3s /usr/local/bin/k3s
sudo env INSTALL_K3S_SKIP_DOWNLOAD=true \
  sh artifacts/mp-kv-events/bootstrap/install.sh server --disable traefik
mkdir -p artifacts/mp-kv-events/kube
sudo install -o "$(id -u)" -g "$(id -g)" -m 0600 /etc/rancher/k3s/k3s.yaml \
  artifacts/mp-kv-events/kube/config
export KUBECONFIG="$PWD/artifacts/mp-kv-events/kube/config"
kubectl get nodes
kubectl get runtimeclass nvidia
```

If the selected node does not already advertise GPUs, install the pinned scoped
plugin. Set `KV_NODE` to that local node's name and `KV_GPUS` to the comma-separated
UUIDs of unused GPUs shown by `nvidia-smi -L`; preserve any GPU used by another
workload. The installer checks the actual visible UUID set and node allocation.
It refuses to add a second plugin when the node already advertises GPUs.

```bash
python3 tests/e2e/mp_kv_events/bootstrap.py render --node "$KV_NODE" \
  --gpu-uuids "$KV_GPUS" --state artifacts/mp-kv-events/bootstrap.json
python3 tests/e2e/mp_kv_events/bootstrap.py install --node "$KV_NODE" \
  --gpu-uuids "$KV_GPUS" --state artifacts/mp-kv-events/bootstrap.json
```

Build tools are Docker (including its buildx plugin), Git, `zstd`, and Python 3.12.
Use a large local filesystem for Docker's build data, source archives and model
cache. The tested compact vLLM image uses about 21.55 GB; the full SGLang image
uses about 31 GB. The optional thin SGLang image uses about 20.4 GB plus 10.4 GB
of unchanged runtime files on the secondary filesystem. Allow additional space
for build intermediates and keep the normal kubelet disk reserve intact. For an
isolated build daemon on a secondary filesystem:

```bash
export LMCACHE_E2E_WORK_ROOT=/mnt/nvme1/mp-kv-events
export KV_BUILD_ROOT="$LMCACHE_E2E_WORK_ROOT/docker-build"
export STATE="$PWD/artifacts/mp-kv-events/state"
export ARTIFACTS="$LMCACHE_E2E_WORK_ROOT/artifacts"
mkdir -p "$KV_BUILD_ROOT" "$STATE" "$ARTIFACTS"
sudo dockerd --host "unix://$KV_BUILD_ROOT/docker.sock" \
  --data-root "$KV_BUILD_ROOT/data" --exec-root "$KV_BUILD_ROOT/exec" \
  --pidfile "$KV_BUILD_ROOT/dockerd.pid" --bridge none --iptables=false \
  --ip-forward=false --ip-masq=false \
  --add-runtime nvidia=/usr/bin/nvidia-container-runtime \
  > "$KV_BUILD_ROOT/dockerd.log" 2>&1 &
export DOCKER_HOST="unix://$KV_BUILD_ROOT/docker.sock"
docker info
```

Prepare companion source checkouts beside LMCache. The companion commits may be
checked out directly from the accompanying local branches, or recreated by
applying the tracked patches to the locked base revisions. Verify the resulting
files against the build manifest; the checked-in wheel repacker verifies both
baseline and replacement hashes and leaves native extension bytes unchanged.

```bash
git clone https://github.com/vllm-project/vllm.git ../vllm
git -C ../vllm checkout db9527a46873454610df6dbedf79a36d6bf1a7f6
git -C ../vllm am "$PWD/tests/e2e/mp_kv_events/patches/vllm-kv-event-admission.patch"
git clone https://github.com/sgl-project/sglang.git ../sglang
git -C ../sglang checkout 43e4e84731119e7ebf5a614826c873cae64e8e51
git -C ../sglang am "$PWD/tests/e2e/mp_kv_events/patches/sglang-native-kv-events.patch"
git clone https://github.com/ai-dynamo/dynamo.git ../dynamo-pinned
git -C ../dynamo-pinned checkout 2093662a4d4e5fd318732935213dde08dc7be818
DYNAMO_PATCH=$(python3 -c 'import json; print(json.load(open("tests/e2e/mp_kv_events/versions.lock.yaml"))["dynamo"]["bridge_patch"])')
git -C ../dynamo-pinned am "$PWD/tests/e2e/mp_kv_events/$DYNAMO_PATCH"
```

For the SGLang Dynamo companion wheel, the build host needs Ubuntu 24.04 amd64,
CUDA development headers in `/usr/local/cuda`, an installed `rustup` executable,
and these build packages:

```bash
sudo apt-get install build-essential clang-18 libclang-18-dev pkg-config \
  protobuf-compiler cmake ninja-build python3.12-dev python3.12-venv
export DYNAMO_WORK="$ARTIFACTS/dynamo-runtime"
python3 tests/e2e/mp_kv_events/build_dynamo_runtime.py \
  --source ../dynamo-pinned --work "$DYNAMO_WORK"
export DYNAMO_WHEEL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["wheel_path"])' \
  "$DYNAMO_WORK/receipt.json")
```

The helper isolates Rust 1.96.1, Cargo artifacts, Python 3.12, Maturin 1.15.0,
and patchelf 0.19.1.0 on the selected large filesystem. It records the locked
release-build command, features, tool versions, source tree, and wheel SHA256.
Pass `--rustup /absolute/path/to/rustup` if it is not on PATH. The aggregated
profile uses the upstream runtime's ordinary dynamic NIXL loading path; this
wheel build does not establish optional NIXL P/D coverage.
Every SGLang `run.py build` receives `--dynamo-wheel "$DYNAMO_WHEEL"` and
`--dynamo-build-receipt "$DYNAMO_WORK/receipt.json"`. Use `--no-import` when
preparing the thin image described below.

## Existing k3s cluster

Prerequisites: the versions of k3s and NVIDIA Container Toolkit in the lock,
working existing NVIDIA drivers, Docker, Python 3.12, and a schedulable node with
at least two free GPUs (four for TP2). This harness does not replace an existing
cluster, install drivers, label unrelated nodes or modify other workloads.
The selected node's `nvidia` RuntimeClass must already resolve to the NVIDIA
container runtime. No GPU time slicing or MIG substitution is used.

From the LMCache checkout:

```bash
python3 tests/e2e/mp_kv_events/run.py preflight --cluster k3s --tp 1 --node control-190
python3 tests/e2e/mp_kv_events/run.py build --engine vllm
python3 tests/e2e/mp_kv_events/run.py deploy --cluster k3s --node control-190 \
  --engine vllm --router dynamo --tp 1 --transport zmq --chunk-size 256
python3 tests/e2e/mp_kv_events/run.py test --suite core \
  --artifacts artifacts/mp-kv-events/vllm-dynamo-tp1
python3 tests/e2e/mp_kv_events/run.py collect \
  --artifacts artifacts/mp-kv-events/vllm-dynamo-tp1
python3 tests/e2e/mp_kv_events/run.py cleanup --namespace kv-events-e2e
```

Build uses source checkouts `../vllm` and `../sglang`, verifies their locked base
revisions and copies only tracked/nonignored source files. Recreated patch commits
may have different commit timestamps; the patch and every replacement file must
match the locked content. It compiles LMCache's
native extension against the selected image's torch/CUDA. The resulting local
image is compacted without changing its installed package/native bytes, then
compressed on the build filesystem. `import_image.py` imports with
`sudo -n python3`: it stages only this task image's unique compressed layer through
a verified read-only bind mount, unpacks with containerd, runs native-import
validation, then removes that temporary blob and mount. The saved import JSON
records every phase. It never relocates shared containerd storage or changes its
metadata databases. An interrupted staging operation must be resolved from that
report before another import; do not delete content-store files manually.
`--build-dir` and `--model-cache` select large local disks; model files are
preserved by cleanup. Set `LMCACHE_E2E_WORK_ROOT` to the selected large filesystem before build/deploy;
without it, paths default to `artifacts/mp-kv-events/work` in the checkout.

`deploy` writes the concrete manifest to `artifacts/mp-kv-events/state/manifest.json`.
Use `render` after build to review the manifest. `test`, `collect` and `cleanup`
verify the saved Kubernetes context and namespace UID. Cleanup deletes only the
saved run label's Pods, Services and ConfigMaps, never the namespace or model
cache. Raw worker traces live under `<model-cache-parent>/runs/<run>/worker-*`
and survive pod deletion. Compiler caches are mounted for both root and the
base image's `/home/dynamo` home. A and B have separate Services and discovered
numeric worker IDs; the runtime's hexadecimal metric label is decoded directly.

Run engines sequentially. After collecting and cleaning up the first run,
repeat build/deploy/test/collect for `--engine sglang`. Reuse the same worker image
for `--tp 2` or `--transport grpc`; use `--chunk-size 16` for C=B and
`--chunk-size 256` for C>B, with native B=16 recorded in the deployment.
`--event-log-size` selects retained history for explicit overflow/restart runs.
The MP server uses one GPU affinity worker per TP rank. This allows the T07
delayed-rank gate to hold one real STORE while the other rank completes its own
D2H transfer and L1 commit.

Use fresh deployments for each profile and fresh processes for each overhead arm.
Execute `core`,
`capacity`, and `recovery` for both engines at TP1/ZMQ/C256; repeat `core` with C16
and gRPC. Run `tp` in TP2, and `overflow` with C256 and `--event-log-size 8`.
These suites preserve independent fixture IDs and emit their own JSON/JUnit:

```bash
python3 tests/e2e/mp_kv_events/run.py test --suite capacity --artifacts artifacts/mp-kv-events/capacity
python3 tests/e2e/mp_kv_events/run.py test --suite recovery --artifacts artifacts/mp-kv-events/recovery
# After cleanup + redeploy with --event-log-size 8 --chunk-size 256:
python3 tests/e2e/mp_kv_events/run.py test --suite overflow --artifacts artifacts/mp-kv-events/overflow
# After cleanup + redeploy with --tp 2:
python3 tests/e2e/mp_kv_events/run.py test --suite tp --artifacts artifacts/mp-kv-events/tp2
```

For performance, run `--suite overhead` once after a fresh enabled deployment and
once after an otherwise identical deployment with `--events-disabled`; use the
same image, GPUs, TP, transport and chunk size. To retain exactly the same
allocated GPU UUIDs, collect the enabled arm, then use:

```bash
python3 tests/e2e/mp_kv_events/run.py reconfigure-events --events-disabled \
  --artifacts artifacts/mp-kv-events/overhead-mode-change
python3 tests/e2e/mp_kv_events/run.py test --suite overhead \
  --artifacts artifacts/mp-kv-events/overhead-disabled
```

The supervisor verifies the projected ConfigMap checksum before stopping both
processes, archives every previous raw trace/provenance file, and starts new MP,
observer and engine processes with empty caches. The pods retain their GPUs.
Omit `--events-disabled` to switch back. Compare the saved measurements
using `overhead.py ENABLED_JSON DISABLED_JSON --output comparison.json`.
A performance result never substitutes for a routing correctness result.

After all serving runs, remove the independently tracked bootstrap plugin with
`bootstrap.py cleanup --state artifacts/mp-kv-events/bootstrap.json` if this
harness installed it. `cleanup --remove-image` removes only the saved serving
image after deleting its pods. Preserve archives on the large filesystem if
another engine's image must replace it to stay above the kubelet disk reserve.
Stop the optional private build daemon by its saved PID after its own containers
are stopped; do not prune unrelated Docker or containerd images.

## Evidence and scope

Core fixtures use exact integer Qwen vocabulary IDs, unique first tokens and an
incomplete terminal token. Every request artifact records the complete token
sequence. The unique first token is followed by fixed natural-language padding
and a fixed chat continuation. The original repeated arbitrary-token padding
produced a whitespace-token mismatch even for the engine's own GPU cache;
`fixture_control.py` verifies strict cold/native-GPU equality on both owners
before the routing matrix. The ordinary SSE oracle requires a successful terminal
choice, positive completion usage, and `[DONE]`; error or partial streams fail.
Correctness generation has temperature zero, one output token and
concurrency one. Before each suite, targeted two-token decode and a repeated
prefix request compile the real attention kernels with a 600-second setup bound.
The supervisor then archives setup traces and restarts both engine and MP to
restore empty caches; the NVIDIA driver cache persists on NVMe. Setup durations
are recorded separately and never counted as measured latency. Targeted warmup
uses discovered Dynamo worker IDs; measured
requests omit targeting/affinity headers. Twenty query-only probes prove router
convergence without warming the GPU, followed by an actual untargeted generation.

The core driver exercises cold/GPU-only, CPU-only, GPU-prefix+CPU-suffix, private
CPU removal and exact native identity/removal scenarios in both A/B directions.
GPU eviction uses actual capacity pressure. A recorded unrelated setup request
provides vLLM's active publication opportunity after MP-only changes; this is not
an idle-publication claim. Before/after snapshots contain native publisher frames,
MP operations and counters, engine lookup/load traces and status. MP retrieve-end
records are generated after the actual CUDA stream completes, with request IDs,
loaded tokens and byte counts. Failed or missing evidence fails the scenario.

`results.json`, `junit.xml`, `REPORT.md`, request records and per-request traces
are written under `--artifacts`. A successful core run alone does not establish
recovery, TP partial-commit faults, request-free idle publication or performance
overhead; their evidence must be reported separately in the complete matrix.

The test supervisor's localhost-only controls require
`LMCACHE_KV_EVENTS_E2E=1`: `POST /restart/engine`, `/restart/mp`, `/pause/engine`
and `/resume/engine` on port 9900. `/reload` is the checksum-verified event-mode
restart used only by the overhead arms. The MP fault-control extension also requires
that environment switch. These controls operate on real transfers and process
groups and are never enabled by normal LMCache startup.

Minikube and llm-d are unselected optional targets. This harness accepts only
`--cluster k3s --router dynamo`; it does not silently switch targets.

Oracle regressions run without either inference engine installed:

```bash
python3 -m unittest discover -s tests/e2e/mp_kv_events -p 'test_*oracle.py'
```

For an admission-only check inside the pinned worker image, reserve one GPU and
run each command in a fresh container with the pinned model cache mounted:

```bash
FLASHINFER_USE_CUDA_NORM=1 python3 /opt/mp-kv-events/negative_sglang_probe.py --output /artifacts/sglang-enabled
FLASHINFER_USE_CUDA_NORM=1 python3 /opt/mp-kv-events/negative_sglang_probe.py --events-disabled --output /artifacts/sglang-disabled
```

The enabled arm requires HTTP 400 for salted, extra-key, and session requests,
then a successful ordinary generation. The disabled arm requires salted
inference to succeed. The script starts and stops its own engine and MP process
groups and records exact argv, versions, responses, and logs. These standalone
results supplement the Kubernetes routing profiles.

A Python-only correction can reuse an already unpacked compact native image:

```bash
python3 tests/e2e/mp_kv_events/incremental_image.py \
  --engine vllm --base-image "$BASE_IMAGE" --base-archive "$BASE_OCI_ARCHIVE" \
  --output "$ARTIFACTS/incremental"
sudo k3s ctr -n k8s.io images import --local "$INCREMENTAL_OCI_ARCHIVE"
```

The builder verifies the base image/config and installed baseline files, rejects
native/dependency source changes, refreshes installed distribution RECORD entries,
and records all native-library hashes. Its small OCI archive intentionally omits
the immutable base layer: import requires the exact base snapshot already present
on that node. Containerd reuses that snapshot; no large base blob is restored.
The JSON report records both required base identity and complete source hashes.
For a thin base image, also pass `--base-state "$STATE/build-sglang.json"` to
preserve its required runtime-volume descriptor in the incremental report.
To apply the Dynamo native replay companion over an older base, also supply
`--dynamo-wheel "$DYNAMO_WHEEL"`,
`--dynamo-build-receipt "$DYNAMO_WORK/receipt.json"`, and
`--dynamo-source ../dynamo-pinned`. This explicit exception records the before
and after hashes of `dynamo/_core.abi3.so` and installs only the verified runtime
wheel and tracked SGLang wrapper. LMCache, SGLang, and CUDA native libraries are
unchanged. The builder rejects changed source pins without the matching wheel.

When the pinned SGLang image cannot fit within the node's disk reserve, preserve
its large immutable runtime directories on NVMe instead of deleting dependencies:

```bash
python3 tests/e2e/mp_kv_events/run.py build --engine sglang --no-import \
  --state "$STATE" --build-dir "$ARTIFACTS/build-sglang" \
  --dynamo-wheel "$DYNAMO_WHEEL" \
  --dynamo-build-receipt "$DYNAMO_WORK/receipt.json" --dynamo-source ../dynamo-pinned
SG_NATIVE_IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' \
  "$STATE/build-sglang.json")
python3 tests/e2e/mp_kv_events/thin_runtime.py \
  --base-image "$SG_NATIVE_IMAGE" --output "$ARTIFACTS/thin-sglang" \
  --state "$STATE/build-sglang.json"
```

This exports `flashinfer_cubin` and the `nvidia` Python runtime directories, hashes
every file and symlink, and produces a smaller image plus a JSON build report.
Keep the reported `external_runtime` descriptor with the image's build state.
`run.py deploy` mounts those exact directories read-only at their original paths
for workers and the frontend. Host paths must already exist. The image and pod
commands run `verify_runtime.py` before any service starts, rejecting missing or
changed mounts. No package metadata, shared library, CUDA header, or JIT dependency
is removed from the combined runtime. The report and content-addressed directory
are required artifacts for reproduction; this thin image is not standalone.

Importing a thin base uses the saved runtime descriptor for its native smoke
check as well:

```bash
sudo python3 tests/e2e/mp_kv_events/import_image.py "$BASE_OCI_ARCHIVE" \
  --scratch "$ARTIFACTS/import-staging" --report "$ARTIFACTS/import.json" \
  --unpacked-bytes "$BASE_UNPACKED_BYTES" --runtime-state "$STATE/build-sglang.json"
```

Take `BASE_OCI_ARCHIVE` and `BASE_UNPACKED_BYTES` from the thin build report's
`compression` record (or `base_compact.compression` after a Python-only update).
Import the small incremental archive afterward with the ordinary `ctr` command.
The staged importer verifies both volume manifests before its native-import
smoke test and preserves the volume descriptor in its report.

For the SGLang MP-incarnation recovery case, pause request admission, restart MP,
observe its changed incarnation and the normal idle CPU withdrawals, then restart
the SGLang engine and refresh its Dynamo worker identity. Resume inference only
after startup registration succeeds. The SGLang heartbeat does not re-register
device handles after MP replacement; its subscriber waits for that registration.
The recovery matrix must prove a fresh routed CPU store and load after this supported
restart sequence. Do not issue an inference barrier between the MP restart and
the SGLang engine restart.
