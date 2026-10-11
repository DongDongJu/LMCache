# L1 cuFile worker comparison

This is an isolated GPU microbenchmark of the public LMCache MP client and server.
It compares the original checkout, the candidate with workers disabled, and the
candidate with one or four ordinary cuFile workers. All arms use one initialized
16 GiB regular-file allocation and the same GPU, object geometry, payload seed,
offset policy, gather/scatter kernels, direct-I/O setting and verification work.
It is not a model-serving or tensor-parallel throughput benchmark.

The package is self-contained. It does not import experiment scripts or the
separate `gds_perf` campaign. Both selected checkouts must have their own matching
native extensions built, with a compatible Python environment containing LMCache's
MP client dependencies, PyTorch, NumPy and the cuFile Python bindings. Linux tools
`nvidia-smi`, `findmnt` and `lsblk` are required; `numactl` is required only when
`--numa-node` is supplied. Use an exclusive GPU/storage measurement window.

## Matched steady comparison

Run from this checkout. Set the paths, GPU UUID and NUMA node to the intended
machine; the runner contains no lab-specific paths or physical GPU defaults.
Choose a dedicated directory on the intended local SSD/filesystem, with at least
16 GiB free. Existing LMCache slabs are never valid inputs.

```bash
python -m benchmarks.gds_optimization.run \
  --source /path/to/candidate-checkout \
  --baseline-source /path/to/original-checkout \
  --scratch-root /path/on/ssd/dedicated-benchmark-directory \
  --gpu-uuid GPU-your-physical-device-uuid --numa-node 0 \
  --cufile-config benchmarks/gds_optimization/cufile-baseline.json \
  --output /path/to/new-comparison-results \
  --repetitions 3 --duration 30 --requests 10000 --warmup 2
```

Create the scratch root before running; the output directory must be new. Each
repetition runs original, workers=0, workers=1 and workers=4, rotating arm order
between repetitions. The original arm omits the candidate-only server flags.
Omit `--baseline-source` and use `--workers 4` for one candidate trial. Set
`--cuda-lib-dir /path/to/cuda/lib64` only if the environment needs an explicit
cuFile library search path. Otherwise the current loader configuration is kept.

The supplied cuFile JSON files preserve the measured configuration, including
`allow_compat_mode=true`, and differ only in logging/profiling. Review them for
the installed GDS version. Effective JSON, source/native hashes, Git state,
selected environment overrides, commands, GPU inventory/topology, storage model
and mount details are retained. The runner replaces only the JSON logging
directory. It does not force or disable `CUFILE_FORCE_COMPAT_MODE`; its inherited
value is recorded. Neither successful `O_DIRECT` nor this config establishes
native GPU DMA. Keep installed GDS path diagnostics and cuFile/POSIX counters
with each hardware report.

Default geometry is FP16 NHD, 32 layers, 8 KV heads, head size 128, block size 16,
1,024 GPU blocks, and 4,096 tokens per request. Each request has sixteen 32 MiB
chunks (512 MiB useful bytes per direction). The engine cache is 2 GiB. Candidate
admission is capped at 128 MiB by default; change it explicitly with
`--max-inflight-bytes`. Existing staging entries are reused; these worker counts
do not imply extra full-payload staging allocations.

For smaller homogeneous objects, pass `--heads 1 --layers 2`, `--heads 1 --layers 8`,
or `--heads 1 --layers 32` for 256 KiB, 1 MiB or 4 MiB chunks respectively.
`--heads 4 --layers 32` produces 16 MiB chunks. Keep the geometry identical across
comparison arms. Run other physical GPUs by selecting their UUID; independent
simultaneous runs require separate outputs, slab allocations and RPC/HTTP ports.
Single-GPU results must not be summed into a concurrent aggregate result.

## Allocation and ownership

Fresh preparation preallocates the whole slab, writes known nonzero splitmix64
bytes, calls `fsync`, and compares every byte on buffered readback. This is a
filesystem coherency check, not independent media-read evidence. Preparation is
outside MP timing. There is no timed `fsync` durability requirement.

The benchmark-only server entry point reopens that exact inode without truncation
on both original and candidate. All production allocation, I/O, RPC and completion
logic remains in the selected source checkout. A nonblocking exclusive flock is
held throughout each campaign and inherited by the server: a surviving server
keeps reuse blocked after its controller exits. Startup also proves the server
owns an open descriptor for the exact private regular-file inode.

A later campaign can reuse only a slab prepared by this package:

```bash
python -m benchmarks.gds_optimization.run \
  --source /path/to/candidate-checkout --workers 4 \
  --scratch-root /path/on/ssd/dedicated-benchmark-directory \
  --reuse-slab /path/on/ssd/dedicated-benchmark-directory/new-comparison-results/lmcache_gds_slab.bin \
  --gpu-uuid GPU-your-physical-device-uuid --numa-node 0 \
  --cufile-config benchmarks/gds_optimization/cufile-baseline.json \
  --output /path/to/new-repeat-results
```

Reuse requires the original preparation marker, path, device/inode, capacity,
private directory and exclusive lease. Symlinks, replaced files and live leases
are rejected. Data is preserved after a run for subsequent comparisons; remove
only the dedicated benchmark directory after all owned processes have exited.

## Completion and verification boundaries

Every request starts with unique keys and a full cache miss. Source GPU regions
are initialized once by the public client (effective seed 42); the benchmark
records full per-chunk CPU digests of every source region before and after replay.
A separate destination is zeroed before each request, then checked unchanged
before retrieve. Every NHD byte on every layer must exactly match its source after
retrieve. A no-op retrieve and last-byte corruption both fail the verifier.

`store_reply_us` is the public STORE completion after its remote device event.
`store_visible_us` extends through the first fresh-session lookup observing every
chunk. `combined_read_ready_us` begins at read lookup admission and ends after the
public retrieve and a consumer CUDA marker. The public client host-waits on the
retrieved remote IPC event; these are not wire-only RPC reply times. Exact byte
comparison follows outside the per-request read-ready interval.

`GBps` counts both useful directions over the entire steady wall-clock window,
including exact verification, zeroing, lookups, session cleanup and periodic cache
clears. It is not completed storage bandwidth. Physical cuFile bytes remain null
unless separately established by diagnostics. The client caps requests and stops
after the duration threshold at a completed-request boundary. Small-sample p99
is explicitly exploratory. Report repetition medians and ranges, rather than
pooling unlike object sizes or instrumented trials.

Every sixteen measured requests, the owned sequential server is checked for no
active sessions, stores, writes, staging objects or prefetch jobs, then its logical
L1 contents are cleared. The forced clear handles read locks retained by
publication-only lookup probes. This is permitted only for this disposable,
quiescent server. The clear latency is included in whole-window throughput and
its before/after status is retained. All prior stores are published and retrieves
are complete and byte-verified before the clear.

No persistent verification staging tensor is allocated. Temporary allocations
inside `count_nonzero`/`equal` are not measured; the artifact explicitly leaves the
verification temporary peak unknown. The process sampler records identity-checked
server/client CPU and `/proc/PID/io` counters at roughly one second cadence.
These process counters include work outside the steady window and need timestamp
alignment; they do not establish exact physical I/O amplification.

## Short diagnostics

Run diagnostics separately from performance trials:

```bash
python -m benchmarks.gds_optimization.run \
  --source /path/to/candidate-checkout --workers 4 \
  --scratch-root /path/on/ssd/dedicated-benchmark-directory \
  --reuse-slab /path/to/package-owned/lmcache_gds_slab.bin \
  --gpu-uuid GPU-your-physical-device-uuid --numa-node 0 \
  --cufile-config benchmarks/gds_optimization/cufile-diagnostic.json \
  --instrument --requests 4 --warmup 1 --duration 30 \
  --output /path/to/new-diagnostic-results
```

`server-events.jsonl` separates descriptor submission/completion, gather/scatter
GPU events, worker admission/work/retirement, slot waiting, initialization and
registration, and engine completion event generations. The optional candidate
input-event record includes identity/generation reuse. An ordinary observer
thread polls completed CUDA events; the instrumentation never issues CUDA calls
from a host callback. It adds timing overhead and is labelled diagnostic. Inspect
actual descriptor byte counts, unresolved events, and installed cuFile statistics
to classify the observed path; compatibility execution must be reported as such.

## Artifacts and CPU checks

`campaign.json` records the plan and terminal arms. `inventory.json` records the
hardware/filesystem evidence. Each arm retains `manifest.json`, effective
`cufile.json`, server/client logs, `result.json`, per-request `samples.jsonl`, and
process counters. A failed arm remains failed, stops the campaign, and keeps its
evidence. The controller terminates and reaps only its own launched process groups.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q \
  --confcutdir=benchmarks/gds_optimization/tests \
  benchmarks/gds_optimization/tests
python -m benchmarks.gds_optimization.run --help
```

The tests use CPU tensors and small temporary regular files. They exercise exact
strided byte comparison, sentinel/no-op guards, stable source regions, busy-server
clear refusal, full preparation, failed writes, inode replacement, symlink
rejection, and a lease inherited by a surviving child. Hardware measurements
remain necessary to establish native DMA, device ordering and performance.
