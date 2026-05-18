# Agentic MP Trace Record and Replay

This benchmark records real LMCache MP storage traces from vLLM agentic
requests, then replays those traces concurrently into one raw-block/FDP device.

It does not add LMCache CLI flags. Recording uses `lmcache server --trace-level
storage --trace-output ...`; replay uses `lmcache trace replay`.

## Record

Offline transcript mode is the default. Local dataset exports are converted into
agent-style chat requests, sent to the vLLM OpenAI-compatible endpoint, and
recorded by the live LMCache MP server.

```bash
uv run --no-sync python benchmarks/agentic_mp_trace/record_agentic_mp_traces.py \
  --config benchmarks/agentic_mp_trace/config.example.yaml \
  --output-dir /mnt/hc-ssd/lmcache-agentic-records \
  --record-suite agentic_mix_v1
```

Dry-run only prints and writes commands/metadata:

```bash
uv run --no-sync python benchmarks/agentic_mp_trace/record_agentic_mp_traces.py \
  --config benchmarks/agentic_mp_trace/config.example.yaml \
  --output-dir /mnt/hc-ssd/lmcache-agentic-records-dry-run \
  --record-suite agentic_mix_v1 \
  --dry-run
```

## Replay

Run the same trace mix three ways on `/dev/ng1n1`:

```bash
uv run --no-sync python benchmarks/agentic_mp_trace/replay_agentic_trace_mix.py \
  --trace-manifest /mnt/hc-ssd/lmcache-agentic-records/trace_manifest.yaml \
  --config benchmarks/agentic_mp_trace/config.example.yaml \
  --mode no_fdp \
  --warmup-iterations 2 \
  --iterations 8 \
  --output-dir /mnt/hc-ssd/agentic-no-fdp

uv run --no-sync python benchmarks/agentic_mp_trace/replay_agentic_trace_mix.py \
  --trace-manifest /mnt/hc-ssd/lmcache-agentic-records/trace_manifest.yaml \
  --config benchmarks/agentic_mp_trace/config.example.yaml \
  --mode fdp_mixed \
  --warmup-iterations 2 \
  --iterations 8 \
  --output-dir /mnt/hc-ssd/agentic-fdp-mixed

uv run --no-sync python benchmarks/agentic_mp_trace/replay_agentic_trace_mix.py \
  --trace-manifest /mnt/hc-ssd/lmcache-agentic-records/trace_manifest.yaml \
  --config benchmarks/agentic_mp_trace/config.example.yaml \
  --mode fdp_separated \
  --warmup-iterations 2 \
  --iterations 8 \
  --output-dir /mnt/hc-ssd/agentic-fdp-separated
```

The default config assumes a 4-RUH device. `no_fdp` is the baseline.
`fdp_mixed` intentionally mixes data into RUHs `[0,1,2]` with metadata on RUH
`3`. `fdp_separated` coarsely separates tool/coding/browser classes across
RUHs `0`, `1`, and `2`, with metadata on RUH `3`.

The replay phase uses L1 buffer-only mode for speed:

- `--l2-store-policy skip_l1`
- `--eviction-policy noop`
- `--l1-size-gb 1`

LMCache still requires an L1 buffer, but stored keys are deleted from L1 after
L2 store.

## Write Target

The default replay target is host writes of at least five times the configured
test-region capacity. The summary records:

- `target_host_write_bytes`
- `host_write_bytes_delta`
- `lmcache_store_attempted_logical_bytes`
- `lmcache_store_committed_logical_bytes`
- `lmcache_eviction_count`
- `lmcache_eviction_logical_bytes`
- `lmcache_successful_data_write_physical_bytes`
- `lmcache_successful_metadata_write_physical_bytes`
- `lmcache_successful_write_physical_bytes`
- `host_vs_lmcache_successful_physical_delta_bytes`
- `host_vs_lmcache_successful_physical_ratio`
- `target_host_write_bytes_reached`
- `waf`
- `waf_status`

Use the committed/successful LMCache counters when validating device-side
accounting. Attempted trace store bytes include failed stores, no-free-slot
cases, and duplicate/existing-key hits that do not necessarily reach the NVMe
namespace. The successful physical counter includes raw-block data payloads,
slot headers, and metadata checkpoint writes exported by each replay process in
`storage_manager_status.json`. The eviction counters report raw-block LRU slot
evictions caused by capacity pressure; they do not include explicit deletes or
failed stores that never acquired a reusable slot.

WAF is only available when a vendor media/NAND write counter is configured.

## Multi-Window Raw-Block Planner

`replay_multi_window_raw_block.py` splits one NVMe namespace into multiple
non-overlapping raw-block byte windows and builds one `lmcache trace replay`
command per active window. The default mode is dry-run/plan-only; it writes
`resolved_plan.json`, `resolved_plan.yaml`, and `commands.sh` without touching
the device.

Example plan for four windows and four available RUHs:

```bash
uv run --no-sync python benchmarks/agentic_mp_trace/replay_multi_window_raw_block.py \
  --trace-manifest /mnt/hc-ssd/lmcache-agentic-records/trace_manifest.yaml \
  --device-path /dev/ng1n1 \
  --block-device-path /dev/nvme1n1 \
  --output-dir /mnt/hc-ssd/lmcache-multi-window-plan \
  --device-capacity-bytes auto \
  --start-offset-bytes 0 \
  --num-windows 4 \
  --num-workloads 4 \
  --window-capacity-policy equal \
  --use-fdp true \
  --ruh-count 4 \
  --ruh-assignment mixed \
  --application-placement fixed \
  --stop-policy iterations \
  --iterations 1 \
  --plan-only
```

The generated replay write path is:

```text
lmcache trace replay
  -> StoreController / PrefetchController
  -> RawBlockL2Adapter
  -> RawBlockCore
  -> Rust raw-block I/O
  -> NVMe namespace
```

For an actual destructive run, pass `--allow-destructive-device-write` and then
type exactly `RUN` at the prompt. For non-interactive automation, both `--yes`
and `--allow-destructive-device-write` are required. The tool does not claim
GPU-direct disk I/O support; use `--gpu-io-mode cpu_stage` to document the
current replay path through CPU/L1 staging and asynchronous LMCache store
workers.

If the SSD exposes a vendor media/NAND write counter, pass it with
`--media-write-counter-command '<command returning JSON or bytes>'`; otherwise
`summary.json` reports `waf: null` and a WAF unavailable status instead of
deriving WAF from host writes.

Optional device-gated smoke command for a dedicated raw NVMe namespace:

```bash
LMCACHE_RAW_BLOCK_TEST_NG_DEVICE=/dev/ng1n1 \
LMCACHE_RAW_BLOCK_TEST_BLOCK_DEVICE=/dev/nvme1n1 \
LMCACHE_ALLOW_DESTRUCTIVE_RAW_BLOCK_TEST=1 \
uv run --no-sync python benchmarks/agentic_mp_trace/replay_multi_window_raw_block.py \
  --trace-manifest /mnt/hc-ssd/lmcache-agentic-records/trace_manifest.yaml \
  --device-path "$LMCACHE_RAW_BLOCK_TEST_NG_DEVICE" \
  --block-device-path "$LMCACHE_RAW_BLOCK_TEST_BLOCK_DEVICE" \
  --output-dir /mnt/hc-ssd/lmcache-multi-window-smoke \
  --device-capacity-bytes auto \
  --num-windows 2 \
  --num-workloads 2 \
  --use-fdp true \
  --ruh-count 4 \
  --ruh-assignment mixed \
  --stop-policy iterations \
  --iterations 1 \
  --allow-destructive-device-write
```
