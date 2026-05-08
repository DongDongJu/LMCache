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
- `target_host_write_bytes_reached`
- `waf`
- `waf_available`
- `waf_unavailable_reason`

WAF is only available when a vendor media/NAND write counter is configured.

