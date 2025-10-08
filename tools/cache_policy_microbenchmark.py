#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Synthetic cache-policy comparison for LRU and SIEVE variants."""

from __future__ import annotations

import argparse
import itertools
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cache_policy.clock_ecl import CLOCKECLCachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve import SIEVECachePolicy


class DummyCacheObj:
    __slots__ = ("can_evict", "size")

    def __init__(self, can_evict: bool = True, size: int = 1) -> None:
        self.can_evict = can_evict
        self.size = size

    def get_size(self) -> int:
        return self.size


@dataclass
class PinConfig:
    pin_probability: float
    pin_duration: int
    max_pinned: int


@dataclass
class Scenario:
    name: str
    description: str
    trace: Sequence[CacheEngineKey]
    pin_config: Optional[PinConfig] = None
    pin_schedule: Optional[list[bool]] = None
    stage_schedule: Optional[list[str]] = None


@dataclass
class SimulationResult:
    hits: int
    misses: int
    evictions: int
    runtime_ms: float
    scan_iterations: int
    extra_metrics: Dict[str, Any]

    @property
    def hit_rate(self) -> float:
        if self.hits + self.misses == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)


def cache_keys(num_keys: int, world_size: int = 1, workers: int = 4) -> list[CacheEngineKey]:
    keys: list[CacheEngineKey] = []
    for idx in range(num_keys):
        worker_id = idx % workers
        keys.append(CacheEngineKey("vllm", "benchmark", world_size, worker_id, idx))
    return keys


def make_prefill_decode_trace(
    keys: Sequence[CacheEngineKey],
    sessions: int,
    decode_reuse: tuple[int, int],
) -> list[CacheEngineKey]:
    trace: list[CacheEngineKey] = []
    key_cycle = list(keys)
    random.shuffle(key_cycle)
    idx = 0
    for _ in range(sessions):
        key = key_cycle[idx % len(key_cycle)]
        idx += 1
        trace.append(key)  # prefill insert
        repeats = random.randint(*decode_reuse)
        trace.extend([key] * repeats)
        if random.random() < 0.15:
            # inject a cold key to grow working set
            cold = random.choice(keys)
            trace.append(cold)
    return trace


def make_zipf_trace(keys: Sequence[CacheEngineKey], size: int, exponent: float) -> list[CacheEngineKey]:
    weights = [1.0 / (rank + 1) ** exponent for rank in range(len(keys))]
    total = sum(weights)
    normalized = [w / total for w in weights]
    indices = random.choices(range(len(keys)), weights=normalized, k=size)
    return [keys[i] for i in indices]


def make_burst_trace(
    hot_keys: Sequence[CacheEngineKey],
    warm_keys: Sequence[CacheEngineKey],
    cold_keys: Sequence[CacheEngineKey],
    bursts: int,
    burst_length: tuple[int, int],
) -> list[CacheEngineKey]:
    trace: list[CacheEngineKey] = []
    for _ in range(bursts):
        population, label = random.choices(
            [(hot_keys, "hot"), (warm_keys, "warm"), (cold_keys, "cold")],
            weights=[0.55, 0.3, 0.15],
            k=1,
        )[0]
        length = random.randint(*burst_length)
        key = random.choice(population)
        trace.extend([key for _ in range(length)])
        if label != "hot":
            trace.append(random.choice(hot_keys))
    return trace


def make_prefill_only_trace(
    keys: Sequence[CacheEngineKey],
    inserts: int,
) -> list[CacheEngineKey]:
    key_cycle = itertools.cycle(keys)
    return [next(key_cycle) for _ in range(inserts)]


def make_prefill_with_reuse_trace(
    keys: Sequence[CacheEngineKey],
    num_docs: int,
    doc_size: int,
    reuse_gap: int,
) -> tuple[list[CacheEngineKey], list[bool]]:
    key_cycle = itertools.cycle(keys)
    doc_tokens: list[list[CacheEngineKey]] = [
        [next(key_cycle) for _ in range(doc_size)] for _ in range(num_docs)
    ]

    trace: list[CacheEngineKey] = []
    pin_schedule: list[bool] = []

    for doc_idx in range(num_docs):
        doc_keys = doc_tokens[doc_idx]
        trace.extend(doc_keys)
        pin_schedule.extend([True] * len(doc_keys))

        reuse_idx = doc_idx - reuse_gap
        if reuse_idx >= 0:
            reuse_keys = doc_tokens[reuse_idx]
            trace.extend(reuse_keys)
            pin_schedule.extend([False] * len(reuse_keys))

    for doc_idx in range(max(0, num_docs - reuse_gap), num_docs):
        reuse_keys = doc_tokens[doc_idx]
        trace.extend(reuse_keys)
        pin_schedule.extend([False] * len(reuse_keys))

    return trace, pin_schedule


def make_anchor_rehydrate_trace(
    keys: Sequence[CacheEngineKey],
    sessions: int,
    prefill_len: int,
    decode_revisits: int,
    revisit_stride: int,
) -> tuple[list[CacheEngineKey], list[str], list[bool]]:
    trace: list[CacheEngineKey] = []
    stages: list[str] = []
    pins: list[bool] = []
    total = len(keys)
    cursor = 0

    for _ in range(sessions):
        session_keys: list[CacheEngineKey] = []
        for _ in range(prefill_len):
            key = keys[cursor % total]
            cursor += 1
            trace.append(key)
            stages.append("prefill")
            pins.append(True)
            session_keys.append(key)

        if not session_keys:
            continue

        for step in range(decode_revisits):
            anchor = session_keys[(step * revisit_stride) % len(session_keys)]
            trace.append(anchor)
            stages.append("decode")
            pins.append(False)

            if step % 3 == 0:
                new_key = keys[cursor % total]
                cursor += 1
                trace.append(new_key)
                stages.append("decode")
                pins.append(False)

    return trace, stages, pins


def make_high_concurrency_trace(
    keys: Sequence[CacheEngineKey],
    concurrency: int,
    prefill_len: int,
    decode_rounds: int,
) -> tuple[list[CacheEngineKey], list[str], list[bool]]:
    trace: list[CacheEngineKey] = []
    stages: list[str] = []
    pins: list[bool] = []

    total = len(keys)
    cursor = 0
    request_prefills: list[list[CacheEngineKey]] = []

    for _ in range(concurrency):
        req_prefill: list[CacheEngineKey] = []
        for _ in range(prefill_len):
            key = keys[cursor % total]
            cursor += 1
            req_prefill.append(key)
        if not req_prefill:
            req_prefill.append(keys[cursor % total])
            cursor += 1
        request_prefills.append(req_prefill)

    for step in range(prefill_len):
        for req_idx in range(concurrency):
            key = request_prefills[req_idx][step % len(request_prefills[req_idx])]
            trace.append(key)
            stages.append("prefill")
            pins.append(True)

    for round_idx in range(decode_rounds):
        for req_idx in range(concurrency):
            prefill_seq = request_prefills[req_idx]
            anchor = prefill_seq[(round_idx + req_idx) % len(prefill_seq)]
            trace.append(anchor)
            stages.append("decode")
            pins.append(False)

            new_key = keys[cursor % total]
            cursor += 1
            trace.append(new_key)
            stages.append("decode")
            pins.append(False)

    return trace, stages, pins


def make_streaming_window_trace(
    keys: Sequence[CacheEngineKey],
    window_size: int,
    stride: int,
    steps: int,
) -> list[CacheEngineKey]:
    key_cycle = itertools.cycle(keys)
    trace: list[CacheEngineKey] = []
    window: list[CacheEngineKey] = [next(key_cycle) for _ in range(window_size)]

    trace.extend(window)  # initial prefill

    for _ in range(steps):
        trace.extend(window)  # decode hits within the sliding window
        new_tokens = [next(key_cycle) for _ in range(stride)]
        trace.extend(new_tokens)
        window.extend(new_tokens)
        if len(window) > window_size:
            window = window[-window_size:]

    return trace


def make_rag_trace(
    keys: Sequence[CacheEngineKey],
    queries: int,
    query_len: int,
    doc_span: int,
    rag_pool: int,
) -> list[CacheEngineKey]:
    trace: list[CacheEngineKey] = []
    key_cycle = itertools.cycle(keys)
    max_chunks = max(1, min(rag_pool, len(keys) // doc_span))
    doc_chunks: list[list[CacheEngineKey]] = []
    for chunk_idx in range(max_chunks):
        start = chunk_idx * doc_span
        end = start + doc_span
        if end <= len(keys):
            doc_chunks.append(list(keys[start:end]))
    if not doc_chunks:
        doc_chunks.append([next(key_cycle) for _ in range(doc_span)])

    for _ in range(queries):
        query_tokens = [next(key_cycle) for _ in range(query_len)]
        trace.extend(query_tokens)
        doc_sequence = random.choice(doc_chunks)
        trace.extend(doc_sequence)

    return trace


def make_long_summary_trace(
    keys: Sequence[CacheEngineKey],
    doc_len: int,
    summary_rounds: int,
    summary_len: int,
) -> list[CacheEngineKey]:
    key_cycle = itertools.cycle(keys)
    document = [next(key_cycle) for _ in range(doc_len)]
    trace: list[CacheEngineKey] = document.copy()

    for _ in range(summary_rounds):
        trace.extend(document)  # revisit the long document
        summary_tokens = [next(key_cycle) for _ in range(summary_len)]
        trace.extend(summary_tokens)

    return trace


def _call_update_on_hit(
    policy: Any,
    key: CacheEngineKey,
    cache_dict: dict[CacheEngineKey, Any],
    stage: str,
) -> None:
    try:
        policy.update_on_hit(key, cache_dict, stage=stage)
    except TypeError:
        policy.update_on_hit(key, cache_dict)


def _call_update_on_put(
    policy: Any,
    key: CacheEngineKey,
    stage: str,
) -> None:
    try:
        policy.update_on_put(key, stage=stage)
    except TypeError:
        policy.update_on_put(key)


def simulate(
    policy_name: str,
    trace: Sequence[CacheEngineKey],
    capacity: int,
    *,
    pin_config: Optional[PinConfig] = None,
    pin_schedule: Optional[Sequence[bool]] = None,
    stage_schedule: Optional[Sequence[str]] = None,
    policy_factory: Optional[Callable[[], Any]] = None,
) -> SimulationResult:
    policy = policy_factory() if policy_factory is not None else get_cache_policy(policy_name)
    cache_dict = policy.init_mutable_mapping()

    hits = 0
    misses = 0
    evictions = 0
    scan_iterations: list[int] = []
    pinned: Dict[CacheEngineKey, int] = {}

    def record_scan_iterations() -> None:
        visited = getattr(policy, "_last_scan_iterations", None)
        if visited is not None:
            scan_iterations.append(int(visited))

    start = time.perf_counter()
    for step, key in enumerate(trace):
        if pinned:
            for pinned_key in list(pinned.keys()):
                remaining = pinned[pinned_key] - 1
                obj = cache_dict.get(pinned_key)
                if obj is None or remaining <= 0:
                    if obj is not None:
                        obj.can_evict = True
                    pinned.pop(pinned_key, None)
                else:
                    pinned[pinned_key] = remaining

        stage_hint = None
        if stage_schedule is not None and step < len(stage_schedule):
            stage_hint = stage_schedule[step]

        if key in cache_dict:
            hits += 1
            _call_update_on_hit(
                policy,
                key,
                cache_dict,
                stage=stage_hint or "decode",
            )
            continue

        misses += 1
        while len(cache_dict) >= capacity:
            victims = policy.get_evict_candidates(cache_dict, 1)
            record_scan_iterations()
            if not victims:
                break
            for victim in victims:
                if victim in cache_dict:
                    cache_dict.pop(victim)
                    policy.update_on_force_evict(victim)
                    evictions += 1
                    pinned.pop(victim, None)
        cache_dict[key] = DummyCacheObj()
        put_stage = stage_hint or "prefill"
        _call_update_on_put(policy, key, stage=put_stage)

        should_pin = False
        if pin_config is not None and pin_schedule is not None:
            if step < len(pin_schedule):
                should_pin = pin_schedule[step]
        if should_pin and len(pinned) < pin_config.max_pinned:
            obj = cache_dict.get(key)
            if obj is not None:
                obj.can_evict = False
                pinned[key] = max(1, pin_config.pin_duration)

    runtime_ms = (time.perf_counter() - start) * 1000

    total_scans = sum(scan_iterations)
    extra_metrics: Dict[str, Any] = {}
    if hasattr(policy, "get_debug_metrics"):
        try:
            metrics = policy.get_debug_metrics()
        except Exception:  # pragma: no cover - diagnostics only
            metrics = {}
        if isinstance(metrics, dict):
            extra_metrics = metrics

    return SimulationResult(
        hits,
        misses,
        evictions,
        runtime_ms,
        total_scans,
        extra_metrics,
    )



def run_scenarios(capacity: int, seed: int) -> None:
    random.seed(seed)

    all_keys = cache_keys(512, world_size=4, workers=16)

    scenarios: list[Scenario] = []

    scenarios.append(
        Scenario(
            name="Prefill/Decode",
            description="Session-oriented bursts: insert once, reuse for decode tokens",
            trace=make_prefill_decode_trace(all_keys, sessions=600, decode_reuse=(2, 10)),
        )
    )

    scenarios.append(
        Scenario(
            name="Zipf(1.1)",
            description="Zipf popularity with long tail",
            trace=make_zipf_trace(all_keys, size=8000, exponent=1.1),
        )
    )

    hot = all_keys[:32]
    warm = all_keys[32:160]
    cold = all_keys[160:512]
    scenarios.append(
        Scenario(
            name="Burst Mix",
            description="Alternating hot/warm/cold bursts",
            trace=make_burst_trace(hot, warm, cold, bursts=600, burst_length=(3, 25)),
        )
    )

    scenarios.append(
        Scenario(
            name="Streaming Window",
            description="Sliding-window decode inspired by StreamingLLM (Xiao et al., 2023)",
            trace=make_streaming_window_trace(all_keys, window_size=128, stride=8, steps=256),
        )
    )

    scenarios.append(
        Scenario(
            name="RAG Burst",
            description="Retrieval-augmented batches similar to Atlas/RETRO style systems",
            trace=make_rag_trace(all_keys, queries=512, query_len=4, doc_span=6, rag_pool=96),
        )
    )

    scenarios.append(
        Scenario(
            name="Long Summary",
            description="Repeated long-context summarization akin to LongChat/Claude reports",
            trace=make_long_summary_trace(all_keys, doc_len=2048, summary_rounds=6, summary_len=64),
        )
    )

    pinned_trace, pinned_schedule = make_prefill_with_reuse_trace(
        all_keys,
        num_docs=200,
        doc_size=6,
        reuse_gap=64,
    )
    scenarios.append(
        Scenario(
            name="Pinned Prefill",
            description="Delayed decode reuse with temporary pins to stress blocked probation",
            trace=pinned_trace,
            pin_config=PinConfig(pin_probability=1.0, pin_duration=256, max_pinned=160),
            pin_schedule=pinned_schedule,
        )
    )

    anchor_trace, anchor_stages, anchor_pins = make_anchor_rehydrate_trace(
        all_keys,
        sessions=96,
        prefill_len=40,
        decode_revisits=48,
        revisit_stride=5,
    )
    scenarios.append(
        Scenario(
            name="Anchored Replay",
            description="Long prefills with sparse decode revisits to test anchor rehydration",
            trace=anchor_trace,
            pin_config=PinConfig(pin_probability=1.0, pin_duration=512, max_pinned=320),
            pin_schedule=anchor_pins,
            stage_schedule=anchor_stages,
        )
    )

    concurrency_trace, concurrency_stages, concurrency_pins = make_high_concurrency_trace(
        all_keys,
        concurrency=12,
        prefill_len=24,
        decode_rounds=64,
    )
    scenarios.append(
        Scenario(
            name="Long Concurrency",
            description="Interleaved long-context sessions with decode windows under high load",
            trace=concurrency_trace,
            pin_config=PinConfig(pin_probability=1.0, pin_duration=384, max_pinned=288),
            pin_schedule=concurrency_pins,
            stage_schedule=concurrency_stages,
        )
    )

    for idx, scenario in enumerate(scenarios):
        if scenario.pin_config is not None and scenario.pin_schedule is None:
            rng = random.Random(seed + idx * 9973)
            scenario.pin_schedule = [
                rng.random() < scenario.pin_config.pin_probability
                for _ in range(len(scenario.trace))
            ]

    policies: list[tuple[str, Optional[Callable[[], Any]]]] = [
        ("ANCHOR_SLIDING_WINDOW", None),
        ("LRU", None),
        ("MRU", None),
        ("FIFO", None),
        ("CLOCK_ECL", lambda: CLOCKECLCachePolicy()),
        ("SIEVE", None),
        ("SIEVE_BASE", lambda: SIEVECachePolicy(enable_ecl=False, enable_cost=False, decode_hot_ttl_ms=0)),
        ("SIEVE_ECL", lambda: SIEVECachePolicy(enable_ecl=True, enable_cost=False, decode_hot_ttl_ms=0)),
        ("SIEVE_COST", lambda: SIEVECachePolicy(enable_ecl=True, enable_cost=True, decode_hot_ttl_ms=0)),
        ("SIEVE_FULL", lambda: SIEVECachePolicy()),
        ("SIEVE_SLRU", None),
        ("SIEVE_PDG", None),
        ("WEB_CACHE", None),
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario.name}")
        print(f"  {scenario.description}")
        print(f"  Trace length: {len(scenario.trace)}")
        for policy_name, factory in policies:
            result = simulate(
                policy_name,
                scenario.trace,
                capacity,
                pin_config=scenario.pin_config,
                pin_schedule=scenario.pin_schedule,
                stage_schedule=scenario.stage_schedule,
                policy_factory=factory,
            )
            avg_scan = "-"
            if result.scan_iterations and result.evictions:
                avg_scan = f"{result.scan_iterations / result.evictions:.1f}"
            metric_suffix = ""
            if result.extra_metrics:
                guard_size = result.extra_metrics.get("decode_guard_size")
                guard_demotes = result.extra_metrics.get("decode_guard_demotions")
                blocked = result.extra_metrics.get("probation_blocked_size")
                guard_ratio = result.extra_metrics.get("dynamic_guard_ratio_percent")
                avg_scans = result.extra_metrics.get("avg_probation_scans")
                decode_ratio = result.extra_metrics.get("decode_hit_ratio")
                fifo_active = result.extra_metrics.get("fifo_override_active")
                ecl_size = result.extra_metrics.get("ecl_size")
                ecl_evictions = result.extra_metrics.get("ecl_evictions")
                ttl_deferrals = result.extra_metrics.get("ttl_deferrals")
                last_scan = result.extra_metrics.get("last_scan_iterations")
                parts: list[str] = []
                if guard_size is not None:
                    parts.append(
                        f"decode_guard={guard_size} demotes={guard_demotes}"
                    )
                if blocked is not None:
                    parts.append(f"blocked_probation={blocked}")
                if guard_ratio is not None:
                    parts.append(f"guard_ratio={guard_ratio}%")
                if avg_scans is not None:
                    parts.append(f"avg_scans={avg_scans}")
                if decode_ratio is not None:
                    parts.append(f"decode_hits={decode_ratio}%")
                if fifo_active:
                    parts.append("fifo_override=1")
                if ecl_size is not None:
                    parts.append(f"ecl={ecl_size}")
                if ecl_evictions:
                    parts.append(f"ecl_victims={ecl_evictions}")
                if ttl_deferrals:
                    parts.append(f"ttl_skip={ttl_deferrals}")
                if last_scan is not None:
                    parts.append(f"last_scan={last_scan}")
                if parts:
                    metric_suffix = " | " + " ".join(parts)
            print(
                f"    {policy_name:9s} | hit_rate={result.hit_rate:.3f}"
                f" | misses={result.misses:5d} | evictions={result.evictions:5d}"
                f" | runtime={result.runtime_ms:6.2f} ms"
                f" | scan iterations={result.scan_iterations} (avg {avg_scan})"
                f"{metric_suffix}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity", type=int, default=256, help="max cache entries to retain")
    parser.add_argument("--seed", type=int, default=20251006, help="RNG seed")
    args = parser.parse_args()

    run_scenarios(capacity=args.capacity, seed=args.seed)


if __name__ == "__main__":
    main()
