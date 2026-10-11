# SPDX-License-Identifier: Apache-2.0
"""Measure real MP RPC completion and verify retrieval into different GPU blocks.

The existing public benchmark client does not clear handle-mode destinations.
This benchmark instead requires a disjoint destination to have a different hash
before retrieval, then compares every retrieved byte through per-chunk digests.
Each concurrent process needs a different server: the public client registers
its single worker with the fixed instance ID 1000.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
import argparse
import hashlib
import json
import os
import socket
import statistics
import time
import uuid

# Third Party
import torch

# First Party
from benchmarks.gds_optimization.client import OwnedBenchClient
from benchmarks.gds_optimization.gpu_verification import GPUVerifier
from benchmarks.gds_optimization.support import FileBarrier
from lmcache.cli.commands.bench.server_bench.client import (
    RequestContext,
    ServerBenchClient,
)
from lmcache.cli.commands.bench.server_bench.config import BenchConfig
from lmcache.logging import init_logger
from lmcache.v1.gpu_connector.gds_context import GDSContext

logger = init_logger(__name__)
GPU_VERIFIER: GPUVerifier | None = None


@dataclass(frozen=True)
class RequestSample:
    """Correlated host timestamps and integrity evidence for one full round trip."""

    sequence: int
    source_block_offset: int
    destination_block_offset: int
    chunks: int
    useful_bytes_each_direction: int
    cold_lookup_us: float
    store_reply_us: float
    store_visible_us: float
    publication_after_reply_us: float
    publication_lookup_us: tuple[float, ...]
    read_lookup_us: float
    retrieve_reply_us: float
    retrieve_gpu_ready_us: float
    combined_read_ready_us: float
    checksum_overhead_us: float
    zero_destination_call_us: float
    store_admitted_monotonic_ns: int
    store_reply_monotonic_ns: int
    store_visible_monotonic_ns: int
    read_admitted_monotonic_ns: int
    completed_monotonic_ns: int
    source_checksums: tuple[str, ...]
    destination_before_checksums: tuple[str, ...]
    destination_after_checksums: tuple[str, ...]
    destination_changed_before_retrieve: bool
    noop_retrieve_would_fail: bool
    checksum_match: bool
    worker_rank: int
    group_id: str
    shared_group_sequence: int


def distribution(values: list[float]) -> dict[str, float | int | bool]:
    """Summarize measured samples without deriving request tails from batches.

    Args:
        values: Nonempty collection of per-request values in the stated unit.

    Returns:
        Mean, nearest-rank percentiles, maximum, sample count, and tail caveat.

    Raises:
        ValueError: If no observations were supplied.
    """
    if not values:
        raise ValueError("a distribution needs at least one sample")
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "p50": ordered[int((len(ordered) - 1) * 0.50)],
        "p95": ordered[int((len(ordered) - 1) * 0.95)],
        "p99": ordered[int((len(ordered) - 1) * 0.99)],
        "max": ordered[-1],
        "samples": len(ordered),
        "p99_exploratory": len(ordered) < 10000,
    }


@lru_cache(maxsize=16)
def zero_checksum(bytes_per_chunk: int) -> str:
    """Return the digest of an all-zero chunk without a full-size allocation.

    Args:
        bytes_per_chunk: Exact byte count covered by one chunk digest.

    Returns:
        MD5 digest using the public server checksum endpoint's convention.
    """
    digest = hashlib.md5()
    block = bytes(1048576)
    remaining = bytes_per_chunk
    while remaining:
        length = min(remaining, len(block))
        digest.update(block[:length])
        remaining -= length
    return digest.hexdigest()


def checked_checksums(
    client: ServerBenchClient, request: RequestContext
) -> tuple[str, ...]:
    """Require one full-payload checksum for every requested logical chunk.

    Args:
        client: Started public benchmark client.
        request: Registered GPU block range to hash.

    Returns:
        Ordered immutable per-chunk digests.

    Raises:
        RuntimeError: If the checksum endpoint fails or omits a chunk.
    """
    if GPU_VERIFIER is not None:
        return GPU_VERIFIER.check(client, request)
    checksums = client.compute_checksums(request, 0, request.num_full_tokens)
    if checksums is None or len(checksums) != request.total_chunks:
        raise RuntimeError(
            f"checksum coverage failure sequence={request.sequence_id}: "
            f"expected {request.total_chunks} chunks, got {checksums}"
        )
    return tuple(checksums)


def run_request(
    client: ServerBenchClient,
    sequence: int,
    ordinal: int,
    run_id: str,
    options: argparse.Namespace,
    consumer_ready: torch.cuda.Event,
    phase_barrier: Callable[[str, int], None] | None = None,
) -> RequestSample:
    """Store, confirm publication, retrieve elsewhere, and compare all digests.

    Args:
        client: Started public client registered with a dedicated MP server.
        sequence: Unique token-prefix ID, used to require an initial cache miss.
        ordinal: Selects an untouched source block range round-robin.
        run_id: Namespace for this process's server sessions.
        options: Parsed geometry, visibility timeout, and polling configuration.
        consumer_ready: Reusable completion event initialized before timing.
        phase_barrier: Optional coordinator called before store/read admission.

    Returns:
        Correlated completion timings and checksums for one verified request.

    Raises:
        RuntimeError: On a cache, transfer, checksum, or sentinel mismatch.
        TimeoutError: If metadata does not become visible within the deadline.
        ValueError: If the server chunk geometry cannot fit the reserved ranges.
    """
    if GPU_VERIFIER is None:
        raise RuntimeError("GPU verifier must be initialized before requests")
    GPU_VERIFIER.begin(ordinal)
    request = client.create_request(sequence, f"{run_id}-{sequence}-store", "gds")
    if request is None:
        raise ValueError("configured tokens do not contain one server chunk")
    source_regions = (options.num_blocks // request.num_blocks) - 1
    if source_regions < 2:
        raise ValueError("need two untouched source ranges and one destination range")
    if phase_barrier is not None and source_regions < options.barrier_workers:
        raise ValueError("group geometry needs a distinct source range for every rank")
    request = replace(
        request, block_offset=(ordinal % source_regions) * request.num_blocks
    )
    destination = replace(
        request,
        request_id=f"{run_id}-{sequence}-read",
        block_offset=options.num_blocks - request.num_blocks,
        request_kind="gds-read",
    )
    sessions = [request]
    try:
        checksum_start = time.monotonic_ns()
        source_hashes = checked_checksums(client, request)
        destination_hashes = checked_checksums(client, destination)
        payload_bytes = (
            request.num_full_tokens
            * options.num_layers
            * 2
            * options.num_heads
            * options.head_size
            * 2
        )
        zero_hash = zero_checksum(payload_bytes // request.total_chunks)
        checksum_ns = time.monotonic_ns() - checksum_start
        if any(digest == zero_hash for digest in source_hashes):
            raise RuntimeError(f"source contains an all-zero chunk sequence={sequence}")
        if any(a == b for a, b in zip(source_hashes, destination_hashes, strict=True)):
            raise RuntimeError(
                f"destination sentinel cannot detect a no-op sequence={sequence}"
            )
        with torch.cuda.nvtx.range("mp_cold_lookup"):
            cold_start = time.monotonic_ns()
            cold = client.lookup(request)
            cold_end = time.monotonic_ns()
        if not cold.is_full_miss:
            raise RuntimeError(
                f"expected fresh cache miss sequence={sequence}: "
                f"{cold.hit_chunks}/{cold.total_chunks} error={cold.error}"
            )
        if phase_barrier is not None:
            phase_barrier("store", sequence)
        with torch.cuda.nvtx.range("mp_store_reply_device_ready"):
            store_start = time.monotonic_ns()
            stored = client.store(request, 0, request.num_full_tokens)
            store_reply = time.monotonic_ns()
        if stored is None or not stored.succeeded:
            raise RuntimeError(f"STORE failed sequence={sequence}: {stored}")

        publication_lookups: list[float] = []
        deadline = time.monotonic() + options.visibility_timeout
        while True:
            probe = replace(
                destination,
                request_id=f"{run_id}-{sequence}-visible-{len(publication_lookups)}",
                request_kind="publication-probe",
            )
            sessions.append(probe)
            with torch.cuda.nvtx.range("mp_publication_lookup"):
                probe_start = time.monotonic_ns()
                visible = client.lookup(probe)
                probe_end = time.monotonic_ns()
            publication_lookups.append((probe_end - probe_start) / 1000)
            if not visible.succeeded:
                raise RuntimeError(f"publication LOOKUP failed: {visible.error}")
            if visible.is_full_hit:
                store_visible = probe_end
                client.end_session(probe)
                sessions.pop()
                break
            client.end_session(probe)
            sessions.pop()
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"publication timeout sequence={sequence}: "
                    f"{visible.hit_chunks}/{visible.total_chunks}"
                )
            if options.poll_interval:
                time.sleep(options.poll_interval)

        zero_start = time.monotonic_ns()
        client.zero_destination(destination, 0, destination.num_full_tokens)
        zero_end = time.monotonic_ns()
        checksum_start = time.monotonic_ns()
        sentinel_hashes = checked_checksums(client, destination)
        checksum_ns += time.monotonic_ns() - checksum_start
        if sentinel_hashes != destination_hashes:
            raise RuntimeError("handle-mode destination changed before RETRIEVE")
        if phase_barrier is not None:
            phase_barrier("read", sequence)

        sessions.append(destination)
        with torch.cuda.nvtx.range("mp_read_lookup_to_consumer_ready"):
            read_start = time.monotonic_ns()
            hit = client.lookup(destination)
            lookup_end = time.monotonic_ns()
            if not hit.is_full_hit:
                raise RuntimeError(
                    f"published request is not a full hit sequence={sequence}: "
                    f"{hit.hit_chunks}/{hit.total_chunks} error={hit.error}"
                )
            retrieve_start = time.monotonic_ns()
            retrieved = client.retrieve(destination, 0, destination.num_full_tokens)
            retrieve_reply = time.monotonic_ns()
            if retrieved is None or not retrieved.succeeded:
                raise RuntimeError(f"RETRIEVE failed sequence={sequence}: {retrieved}")
            consumer_ready.record(torch.cuda.current_stream())
            consumer_ready.synchronize()
            read_ready = time.monotonic_ns()

        checksum_start = time.monotonic_ns()
        retrieved_hashes = checked_checksums(client, destination)
        checksum_ns += time.monotonic_ns() - checksum_start
        if retrieved_hashes != source_hashes:
            raise RuntimeError(
                f"round-trip corruption sequence={sequence}: "
                f"source={source_hashes}, retrieved={retrieved_hashes}"
            )
        return RequestSample(
            sequence=sequence,
            source_block_offset=request.block_offset,
            destination_block_offset=destination.block_offset,
            chunks=request.total_chunks,
            useful_bytes_each_direction=payload_bytes,
            cold_lookup_us=(cold_end - cold_start) / 1000,
            store_reply_us=(store_reply - store_start) / 1000,
            store_visible_us=(store_visible - store_start) / 1000,
            publication_after_reply_us=(store_visible - store_reply) / 1000,
            publication_lookup_us=tuple(publication_lookups),
            read_lookup_us=(lookup_end - read_start) / 1000,
            retrieve_reply_us=(retrieve_reply - retrieve_start) / 1000,
            retrieve_gpu_ready_us=(read_ready - retrieve_start) / 1000,
            combined_read_ready_us=(read_ready - read_start) / 1000,
            checksum_overhead_us=checksum_ns / 1000,
            zero_destination_call_us=(zero_end - zero_start) / 1000,
            store_admitted_monotonic_ns=store_start,
            store_reply_monotonic_ns=store_reply,
            store_visible_monotonic_ns=store_visible,
            read_admitted_monotonic_ns=read_start,
            completed_monotonic_ns=read_ready,
            source_checksums=source_hashes,
            destination_before_checksums=destination_hashes,
            destination_after_checksums=retrieved_hashes,
            destination_changed_before_retrieve=False,
            noop_retrieve_would_fail=True,
            checksum_match=True,
            worker_rank=options.barrier_rank,
            group_id=options.group_id,
            shared_group_sequence=sequence,
        )
    finally:
        for session in reversed(sessions):
            client.end_session(session)


def parse_args() -> argparse.Namespace:
    """Parse and validate benchmark-only CLI settings.

    Returns:
        Namespace with RPC endpoints, request counts, geometry, and output paths.

    Raises:
        SystemExit: On invalid CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc-url", default="tcp://127.0.0.1:18555")
    parser.add_argument("--http-url", default="http://127.0.0.1:18880")
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--sequence-start", type=int, default=1_000_000)
    parser.add_argument("--visibility-timeout", type=float, default=30)
    parser.add_argument("--poll-interval", type=float, default=0.001)
    parser.add_argument("--start-at-ns", type=int, default=0)
    parser.add_argument("--barrier-dir", type=Path)
    parser.add_argument("--barrier-rank", type=int, default=0)
    parser.add_argument("--barrier-workers", type=int, default=1)
    parser.add_argument("--barrier-timeout", type=float, default=60)
    parser.add_argument("--group-id", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=Path)
    options = parser.parse_args()
    if (
        options.requests < 1
        or options.warmup < 0
        or options.visibility_timeout <= 0
        or options.duration <= 0
    ):
        parser.error("positive request count/timeout and nonnegative warmup required")
    if options.poll_interval < 0:
        parser.error("poll interval must be nonnegative")
    if not 0 <= options.barrier_rank < options.barrier_workers:
        parser.error("barrier rank must identify one configured worker")
    if (
        min(
            options.num_tokens,
            options.num_layers,
            options.num_heads,
            options.head_size,
            options.num_blocks,
            options.block_size,
        )
        <= 0
    ):
        parser.error("all geometry dimensions must be positive")
    if options.samples is None:
        options.samples = options.output.with_suffix(".samples.jsonl")
    return options


def run(options: argparse.Namespace) -> dict[str, object]:
    """Execute the real GPU/RPC workload and retain every completed request.

    Args:
        options: Validated CLI namespace.

    Returns:
        Aggregate result with explicit completion boundaries and source audit.

    Raises:
        RuntimeError: On unavailable CUDA, missed start barrier, or bad I/O.
        OSError: If result artifacts cannot be written.
    """
    global GPU_VERIFIER
    GPU_VERIFIER = None
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.cuda.set_device(options.gpu)
    config = BenchConfig(
        rpc_url=options.rpc_url,
        http_url=options.http_url,
        mode="gpu",
        transfer_mode="lmcache_driven",
        tp_size=1,
        use_mla=False,
        num_tokens=options.num_tokens,
        kvcache_shape_spec=(
            f"(2,{options.num_blocks},{options.block_size},{options.num_heads},"
            f"{options.head_size}):float16:{options.num_layers}"
        ),
        num_blocks=options.num_blocks,
        block_size=options.block_size,
    )
    options.samples.parent.mkdir(parents=True, exist_ok=True)
    client = OwnedBenchClient(config, lambda message: logger.debug("%s", message))
    run_id = f"gds-mp-{uuid.uuid4().hex}"
    samples: list[RequestSample] = []
    barrier = (
        FileBarrier(
            options.barrier_dir,
            options.barrier_rank,
            options.barrier_workers,
            options.barrier_timeout,
        )
        if options.barrier_dir is not None
        else None
    )
    consumer_ready = torch.cuda.Event()
    consumer_ready.record(torch.cuda.current_stream())
    consumer_ready.synchronize()
    setup_start = time.monotonic_ns()
    try:
        client.start()
        GPU_VERIFIER = GPUVerifier(client, options, checked_checksums, zero_checksum)
        setup_end = time.monotonic_ns()
        for ordinal in range(options.warmup):
            run_request(
                client,
                options.sequence_start + ordinal,
                ordinal + options.barrier_rank if barrier is not None else ordinal,
                run_id,
                options,
                consumer_ready,
                barrier.wait if barrier is not None else None,
            )
        if options.start_at_ns:
            remaining = (options.start_at_ns - time.monotonic_ns()) / 1e9
            if remaining < 0:
                raise RuntimeError("missed requested measurement start barrier")
            time.sleep(remaining)
        GPU_VERIFIER.clear(-1)
        start = time.monotonic_ns()
        cpu_start = time.process_time()
        with options.samples.open("w", encoding="utf-8") as output:
            for ordinal in range(options.warmup, options.warmup + options.requests):
                if samples and (time.monotonic_ns() - start) / 1e9 >= options.duration:
                    break
                if samples and len(samples) % 16 == 0:
                    GPU_VERIFIER.clear(len(samples))
                sample = run_request(
                    client,
                    options.sequence_start + ordinal,
                    ordinal + options.barrier_rank if barrier is not None else ordinal,
                    run_id,
                    options,
                    consumer_ready,
                    barrier.wait if barrier is not None else None,
                )
                samples.append(sample)
                output.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
                output.flush()
        end = time.monotonic_ns()
        measured_cpu_seconds = time.process_time() - cpu_start
        verifier = GPU_VERIFIER
        GPU_VERIFIER = None
        verifier.postflight()
    finally:
        GPU_VERIFIER = None
        client.close()

    useful_bytes = sum(sample.useful_bytes_each_direction for sample in samples)
    timing_fields = (
        "cold_lookup_us",
        "store_reply_us",
        "store_visible_us",
        "publication_after_reply_us",
        "read_lookup_us",
        "retrieve_reply_us",
        "retrieve_gpu_ready_us",
        "combined_read_ready_us",
        "checksum_overhead_us",
    )
    timing = {
        name: distribution([float(getattr(sample, name)) for sample in samples])
        for name in timing_fields
    }
    return {
        "verification": (
            "GPU exact byte equality on every NHD layer/range; full CPU source "
            "digests before/after steady window"
        ),
        "clear_events": verifier.clear_events,
        "gpu_verification_prepare_seconds": verifier.prepare_seconds,
        "verification_persistent_extra_gpu_bytes": 0,
        "verification_peak_extra_gpu_bytes": None,
        "verification_peak_extra_gpu_bytes_reason": (
            "Temporary allocations inside count_nonzero/equal are not measured; "
            "no extra persistent full-payload staging tensor is allocated"
        ),
        "initial_process_rng_seed": 0,
        "effective_payload_rng_seed": 42,
        "payload_seed_source": (
            "ServerBenchClient allocation resets torch.random.manual_seed(42)"
        ),
        "implementation": "mp_rpc",
        "mode": "lmcache_driven",
        "direction": "roundtrip",
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "worker_rank": options.barrier_rank,
        "group_id": options.group_id,
        "source_ordinal_policy": (
            "request ordinal plus group rank, modulo untouched source regions"
            if barrier is not None
            else "request ordinal modulo untouched source regions"
        ),
        "gpu": options.gpu,
        "gpu_name": torch.cuda.get_device_name(options.gpu),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "rpc_url": options.rpc_url,
        "http_url": options.http_url,
        "config": asdict(config),
        "seed": 42,
        "size": samples[0].useful_bytes_each_direction,
        "requested_depth": 1,
        "streams": 1,
        "completed_operations": len(samples) * 2,
        "completed_roundtrips": len(samples),
        "useful_bytes": useful_bytes * 2,
        "physical_bytes": None,
        "physical_bytes_reason": "RPC client cannot observe internal cuFile requests",
        "duration_seconds": (end - start) / 1e9,
        "measured_process_cpu_seconds": measured_cpu_seconds,
        "cpu_percent_one_core": measured_cpu_seconds * 1e11 / (end - start),
        "cpu_scope": "benchmark client process; server CPU not measured here",
        "measurement_start_ns": start,
        "measurement_stop_ns": end,
        "GBps": useful_bytes * 2 / (end - start),
        "GBps_definition": (
            "matched whole window; includes checksums and session cleanup"
        ),
        "checksum_seconds": sum(s.checksum_overhead_us for s in samples) / 1e6,
        "setup_seconds": (setup_end - setup_start) / 1e9,
        "timing_us": timing,
        "latency_us": timing["combined_read_ready_us"],
        "latency_definition": (
            "same request LOOKUP admission through consumer CUDA event; exact GPU "
            "verification follows outside this per-request interval"
        ),
        "store_reply_definition": (
            "public STORE returns after the remote device event; raw message arrival "
            "is not independently observable through ServerBenchClient"
        ),
        "store_visible_definition": (
            "STORE admission through first new-session LOOKUP reporting all chunks; "
            "includes lookup RPC and publication-observation delay"
        ),
        "destination_guard": (
            "disjoint original GPU blocks must differ from source before RETRIEVE; "
            "all chunk digests must match source afterwards"
        ),
        "noop_retrieve_would_fail": all(s.noop_retrieve_would_fail for s in samples),
        "correctness": "passed",
        "errors": 0,
        "path_verdict": "inconclusive",
        "samples_file": str(options.samples),
        "source_audit": {
            "zero_destination": (
                "ServerBenchClient.zero_destination is a no-op in handle mode; "
                "the benchmark writes an exact zero sentinel before each request, "
                "then verifies the handle-mode no-op leaves it unchanged"
            ),
            "publication": (
                "STORE device completion and metadata visibility are "
                "observed separately"
            ),
            "concurrency": (
                "tp_size1 registers fixed instance_id1000; use separate servers for "
                "concurrent processes"
            ),
            "production_cufile": (
                "context has explicit deferred-result validation and optional "
                "checked synchronous workers"
                if hasattr(GDSContext, "synchronize_and_check")
                else "original context has no explicit deferred-result validation; "
                "standalone replay verifies retrieved payload"
            ),
        },
    }


def main() -> None:
    """Write one success/error artifact and propagate failed real-workload checks."""
    options = parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run(options)
    except (RuntimeError, ValueError, TimeoutError, OSError) as error:
        logger.exception("MP replay failed")
        options.output.write_text(
            json.dumps(
                {
                    "implementation": "mp_rpc",
                    "errors": 1,
                    "correctness": "failed",
                    "error": str(error),
                    "samples_file": str(options.samples),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    options.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
