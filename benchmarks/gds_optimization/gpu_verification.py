# SPDX-License-Identifier: Apache-2.0
"""Exact, bounded NHD GPU verification for the owned single-worker MP replay."""

# Standard
from argparse import Namespace
from collections.abc import Callable
from dataclasses import replace
from urllib.request import Request, urlopen
import json
import time

# Third Party
import torch

# First Party
from benchmarks.gds_optimization.client import OwnedBenchClient
from lmcache.cli.commands.bench.server_bench.client import (
    RequestContext,
    ServerBenchClient,
)


class GPUVerifier:
    """Verify every requested FP16 NHD byte on a benchmark-owned TP1 client.

    Source CPU digests are captured outside measurement and checked afterwards.
    Within each request, checks progress through source, two zero sentinels and
    exact destination equality. No-op retrieve must fail the final comparison.
    """

    def __init__(
        self,
        client: OwnedBenchClient,
        options: Namespace,
        checked_checksums: Callable[
            [ServerBenchClient, RequestContext], tuple[str, ...]
        ],
        zero_checksum: Callable[[int], str],
    ) -> None:
        """Capture source digests and owned tensor views without new GPU storage.

        Args:
            client: Started benchmark client exposing its owned TP1 tensors.
            options: Replay geometry, sequence start and owned server HTTP URL.
            checked_checksums: Full-payload CPU digest callback, outside timing.
            zero_checksum: Digest callback for an all-zero logical chunk.

        Raises:
            ValueError: Geometry has no full request or separate source region.
            RuntimeError: A source logical chunk is entirely zero.
        """
        self.client = client
        self.options = options
        self.cpu_checksums = checked_checksums
        self.hashes: dict[int, tuple[str, ...]] = {}
        self.clear_events: list[dict[str, object]] = []
        self.prepare_seconds = 0.0
        self.calls = 0
        request = client.create_request(
            options.sequence_start, "verification-preflight", "gds"
        )
        if request is None or request.num_blocks <= 0:
            raise ValueError("GPU exact verifier needs at least one full request")
        self.blocks = request.num_blocks
        self.regions = options.num_blocks // self.blocks - 1
        if self.regions <= 0:
            raise ValueError("GPU exact verifier needs separate source and destination")
        self.tensors = list(client.verification_tensors())
        if len(self.tensors) != options.num_layers or any(
            tuple(t.shape)
            != (
                2,
                options.num_blocks,
                options.block_size,
                options.num_heads,
                options.head_size,
            )
            for t in self.tensors
        ):
            raise ValueError("GPU exact verifier requires the explicit NHD geometry")
        self.references: list[RequestContext] = []
        self.region = 0
        self.source: list[torch.Tensor] = []
        self.destination: list[torch.Tensor] = []
        for region in range(self.regions):
            reference = replace(request, block_offset=region * self.blocks)
            values = checked_checksums(client, reference)
            logical_chunk_bytes = (
                2
                * options.num_layers
                * request.chunk_size
                * options.num_heads
                * options.head_size
                * 2
            )
            if any(value == zero_checksum(logical_chunk_bytes) for value in values):
                raise RuntimeError("Preflight source has an all-zero logical chunk")
            self.hashes[region] = values
            self.references.append(reference)

    def views(self, block: int) -> list[torch.Tensor]:
        """Return byte views covering both KV planes at the specified block.

        Args:
            block: First block of one complete request region.

        Returns:
            One strided uint8 view per layer, without copying source data.
        """
        return [
            t[:, block : block + self.blocks].view(torch.uint8) for t in self.tensors
        ]

    def begin(self, ordinal: int) -> None:
        """Select a source region and finish zeroing the destination sentinel.

        Args:
            ordinal: Request ordinal used to cycle through source regions.
        """
        started = time.monotonic_ns()
        self.region = ordinal % self.regions
        self.source = self.views(self.region * self.blocks)
        self.destination = self.views(self.options.num_blocks - self.blocks)
        for tensor in self.destination:
            tensor.zero_()
        torch.cuda.current_stream().synchronize()
        self.calls = 0
        self.prepare_seconds += (time.monotonic_ns() - started) / 1e9

    def check(
        self, client: ServerBenchClient, request: RequestContext
    ) -> tuple[str, ...]:
        """Apply the next source/sentinel/returned-data check for one request.

        Args:
            client: Benchmark client retained for checksum callback compatibility.
            request: Request supplying the expected logical chunk count.

        Returns:
            Source digests after success, or explicit zero-sentinel markers.

        Raises:
            RuntimeError: A sentinel changed, a byte differs, or phases are wrong.
        """
        self.calls += 1
        if self.calls == 1:
            return self.hashes[self.region]
        if self.calls in (2, 3):
            if any(torch.count_nonzero(t).item() != 0 for t in self.destination):
                raise RuntimeError("GPU sentinel changed before retrieve")
            return ("exact-zero-sentinel",) * request.total_chunks
        if self.calls != 4 or not all(
            torch.equal(a, b)
            for a, b in zip(self.source, self.destination, strict=True)
        ):
            raise RuntimeError("GPU exact byte comparison failed after retrieve")
        # Exact equality proves the destination has the preflight source digest.
        return self.hashes[self.region]

    def clear(self, after_request: int) -> None:
        """Clear only an idle, dedicated benchmark server and verify it emptied.

        Args:
            after_request: Completed request ordinal for the retained event log.

        Raises:
            RuntimeError: Public status reports busy work, or clear is incomplete.
            OSError: The public HTTP request fails.

        The caller owns this sequential server; status is not a lease for shared
        servers. No other caller may start work between status and forced clear.
        """
        started = time.monotonic_ns()
        with urlopen(self.options.http_url + "/status", timeout=30) as response:
            before = json.load(response)
        manager = before["storage_manager"]["l1_manager"]
        controller = before["storage_manager"]["store_controller"]
        if (
            before["active_sessions"]
            or manager["write_locked_count"]
            or manager["staging_object_count"]
            or controller["in_flight_task_count"]
            or controller["pending_keys_count"]
            or before["active_prefetch_jobs"]
        ):
            raise RuntimeError("Owned server is not quiescent; refusing forced clear")
        request = Request(
            self.options.http_url + "/cache/clear",
            data=json.dumps({"tier": "l1", "force": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        if result.get("status") != "ok":
            raise RuntimeError(f"Owned benchmark cache clear failed: {result}")
        with urlopen(self.options.http_url + "/status", timeout=30) as response:
            after = json.load(response)
        manager_after = after["storage_manager"]["l1_manager"]
        if manager_after["memory_used_bytes"] or manager_after["total_object_count"]:
            raise RuntimeError("Owned clear did not release all logical slab regions")
        self.clear_events.append(
            {
                "after_request": after_request,
                "start_ns": started,
                "end_ns": time.monotonic_ns(),
                "force": True,
                "quiescent_scope": (
                    "owned sequential server only; every prior store published, "
                    "retrieve GPU-ready and byte-verified, all sessions ended"
                ),
                "reason": (
                    "publication-only LOOKUP probes retain read locks after END_SESSION"
                ),
                "status_before": before,
                "status_after": after,
            }
        )

    def postflight(self) -> None:
        """Rehash all source regions after measurement to catch corruption.

        Raises:
            RuntimeError: Any full CPU source digest differs from preflight.
        """
        for region, reference in enumerate(self.references):
            if self.cpu_checksums(self.client, reference) != self.hashes[region]:
                raise RuntimeError("Untouched source changed during measured replay")
