# SPDX-License-Identifier: Apache-2.0
"""Benchmark-owned tensor access without changing the measured LMCache source."""

# Third Party
import torch

# First Party
from lmcache.cli.commands.bench.server_bench.client import ServerBenchClient


class OwnedBenchClient(ServerBenchClient):
    """Expose this benchmark instance's GPU tensors for exact byte verification."""

    def verification_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the owned TP1 layer tensors without copying or allocating.

        Raises:
            ValueError: The client has not started exactly one worker.
        """
        if len(self._workers) != 1:
            raise ValueError("GPU exact verifier requires one benchmark worker")
        return tuple(self._workers[0].kv_caches.values())
