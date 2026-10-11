# SPDX-License-Identifier: Apache-2.0
"""CPU checks of exact verification through its public construction contract."""

# Standard
from argparse import Namespace
from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock
import hashlib
import json

# Third Party
import pytest
import torch

# First Party
from benchmarks.gds_optimization import gpu_verification
from benchmarks.gds_optimization.client import OwnedBenchClient
from benchmarks.gds_optimization.gpu_verification import GPUVerifier
from lmcache.cli.commands.bench.server_bench.client import (
    RequestContext,
    ServerBenchClient,
)


class CPUClient(OwnedBenchClient):
    """Public client fixture with deterministic nonzero NHD layer tensors."""

    def __init__(self) -> None:
        self.tensors = tuple(
            (torch.arange(48, dtype=torch.float16) + 1 + layer * 48).reshape(
                2, 6, 2, 1, 2
            )
            for layer in range(2)
        )

    def verification_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return this fixture's owned layers through the production interface."""
        return self.tensors

    def create_request(
        self, sequence_id: int, request_id: str, request_kind: str
    ) -> RequestContext:
        """Create one four-token chunk spanning two engine blocks."""
        return RequestContext(
            sequence_id=sequence_id,
            request_id=request_id,
            request_kind=request_kind,
            token_ids=(1, 2, 3, 4),
            num_full_tokens=4,
            total_chunks=1,
            chunk_size=4,
            block_offset=0,
            num_blocks=2,
        )


def checksums(client: ServerBenchClient, request: RequestContext) -> tuple[str, ...]:
    """Hash every requested byte of both KV planes and both fixture layers."""
    assert isinstance(client, CPUClient)
    digest = hashlib.md5()
    for layer in client.verification_tensors():
        view = layer[
            :, request.block_offset : request.block_offset + request.num_blocks
        ]
        digest.update(view.contiguous().view(torch.uint8).numpy().tobytes())
    return (digest.hexdigest(),)


def zero_checksum(length: int) -> str:
    """Use the same CPU checksum convention for a completely zero chunk."""
    return hashlib.md5(bytes(length)).hexdigest()


@pytest.fixture
def verifier(monkeypatch: pytest.MonkeyPatch) -> tuple[GPUVerifier, CPUClient]:
    """Construct normally and replace only the unavailable CUDA stream wait."""
    monkeypatch.setattr(gpu_verification.torch.cuda, "current_stream", MagicMock())
    client = CPUClient()
    options = Namespace(
        sequence_start=0,
        num_blocks=6,
        num_layers=2,
        block_size=2,
        num_heads=1,
        head_size=2,
        http_url="http://owned-server",
    )
    return GPUVerifier(client, options, checksums, zero_checksum), client


def start_checks(verifier: GPUVerifier, client: CPUClient) -> RequestContext:
    """Follow the request's source and two zero-sentinel checks in order."""
    verifier.begin(1)
    source = client.create_request(1, "source", "gds")
    source = replace(source, block_offset=2)
    destination = replace(source, block_offset=4)
    assert verifier.check(client, source) == checksums(client, source)
    for _ in range(2):
        assert verifier.check(client, destination) == ("exact-zero-sentinel",)
    return destination


def test_exact_nonzero_strided_roundtrip(
    verifier: tuple[GPUVerifier, CPUClient],
) -> None:
    """Exact equality covers every byte of noncontiguous NHD block regions."""
    checker, client = verifier
    request = start_checks(checker, client)
    source = checker.views(2)
    destination = checker.views(4)
    assert all(not view.is_contiguous() for view in source)
    for expected, actual in zip(source, destination, strict=True):
        actual.copy_(expected)
    assert checker.check(client, request) == checksums(client, request)
    checker.postflight()


@pytest.mark.parametrize("corrupt", [False, True], ids=["noop", "last-byte"])
def test_noop_and_last_byte_corruption_fail(
    verifier: tuple[GPUVerifier, CPUClient], corrupt: bool
) -> None:
    """A missing transfer or final byte corruption cannot pass GPU equality."""
    checker, client = verifier
    request = start_checks(checker, client)
    if corrupt:
        for expected, actual in zip(checker.views(2), checker.views(4), strict=True):
            actual.copy_(expected)
        checker.views(4)[-1][-1, -1, -1, -1, -1] ^= 1
    with pytest.raises(RuntimeError, match="exact byte comparison"):
        checker.check(client, request)


def test_changed_zero_sentinel_fails(verifier: tuple[GPUVerifier, CPUClient]) -> None:
    """Destination mutation before retrieve must fail its sentinel check."""
    checker, client = verifier
    checker.begin(0)
    request = client.create_request(0, "source", "gds")
    checker.check(client, request)
    checker.views(4)[0][0, 0, 0, 0, 0] = 1
    with pytest.raises(RuntimeError, match="sentinel changed"):
        checker.check(client, request)


def test_source_postflight_detects_mutation(
    verifier: tuple[GPUVerifier, CPUClient],
) -> None:
    """Source integrity is independently rehashed after the measured phase."""
    checker, client = verifier
    client.verification_tensors()[-1][-1, 3, -1, -1, -1] += 1
    with pytest.raises(RuntimeError, match="Untouched source changed"):
        checker.postflight()


@pytest.mark.parametrize(
    "busy_field",
    [
        "active_sessions",
        "active_prefetch_jobs",
        "write_locked_count",
        "staging_object_count",
        "in_flight_task_count",
        "pending_keys_count",
    ],
)
def test_clear_refuses_every_busy_server_state(
    verifier: tuple[GPUVerifier, CPUClient],
    monkeypatch: pytest.MonkeyPatch,
    busy_field: str,
) -> None:
    """Each public busy indicator blocks forced clear before any POST."""
    manager = {"write_locked_count": 0, "staging_object_count": 0}
    controller = {"in_flight_task_count": 0, "pending_keys_count": 0}
    status: dict[str, object] = {
        "active_sessions": 0,
        "active_prefetch_jobs": 0,
        "storage_manager": {"l1_manager": manager, "store_controller": controller},
    }
    if busy_field in manager:
        manager[busy_field] = 1
    elif busy_field in controller:
        controller[busy_field] = 1
    else:
        status[busy_field] = 1
    request = MagicMock(return_value=BytesIO(json.dumps(status).encode()))
    monkeypatch.setattr(gpu_verification, "urlopen", request)
    with pytest.raises(RuntimeError, match="not quiescent"):
        verifier[0].clear(16)
    request.assert_called_once_with("http://owned-server/status", timeout=30)
    assert verifier[0].clear_events == []
