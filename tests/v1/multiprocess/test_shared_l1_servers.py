# SPDX-License-Identifier: Apache-2.0
"""Two MP servers on one host share one DRAM L1 region through an orchestrator.

The region is a tmpfs file both servers map, so stores and retrieves are real
D2H/H2D copies through the shared mapping.
"""

# Standard
from collections.abc import Iterator
from dataclasses import replace
from multiprocessing.process import BaseProcess
from pathlib import Path
import json
import multiprocessing as mp
import os
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import cache_flush_supported
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.memory_orchestrator.api import DEFAULT_LAYOUT_FINGERPRINT, RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from tests.v1.memory_orchestrator.harness import (
    REGION_ID,
    free_port,
    start_orchestrator,
)

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402

CHUNK_SIZE = 256
PAGE_SIZE = 16
BLOCKS_PER_KEY = CHUNK_SIZE // PAGE_SIZE
NUM_LAYERS = 4
NUM_PAGES = 512
NUM_KEYS = 6
REGION_BYTES = 256 << 20
# Cross-host fences where the CPU implements them; the region is one host here.
MODE = "software_fenced" if cache_flush_supported() else "coherent"
TIMEOUT = 30.0
# A CUDA event handle imports only while the exporting event is alive.
_EVENTS: list[object] = []


def _key(index: int, world_size: int, rank: int = 0) -> IPCCacheServerKey:
    tokens = [index + 1] * CHUNK_SIZE
    return IPCCacheServerKey.from_token_ids(
        "shared-model",
        world_size,
        rank,
        tokens,
        end=CHUNK_SIZE,
        request_id=f"shared-{index}",
    )


def _blocks(slot: int) -> list[int]:
    """Block ids of the ``slot``-th chunk in a paged KV cache."""
    return list(range(slot * BLOCKS_PER_KEY, (slot + 1) * BLOCKS_PER_KEY))


def _event_handle(device: int) -> bytes:
    backend = get_event_ipc_backend(device)
    event = backend.create_event(device)
    backend.record_event(event, None)
    _EVENTS.append(event)
    return backend.export_event(event, device)


def _run_server(port: int, region: Path, endpoint: str) -> None:
    spec = {
        "type": "DRAM",
        "tag": "pool",
        "size_gb": REGION_BYTES / (1 << 30),
        "path": str(region),
        "shared": {
            "orchestrator": endpoint,
            "region_id": REGION_ID,
            "client_id": f"mp-{port}",
        },
    }
    run_cache_server(
        mp_config=MPServerConfig(
            host="localhost",
            port=port,
            chunk_size=CHUNK_SIZE,
            supported_transfer_mode="lmcache_driven",
        ),
        storage_manager_config=parse_args(
            ["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)]
        ),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
    )


def _connect(port: int, process: BaseProcess) -> RequestClient:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        assert process.is_alive(), "MP server exited during startup"
        client = RequestClientFactory.create(f"tcp://localhost:{port}")
        try:
            assert client.get_chunk_size().result(timeout=2) == CHUNK_SIZE
            return client
        except Exception:
            client.close()
            time.sleep(0.5)
    raise RuntimeError("MP server did not come up")


def _region_usage(endpoint: str) -> RegionUsage:
    """Ask the orchestrator directly, as an extra registered client."""
    probe = MemoryOrchestratorClient(endpoint, REGION_ID, f"probe-{uuid.uuid4().hex}")
    try:
        probe.register(
            layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
            mapped_bytes=REGION_BYTES,
            visibility_mode=MODE,
        )
        return probe.usage()
    finally:
        probe.close()


def _lookup(client: RequestClient, key: IPCCacheServerKey) -> int:
    lookup_key = key.no_worker_id_version()
    client.lookup(lookup_key, lookup_key.world_size).result(timeout=TIMEOUT)
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        status = client.query_prefetch_status(lookup_key.request_id)
        result = status.result(timeout=TIMEOUT)
        if result is not None:
            return result
    raise TimeoutError("prefetch status never resolved")


@pytest.fixture
def shared_servers(
    tmp_path: Path,
) -> Iterator[tuple[RequestClient, RequestClient, str]]:
    """Start an orchestrator and two MP servers that map one tmpfs region.

    Args:
        tmp_path: Directory for the orchestrator's state.

    Yields:
        A client of each server, and the orchestrator endpoint.

    Raises:
        RuntimeError: If the orchestrator or a server does not come up.
    """
    region = Path("/dev/shm") / f"lmcache-shared-l1-test-{uuid.uuid4().hex}"
    orchestrator = start_orchestrator(
        tmp_path / "state",
        capacity_bytes=REGION_BYTES,
        alignment_bytes=2 << 20,
        visibility_mode=MODE,
    )
    servers: list[BaseProcess] = []
    clients: list[RequestClient] = []
    try:
        with open(region, "wb") as backing:
            backing.truncate(REGION_BYTES)
        endpoint = orchestrator.endpoint
        ctx = mp.get_context("spawn")
        ports = [free_port(), free_port()]
        servers = [
            ctx.Process(target=_run_server, args=(port, region, endpoint), daemon=True)
            for port in ports
        ]
        for server in servers:
            server.start()
        clients = [_connect(port, s) for port, s in zip(ports, servers, strict=True)]
        yield clients[0], clients[1], endpoint
    finally:
        for client in clients:
            client.close()
        for server in servers:
            if server.is_alive():
                server.terminate()
                server.join(timeout=10)
            if server.is_alive():
                server.kill()
                server.join()
        orchestrator.stop()
        region.unlink(missing_ok=True)


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_chunks_stored_by_one_server_are_retrieved_by_another(
    shared_servers: tuple[RequestClient, RequestClient, str],
    world_size: int,
) -> None:
    """Every TP shard crosses the mapping intact, including overlapping reads.

    Args:
        shared_servers: The writer and reader clients and the orchestrator endpoint.
        world_size: Number of tensor-parallel workers, each on its own GPU.
    """
    if torch_dev.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    writer, reader, endpoint = shared_servers
    torch.random.manual_seed(7)
    shape = (2, NUM_PAGES, PAGE_SIZE, 8, 128)
    sources, targets = [], []
    for rank in range(world_size):
        with torch_dev.device(rank):
            source = [
                torch.rand(shape, dtype=torch.bfloat16, device=f"cuda:{rank}")
                for _ in range(NUM_LAYERS)
            ]
            target = [torch.zeros_like(layer) for layer in source]
            sources.append(source)
            targets.append(target)
            for client, kv in ((writer, source), (reader, target)):
                wrappers: KVCache = [CudaIPCWrapper(tensor) for tensor in kv]
                client.register_kv_cache(
                    os.getpid() * 8 + rank,
                    wrappers,
                    "shared-model",
                    world_size,
                    EngineType.VLLM,
                    {},
                    [],
                ).result(timeout=TIMEOUT)
            for i in range(NUM_KEYS):
                if rank == world_size - 1 and i == NUM_KEYS - 1:
                    continue
                stored = writer.store(
                    _key(i, world_size, rank),
                    os.getpid() * 8 + rank,
                    [_blocks(i)],
                    _event_handle(rank),
                )
                assert stored.to_device_future().result(timeout=TIMEOUT) is True

    # Commits run in a stream callback after the copies; wait for all of them.
    deadline = time.monotonic() + TIMEOUT
    while (usage := _region_usage(endpoint)).valid != NUM_KEYS * world_size - 1:
        assert time.monotonic() < deadline, f"commits never landed: {usage}"
        time.sleep(0.2)
    late_key = _key(NUM_KEYS - 1, world_size, world_size - 1)
    assert _lookup(reader, replace(late_key, request_id="missing-shard")) == 0
    assert _region_usage(endpoint).read_leases == 0
    with torch_dev.device(world_size - 1):
        stored = writer.store(
            late_key,
            os.getpid() * 8 + world_size - 1,
            [_blocks(NUM_KEYS - 1)],
            _event_handle(world_size - 1),
        )
        assert stored.to_device_future().result(timeout=TIMEOUT) is True
    deadline = time.monotonic() + TIMEOUT
    while (usage := _region_usage(endpoint)).valid != NUM_KEYS * world_size:
        assert time.monotonic() < deadline, f"late commit never landed: {usage}"
        time.sleep(0.1)
    keys = [_key(i, world_size) for i in range(NUM_KEYS)]
    assert sum(_lookup(reader, key) for key in keys) == NUM_KEYS
    overlapping = replace(keys[0], request_id="overlapping-read")
    assert _lookup(reader, overlapping) == 1

    for rank in range(world_size):
        with torch_dev.device(rank):
            for i, key in enumerate(keys):
                loaded = reader.retrieve(
                    replace(key, worker_id=rank),
                    os.getpid() * 8 + rank,
                    [_blocks(NUM_KEYS + i)],
                    _event_handle(rank),
                    0,
                )
                assert loaded.to_device_future().result(timeout=TIMEOUT)
            loaded = reader.retrieve(
                replace(overlapping, worker_id=rank),
                os.getpid() * 8 + rank,
                [_blocks(NUM_KEYS)],
                _event_handle(rank),
                0,
            )
            assert loaded.to_device_future().result(timeout=TIMEOUT)
            torch_dev.synchronize()
            for i in range(NUM_KEYS):
                for layer in range(NUM_LAYERS):
                    assert torch.equal(
                        sources[rank][layer][:, _blocks(i)],
                        targets[rank][layer][:, _blocks(NUM_KEYS + i)],
                    ), f"rank {rank} chunk {i} layer {layer} differs"

    # The reader storing the same chunk allocates nothing: one copy per key.
    with torch_dev.device(0):
        duplicate = reader.store(
            keys[0], os.getpid() * 8, [_blocks(0)], _event_handle(0)
        )
        duplicate.to_device_future().result(timeout=TIMEOUT)
    after = _region_usage(endpoint)
    assert (after.valid, after.writing) == (NUM_KEYS * world_size, 0)
    assert after.allocated_bytes == usage.allocated_bytes
    deadline = time.monotonic() + TIMEOUT
    while (after := _region_usage(endpoint)).read_leases:
        assert time.monotonic() < deadline, f"read leases never drained: {after}"
        time.sleep(0.1)
    for client in (writer, reader):
        for rank in range(world_size):
            client.unregister_kv_cache(os.getpid() * 8 + rank).result(timeout=TIMEOUT)
    _EVENTS.clear()
