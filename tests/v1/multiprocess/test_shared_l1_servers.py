# SPDX-License-Identifier: Apache-2.0
"""Two MP servers on one host share one region through an orchestrator.

Server A stores KV chunks from GPU memory; server B, which never saw them,
looks them up and retrieves them into its own GPU blocks. The region is a
tmpfs file both servers map as a shared DRAM L1, so the copies are real
D2H/H2D DMA through the shared mapping and every byte must survive the trip.
"""

# Standard
from collections.abc import Iterator
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any
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
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import parse_args
from lmcache.v1.memory_orchestrator.api import DEFAULT_LAYOUT_FINGERPRINT, RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from tests.v1.distributed.shared_l1.utils import (
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
TIMEOUT = 30.0
_EVENTS: list[Any] = []


def _kv_cache(fill_random: bool) -> list[torch.Tensor]:
    device = torch.device(torch_device_type)
    shape = (2, NUM_PAGES, PAGE_SIZE, 8, 128)
    if fill_random:
        torch.random.manual_seed(7)
        return [
            torch.rand(shape, dtype=torch.bfloat16, device=device)
            for _ in range(NUM_LAYERS)
        ]
    return [
        torch.zeros(shape, dtype=torch.bfloat16, device=device)
        for _ in range(NUM_LAYERS)
    ]


def _key(index: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "shared-model",
        1,
        0,
        [index + 1] * CHUNK_SIZE,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"shared-{index}",
    )


def _event_handle() -> bytes:
    backend = get_event_ipc_backend(0)
    event = backend.create_event(0)
    backend.record_event(event, None)
    _EVENTS.append(event)
    return backend.export_event(event, 0)


def _run_server(port: int, region: str, endpoint: str) -> None:
    storage = parse_args(
        [
            "--eviction-policy",
            "noop",
            "--l1-manager",
            json.dumps(
                {
                    "type": "DRAM",
                    "tag": "pool",
                    "size_gb": REGION_BYTES / (1 << 30),
                    "path": region,
                    "shared": {"orchestrator": endpoint, "region_id": REGION_ID},
                }
            ),
        ]
    )
    run_cache_server(
        mp_config=MPServerConfig(
            host="localhost",
            port=port,
            chunk_size=CHUNK_SIZE,
            supported_transfer_mode="lmcache_driven",
        ),
        storage_manager_config=storage,
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


@pytest.fixture
def shared_servers(
    tmp_path: Path,
) -> Iterator[tuple[RequestClient, RequestClient, str]]:
    region = Path("/dev/shm") / f"lmcache-shared-l1-test-{uuid.uuid4().hex}"
    with open(region, "wb") as backing:
        backing.truncate(REGION_BYTES)
    orchestrator = start_orchestrator(
        tmp_path / "state",
        capacity_bytes=REGION_BYTES,
        alignment_bytes=2 << 20,
        visibility_mode="software_fenced",
    )
    context = mp.get_context("spawn")
    ports = [free_port(), free_port()]
    servers = [
        context.Process(
            target=_run_server,
            args=(port, str(region), orchestrator.endpoint),
            daemon=True,
        )
        for port in ports
    ]
    clients: list[RequestClient] = []
    try:
        for server in servers:
            server.start()
        clients = [_connect(port, s) for port, s in zip(ports, servers, strict=True)]
        yield clients[0], clients[1], orchestrator.endpoint
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


def _register(client: RequestClient, kv: list[torch.Tensor]) -> int:
    instance_id = os.getpid()
    client.register_kv_cache(
        instance_id,
        [CudaIPCWrapper(tensor) for tensor in kv],
        "shared-model",
        1,
        EngineType.VLLM,
        {},
        [],
    ).result(timeout=TIMEOUT)
    return instance_id


def _region_usage(endpoint: str) -> RegionUsage:
    """Ask the orchestrator directly, as an extra registered client."""
    probe = MemoryOrchestratorClient(endpoint, REGION_ID, f"probe-{uuid.uuid4().hex}")
    try:
        probe.register(
            layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
            mapped_bytes=REGION_BYTES,
            visibility_mode="software_fenced",
        )
        return probe.usage()
    finally:
        probe.close()


def _lookup(client: RequestClient, key: IPCCacheServerKey) -> int:
    lookup_key = key.no_worker_id_version()
    client.lookup(lookup_key, 1).result(timeout=TIMEOUT)
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        result = client.query_prefetch_status(lookup_key.request_id).result(
            timeout=TIMEOUT
        )
        if result is not None:
            return result
    raise TimeoutError("prefetch status never resolved")


def test_chunks_stored_by_one_server_are_retrieved_by_another(
    shared_servers: tuple[RequestClient, RequestClient, str],
) -> None:
    writer, reader, endpoint = shared_servers
    source = _kv_cache(fill_random=True)
    target = _kv_cache(fill_random=False)
    writer_id = _register(writer, source)
    reader_id = _register(reader, target)
    keys = [_key(i) for i in range(NUM_KEYS)]

    handle = _event_handle()
    for i, key in enumerate(keys):
        blocks = list(range(i * BLOCKS_PER_KEY, (i + 1) * BLOCKS_PER_KEY))
        stored = writer.store(key, writer_id, [blocks], handle)
        assert stored.to_device_future().result(timeout=TIMEOUT) is True

    # Commits run in a stream callback after the copies; wait for all of them.
    deadline = time.monotonic() + TIMEOUT
    while (usage := _region_usage(endpoint)).valid != NUM_KEYS:
        assert time.monotonic() < deadline, f"commits never landed: {usage}"
        time.sleep(0.2)
    assert sum(_lookup(reader, key) for key in keys) == NUM_KEYS

    handle = _event_handle()
    offset = NUM_KEYS * BLOCKS_PER_KEY
    for i, key in enumerate(keys):
        blocks = list(
            range(offset + i * BLOCKS_PER_KEY, offset + (i + 1) * BLOCKS_PER_KEY)
        )
        retrieved = reader.retrieve(key, reader_id, [blocks], handle, 0)
        assert retrieved.to_device_future().result(timeout=TIMEOUT)
    torch_dev.synchronize()

    for i in range(NUM_KEYS):
        stored_pages = slice(i * BLOCKS_PER_KEY, (i + 1) * BLOCKS_PER_KEY)
        loaded_pages = slice(
            offset + i * BLOCKS_PER_KEY, offset + (i + 1) * BLOCKS_PER_KEY
        )
        for layer in range(NUM_LAYERS):
            assert torch.equal(
                source[layer][:, stored_pages], target[layer][:, loaded_pages]
            ), f"chunk {i} layer {layer} differs after the cross-server trip"

    # The reader storing the same chunk allocates nothing: one copy per key.
    handle = _event_handle()
    blocks = list(range(0, BLOCKS_PER_KEY))
    reader.store(keys[0], reader_id, [blocks], handle).to_device_future().result(
        timeout=TIMEOUT
    )
    after = _region_usage(endpoint)
    assert (after.valid, after.writing, after.allocated_bytes) == (
        NUM_KEYS,
        0,
        usage.allocated_bytes,
    )
    for client, instance_id in ((writer, writer_id), (reader, reader_id)):
        client.unregister_kv_cache(instance_id).result(timeout=TIMEOUT)
