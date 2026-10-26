# SPDX-License-Identifier: Apache-2.0
"""Two MP servers share one Device-DAX region: KV stored through one server
is retrieved through the other, bit-exact, on the GPU.

The region is a tmpfs file mapped through the ``DEVDAX`` option, or the
Device-DAX device named by ``LMCACHE_TEST_SHARED_DEVDAX_PATH`` (its first
``REGION_BYTES`` are overwritten).
"""

# Standard
from collections.abc import Callable, Iterator
from functools import partial
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
from lmcache.utils import EngineType
from lmcache.v1.distributed.config import EvictionConfig, parse_args
from lmcache.v1.memory_orchestrator.api import RegionUsage
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.server import run_cache_server
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from tests.v1.distributed.shared_l1.utils import region_usage
from tests.v1.memory_orchestrator.harness import (
    REGION_ID,
    free_port,
    running_orchestrator,
)

pytestmark = pytest.mark.cuda

if not (torch_dev.is_available() and torch_device_type == "cuda"):
    pytest.skip("requires available CUDA runtime", allow_module_level=True)

# First Party
from lmcache.v1.platform.devices.cuda.ipc_wrapper import CudaIPCWrapper  # noqa: E402
from tests.v1.multiprocess.test_cache_server import (  # noqa: E402
    BLOCKS_PER_KEY,
    CHUNK_SIZE,
    lookup_all,
    retrieve_keys,
    store_keys,
)

PAGE_SIZE = 16
NUM_KEYS = 4
NUM_LAYERS = 4
NUM_PAGES = 256
# One chunk is 4 layers x 2 x 256 tokens x 8 heads x 128 x 2 bytes = 4 MiB.
REGION_BYTES = 64 << 20
TIMEOUT = 30.0


def _serve(port: int, region: str, endpoint: str, client_id: str) -> None:
    """Run one MP server whose only L1 is the shared region."""
    shared = {
        "type": "DEVDAX",
        "tag": "cxl",
        "size_gb": REGION_BYTES / (1 << 30),
        "path": region,
        "shared": {
            "orchestrator": endpoint,
            "region_id": REGION_ID,
            "client_id": client_id,
        },
    }
    storage = parse_args(["--l1-manager", json.dumps(shared)])
    storage.eviction_config = EvictionConfig("noop")
    run_cache_server(
        MPServerConfig(
            transport="zmq",
            host="localhost",
            port=port,
            chunk_size=CHUNK_SIZE,
            supported_transfer_mode="lmcache_driven",
        ),
        storage,
        DEFAULT_OBSERVABILITY_CONFIG,
        start_prometheus_http_server=False,
    )


def _event_handle(keepalive: list[object]) -> bytes:
    backend = get_event_ipc_backend(0)
    event = backend.create_event(0)
    backend.record_event(event, None)
    keepalive.append(event)
    return backend.export_event(event, 0)


def _key(index: int) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "shared-model",
        1,
        0,
        [index + 1] * CHUNK_SIZE,
        start=0,
        end=CHUNK_SIZE,
        request_id=f"shared-{uuid.uuid4().hex}",
    )


def _wait_until(check: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def _responds(client: RequestClient) -> bool:
    try:
        client.get_chunk_size().result(timeout=1)
    except Exception:
        return False
    return True


@pytest.fixture
def region(tmp_path: Path) -> Iterator[str]:
    device = os.environ.get("LMCACHE_TEST_SHARED_DEVDAX_PATH")
    if device:
        yield device
        return
    path = Path("/dev/shm") / f"lmcache-shared-{uuid.uuid4().hex}"
    with open(path, "wb") as backing:
        backing.truncate(REGION_BYTES)
    yield str(path)
    path.unlink()


@pytest.fixture
def servers(
    tmp_path: Path, region: str
) -> Iterator[tuple[RequestClient, RequestClient, Callable[[], RegionUsage]]]:
    with running_orchestrator(tmp_path / "state", capacity_bytes=REGION_BYTES) as orch:
        ctx = mp.get_context("spawn")
        ports = [free_port(), free_port()]
        processes = [
            ctx.Process(
                target=_serve,
                args=(port, region, orch.endpoint, f"server-{index}"),
                daemon=True,
            )
            for index, port in enumerate(ports)
        ]
        clients: list[RequestClient] = []
        try:
            for process in processes:
                process.start()
            for port in ports:
                client = RequestClientFactory.create(f"tcp://localhost:{port}")
                clients.append(client)
                _wait_until(partial(_responds, client))
            with region_usage(orch.endpoint) as usage:
                yield clients[0], clients[1], usage
        finally:
            for client in clients:
                client.close()
            for process in processes:
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()


def test_store_on_one_server_retrieve_on_the_other(
    servers: tuple[RequestClient, RequestClient, Callable[[], RegionUsage]],
) -> None:
    writer, reader, usage = servers
    torch.manual_seed(7)
    kv = [
        torch.rand((2, NUM_PAGES, PAGE_SIZE, 8, 128), dtype=torch.bfloat16).to(
            torch_device_type
        )
        for _ in range(NUM_LAYERS)
    ]
    instance_id = os.getpid()
    for client in (writer, reader):
        client.register_kv_cache(
            instance_id,
            [CudaIPCWrapper(tensor) for tensor in kv],
            "shared-model",
            1,
            EngineType.VLLM,
            {},
            [],
        ).result(timeout=TIMEOUT)
    keepalive: list[object] = []
    keys = [_key(i) for i in range(NUM_KEYS)]
    stored = list(range(NUM_KEYS * BLOCKS_PER_KEY))
    store_keys(writer, keys, instance_id, stored, _event_handle(keepalive), TIMEOUT)
    _wait_until(lambda: usage().valid == NUM_KEYS and usage().writing == 0)

    # Every chunk is served from the region.
    assert lookup_all(reader, keys, TIMEOUT) == NUM_KEYS
    loaded = [block + 8 * BLOCKS_PER_KEY for block in stored]
    handle = _event_handle(keepalive)
    assert all(retrieve_keys(reader, keys, instance_id, loaded, handle, TIMEOUT))
    torch_dev.synchronize()

    pages = len(stored)
    for layer in kv:
        original = layer[:, :pages]
        copied = layer[:, 8 * BLOCKS_PER_KEY : 8 * BLOCKS_PER_KEY + pages]
        assert torch.equal(original, copied)
    # Every grant committed and every lease released once the copies drained.
    _wait_until(lambda: usage().read_leases == 0)
    assert (usage().writing, usage().valid) == (0, NUM_KEYS)
