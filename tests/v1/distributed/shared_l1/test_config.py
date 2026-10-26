# SPDX-License-Identifier: Apache-2.0
"""Validation of the ``shared`` object of a Device-DAX L1 and of server modes."""

# Standard
from dataclasses import replace
from typing import Any
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    SharedDevDaxConfig,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    P2PConfig,
    RuntimePluginConfig,
)
from lmcache.v1.multiprocess.server import check_shared_devdax_support

SHARED = {"orchestrator": "h:1", "region_id": "region", "client_id": "host"}


@pytest.fixture
def spec() -> dict[str, object]:
    return {
        "type": "DEVDAX",
        "tag": "pool",
        "size_gb": 1,
        "path": "/dev/dax0.0",
        "offset_gb": 1,
        "shared": dict(SHARED),
    }


def args(specs: list[dict[str, object]]) -> list[str]:
    return [
        "--eviction-policy",
        "LRU",
        *[arg for spec in specs for arg in ("--l1-manager", json.dumps(spec))],
    ]


def test_shared_object_reuses_the_dax_fields(spec: dict[str, object]) -> None:
    private = {"type": "DRAM", "tag": "_default", "size_gb": 1}
    config = parse_args(args([spec, private]))
    shared = config.l1_manager_configs[0]
    assert isinstance(shared, DevDaxL1ManagerConfig)
    assert shared.shared == SharedDevDaxConfig("h:1", "region", "host", 5.0)
    memory = shared.memory_config
    assert (memory.devdax_path, memory.devdax_offset_in_bytes) == (
        "/dev/dax0.0",
        1 << 30,
    )
    # The global LRU default does not reach the shared L1.
    assert shared.eviction is not None and shared.eviction.eviction_policy == "noop"
    check_shared_devdax_support(
        MPServerConfig(supported_transfer_mode="lmcache_driven"), config
    )


def test_omitted_shared_keeps_the_private_l1(spec: dict[str, object]) -> None:
    del spec["shared"]
    config = parse_args(args([spec])).l1_manager_configs[0]
    assert isinstance(config, DevDaxL1ManagerConfig) and config.shared is None
    assert config.eviction is not None and config.eviction.eviction_policy == "LRU"


@pytest.mark.parametrize(
    "field,value",
    [
        ("orchestrator", ""),
        ("region_id", None),
        ("client_id", " "),
        ("rpc_timeout_seconds", 0),
        ("rpc_timeout_seconds", float("inf")),
        ("rpc_timeout_seconds", True),
        ("path", "/dev/dax0.0"),
        ("unknown", 1),
    ],
)
def test_malformed_shared_fields(
    spec: dict[str, object], field: str, value: object
) -> None:
    shared = spec["shared"]
    assert isinstance(shared, dict)
    shared[field] = value
    with pytest.raises(ValueError):
        parse_args(args([spec]))


@pytest.mark.parametrize(
    "change",
    [
        {"shared": "h:1"},
        {"type": "DRAM"},
        {"type": "GDS"},
        {"eviction": {"eviction_policy": "LRU"}},
        {"eviction": {"trigger_watermark": 0.5}},
        {"path": None},
        {"shared": {"orchestrator": "h:1", "region_id": "r"}},
    ],
)
def test_invalid_shared_settings(
    spec: dict[str, object], change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        parse_args(args([{**spec, **change}]))


def test_only_one_shared_region(spec: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Only one shared"):
        parse_args(args([spec, {**spec, "tag": "peer", "offset_gb": 3}]))


def programmatic(**changes: Any) -> DevDaxL1ManagerConfig:
    config = DevDaxL1ManagerConfig(
        L1MemoryManagerConfig(
            1 << 30, use_lazy=False, shm_name="", devdax_path="/dev/dax0.0"
        ),
        tag="pool",
        shared=SharedDevDaxConfig("h:1", "region", "host"),
    )
    return replace(config, **changes)


@pytest.mark.parametrize(
    "build",
    [
        # Inherits the global LRU policy, which a shared L1 cannot use.
        lambda: programmatic(),
        lambda: programmatic(eviction=EvictionConfig("ARC")),
        lambda: programmatic(
            eviction=EvictionConfig("noop"),
            memory_config=L1MemoryManagerConfig(
                1 << 30,
                use_lazy=False,
                shm_name="",
                devdax_path="/dev/dax0.0",
                devdax_size_in_bytes=1 << 30,
            ),
        ),
        lambda: programmatic(
            eviction=EvictionConfig("noop"),
            shared=SharedDevDaxConfig("h:1", "region", "host", float("nan")),
        ),
    ],
)
def test_programmatic_construction_is_validated(build) -> None:
    with pytest.raises(ValueError):
        config = build()
        StorageManagerConfig(config, EvictionConfig("LRU"), l1_manager_configs=[config])


BASE_MP = MPServerConfig(supported_transfer_mode="lmcache_driven")


@pytest.mark.parametrize(
    "mp_config",
    [
        replace(BASE_MP, supported_transfer_mode="auto"),
        replace(BASE_MP, supported_transfer_mode="engine_driven"),
        replace(BASE_MP, engine_type="blend"),
        replace(BASE_MP, p2p_config=P2PConfig(advertise_url="h:9000")),
        replace(
            BASE_MP, runtime_plugin_config=RuntimePluginConfig(locations=["plugin.py"])
        ),
    ],
)
def test_unsupported_server_modes(
    spec: dict[str, object], mp_config: MPServerConfig
) -> None:
    with pytest.raises(ValueError, match="shared Device-DAX"):
        check_shared_devdax_support(mp_config, parse_args(args([spec])))


def test_private_servers_keep_every_mode(spec: dict[str, object]) -> None:
    del spec["shared"]
    check_shared_devdax_support(
        replace(BASE_MP, supported_transfer_mode="auto", engine_type="blend"),
        parse_args(args([spec])),
    )
