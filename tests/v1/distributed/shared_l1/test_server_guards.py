# SPDX-License-Identifier: Apache-2.0
"""MP server settings a shared L1 refuses, and its explicit client identity."""

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    SharedL1Config,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    P2PConfig,
    RuntimePluginConfig,
)
from lmcache.v1.multiprocess.server import prepare_shared_l1


def _storage(tmp_path: Path, **shared: object) -> StorageManagerConfig:
    spec = {
        "type": "DEVDAX",
        "tag": "cxl",
        "size_gb": 1,
        "path": str(tmp_path / "region"),
        "shared": {
            "orchestrator": "127.0.0.1:7700",
            "region_id": "pool-a",
            "client_id": "host-a",
            **shared,
        },
    }
    return parse_args(["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)])


def _shared(storage: StorageManagerConfig) -> SharedL1Config:
    l1 = storage.l1_manager_configs[0]
    assert isinstance(l1, DevDaxL1ManagerConfig) and l1.shared is not None
    return l1.shared


def test_explicit_client_id_is_kept(tmp_path: Path) -> None:
    storage = _storage(tmp_path, client_id="rack1-host3")
    prepare_shared_l1(MPServerConfig(supported_transfer_mode="lmcache_driven"), storage)
    assert _shared(storage).client_id == "rack1-host3"


@pytest.mark.parametrize(
    ("mp_config", "message"),
    [
        (MPServerConfig(supported_transfer_mode="auto"), "lmcache_driven"),
        (MPServerConfig(supported_transfer_mode="engine_driven"), "lmcache_driven"),
        (
            MPServerConfig(
                supported_transfer_mode="lmcache_driven", engine_type="blend"
            ),
            "CacheBlend",
        ),
        (
            MPServerConfig(
                supported_transfer_mode="lmcache_driven",
                p2p_config=P2PConfig(advertise_url="10.0.0.1:9000"),
            ),
            "P2P",
        ),
        (
            MPServerConfig(
                supported_transfer_mode="lmcache_driven",
                runtime_plugin_config=RuntimePluginConfig(locations=["plugin.py"]),
            ),
            "runtime plugins",
        ),
    ],
)
def test_unsupported_features_are_rejected(
    tmp_path: Path, mp_config: MPServerConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        prepare_shared_l1(mp_config, _storage(tmp_path))


def test_private_l1_is_untouched(tmp_path: Path) -> None:
    storage = parse_args(
        [
            "--eviction-policy",
            "noop",
            "--l1-manager",
            json.dumps({"type": "DRAM", "tag": "_default", "size_gb": 1}),
        ]
    )
    prepare_shared_l1(MPServerConfig(supported_transfer_mode="auto"), storage)
