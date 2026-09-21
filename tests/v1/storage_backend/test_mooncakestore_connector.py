# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
    setup_mooncake_store,
)


class FakeStore:
    def __init__(self, result: int = 0, dict_api: bool = True) -> None:
        self.result = result
        self.dict_api = dict_api
        self.args: tuple[Any, ...] = ()

    def setup(self, *args: Any) -> int:
        if self.dict_api is False and len(args) == 1:
            raise TypeError
        self.args = args
        return self.result


@pytest.mark.parametrize(
    ("setup_config", "expected"),
    [
        (
            {
                "device_name": "legacy-device",
                "master_server_address": "legacy-master",
            },
            {
                "rdma_devices": "legacy-device",
                "master_server_addr": "legacy-master",
            },
        ),
        (
            {
                "device_name": "legacy-device",
                "rdma_devices": "",
                "master_server_address": "legacy-master",
                "master_server_addr": "canonical-master",
            },
            {"rdma_devices": "", "master_server_addr": "canonical-master"},
        ),
    ],
)
def test_setup_translates_legacy_keys_for_dict_api(
    setup_config: dict[str, str], expected: dict[str, str]
) -> None:
    store = FakeStore()
    config = MooncakeStoreConfig(setup_config)

    setup_mooncake_store(store, config)

    assert store.args == (expected,)
    assert config.setup_config == setup_config


def test_setup_fallback_reads_canonical_dict_keys() -> None:
    store = FakeStore(dict_api=False)
    config = MooncakeStoreConfig(
        {
            "local_hostname": "host",
            "metadata_server": "metadata",
            "global_segment_size": "1",
            "local_buffer_size": "2",
            "protocol": "rdma",
            "device_name": "legacy-device",
            "rdma_devices": "device",
            "master_server_address": "legacy-master",
            "master_server_addr": "master",
        }
    )

    setup_mooncake_store(store, config)

    assert store.args == ("host", "metadata", 1, 2, "rdma", "device", "master")


def test_config_repr_redacts_canonical_master_address() -> None:
    assert "secret-master" not in repr(
        MooncakeStoreConfig({"master_server_addr": "secret-master"})
    )


@pytest.mark.parametrize("dict_api", [True, False])
def test_setup_rejects_nonzero_result(dict_api: bool) -> None:
    with pytest.raises(RuntimeError, match="error code -1"):
        setup_mooncake_store(
            FakeStore(result=-1, dict_api=dict_api), MooncakeStoreConfig({})
        )
