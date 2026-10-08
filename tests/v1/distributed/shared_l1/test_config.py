# SPDX-License-Identifier: Apache-2.0
"""Parsing and validation of an L1's ``shared`` section."""

# Standard
from pathlib import Path
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    SharedL1Config,
    parse_args,
)


def _args(specs: list[dict[str, object]], *extra: str) -> list[str]:
    return [
        "--eviction-policy",
        "LRU",
        *[item for spec in specs for item in ("--l1-manager", json.dumps(spec))],
        *extra,
    ]


def _shared(
    tmp_path: Path, l1_type: str = "DEVDAX", tag: str = "cxl", **shared: object
) -> dict:
    return {
        "type": l1_type,
        "tag": tag,
        "size_gb": 1,
        "path": str(tmp_path / tag),
        "shared": {"orchestrator": "10.0.0.5:7700", "region_id": "pool-a", **shared},
    }


@pytest.mark.parametrize(
    ("l1_type", "config_class"),
    [("DEVDAX", DevDaxL1ManagerConfig), ("DRAM", DRAML1ManagerConfig)],
)
def test_shared_section_parses_and_defaults_to_noop_eviction(
    tmp_path: Path, l1_type: str, config_class: type
) -> None:
    config = parse_args(_args([_shared(tmp_path, l1_type)]))
    l1 = config.l1_manager_configs[0]
    assert isinstance(l1, config_class)
    assert l1.shared == SharedL1Config("10.0.0.5:7700", "pool-a", str(tmp_path / "cxl"))
    # The global LRU default does not apply: a shared L1 never evicts.
    assert l1.eviction is not None and l1.eviction.eviction_policy == "noop"


def test_shared_section_accepts_client_id_and_timeout(tmp_path: Path) -> None:
    config = parse_args(
        _args([_shared(tmp_path, client_id="host-a:6555", rpc_timeout_seconds=2)])
    )
    l1 = config.l1_manager_configs[0]
    assert l1.shared is not None
    assert l1.shared.client_id == "host-a:6555"
    assert l1.shared.rpc_timeout_seconds == 2.0


def test_private_l1s_have_no_shared_section(tmp_path: Path) -> None:
    specs = [
        {"type": "DEVDAX", "tag": "dax", "size_gb": 1, "path": str(tmp_path)},
        {"type": "DRAM", "tag": "_default", "size_gb": 1},
    ]
    config = parse_args(_args(specs))
    assert all(l1.shared is None for l1 in config.l1_manager_configs)


@pytest.mark.parametrize(
    ("shared", "message"),
    [
        ({"region_id": "pool-a"}, "shared.orchestrator"),
        ({"orchestrator": "h:1"}, "shared.region_id"),
        ({"orchestrator": "h:1", "region_id": " "}, "shared.region_id"),
        ({"orchestrator": "h:1", "region_id": "r", "extra": 1}, "Unknown shared"),
        ({"orchestrator": "h:1", "region_id": "r", "client_id": ""}, "client_id"),
        (
            {"orchestrator": "h:1", "region_id": "r", "rpc_timeout_seconds": 0},
            "rpc_timeout_seconds",
        ),
        ("h:1", "JSON object"),
    ],
)
def test_malformed_shared_section_is_rejected(
    tmp_path: Path, shared: object, message: str
) -> None:
    spec = {
        "type": "DEVDAX",
        "tag": "cxl",
        "size_gb": 1,
        "path": str(tmp_path),
        "shared": shared,
    }
    with pytest.raises(ValueError, match=message):
        parse_args(_args([spec]))


def test_shared_l1_rejects_an_evicting_policy(tmp_path: Path) -> None:
    spec = _shared(tmp_path)
    spec["eviction"] = {"eviction_policy": "LRU"}
    with pytest.raises(ValueError, match="noop"):
        parse_args(_args([spec]))


def test_shared_dram_l1_needs_a_path(tmp_path: Path) -> None:
    spec = _shared(tmp_path, "DRAM", tag="pool")
    del spec["path"]
    with pytest.raises(ValueError, match="needs path"):
        parse_args(_args([spec]))


def test_path_on_a_private_dram_l1_is_rejected(tmp_path: Path) -> None:
    spec = {"type": "DRAM", "tag": "pool", "size_gb": 1, "path": str(tmp_path)}
    with pytest.raises(ValueError, match="only used with a shared section"):
        parse_args(_args([spec]))


def test_gds_l1_cannot_be_shared(tmp_path: Path) -> None:
    spec = _shared(tmp_path, "GDS", tag="gds")
    with pytest.raises(ValueError, match="GDS L1 cannot be shared"):
        parse_args(_args([spec]))


def test_at_most_one_shared_l1_per_process(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only one shared"):
        parse_args(
            _args(
                [
                    _shared(tmp_path, "DEVDAX", "cxl"),
                    _shared(tmp_path, "DRAM", "pool"),
                ]
            )
        )


def test_l2_adapters_cannot_bind_to_the_shared_l1(tmp_path: Path) -> None:
    specs = [
        {"type": "DRAM", "tag": "_default", "size_gb": 1},
        _shared(tmp_path),
    ]
    adapter = '{"type":"fs","base_path":"/unused","affinity_tag":"%s"}'
    parse_args(_args(specs, "--l2-adapter", adapter % "_default"))
    with pytest.raises(ValueError, match="cannot bind to the shared"):
        parse_args(_args(specs, "--l2-adapter", adapter % "cxl"))
