# SPDX-License-Identifier: Apache-2.0
"""Parsing and validation of an L1's ``shared`` section."""

# Standard
from collections.abc import Callable
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

    spec = _shared(tmp_path, l1_type, client_id="host-a:6555", rpc_timeout_seconds=2)
    shared = parse_args(_args([spec])).l1_manager_configs[0].shared
    assert shared is not None
    assert (shared.client_id, shared.rpc_timeout_seconds) == ("host-a:6555", 2.0)


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


def _without_path(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k != "path"}


@pytest.mark.parametrize(
    ("specs", "message"),
    [
        (lambda p: [{**_shared(p), "eviction": {"eviction_policy": "LRU"}}], "noop"),
        (lambda p: [_without_path(_shared(p, "DRAM", "pool"))], "needs path"),
        (
            lambda p: [{"type": "DRAM", "tag": "pool", "size_gb": 1, "path": str(p)}],
            "only used with a shared section",
        ),
        (lambda p: [_shared(p, "GDS", "gds")], "GDS L1 cannot be shared"),
        (
            lambda p: [_shared(p, "DEVDAX", "cxl"), _shared(p, "DRAM", "pool")],
            "Only one shared",
        ),
    ],
    ids=["evicting-policy", "no-path", "path-without-shared", "gds", "two-shared"],
)
def test_invalid_shared_l1_configurations_are_rejected(
    tmp_path: Path, specs: Callable[[Path], list[dict]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_args(_args(specs(tmp_path)))


def test_l2_adapters_cannot_bind_to_the_shared_l1(tmp_path: Path) -> None:
    specs = [
        {"type": "DRAM", "tag": "_default", "size_gb": 1},
        _shared(tmp_path),
    ]
    adapter = '{"type":"fs","base_path":"/unused","affinity_tag":"%s"}'
    parse_args(_args(specs, "--l2-adapter", adapter % "_default"))
    with pytest.raises(ValueError, match="cannot bind to the shared"):
        parse_args(_args(specs, "--l2-adapter", adapter % "cxl"))
