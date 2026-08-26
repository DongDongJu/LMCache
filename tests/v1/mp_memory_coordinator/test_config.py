# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP Memory Coordinator configuration loader."""

# Standard
from pathlib import Path

# Third Party
import pytest
import yaml

# First Party
from lmcache.v1.mp_memory_coordinator.config import (
    LeaderElectionMode,
    MPMemoryCoordinatorConfig,
    config_from_mapping,
    load_config,
)

_REQUIRED_DEFAULTS = {
    "mp_coordinator_url": "http://lmcache-mp-coordinator:8000",
    "memory_allocation_url": "http://memory-allocation-service:8080",
    "poll_interval_seconds": 10,
    "stable_samples": 3,
    "high_ratio": 0.75,
    "low_ratio": 0.40,
    "minimum_ratio_gap": 0.25,
    "projected_donor_max_ratio": 0.70,
    "cooldown_seconds": 300,
    "adapter_index": 0,
    "min_devices_per_instance": 1,
    "allowed_device_path_prefix": "/dev/dax-cxl/",
    "drain_timeout_seconds": 300,
    "state_directory": "/var/lib/lmcache-memory-coordinator",
    "actuation_enabled": False,
}


def test_defaults_match_the_required_configuration() -> None:
    config = MPMemoryCoordinatorConfig()
    for key, value in _REQUIRED_DEFAULTS.items():
        assert getattr(config, key) == value, key
    assert config.leader_election is LeaderElectionMode.NONE


def test_mutation_is_disabled_by_default() -> None:
    assert MPMemoryCoordinatorConfig().actuation_enabled is False


def test_load_config_reads_yaml_and_keeps_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "mp_coordinator_url": "http://coord:9300",
                "memory_allocation_url": "http://alloc:8080",
                "poll_interval_seconds": 1,
                "state_directory": str(tmp_path / "state"),
                "leader_election": "kubernetes",
            }
        )
    )
    config = load_config(path)
    assert config.mp_coordinator_url == "http://coord:9300"
    assert config.poll_interval_seconds == 1.0
    assert config.stable_samples == 3
    assert config.leader_election is LeaderElectionMode.KUBERNETES


def test_load_config_empty_file_is_all_defaults(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("")
    assert load_config(path) == MPMemoryCoordinatorConfig()


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown configuration keys"):
        config_from_mapping({"poll_interval": 5})


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("actuation_enabled", "yes", "boolean"),
        ("stable_samples", True, "integer"),
        ("stable_samples", "3", "integer"),
        ("poll_interval_seconds", "10", "number"),
        ("poll_interval_seconds", True, "number"),
        ("state_directory", 5, "string"),
        ("leader_election", "raft", "one of"),
        ("leader_election", 1, "string"),
    ],
)
def test_strict_types(key: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config_from_mapping({key: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"low_ratio": 0.8, "high_ratio": 0.7},
        {"low_ratio": -0.1},
        {"high_ratio": 1.5},
        {"low_ratio": 0.5, "high_ratio": 0.5},
        {"poll_interval_seconds": 0},
        {"poll_interval_seconds": -1},
        {"cooldown_seconds": 0},
        {"drain_timeout_seconds": 0},
        {"stable_samples": 0},
        {"min_devices_per_instance": 0},
        {"adapter_index": 1},
        {"state_directory": "relative/dir"},
        {"mp_coordinator_url": "coord:9300"},
        {"memory_allocation_url": ""},
        {"allowed_device_path_prefix": "dev/dax"},
        {"projected_donor_max_ratio": 0.0},
        {"minimum_ratio_gap": 2.0},
        {"lease_renew_interval_seconds": 20.0, "lease_duration_seconds": 15.0},
        {"http_port": 0},
        {"get_retry_attempts": 0},
    ],
)
def test_validation_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        MPMemoryCoordinatorConfig(**overrides)  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    config = MPMemoryCoordinatorConfig()
    with pytest.raises(AttributeError):
        config.actuation_enabled = True  # type: ignore[misc]
