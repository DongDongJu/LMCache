# SPDX-License-Identifier: Apache-2.0
"""Wire-model compatibility: presence-watcher fields are optional on the wire."""

# Standard
from pathlib import Path
import json

# First Party
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_PHYSICAL_DEVDAX,
    DaxHotplugStatus,
    DaxReconfigureStatus,
    DaxWatcherStatus,
    JournalDocument,
    MoveCounters,
)

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "e2e"
    / "mp_memory_coordinator"
    / "fixtures"
    / "golden"
)


def _hotplug(name: str) -> dict[str, object]:
    body = json.loads((GOLDEN / name).read_text())
    return dict(body["adapters"][0]["status"])


def _without_watcher_fields(status: dict[str, object]) -> dict[str, object]:
    """Reproduce the status an MP server predating the watcher reports."""
    legacy = dict(status)
    legacy.pop("watcher", None)
    devices = legacy["devices"]
    assert isinstance(devices, list)
    legacy["devices"] = [
        {k: v for k, v in device.items() if k != "physical"} for device in devices
    ]
    return legacy


def test_status_without_watcher_or_physical_still_parses() -> None:
    legacy = _without_watcher_fields(_hotplug("mp_reconfigure_dax_status.json"))
    assert "watcher" not in legacy
    status = DaxHotplugStatus.model_validate(legacy)
    assert status.watcher == DaxWatcherStatus(enabled=False)
    assert status.watcher.present_devices == []
    assert all(device.physical is None for device in status.devices)


def test_status_with_watcher_and_physical_parses() -> None:
    body = json.loads((GOLDEN / "mp_reconfigure_dax_status.json").read_text())
    status = DaxReconfigureStatus.model_validate(body).adapters[0].status
    assert status.watcher.enabled is True
    assert status.watcher.directory.startswith("/dev/dax-cxl/")
    assert status.watcher.interval_seconds > 0
    present = {p.device_path: p for p in status.watcher.present_devices}
    assert set(present) >= {d.device_path for d in status.devices}
    for device in status.devices:
        physical = device.physical
        assert physical is not None
        assert physical.device_path == device.device_path
        assert physical.mode == DAX_PHYSICAL_DEVDAX
        assert physical.present is True
        assert physical.driver == "device_dax"
        assert physical.size_bytes >= device.max_dax_size_bytes


def test_disabled_watcher_block_parses_with_defaults() -> None:
    status = _hotplug("mp_reconfigure_dax_status.json")
    status["watcher"] = {"enabled": False}
    parsed = DaxHotplugStatus.model_validate(status)
    assert parsed.watcher.enabled is False
    assert parsed.watcher.directory == ""
    assert parsed.watcher.present_devices == []


def test_attach_count_is_not_part_of_the_persisted_journal_counters() -> None:
    # The attach success count lives in memory: it is never a persisted
    # counter. The GROW counters are persisted but defaulted, so a journal
    # written before they existed still loads.
    persisted = JournalDocument().model_dump(mode="json")["counters"]
    assert "attached" not in persisted
    assert set(persisted) == {
        "proposed",
        "succeeded",
        "rolled_back",
        "blocked",
        "not_served",
        "grown",
    }
    assert JournalDocument.model_validate(
        {"counters": {"proposed": 2, "succeeded": 1, "rolled_back": 0, "blocked": 0}}
    ).counters == MoveCounters(proposed=2, succeeded=1, rolled_back=0, blocked=0)
