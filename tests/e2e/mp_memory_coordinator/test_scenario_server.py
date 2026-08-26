# SPDX-License-Identifier: Apache-2.0
"""Tests for the scenario server (fake MP Coordinator + fake MP servers).

Everything runs in-process over ``build_apps`` with ``TestClient`` except the
subprocess smoke test and the import-boundary check.
"""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

# Third Party
from fastapi.testclient import TestClient

# ``TestClient`` responses come from whichever httpx flavour Starlette picked
# (``httpx2`` on current Starlette, ``httpx`` on older releases).
from starlette.testclient import httpx as testclient_httpx
import httpx
import pytest

# First Party
from tests.e2e.mp_memory_coordinator.scenario_server.app import build_apps
from tests.e2e.mp_memory_coordinator.scenario_server.state import (
    DONOR_ID,
    GIB,
    RECEIVER_ID,
    InstanceEndpoint,
    ScenarioState,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
GOLDEN_DIR = HERE / "fixtures" / "golden"
FIXTURE_PATH = HERE / "fixtures" / "two_server_local_dax.yaml"
MIB = 1 << 20
DONOR_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
DONOR_RUNTIME = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
RECEIVER_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.0"
RECEIVER_RUNTIME = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"
SIZE_ERROR = "size must be a positive integer byte count or a string like '100GiB'"
ENDPOINTS = {
    DONOR_ID: InstanceEndpoint(ip="127.0.0.1", http_port=8081, alt_http_port=8181),
    RECEIVER_ID: InstanceEndpoint(ip="127.0.0.1", http_port=8082, alt_http_port=8182),
}
# Verbatim copy of PLAN.md Phase 1A, used only when the shared fixture is absent.
FIXTURE_YAML = """\
schema_version: 1
nodes:
  "192.0.2.40":
    name: worker-196
    devices:
      - path: /dev/dax-cxl/lmcache-e2e--mp-196/dax0.0
        size_gib: 64
        role: bootstrap
        state: assigned
      - path: /dev/dax-cxl/lmcache-e2e--mp-196/dax0.1
        size_gib: 64
        role: runtime
        state: assigned
      - path: /dev/dax-cxl/lmcache-e2e--mp-196/dax0.2
        size_gib: 64
        role: runtime
        state: free
  "192.0.2.41":
    name: worker-197
    devices:
      - path: /dev/dax-cxl/lmcache-e2e--mp-197/dax0.0
        size_gib: 64
        role: bootstrap
        state: assigned
      - path: /dev/dax-cxl/lmcache-e2e--mp-197/dax0.1
        size_gib: 64
        role: runtime
        state: free
      - path: /dev/dax-cxl/lmcache-e2e--mp-197/dax0.2
        size_gib: 64
        role: runtime
        state: free
"""


def golden(name: str) -> object:
    """Load one golden fixture."""
    return json.loads((GOLDEN_DIR / name).read_text())


def assert_same_shape(expected: object, actual: object, path: str = "$") -> None:
    """Recursively compare key sets, ignoring values and list lengths.

    Every element of an actual list is compared against the first element of
    the expected list.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected object"
        assert set(expected) == set(actual), (
            f"{path}: keys differ: missing={set(expected) - set(actual)} "
            f"extra={set(actual) - set(expected)}"
        )
        for key, value in expected.items():
            assert_same_shape(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected list"
        if expected:
            for index, item in enumerate(actual):
                assert_same_shape(expected[0], item, f"{path}[{index}]")


def wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll ``predicate`` until true or the deadline passes."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met before deadline")
        time.sleep(0.01)


def free_port() -> int:
    """Return a currently free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Harness:
    """In-process scenario server with one client per listener."""

    state: ScenarioState
    clock: FakeClock
    coordinator: TestClient
    donor: TestClient
    receiver: TestClient
    admin: TestClient

    def devices(self, client: TestClient) -> list[dict[str, object]]:
        body = client.get("/reconfigure/dax/status").json()
        return list(body["adapters"][0]["status"]["devices"])

    def capacity(self, client: TestClient) -> int:
        body = client.get("/reconfigure/dax/status").json()
        return int(body["adapters"][0]["status"]["total_capacity_bytes"])

    def usage(self, instance_id: str) -> dict[str, object]:
        body = self.coordinator.get(f"/instances/{instance_id}/usage").json()
        return dict(body["modules"][1])

    def instances(self) -> dict[str, dict[str, object]]:
        body = self.coordinator.get("/instances").json()
        return {entry["instance_id"]: entry for entry in body["instances"]}

    def remove(
        self, client: TestClient, path: str, mode: str
    ) -> testclient_httpx.Response:
        return client.post(
            "/reconfigure/dax/remove",
            json={
                "adapter_index": 0,
                "device_path": path,
                "mode": mode,
                "force": False,
            },
        )

    def add(
        self, client: TestClient, path: str, size: object
    ) -> testclient_httpx.Response:
        return client.post(
            "/reconfigure/dax/add",
            json={"adapter_index": 0, "device_path": path, "size": size},
        )


@pytest.fixture
def fixture_path(tmp_path: Path) -> Path:
    if FIXTURE_PATH.exists():
        return FIXTURE_PATH
    path = tmp_path / "two_server_local_dax.yaml"
    path.write_text(FIXTURE_YAML)
    return path


@pytest.fixture
def harness(fixture_path: Path) -> Iterator[Harness]:
    clock = FakeClock()
    state = ScenarioState(fixture_path, ENDPOINTS, clock=clock)
    apps = build_apps(state)
    yield Harness(
        state=state,
        clock=clock,
        coordinator=TestClient(apps.coordinator),
        donor=TestClient(apps.donor),
        receiver=TestClient(apps.receiver),
        admin=TestClient(apps.admin),
    )
    state.barriers.clear()


# ----------------------------------------------------------------- golden


def test_instances_matches_golden_shape(harness: Harness) -> None:
    response = harness.coordinator.get("/instances")
    assert response.status_code == 200
    assert_same_shape(golden("coordinator_instances.json"), response.json())
    ids = [entry["instance_id"] for entry in response.json()["instances"]]
    assert ids == [DONOR_ID, RECEIVER_ID]


def test_usage_matches_golden_shape(harness: Harness) -> None:
    fleet = harness.coordinator.get("/instances/usage")
    assert fleet.status_code == 200
    assert_same_shape(golden("coordinator_instances_usage.json"), fleet.json())
    single = harness.coordinator.get(f"/instances/{DONOR_ID}/usage")
    assert single.status_code == 200
    fleet_golden = golden("coordinator_instances_usage.json")
    assert isinstance(fleet_golden, dict)
    fleet_instances = fleet_golden["instances"]
    assert isinstance(fleet_instances, list)
    assert_same_shape(fleet_instances[0], single.json())
    assert harness.coordinator.get("/instances/nope/usage").status_code == 404


def test_status_matches_golden_shape(harness: Harness) -> None:
    for client in (harness.donor, harness.receiver):
        response = client.get("/status")
        assert response.status_code == 200
        assert_same_shape(golden("mp_status.json"), response.json())
    assert harness.remove(harness.donor, DONOR_RUNTIME, "evict").status_code == 200
    assert_same_shape(golden("mp_status.json"), harness.donor.get("/status").json())


def test_dax_status_and_responses_match_golden_shape(harness: Harness) -> None:
    donor = harness.donor
    assert_same_shape(
        golden("mp_reconfigure_dax_status.json"),
        donor.get("/reconfigure/dax/status").json(),
    )
    drain = harness.remove(donor, DONOR_RUNTIME, "drain")
    assert drain.status_code == 200
    assert_same_shape(
        golden("mp_reconfigure_dax_remove_drain_response.json"), drain.json()
    )
    assert_same_shape(
        golden("mp_reconfigure_dax_status.json"),
        donor.get("/reconfigure/dax/status").json(),
    )
    evict = harness.remove(donor, DONOR_RUNTIME, "evict")
    assert evict.status_code == 200
    assert_same_shape(
        golden("mp_reconfigure_dax_remove_evict_response.json"), evict.json()
    )
    assert_same_shape(
        golden("mp_reconfigure_dax_status_after_evict.json"),
        donor.get("/reconfigure/dax/status").json(),
    )
    missing = harness.remove(donor, DONOR_RUNTIME, "evict")
    assert missing.status_code == 404
    assert missing.json() == golden("mp_reconfigure_dax_remove_404_response.json")
    add = harness.add(donor, DONOR_RUNTIME, "64GiB")
    assert add.status_code == 200
    assert_same_shape(golden("mp_reconfigure_dax_add_response.json"), add.json())
    assert_same_shape(
        golden("mp_reconfigure_dax_status.json"),
        donor.get("/reconfigure/dax/status").json(),
    )


# --------------------------------------------------------------- topology


def test_initial_topology(harness: Harness) -> None:
    instances = harness.instances()
    donor, receiver = instances[DONOR_ID], instances[RECEIVER_ID]
    assert (donor["ip"], donor["http_port"]) == ("127.0.0.1", 8081)
    assert (receiver["ip"], receiver["http_port"]) == ("127.0.0.1", 8082)
    assert donor["metadata"] == {"worker_ip": "192.0.2.40"}
    assert receiver["metadata"] == {"worker_ip": "192.0.2.41"}
    assert donor["p2p_advertised_url"] == "" and donor["mq_port"] == 0
    assert isinstance(donor["registration_time"], float)

    donor_devices = harness.devices(harness.donor)
    assert [(d["index"], d["device_path"], d["state"]) for d in donor_devices] == [
        (0, DONOR_BOOT, "active"),
        (1, DONOR_RUNTIME, "active"),
    ]
    assert [d["device_id"] for d in donor_devices] == [0, 1]
    assert all(
        d["slot_bytes"] == MIB and d["max_slots"] == 65536 for d in donor_devices
    )
    assert [d["live_slot_count"] for d in donor_devices] == [4096, 4096]
    receiver_devices = harness.devices(harness.receiver)
    assert [(d["index"], d["device_path"], d["state"]) for d in receiver_devices] == [
        (0, RECEIVER_BOOT, "active")
    ]
    assert receiver_devices[0]["live_slot_count"] == 57344
    assert harness.capacity(harness.donor) == 128 * GIB
    assert harness.capacity(harness.receiver) == 64 * GIB

    fleet = harness.coordinator.get("/instances/usage").json()
    assert fleet["shared_modules"] == []
    donor_usage, receiver_usage = fleet["instances"]
    assert donor_usage["registered"] and donor_usage["declared_capacity"]
    assert donor_usage["modules"][0] == {
        "tier": "l1",
        "backend": "dram",
        "shared": False,
        "used_bytes": 0,
        "capacity_bytes": 4 * GIB,
        "usage_ratio": 0.0,
    }
    assert donor_usage["modules"][1] == {
        "tier": "l2",
        "backend": "dax",
        "shared": False,
        "used_bytes": 8 * GIB,
        "capacity_bytes": 128 * GIB,
        "usage_ratio": 0.0625,
    }
    assert receiver_usage["modules"][1]["used_bytes"] == 56 * GIB
    assert receiver_usage["modules"][1]["capacity_bytes"] == 64 * GIB
    assert receiver_usage["modules"][1]["usage_ratio"] == 0.875

    status = harness.donor.get("/status").json()
    adapter = status["storage_manager"]["l2_adapters"][0]
    assert adapter["device_path"] == DONOR_BOOT
    assert adapter["max_dax_size_bytes"] == 128 * GIB
    assert adapter["num_devices"] == 2 and adapter["live_slot_count"] == 8192
    assert status["is_healthy"] and adapter["hotplug_enabled"]
    assert harness.coordinator.get("/healthz").json() == {"status": "ok"}
    assert harness.donor.get("/healthcheck").json() == {"status": "healthy"}
    assert harness.donor.get("/").status_code == 200


# -------------------------------------------------------------- lifecycle


def test_drain_marks_device_draining_and_add_does_not_reactivate(
    harness: Harness,
) -> None:
    donor = harness.donor
    drain = harness.remove(donor, DONOR_RUNTIME, "drain")
    assert drain.json() == {
        "status": "ok",
        "operation": "drain",
        "adapter_index": 0,
        "device_path": DONOR_RUNTIME,
        "index": 1,
        "state": "draining",
    }
    assert harness.devices(donor)[1]["state"] == "draining"
    assert harness.capacity(donor) == 128 * GIB
    assert donor.get("/status").json()["is_healthy"]
    assert harness.remove(donor, DONOR_RUNTIME, "drain").json() == drain.json()

    add = harness.add(donor, DONOR_RUNTIME, 64 * GIB)
    assert add.status_code == 200
    assert add.json()["device"]["state"] == "draining"
    assert add.json()["device"]["index"] == 1
    assert len(harness.devices(donor)) == 2
    assert harness.devices(donor)[1]["state"] == "draining"

    conflict = harness.add(donor, DONOR_RUNTIME, "32GiB")
    assert conflict.status_code == 409
    assert conflict.json() == {
        "error": "device_path already active with a different size"
    }


def test_evict_busy_returns_409_then_removes_tombstone(harness: Harness) -> None:
    donor = harness.donor
    assert (
        harness.admin.post(
            "/__test/devices",
            json={
                "instance_id": DONOR_ID,
                "device_path": DONOR_RUNTIME,
                "locked_key_count": 3,
            },
        ).status_code
        == 200
    )
    blocked = harness.remove(donor, DONOR_RUNTIME, "evict")
    assert blocked.status_code == 409
    assert blocked.json() == {
        "status": "blocked",
        "reason": "device has externally locked or borrowed slots",
        "locked_key_count": 3,
        "borrowed_slot_count": 0,
    }
    assert harness.devices(donor)[1]["state"] == "draining"

    harness.admin.post(
        "/__test/devices",
        json={
            "instance_id": DONOR_ID,
            "device_path": DONOR_RUNTIME,
            "locked_key_count": 0,
        },
    )
    faults = harness.admin.post(
        "/__test/faults", json={"mp": {DONOR_ID: {"evict_409_count": 2}}}
    )
    assert faults.status_code == 200
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 409
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 409
    assert harness.devices(donor)[1]["state"] == "draining"

    removed = harness.remove(donor, DONOR_RUNTIME, "evict")
    assert removed.status_code == 200
    assert removed.json() == {
        "status": "ok",
        "operation": "remove",
        "adapter_index": 0,
        "device_path": DONOR_RUNTIME,
        "index": 1,
        "moved_keys": 0,
        "moved_bytes": 0,
        "deleted_keys": 4096,
        "source_slots_freed": 4096,
        "state": "removed",
    }
    devices = harness.devices(donor)
    assert len(devices) == 2
    tombstone = devices[1]
    assert tombstone["state"] == "removed"
    assert tombstone["is_healthy"] is False and tombstone["closing"] is True
    assert tombstone["live_slot_count"] == 0
    assert tombstone["max_dax_size_bytes"] == 64 * GIB
    assert harness.capacity(donor) == 64 * GIB
    status = donor.get("/status").json()
    assert status["is_healthy"] is True
    assert status["storage_manager"]["l2_adapters"][0]["is_healthy"] is True
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 64 * GIB
    assert harness.usage(DONOR_ID)["usage_ratio"] == 0.125
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 404
    assert harness.remove(donor, DONOR_RUNTIME, "drain").status_code == 404


def test_migrate_behaves_like_evict(harness: Harness) -> None:
    response = harness.remove(harness.donor, DONOR_RUNTIME, "migrate")
    assert response.status_code == 200
    assert response.json()["operation"] == "remove"
    assert harness.devices(harness.donor)[1]["state"] == "removed"


def test_force_skips_busy_check(harness: Harness) -> None:
    harness.admin.post(
        "/__test/devices",
        json={
            "instance_id": DONOR_ID,
            "device_path": DONOR_RUNTIME,
            "borrowed_slot_count": 1,
        },
    )
    forced = harness.donor.post(
        "/reconfigure/dax/remove",
        json={"device_path": DONOR_RUNTIME, "mode": "evict", "force": True},
    )
    assert forced.status_code == 200
    assert forced.json()["state"] == "removed"


def test_add_after_evict_creates_new_entry_and_restores_capacity(
    harness: Harness,
) -> None:
    donor = harness.donor
    assert harness.capacity(donor) == 128 * GIB
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 200
    assert harness.capacity(donor) == 64 * GIB
    add = harness.add(donor, DONOR_RUNTIME, "64GiB")
    assert add.status_code == 200
    device = add.json()["device"]
    assert (device["index"], device["device_id"], device["state"]) == (2, 2, "active")
    assert device["live_slot_count"] == 0 and device["max_slots"] == 65536
    devices = harness.devices(donor)
    assert [d["state"] for d in devices] == ["active", "removed", "active"]
    assert harness.capacity(donor) == 128 * GIB
    assert harness.usage(DONOR_ID)["usage_ratio"] == 0.0625

    receiver = harness.receiver
    added = harness.add(receiver, RECEIVER_RUNTIME, "64GiB")
    assert added.status_code == 200
    assert (added.json()["device"]["index"], added.json()["device"]["device_id"]) == (
        1,
        1,
    )
    assert harness.capacity(receiver) == 128 * GIB
    assert harness.usage(RECEIVER_ID)["usage_ratio"] == 0.4375


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (64 * GIB, 64 * GIB),
        ("64GiB", 64 * GIB),
        ("64 GiB", 64 * GIB),
        ("64gb", 64 * GIB),
        ("65536MiB", 64 * GIB),
        ("1T", 1024**4),
        ("0.5GiB", 512 * MIB),
        ("2097152", 2 * MIB),
    ],
)
def test_add_size_parsing(harness: Harness, size: object, expected: int) -> None:
    response = harness.add(harness.receiver, RECEIVER_RUNTIME, size)
    assert response.status_code == 200, response.text
    assert response.json()["device"]["max_dax_size_bytes"] == expected


@pytest.mark.parametrize("size", ["abc", "-1", 0, -5, "64XB", "", "1.GiB", True])
def test_add_rejects_invalid_sizes(harness: Harness, size: object) -> None:
    response = harness.add(harness.receiver, RECEIVER_RUNTIME, size)
    assert response.status_code == 400
    assert len(harness.devices(harness.receiver)) == 1


def test_request_validation(harness: Harness) -> None:
    donor = harness.donor
    extra = donor.post(
        "/reconfigure/dax/add",
        json={"device_path": DONOR_RUNTIME, "size": "64GiB", "bogus": 1},
    )
    assert extra.status_code == 422 and "detail" in extra.json()
    assert donor.post("/reconfigure/dax/add", json={"size": "64GiB"}).status_code == 422
    assert (
        donor.post(
            "/reconfigure/dax/remove",
            json={"device_path": DONOR_RUNTIME, "mode": "purge"},
        ).status_code
        == 422
    )
    assert (
        donor.post(
            "/reconfigure/dax/remove",
            json={"device_path": DONOR_RUNTIME, "mode": "drain", "extra": True},
        ).status_code
        == 422
    )
    assert donor.post("/reconfigure/dax/remove", json=[1]).status_code == 422
    empty = donor.post("/reconfigure/dax/remove", json={"device_path": "  "})
    assert empty.status_code == 400
    wrong_adapter = donor.post(
        "/reconfigure/dax/remove",
        json={"adapter_index": 1, "device_path": DONOR_RUNTIME, "mode": "drain"},
    )
    assert wrong_adapter.status_code == 404
    assert wrong_adapter.json() == {"error": "dax adapter not found"}
    assert (
        donor.post(
            "/reconfigure/dax/add",
            json={"adapter_index": 1, "device_path": DONOR_RUNTIME, "size": "64GiB"},
        ).status_code
        == 404
    )
    assert harness.devices(donor)[1]["state"] == "active"


# ---------------------------------------------------------------- faults


def test_delayed_capacity_publication(harness: Harness) -> None:
    donor = harness.donor
    assert (
        harness.admin.post(
            "/__test/faults", json={"coordinator": {"delayed_capacity_seconds": 5.0}}
        ).status_code
        == 200
    )
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 200
    assert harness.capacity(donor) == 64 * GIB
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 128 * GIB
    harness.clock.advance(4.9)
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 128 * GIB
    harness.clock.advance(0.2)
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 64 * GIB
    assert harness.usage(DONOR_ID)["usage_ratio"] == 0.125

    assert harness.add(donor, DONOR_RUNTIME, "64GiB").status_code == 200
    assert harness.capacity(donor) == 128 * GIB
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 64 * GIB
    harness.clock.advance(5.1)
    assert harness.usage(DONOR_ID)["capacity_bytes"] == 128 * GIB
    state = harness.admin.get("/__test/state").json()
    assert state["instances"][0]["published_capacity_bytes"] == 128 * GIB


def test_coordinator_faults(harness: Harness) -> None:
    admin, coordinator = harness.admin, harness.coordinator
    assert (
        admin.post(
            "/__test/faults", json={"coordinator": {"unavailable": True}}
        ).json()["coordinator"]["unavailable"]
        is True
    )
    for path in ("/instances", "/instances/usage", f"/instances/{DONOR_ID}/usage"):
        response = coordinator.get(path)
        assert response.status_code == 503 and response.json() == {
            "error": "unavailable"
        }
    assert coordinator.get("/healthz").status_code == 503
    assert admin.delete("/__test/faults").status_code == 200
    assert coordinator.get("/instances").status_code == 200

    admin.post(
        "/__test/faults",
        json={
            "coordinator": {
                "undeclared_capacity": [DONOR_ID],
                "null_ratio": [RECEIVER_ID],
                "shared_dax": [DONOR_ID],
            }
        },
    )
    fleet = coordinator.get("/instances/usage").json()["instances"]
    donor_usage, receiver_usage = fleet
    assert donor_usage["declared_capacity"] is False
    assert [m["capacity_bytes"] for m in donor_usage["modules"]] == [0, 0]
    assert [m["usage_ratio"] for m in donor_usage["modules"]] == [None, None]
    assert donor_usage["modules"][1]["used_bytes"] == 8 * GIB
    assert donor_usage["modules"][1]["shared"] is True
    assert receiver_usage["declared_capacity"] is True
    assert receiver_usage["modules"][1]["capacity_bytes"] == 64 * GIB
    assert receiver_usage["modules"][1]["usage_ratio"] is None

    admin.post(
        "/__test/faults",
        json={
            "coordinator": {
                "unregistered": [RECEIVER_ID],
                "worker_ip_override": {DONOR_ID: None},
            }
        },
    )
    instances = coordinator.get("/instances").json()["instances"]
    assert [entry["instance_id"] for entry in instances] == [DONOR_ID]
    assert instances[0]["metadata"] == {}
    fleet = coordinator.get("/instances/usage").json()["instances"]
    assert [(e["instance_id"], e["registered"]) for e in fleet] == [
        (DONOR_ID, True),
        (RECEIVER_ID, False),
    ]
    # Merge semantics: earlier keys survive, only the given key changes.
    admin.post(
        "/__test/faults",
        json={"coordinator": {"worker_ip_override": {RECEIVER_ID: "192.0.2.40"}}},
    )
    active = admin.get("/__test/state").json()["faults"]["coordinator"]
    assert active["unregistered"] == [RECEIVER_ID]
    assert active["shared_dax"] == [DONOR_ID]
    admin.post("/__test/faults", json={"coordinator": {"unregistered": []}})
    instances = coordinator.get("/instances").json()["instances"]
    assert [entry["metadata"] for entry in instances] == [
        {"worker_ip": "192.0.2.40"},
        {"worker_ip": "192.0.2.40"},
    ]

    assert (
        admin.post("/__test/faults", json={"coordinator": {"nope": 1}}).status_code
        == 422
    )
    assert admin.post("/__test/faults", json={"bogus": {}}).status_code == 422
    assert admin.post("/__test/faults", json={"mp": {"mp-x": {}}}).status_code == 404
    assert (
        admin.post(
            "/__test/faults", json={"mp": {DONOR_ID: {"adapters": 3}}}
        ).status_code
        == 422
    )


def test_mp_faults(harness: Harness) -> None:
    admin, donor, receiver = harness.admin, harness.donor, harness.receiver

    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"status_unavailable": True}}})
    assert donor.get("/status").status_code == 503
    assert donor.get("/reconfigure/dax/status").status_code == 503
    assert donor.get("/healthcheck").status_code == 503
    assert receiver.get("/status").status_code == 200
    admin.delete("/__test/faults")

    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"adapters": 0}}})
    status = donor.get("/status").json()
    assert status["storage_manager"]["l2_adapters"] == []
    assert status["storage_manager"]["num_l2_adapters"] == 0
    assert donor.get("/reconfigure/dax/status").json() == {
        "enabled": False,
        "backend": "dax",
        "num_adapters": 0,
        "adapters": [],
    }
    assert harness.remove(donor, DONOR_RUNTIME, "drain").status_code == 404
    assert harness.add(donor, DONOR_RUNTIME, "64GiB").status_code == 404
    # Merge semantics across the two fault groups.
    admin.post("/__test/faults", json={"coordinator": {"unavailable": False}})
    assert donor.get("/reconfigure/dax/status").json()["num_adapters"] == 0

    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"adapters": 2}}})
    dax = donor.get("/reconfigure/dax/status").json()
    assert dax["num_adapters"] == 2 and len(dax["adapters"]) == 2
    assert [a["adapter_index"] for a in dax["adapters"]] == [0, 1]
    assert len(donor.get("/status").json()["storage_manager"]["l2_adapters"]) == 2
    assert (
        donor.post(
            "/reconfigure/dax/remove",
            json={"adapter_index": 1, "device_path": DONOR_RUNTIME, "mode": "drain"},
        ).status_code
        == 200
    )
    admin.delete("/__test/faults")

    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"unhealthy": True}}})
    status = donor.get("/status").json()
    assert status["is_healthy"] is False
    assert status["storage_manager"]["is_healthy"] is False
    assert status["storage_manager"]["l2_adapters"][0]["is_healthy"] is False
    assert all(d["is_healthy"] for d in harness.devices(donor))
    admin.post(
        "/__test/faults", json={"mp": {DONOR_ID: {"unhealthy": False, "closing": True}}}
    )
    status = donor.get("/status").json()
    assert status["storage_manager"]["l2_adapters"][0]["closing"] is True
    admin.delete("/__test/faults")

    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"hotplug_disabled": True}}})
    assert (
        donor.get("/reconfigure/dax/status").json()["adapters"][0]["status"][
            "hotplug_enabled"
        ]
        is False
    )
    assert (
        donor.get("/status").json()["storage_manager"]["l2_adapters"][0][
            "hotplug_enabled"
        ]
        is False
    )
    disabled = harness.remove(donor, DONOR_RUNTIME, "drain")
    assert disabled.status_code == 403
    assert disabled.json() == {"error": "DAX hotplug is disabled"}
    assert harness.add(donor, DONOR_RUNTIME, "64GiB").status_code == 403
    admin.delete("/__test/faults")

    admin.post("/__test/faults", json={"mp": {RECEIVER_ID: {"add_fail_count": 2}}})
    for _ in range(2):
        failed = harness.add(receiver, RECEIVER_RUNTIME, "64GiB")
        assert failed.status_code == 400
        assert failed.json() == {"error": "failed to map DAX device"}
    assert len(harness.devices(receiver)) == 1
    assert harness.add(receiver, RECEIVER_RUNTIME, "64GiB").status_code == 200
    admin.delete("/__test/faults")

    admin.post("/__test/faults", json={"mp": {RECEIVER_ID: {"add_always_fail": True}}})
    for _ in range(3):
        assert (
            harness.add(
                receiver, "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.2", "64GiB"
            ).status_code
            == 400
        )
    # Re-adding an already mapped path never touches the mapping, so it succeeds.
    assert harness.add(receiver, RECEIVER_RUNTIME, "64GiB").status_code == 200
    admin.delete("/__test/faults")

    before = harness.devices(donor)[1]["state"]
    admin.post(
        "/__test/faults", json={"mp": {DONOR_ID: {"remove_route_failure": True}}}
    )
    assert harness.remove(donor, DONOR_RUNTIME, "evict").status_code == 500
    assert harness.devices(donor)[1]["state"] == before


def test_identity_flip_every_second_read(harness: Harness) -> None:
    base = harness.instances()[DONOR_ID]
    harness.admin.post(
        "/__test/faults",
        json={
            "coordinator": {
                "identity_flip": {
                    "instance_id": DONOR_ID,
                    "field": "registration_time",
                    "every_n_reads": 2,
                }
            }
        },
    )
    base_time = base["registration_time"]
    assert isinstance(base_time, float)
    reads = [harness.instances() for _ in range(4)]
    times = [read[DONOR_ID]["registration_time"] for read in reads]
    assert times == [base_time, base_time + 1.0, base_time, base_time + 1.0]
    assert all(
        read[RECEIVER_ID]["registration_time"]
        == reads[0][RECEIVER_ID]["registration_time"]
        for read in reads
    )
    assert all(read[DONOR_ID]["http_port"] == 8081 for read in reads)

    harness.admin.post(
        "/__test/faults",
        json={
            "coordinator": {
                "identity_flip": {"instance_id": RECEIVER_ID, "field": "endpoint"}
            }
        },
    )
    ports = [harness.instances()[RECEIVER_ID]["http_port"] for _ in range(4)]
    assert ports == [8082, 8083, 8082, 8083]
    harness.admin.post("/__test/faults", json={"coordinator": {"identity_flip": None}})
    assert [harness.instances()[RECEIVER_ID]["http_port"] for _ in range(3)] == [
        8082
    ] * 3


def test_reregister_changes_identity(harness: Harness) -> None:
    admin = harness.admin
    base = harness.instances()[DONOR_ID]
    bumped = admin.post(
        f"/__test/instances/{DONOR_ID}/reregister", json={"bump": "registration_time"}
    )
    assert bumped.status_code == 200
    assert bumped.json()["registration_time"] > base["registration_time"]
    current = harness.instances()[DONOR_ID]
    assert current["registration_time"] == bumped.json()["registration_time"]
    assert current["http_port"] == 8081

    assert (
        admin.post(
            f"/__test/instances/{DONOR_ID}/reregister", json={"bump": "endpoint"}
        ).json()["http_port"]
        == 8181
    )
    assert harness.instances()[DONOR_ID]["http_port"] == 8181
    assert (
        harness.instances()[DONOR_ID]["registration_time"]
        == current["registration_time"]
    )
    assert (
        admin.post(
            f"/__test/instances/{DONOR_ID}/reregister", json={"bump": "endpoint"}
        ).json()["http_port"]
        == 8081
    )

    before = harness.instances()[RECEIVER_ID]
    both = admin.post(
        f"/__test/instances/{RECEIVER_ID}/reregister", json={"bump": "both"}
    )
    assert both.json()["http_port"] == 8182
    assert both.json()["registration_time"] > before["registration_time"]
    assert (
        admin.post(
            "/__test/instances/mp-x/reregister", json={"bump": "both"}
        ).status_code
        == 404
    )
    assert (
        admin.post(
            f"/__test/instances/{DONOR_ID}/reregister", json={"bump": "ip"}
        ).status_code
        == 422
    )


# ------------------------------------------------------------ isolation


def test_route_isolation(harness: Harness) -> None:
    for client in (harness.coordinator, harness.donor, harness.receiver):
        for path in (
            "/__test/health",
            "/__test/state",
            "/docs",
            "/redoc",
            "/openapi.json",
        ):
            assert client.get(path).status_code == 404, path
        assert client.post("/__test/reset").status_code == 404
        assert client.post("/__test/faults", json={}).status_code == 404
    assert harness.coordinator.get("/").status_code == 404
    assert harness.coordinator.get("/status").status_code == 404
    assert harness.coordinator.get("/reconfigure/dax/status").status_code == 404
    assert harness.donor.get("/instances").status_code == 404
    assert harness.donor.get("/instances/usage").status_code == 404
    for path in (
        "/",
        "/instances",
        "/instances/usage",
        "/status",
        "/healthcheck",
        "/reconfigure/dax/status",
        "/docs",
        "/openapi.json",
    ):
        assert harness.admin.get(path).status_code == 404, path
    assert harness.admin.post("/reconfigure/dax/remove", json={}).status_code == 404


# ----------------------------------------------------------------- audit


def test_audit_records_every_request_and_mutation(harness: Harness) -> None:
    admin = harness.admin
    assert admin.get("/__test/health").json() == {"status": "ok", "seq": 0}
    harness.coordinator.get("/instances")
    body = {
        "adapter_index": 0,
        "device_path": DONOR_RUNTIME,
        "mode": "drain",
        "force": False,
    }
    harness.donor.post("/reconfigure/dax/remove", json=body)
    harness.receiver.get("/nope")
    harness.donor.post(
        "/reconfigure/dax/add",
        content=b"not json",
        headers={"content-type": "application/json"},
    )

    records = admin.get("/__test/audit", params={"after_seq": 0}).json()["records"]
    seqs = [r["seq"] for r in records]
    assert seqs == list(range(1, len(records) + 1))
    assert admin.get("/__test/health").json()["seq"] == len(records)
    for record in records:
        assert set(record) == {
            "seq",
            "kind",
            "service",
            "method",
            "path",
            "body",
            "status_code",
            "response",
            "mutation",
            "timestamp",
        }

    assert records[0]["kind"] == "request"
    assert (records[0]["service"], records[0]["method"], records[0]["path"]) == (
        "coordinator",
        "GET",
        "/instances",
    )
    assert records[0]["body"] is None
    assert records[1]["kind"] == "response" and records[1]["status_code"] == 200
    assert [i["instance_id"] for i in records[1]["response"]["instances"]] == [
        DONOR_ID,
        RECEIVER_ID,
    ]

    assert records[2]["kind"] == "request" and records[2]["service"] == DONOR_ID
    assert (records[2]["method"], records[2]["path"]) == (
        "POST",
        "/reconfigure/dax/remove",
    )
    assert records[2]["body"] == body
    assert records[3]["kind"] == "mutation"
    assert records[3]["mutation"]["device_path"] == DONOR_RUNTIME
    assert (
        records[3]["mutation"]["from_state"],
        records[3]["mutation"]["to_state"],
    ) == ("active", "draining")
    assert records[4]["kind"] == "response" and records[4]["status_code"] == 200
    assert records[4]["response"]["operation"] == "drain"

    assert (records[5]["service"], records[5]["path"]) == (RECEIVER_ID, "/nope")
    assert records[6]["status_code"] == 404
    assert records[7]["body"] == "not json"
    assert records[8]["status_code"] == 422

    tail = admin.get("/__test/audit", params={"after_seq": records[4]["seq"]}).json()[
        "records"
    ]
    assert [r["seq"] for r in tail] == seqs[5:]
    # Admin traffic itself is not audited.
    assert admin.get("/__test/health").json()["seq"] == len(records)


def test_reset_restores_defaults(harness: Harness) -> None:
    admin = harness.admin
    harness.remove(harness.donor, DONOR_RUNTIME, "evict")
    admin.post("/__test/faults", json={"mp": {DONOR_ID: {"adapters": 0}}})
    admin.post("/__test/usage", json={"instance_id": DONOR_ID, "used_bytes": GIB})
    admin.post(
        "/__test/barriers",
        json={
            "instance_id": DONOR_ID,
            "operation": "add",
            "when": "before",
            "name": "b",
        },
    )
    admin.post(f"/__test/instances/{DONOR_ID}/reregister", json={"bump": "endpoint"})
    snapshot = admin.post("/__test/reset").json()
    assert snapshot["seq"] == 0 and snapshot["barriers"] == []
    assert snapshot["faults"]["mp"][DONOR_ID]["adapters"] == 1
    assert admin.get("/__test/audit").json()["records"] == []
    devices = harness.devices(harness.donor)
    assert [d["state"] for d in devices] == ["active", "active"]
    assert harness.usage(DONOR_ID) == {
        "tier": "l2",
        "backend": "dax",
        "shared": False,
        "used_bytes": 8 * GIB,
        "capacity_bytes": 128 * GIB,
        "usage_ratio": 0.0625,
    }
    assert harness.instances()[DONOR_ID]["http_port"] == 8081
    assert harness.add(harness.receiver, RECEIVER_RUNTIME, "64GiB").status_code == 200


def test_admin_usage_and_device_counters(harness: Harness) -> None:
    admin = harness.admin
    usage = admin.post(
        "/__test/usage", json={"instance_id": DONOR_ID, "used_bytes": 32 * GIB}
    )
    assert usage.status_code == 200
    assert harness.usage(DONOR_ID)["usage_ratio"] == 0.25
    assert (
        admin.post(
            "/__test/usage", json={"instance_id": "mp-x", "used_bytes": 1}
        ).status_code
        == 404
    )

    updated = admin.post(
        "/__test/devices",
        json={
            "instance_id": DONOR_ID,
            "device_path": DONOR_RUNTIME,
            "used_bytes": GIB,
            "inflight_store_tasks": 2,
            "active_read_count": 1,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["live_slot_count"] == 1024
    dax = harness.donor.get("/reconfigure/dax/status").json()["adapters"][0]["status"]
    assert dax["total_used_bytes"] == 5 * GIB
    device = dax["devices"][1]
    assert (device["inflight_store_tasks"], device["active_read_count"]) == (2, 1)
    assert device["locked_key_count"] == 0
    adapter = harness.donor.get("/status").json()["storage_manager"]["l2_adapters"][0]
    assert adapter["inflight_store_tasks"] == 2 and adapter["live_slot_count"] == 5120
    assert (
        admin.post(
            "/__test/devices",
            json={"instance_id": DONOR_ID, "device_path": "/dev/none"},
        ).status_code
        == 404
    )
    assert (
        admin.post(
            "/__test/devices",
            json={"instance_id": DONOR_ID, "device_path": DONOR_RUNTIME, "x": 1},
        ).status_code
        == 422
    )
    harness.remove(harness.donor, DONOR_RUNTIME, "evict")
    assert (
        admin.post(
            "/__test/devices",
            json={"instance_id": DONOR_ID, "device_path": DONOR_RUNTIME},
        ).status_code
        == 404
    )


# -------------------------------------------------------------- barriers


def test_barrier_blocks_mutation_until_release(harness: Harness) -> None:
    admin, donor = harness.admin, harness.donor
    armed = admin.post(
        "/__test/barriers",
        json={
            "instance_id": DONOR_ID,
            "operation": "evict",
            "when": "before",
            "name": "b1",
        },
    )
    assert armed.status_code == 200 and armed.json()["hit"] is False
    assert (
        admin.post(
            "/__test/barriers",
            json={
                "instance_id": DONOR_ID,
                "operation": "evict",
                "when": "before",
                "name": "b1",
            },
        ).status_code
        == 409
    )
    assert (
        admin.post(
            "/__test/barriers",
            json={
                "instance_id": "mp-x",
                "operation": "evict",
                "when": "before",
                "name": "b9",
            },
        ).status_code
        == 404
    )

    results: list[testclient_httpx.Response] = []
    worker = threading.Thread(
        target=lambda: results.append(harness.remove(donor, DONOR_RUNTIME, "evict"))
    )
    worker.start()
    wait_until(lambda: admin.get("/__test/state").json()["barriers"][0]["hit"])
    assert worker.is_alive()
    last = admin.get("/__test/audit").json()["records"][-1]
    assert (last["kind"], last["path"]) == ("request", "/reconfigure/dax/remove")
    assert harness.devices(donor)[1]["state"] == "active"
    assert admin.post("/__test/barriers/b1/release").json()["released"] is True
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert results[0].status_code == 200 and results[0].json()["state"] == "removed"
    # One-shot: a second evict passes straight through.
    assert harness.remove(donor, DONOR_BOOT, "drain").status_code == 200
    assert admin.post("/__test/barriers/nope/release").status_code == 404

    admin.post(
        "/__test/barriers",
        json={
            "instance_id": DONOR_ID,
            "operation": "add",
            "when": "after",
            "name": "b2",
        },
    )
    results.clear()
    worker = threading.Thread(
        target=lambda: results.append(harness.add(donor, DONOR_RUNTIME, "64GiB"))
    )
    worker.start()
    wait_until(lambda: admin.get("/__test/state").json()["barriers"][-1]["hit"])
    assert worker.is_alive()
    records = admin.get("/__test/audit").json()["records"]
    assert records[-1]["kind"] == "mutation"
    assert records[-1]["mutation"]["to_state"] == "active"
    assert harness.devices(donor)[2]["state"] == "active"
    admin.post("/__test/barriers/b2/release")
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert results[0].status_code == 200
    assert admin.get("/__test/audit").json()["records"][-1]["kind"] == "response"


# ------------------------------------------------------------ subprocess


def test_import_boundary() -> None:
    code = (
        "import sys\n"
        "import tests.e2e.mp_memory_coordinator.scenario_server.app\n"
        "import tests.e2e.mp_memory_coordinator.scenario_server.__main__\n"
        "print(sorted(m for m in sys.modules if m.startswith('lmcache')"
        " or 'mock_memory_allocation_service' in m))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


def test_subprocess_smoke(fixture_path: Path) -> None:
    ports = {
        name: free_port()
        for name in (
            "coordinator",
            "donor",
            "donor_alt",
            "receiver",
            "receiver_alt",
            "admin",
        )
    }
    env = dict(os.environ)
    env["SCENARIO_ADVERTISE_IP"] = "10.9.8.7"
    command = [
        sys.executable,
        "-m",
        "tests.e2e.mp_memory_coordinator.scenario_server",
        "--fixture",
        str(fixture_path),
        "--host",
        "127.0.0.1",
        "--coordinator-port",
        str(ports["coordinator"]),
        "--donor-port",
        str(ports["donor"]),
        "--donor-alt-port",
        str(ports["donor_alt"]),
        "--receiver-port",
        str(ports["receiver"]),
        "--receiver-alt-port",
        str(ports["receiver_alt"]),
        "--admin-port",
        str(ports["admin"]),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        admin = f"http://127.0.0.1:{ports['admin']}"

        def healthy() -> bool:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"server exited early:\n{output}")
            try:
                return (
                    httpx.get(f"{admin}/__test/health", timeout=1.0).json()["status"]
                    == "ok"
                )
            except httpx.HTTPError:
                return False

        wait_until(healthy, timeout=30.0)
        instances = httpx.get(
            f"http://127.0.0.1:{ports['coordinator']}/instances"
        ).json()["instances"]
        assert [(i["instance_id"], i["ip"], i["http_port"]) for i in instances] == [
            (DONOR_ID, "10.9.8.7", ports["donor"]),
            (RECEIVER_ID, "10.9.8.7", ports["receiver"]),
        ]
        assert httpx.get(f"http://127.0.0.1:{ports['donor']}/healthcheck").json() == {
            "status": "healthy"
        }
        assert httpx.get(
            f"http://127.0.0.1:{ports['donor_alt']}/healthcheck"
        ).json() == {"status": "healthy"}
        assert (
            httpx.get(f"http://127.0.0.1:{ports['receiver_alt']}/status").status_code
            == 200
        )
        assert (
            httpx.get(f"http://127.0.0.1:{ports['coordinator']}/docs").status_code
            == 404
        )
        # Five production-port requests above: one request + one response each.
        assert httpx.get(f"{admin}/__test/health").json()["seq"] == 10
    finally:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)
    assert process.returncode == 0
