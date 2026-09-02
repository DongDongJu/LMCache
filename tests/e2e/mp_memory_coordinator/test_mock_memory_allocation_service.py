# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the strict mock Memory Allocation service (PLAN.md Phase 1A).

The golden request/response key sets below are written out from PLAN.md
Section 2 on purpose and never imported from production code, so a drift on
either side is caught here.
"""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import ast
import asyncio
import json
import socket
import subprocess
import sys
import time

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
import httpx
import pytest

# First Party
from tests.e2e.mp_memory_coordinator.mock_memory_allocation_service import (
    BarrierRegistry,
    FaultRegistry,
    MockAllocatorState,
    build_apps,
    create_state,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "two_server_local_dax.yaml"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE = "tests.e2e.mp_memory_coordinator.mock_memory_allocation_service"

STATUS_PATH = "/api/v2/apps/lmcache"
DEALLOCATIONS_PATH = "/api/v2/apps/lmcache/deallocations"
ALLOCATIONS_PATH = "/api/v2/apps/lmcache/allocations"

NODE_196 = "192.0.2.40"
NODE_197 = "192.0.2.41"
DAX196_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
DAX196_1 = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
DAX196_2 = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.2"
DAX197_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.0"
DAX197_1 = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"
DAX197_2 = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.2"

INITIAL_STATUS: dict[str, list[str]] = {NODE_196: [DAX196_1], NODE_197: []}

# Golden key sets from PLAN.md Section 2 (literal on purpose).
DEALLOCATION_REQUEST_KEYS = {"request_id", "target_node", "device_path"}
DEALLOCATION_RESPONSE_KEYS = {
    "status",
    "request_id",
    "target_node",
    "device_path",
    "released_size_gib",
}
ALLOCATION_REQUEST_KEYS = {
    "request_id",
    "target_node",
    "request_size_gib",
    "mode",
    "purpose",
    "access",
}
ALLOCATION_RESPONSE_KEYS = {
    "status",
    "request_id",
    "target_node",
    "device_path",
    "requested_size_gib",
    "granted_size_gib",
}

MOCK_ONLY_FIELDS = (
    "role",
    "state",
    "free_runtime_gib",
    "assigned_runtime_gib",
    "audit",
    "fault",
    "barrier",
    "instance_id",
    "adapter_index",
    "inventory",
)

PUBLIC_REJECTED_PATHS = (
    "/",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/__test/health",
    "/__test/state",
    "/__test/audit",
    "/__test/reset",
    "/__test/faults",
    "/__test/barriers",
    "/__test/barriers/x/release",
)


def deallocation_body(
    request_id: str, target_node: str = NODE_196, device_path: str = DAX196_1
) -> dict[str, object]:
    """Exact deallocation request body."""
    return {
        "request_id": request_id,
        "target_node": target_node,
        "device_path": device_path,
    }


def allocation_body(
    request_id: str, target_node: str = NODE_197, request_size_gib: int = 64
) -> dict[str, object]:
    """Exact allocation request body."""
    return {
        "request_id": request_id,
        "target_node": target_node,
        "request_size_gib": request_size_gib,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }


def assert_no_mock_only_fields(body: object) -> None:
    """Assert no mock-only name appears anywhere in a public body."""
    text = json.dumps(body)
    for name in MOCK_ONLY_FIELDS:
        assert name not in text, f"mock-only name {name!r} leaked: {text}"


def assert_status_shape(body: object) -> None:
    """Assert a status body is recursively ``str -> list[str]`` and nothing else."""
    assert isinstance(body, dict)
    for node, paths in body.items():
        assert isinstance(node, str)
        assert isinstance(paths, list)
        assert all(isinstance(path, str) for path in paths)


@dataclass
class Service:
    """A mock service built in-process."""

    state: MockAllocatorState
    faults: FaultRegistry
    barriers: BarrierRegistry
    public_app: FastAPI
    admin_app: FastAPI


def make_service(state_file: Path | None) -> Service:
    """Build a fresh service from the shared fixture."""
    state = create_state(FIXTURE_PATH, state_file)
    faults = FaultRegistry()
    barriers = BarrierRegistry()
    public_app, admin_app = build_apps(state, faults, barriers)
    return Service(state, faults, barriers, public_app, admin_app)


@pytest.fixture
def service() -> Service:
    return make_service(None)


@pytest.fixture
def public(service: Service) -> Iterator[TestClient]:
    with TestClient(service.public_app) as client:
        yield client


@pytest.fixture
def admin(service: Service) -> Iterator[TestClient]:
    with TestClient(service.admin_app) as client:
        yield client


def admin_state(admin: TestClient) -> dict[str, object]:
    response = admin.get("/__test/state")
    assert response.status_code == 200
    body: dict[str, object] = response.json()
    return body


def global_accounting(admin: TestClient) -> dict[str, int]:
    accounting = admin_state(admin)["global"]
    assert isinstance(accounting, dict)
    return accounting


def node_accounting(admin: TestClient, node: str) -> dict[str, int]:
    nodes = admin_state(admin)["nodes"]
    assert isinstance(nodes, dict)
    view = nodes[node]
    return {
        key: view[key]
        for key in (
            "free_runtime_gib",
            "assigned_runtime_gib",
            "fixed_runtime_inventory_gib",
        )
    }


def seen_request_ids(admin: TestClient) -> list[str]:
    ids = admin_state(admin)["seen_request_ids"]
    assert isinstance(ids, list)
    return ids


def device_state(admin: TestClient, node: str, path: str) -> str:
    nodes = admin_state(admin)["nodes"]
    assert isinstance(nodes, dict)
    for device in nodes[node]["devices"]:
        if device["path"] == path:
            state: str = device["state"]
            return state
    raise AssertionError(f"{path} not on {node}")


def assert_conservation(admin: TestClient) -> None:
    """Assert free + assigned == fixed inventory per node and globally."""
    for node in (NODE_196, NODE_197):
        accounting = node_accounting(admin, node)
        assert (
            accounting["free_runtime_gib"] + accounting["assigned_runtime_gib"]
            == accounting["fixed_runtime_inventory_gib"]
        )
    total = global_accounting(admin)
    assert (
        total["free_runtime_gib"] + total["assigned_runtime_gib"]
        == total["fixed_runtime_inventory_gib"]
    )


async def wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll a predicate from the event loop until true or timeout."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
        return port


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def test_status_lists_only_assigned_runtime_paths_per_node(public: TestClient) -> None:
    response = public.get(STATUS_PATH)
    assert response.status_code == 200
    body = response.json()
    assert body == INITIAL_STATUS
    assert_status_shape(body)
    assert_no_mock_only_fields(body)
    assert DAX196_BOOT not in json.dumps(body)
    assert DAX197_BOOT not in json.dumps(body)
    assert body[NODE_197] == []


def test_status_reflects_mutations_and_stays_bare(public: TestClient) -> None:
    public.post(DEALLOCATIONS_PATH, json=deallocation_body("d1"))
    public.post(ALLOCATIONS_PATH, json=allocation_body("a1"))
    public.post(ALLOCATIONS_PATH, json=allocation_body("a2"))
    body = public.get(STATUS_PATH).json()
    assert body == {NODE_196: [], NODE_197: [DAX197_1, DAX197_2]}
    assert_status_shape(body)


# --------------------------------------------------------------------------- #
# Exact key sets and spellings
# --------------------------------------------------------------------------- #


def test_deallocation_success_has_exact_key_set_and_echo(public: TestClient) -> None:
    request = deallocation_body("shrink-0001")
    assert set(request) == DEALLOCATION_REQUEST_KEYS
    response = public.post(DEALLOCATIONS_PATH, json=request)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == DEALLOCATION_RESPONSE_KEYS
    assert body == {
        "status": "DONE",
        "request_id": "shrink-0001",
        "target_node": NODE_196,
        "device_path": DAX196_1,
        "released_size_gib": 64,
    }
    assert_no_mock_only_fields(body)


def test_allocation_success_has_exact_key_set_and_distinct_size_spelling(
    public: TestClient,
) -> None:
    request = allocation_body("grow-0001")
    assert set(request) == ALLOCATION_REQUEST_KEYS
    assert "request_size_gib" in request and "requested_size_gib" not in request
    response = public.post(ALLOCATIONS_PATH, json=request)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == ALLOCATION_RESPONSE_KEYS
    assert "requested_size_gib" in body and "request_size_gib" not in body
    assert body == {
        "status": "DONE",
        "request_id": "grow-0001",
        "target_node": NODE_197,
        "device_path": DAX197_1,
        "requested_size_gib": 64,
        "granted_size_gib": 64,
    }
    assert_no_mock_only_fields(body)


# --------------------------------------------------------------------------- #
# Strict request validation
# --------------------------------------------------------------------------- #


def _without(body: dict[str, object], key: str) -> dict[str, object]:
    return {k: v for k, v in body.items() if k != key}


def _renamed(body: dict[str, object], old: str, new: str) -> dict[str, object]:
    return {(new if k == old else k): v for k, v in body.items()}


INVALID_REQUESTS: list[tuple[str, dict[str, object]]] = [
    (DEALLOCATIONS_PATH, _without(deallocation_body("x"), "device_path")),
    (DEALLOCATIONS_PATH, _without(deallocation_body("x"), "request_id")),
    (DEALLOCATIONS_PATH, _renamed(deallocation_body("x"), "device_path", "path")),
    (DEALLOCATIONS_PATH, {**deallocation_body("x"), "size_gib": 64}),
    (DEALLOCATIONS_PATH, {**deallocation_body("x"), "device_path": 5}),
    (ALLOCATIONS_PATH, _without(allocation_body("x"), "mode")),
    (ALLOCATIONS_PATH, _without(allocation_body("x"), "request_size_gib")),
    (
        ALLOCATIONS_PATH,
        _renamed(allocation_body("x"), "request_size_gib", "requested_size_gib"),
    ),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "worker_ip": NODE_197}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "mode": "fsdax"}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "purpose": "other"}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "access": "shared"}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "request_size_gib": "64"}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "request_size_gib": True}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "request_size_gib": 0}),
    (ALLOCATIONS_PATH, {**allocation_body("x"), "request_size_gib": 64.0}),
]


@pytest.mark.parametrize("path,body", INVALID_REQUESTS)
def test_invalid_request_is_422_without_mutation(
    public: TestClient, admin: TestClient, path: str, body: dict[str, object]
) -> None:
    before = admin_state(admin)
    response = public.post(path, json=body)
    assert response.status_code == 422
    assert_no_mock_only_fields(response.json())
    after = admin_state(admin)
    assert after["nodes"] == before["nodes"]
    assert after["seen_request_ids"] == before["seen_request_ids"] == []
    assert public.get(STATUS_PATH).json() == INITIAL_STATUS


def test_non_json_and_non_object_bodies_are_422(public: TestClient) -> None:
    assert public.post(ALLOCATIONS_PATH, content=b"not json").status_code == 422
    assert public.post(ALLOCATIONS_PATH, json=[1, 2]).status_code == 422
    assert public.get(STATUS_PATH).json() == INITIAL_STATUS


# --------------------------------------------------------------------------- #
# Donor -> receiver move and identity
# --------------------------------------------------------------------------- #


def test_worker196_deallocation_then_worker197_allocation(
    public: TestClient, admin: TestClient
) -> None:
    assert global_accounting(admin)["assigned_runtime_gib"] == 64

    released = public.post(DEALLOCATIONS_PATH, json=deallocation_body("shrink-1"))
    assert released.status_code == 200
    assert released.json()["released_size_gib"] == 64
    assert global_accounting(admin)["assigned_runtime_gib"] == 0
    assert_conservation(admin)

    granted = public.post(
        ALLOCATIONS_PATH,
        json=allocation_body(
            "grow-1", request_size_gib=released.json()["released_size_gib"]
        ),
    )
    assert granted.status_code == 200
    body = granted.json()
    assert body["device_path"] == DAX197_1
    assert body["requested_size_gib"] == body["granted_size_gib"] == 64

    assert global_accounting(admin)["assigned_runtime_gib"] == 64
    assert node_accounting(admin, NODE_196)["assigned_runtime_gib"] == 0
    assert node_accounting(admin, NODE_197)["assigned_runtime_gib"] == 64
    assert_conservation(admin)
    assert public.get(STATUS_PATH).json() == {NODE_196: [], NODE_197: [DAX197_1]}


def test_allocation_never_returns_a_path_from_another_node(
    public: TestClient, admin: TestClient
) -> None:
    paths: list[str] = []
    for i in range(2):
        response = public.post(ALLOCATIONS_PATH, json=allocation_body(f"g{i}"))
        assert response.status_code == 200
        paths.append(response.json()["device_path"])
    assert paths == [DAX197_1, DAX197_2]
    assert all("mp-197" in path for path in paths)

    exhausted = public.post(ALLOCATIONS_PATH, json=allocation_body("g2"))
    assert exhausted.status_code == 409
    assert set(exhausted.json()) == {"error"}

    on_196 = public.post(
        ALLOCATIONS_PATH, json=allocation_body("g3", target_node=NODE_196)
    )
    assert on_196.status_code == 200
    assert on_196.json()["device_path"] == DAX196_2
    assert_conservation(admin)


def test_allocation_requires_exact_size_match(
    public: TestClient, admin: TestClient
) -> None:
    for size in (32, 65, 128):
        response = public.post(
            ALLOCATIONS_PATH, json=allocation_body(f"s{size}", request_size_gib=size)
        )
        assert response.status_code == 409, size
        assert set(response.json()) == {"error"}
    assert global_accounting(admin)["assigned_runtime_gib"] == 64
    exact = public.post(ALLOCATIONS_PATH, json=allocation_body("s64"))
    assert exact.status_code == 200
    assert exact.json()["granted_size_gib"] == 64


# --------------------------------------------------------------------------- #
# Error codes
# --------------------------------------------------------------------------- #


def test_error_codes(public: TestClient, admin: TestClient) -> None:
    cases: list[tuple[str, dict[str, object], int]] = [
        (DEALLOCATIONS_PATH, deallocation_body("e1", target_node="192.0.2.99"), 404),
        (ALLOCATIONS_PATH, allocation_body("e2", target_node="192.0.2.99"), 404),
        (DEALLOCATIONS_PATH, deallocation_body("e3", target_node=NODE_197), 409),
        (DEALLOCATIONS_PATH, deallocation_body("e4", device_path="/dev/dax9.9"), 404),
        (DEALLOCATIONS_PATH, deallocation_body("e5", device_path=DAX196_BOOT), 403),
        (DEALLOCATIONS_PATH, deallocation_body("e6", device_path=DAX196_2), 409),
    ]
    for path, body, expected in cases:
        response = public.post(path, json=body)
        assert response.status_code == expected, body
        assert set(response.json()) == {"error"}
        assert isinstance(response.json()["error"], str)
        assert_no_mock_only_fields(response.json())
    assert public.get(STATUS_PATH).json() == INITIAL_STATUS
    assert_conservation(admin)


def test_duplicate_path_after_release_is_409(public: TestClient) -> None:
    assert (
        public.post(DEALLOCATIONS_PATH, json=deallocation_body("d1")).status_code == 200
    )
    again = public.post(DEALLOCATIONS_PATH, json=deallocation_body("d2"))
    assert again.status_code == 409
    assert set(again.json()) == {"error"}


def test_duplicate_request_id_is_409_and_not_idempotent(
    public: TestClient, admin: TestClient
) -> None:
    first = public.post(ALLOCATIONS_PATH, json=allocation_body("grow-dup"))
    assert first.status_code == 200
    second = public.post(ALLOCATIONS_PATH, json=allocation_body("grow-dup"))
    assert second.status_code == 409
    assert set(second.json()) == {"error"}
    # The duplicate neither replayed the first response nor consumed a device.
    assert global_accounting(admin)["assigned_runtime_gib"] == 128
    # IDs are shared across operations and remembered even for rejected calls.
    rejected = public.post(
        DEALLOCATIONS_PATH, json=deallocation_body("x-1", device_path=DAX196_2)
    )
    assert rejected.status_code == 409
    reused = public.post(DEALLOCATIONS_PATH, json=deallocation_body("x-1"))
    assert reused.status_code == 409
    assert device_state(admin, NODE_196, DAX196_1) == "assigned"
    assert "grow-dup" in seen_request_ids(admin)


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_allocations_never_overcommit() -> None:
    service = make_service(None)
    transport = ASGITransport(app=service.public_app)
    async with AsyncClient(transport=transport, base_url="http://public") as client:
        responses = await asyncio.gather(
            *(
                client.post(ALLOCATIONS_PATH, json=allocation_body(f"c{i}"))
                for i in range(8)
            )
        )
    codes = sorted(response.status_code for response in responses)
    assert codes == [200, 200] + [409] * 6
    granted = [r.json()["device_path"] for r in responses if r.status_code == 200]
    assert sorted(granted) == [DAX197_1, DAX197_2]
    assert len(set(granted)) == 2
    admin_transport = ASGITransport(app=service.admin_app)
    async with AsyncClient(transport=admin_transport, base_url="http://admin") as admin:
        view = (await admin.get("/__test/state")).json()
    assert view["global"]["assigned_runtime_gib"] == 64 + 128
    assert view["global"]["free_runtime_gib"] == 64


# --------------------------------------------------------------------------- #
# Persistence and reset
# --------------------------------------------------------------------------- #


def test_restart_from_persisted_state_and_explicit_reset(tmp_path: Path) -> None:
    state_file = tmp_path / "mock-state.json"
    first = make_service(state_file)
    with TestClient(first.public_app) as public:
        assert (
            public.post(DEALLOCATIONS_PATH, json=deallocation_body("d1")).status_code
            == 200
        )
        assert (
            public.post(ALLOCATIONS_PATH, json=allocation_body("a1")).status_code == 200
        )
    assert state_file.exists()
    assert not state_file.with_name(state_file.name + ".tmp").exists()

    restarted = make_service(state_file)
    with (
        TestClient(restarted.public_app) as public,
        TestClient(restarted.admin_app) as admin,
    ):
        assert public.get(STATUS_PATH).json() == {NODE_196: [], NODE_197: [DAX197_1]}
        assert (
            public.post(ALLOCATIONS_PATH, json=allocation_body("a1")).status_code == 409
        )
        assert_conservation(admin)
        reset = admin.post("/__test/reset")
        assert reset.status_code == 200
        assert reset.json()["seen_request_ids"] == []
        assert admin.get("/__test/audit").json()["records"] == []
        assert admin.get("/__test/health").json()["seq"] == 0
        assert public.get(STATUS_PATH).json() == INITIAL_STATUS

    from_fixture_again = make_service(state_file)
    with TestClient(from_fixture_again.public_app) as public:
        assert public.get(STATUS_PATH).json() == INITIAL_STATUS


def test_invalid_fixture_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "schema_version: 1\nnodes:\n  '10.0.0.1':\n    name: w\n    devices:\n"
        "      - path: /dev/dax0.0\n        size_gib: 64\n        role: bootstrap\n"
        "        state: free\n"
    )
    with pytest.raises(ValueError):
        create_state(bad, None)
    bad.write_text("schema_version: 2\nnodes: {}\n")
    with pytest.raises(ValueError):
        create_state(bad, None)


# --------------------------------------------------------------------------- #
# Faults
# --------------------------------------------------------------------------- #


def install_fault(admin: TestClient, **spec: object) -> None:
    response = admin.post("/__test/faults", json=spec)
    assert response.status_code == 200, response.text


def test_fault_fail_before_mutation(public: TestClient, admin: TestClient) -> None:
    install_fault(admin, operation="allocate", mode="fail_before_mutation")
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("f1"))
    assert response.status_code == 500
    assert set(response.json()) == {"error"}
    assert_no_mock_only_fields(response.json())
    assert global_accounting(admin)["assigned_runtime_gib"] == 64
    install_fault(
        admin, operation="deallocate", mode="fail_before_mutation", status_code=503
    )
    response = public.post(DEALLOCATIONS_PATH, json=deallocation_body("f2"))
    assert response.status_code == 503
    assert device_state(admin, NODE_196, DAX196_1) == "assigned"
    assert admin_state(admin)["faults"] == {"active": []}


def test_fault_commit_then_drop(service: Service, admin: TestClient) -> None:
    install_fault(admin, operation="allocate", mode="commit_then_drop")
    with TestClient(service.public_app, raise_server_exceptions=False) as public:
        response = public.post(ALLOCATIONS_PATH, json=allocation_body("drop-1"))
        assert response.content == b""
        with pytest.raises(ValueError):
            response.json()
        assert public.get(STATUS_PATH).json() == {
            NODE_196: [DAX196_1],
            NODE_197: [DAX197_1],
        }
    assert device_state(admin, NODE_197, DAX197_1) == "assigned"
    records = admin.get("/__test/audit").json()["records"]
    kinds = [
        (r["kind"], r["status_code"]) for r in records if r["operation"] == "allocate"
    ]
    assert kinds[:3] == [("request", 0), ("mutation", 0), ("response", 0)]


def test_fault_delay(public: TestClient, admin: TestClient) -> None:
    install_fault(admin, operation="deallocate", mode="delay", delay_seconds=0.3)
    started = time.monotonic()
    response = public.post(DEALLOCATIONS_PATH, json=deallocation_body("slow"))
    assert time.monotonic() - started >= 0.3
    assert response.status_code == 200
    assert set(response.json()) == DEALLOCATION_RESPONSE_KEYS


def test_fault_wrong_echo(public: TestClient, admin: TestClient) -> None:
    install_fault(
        admin, operation="deallocate", mode="wrong_echo", echo_field="target_node"
    )
    response = public.post(DEALLOCATIONS_PATH, json=deallocation_body("echo"))
    assert response.status_code == 200
    assert response.json()["target_node"] == "wrong-" + NODE_196
    assert response.json()["request_id"] == "echo"
    assert device_state(admin, NODE_196, DAX196_1) == "free"


def test_fault_missing_field(public: TestClient, admin: TestClient) -> None:
    install_fault(
        admin,
        operation="deallocate",
        mode="missing_field",
        missing_field_name="released_size_gib",
    )
    response = public.post(DEALLOCATIONS_PATH, json=deallocation_body("miss"))
    assert response.status_code == 200
    assert set(response.json()) == DEALLOCATION_RESPONSE_KEYS - {"released_size_gib"}
    assert device_state(admin, NODE_196, DAX196_1) == "free"


def test_fault_wrong_size(public: TestClient, admin: TestClient) -> None:
    install_fault(admin, operation="allocate", mode="wrong_size", size_gib_override=1)
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("size"))
    assert response.status_code == 200
    assert response.json()["requested_size_gib"] == 1
    assert response.json()["granted_size_gib"] == 1
    assert node_accounting(admin, NODE_197)["assigned_runtime_gib"] == 64


def test_fault_invalid_path(public: TestClient, admin: TestClient) -> None:
    # Two free devices on worker-197: one bogus path and one donor path.
    for index, override in enumerate(("../etc/x", DAX196_1)):
        install_fault(
            admin, operation="allocate", mode="invalid_path", path_override=override
        )
        response = public.post(ALLOCATIONS_PATH, json=allocation_body(f"p{index}"))
        assert response.status_code == 200
        assert response.json()["device_path"] == override
    assert node_accounting(admin, NODE_197)["assigned_runtime_gib"] == 128
    assert public.get(STATUS_PATH).json() == {
        NODE_196: [DAX196_1],
        NODE_197: [DAX197_1, DAX197_2],
    }


def test_fault_insufficient_capacity(public: TestClient, admin: TestClient) -> None:
    install_fault(admin, operation="allocate", mode="insufficient_capacity")
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("cap"))
    assert response.status_code == 409
    assert set(response.json()) == {"error"}
    assert node_accounting(admin, NODE_197)["free_runtime_gib"] == 128


def test_pool_budget_refuses_without_mutation_and_frees_on_deallocation(
    public: TestClient, admin: TestClient
) -> None:
    assert admin_state(admin)["pool_budget_gib"] is None
    view = admin.post("/__test/pool_budget", json={"pool_budget_gib": 64}).json()
    assert view["pool_budget_gib"] == 64
    # 64 GiB already assigned: another 64 GiB is refused although free
    # devices exist on the node, and nothing changes.
    before = admin_state(admin)
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("budget-1"))
    assert response.status_code == 409
    assert "pool budget" in response.json()["error"]
    after = admin_state(admin)
    assert after["nodes"] == before["nodes"] and after["global"] == before["global"]
    assert "budget-1" in seen_request_ids(admin)
    assert (
        public.post(ALLOCATIONS_PATH, json=allocation_body("budget-1")).status_code
        == 409
    )
    # Deallocating frees budget: the next allocation is served.
    assert (
        public.post(DEALLOCATIONS_PATH, json=deallocation_body("free-1")).status_code
        == 200
    )
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("budget-2"))
    assert response.status_code == 200
    assert global_accounting(admin)["assigned_runtime_gib"] == 64
    # Unlimited again.
    admin.post("/__test/pool_budget", json={"pool_budget_gib": None})
    assert (
        public.post(ALLOCATIONS_PATH, json=allocation_body("budget-3")).status_code
        == 200
    )
    assert (
        admin.post("/__test/pool_budget", json={"pool_budget_gib": -1}).status_code
        == 422
    )


def test_pool_budget_refusal_consumes_no_fault_and_reset_applies_a_budget(
    public: TestClient, admin: TestClient
) -> None:
    admin.post("/__test/pool_budget", json={"pool_budget_gib": 64})
    install_fault(
        admin, operation="allocate", mode="wrong_echo", echo_field="request_id"
    )
    assert (
        public.post(ALLOCATIONS_PATH, json=allocation_body("refused")).status_code
        == 409
    )
    faults = admin_state(admin)["faults"]
    assert isinstance(faults, dict)
    assert len(faults["active"]) == 1, "fault must survive"
    admin.post("/__test/pool_budget", json={"pool_budget_gib": None})
    response = public.post(ALLOCATIONS_PATH, json=allocation_body("served"))
    assert response.status_code == 200
    assert response.json()["request_id"] == "wrong-served"
    # Reset clears the budget unless the body sets one.
    admin.post("/__test/pool_budget", json={"pool_budget_gib": 0})
    assert admin.post("/__test/reset").json()["pool_budget_gib"] is None
    view = admin.post("/__test/reset", json={"pool_budget_gib": 64}).json()
    assert view["pool_budget_gib"] == 64
    assert view["global"]["assigned_runtime_gib"] == 64
    assert (
        public.post(ALLOCATIONS_PATH, json=allocation_body("after-reset")).status_code
        == 409
    )


def test_fault_count_and_clear(public: TestClient, admin: TestClient) -> None:
    install_fault(admin, operation="allocate", mode="insufficient_capacity", count=2)
    assert public.post(ALLOCATIONS_PATH, json=allocation_body("n1")).status_code == 409
    assert public.post(ALLOCATIONS_PATH, json=allocation_body("n2")).status_code == 409
    assert public.post(ALLOCATIONS_PATH, json=allocation_body("n3")).status_code == 200
    install_fault(admin, operation="allocate", mode="fail_before_mutation", count=5)
    assert admin.delete("/__test/faults").json() == {"faults": []}
    assert public.post(ALLOCATIONS_PATH, json=allocation_body("n4")).status_code == 200


def test_fault_spec_rejects_unknown_fields(admin: TestClient) -> None:
    assert admin.post("/__test/faults", json={"mode": "explode"}).status_code == 422
    assert admin.post("/__test/faults", json={"bogus": 1}).status_code == 422


# --------------------------------------------------------------------------- #
# Barriers
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "after"])
async def test_barrier_blocks_until_released(when: str) -> None:
    service = make_service(None)
    public_transport = ASGITransport(app=service.public_app)
    admin_transport = ASGITransport(app=service.admin_app)
    async with (
        AsyncClient(transport=public_transport, base_url="http://public") as public,
        AsyncClient(transport=admin_transport, base_url="http://admin") as admin,
    ):
        installed = await admin.post(
            "/__test/barriers",
            json={"operation": "allocate", "when": when, "name": "b1"},
        )
        assert installed.status_code == 200
        assert installed.json()["barriers"]["b1"]["status"] == "armed"

        task = asyncio.create_task(
            public.post(ALLOCATIONS_PATH, json=allocation_body("blocked"))
        )
        observed_assigned_gib: list[int] = []

        async def barrier_waiting() -> bool:
            view = (await admin.get("/__test/state")).json()
            observed_assigned_gib.append(view["global"]["assigned_runtime_gib"])
            barriers = view["barriers"]
            return "b1" in barriers and barriers["b1"]["status"] == "waiting"

        deadline = time.monotonic() + 5
        while not await barrier_waiting():
            assert time.monotonic() < deadline
            await asyncio.sleep(0.01)
        assert not task.done()
        expected_assigned = 64 if when == "before" else 128
        assert observed_assigned_gib[-1] == expected_assigned

        released = await admin.post("/__test/barriers/b1/release")
        assert released.status_code == 204
        response = await task
        assert response.status_code == 200
        assert response.json()["device_path"] == DAX197_1
        final = (await admin.get("/__test/state")).json()
        assert final["barriers"] == {}
        assert final["global"]["assigned_runtime_gib"] == 128
        assert (await admin.post("/__test/barriers/b1/release")).status_code == 404


def test_barrier_names_are_unique_and_reset_clears_them(admin: TestClient) -> None:
    spec = {"operation": "deallocate", "when": "before", "name": "dup"}
    assert admin.post("/__test/barriers", json=spec).status_code == 200
    assert admin.post("/__test/barriers", json=spec).status_code == 409
    assert admin.post("/__test/reset").json()["barriers"] == {}


# --------------------------------------------------------------------------- #
# Listener separation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", PUBLIC_REJECTED_PATHS)
def test_public_port_rejects_non_frozen_routes(public: TestClient, path: str) -> None:
    for method in ("GET", "POST", "DELETE"):
        response = public.request(method, path)
        assert response.status_code == 404, (method, path)
        assert set(response.json()) == {"error"}
        assert_no_mock_only_fields(response.json())


def test_admin_port_rejects_frozen_outside_routes(admin: TestClient) -> None:
    assert admin.get(STATUS_PATH).status_code == 404
    assert (
        admin.post(DEALLOCATIONS_PATH, json=deallocation_body("x")).status_code == 404
    )
    assert admin.post(ALLOCATIONS_PATH, json=allocation_body("x")).status_code == 404
    assert admin.get("/").status_code == 404
    assert admin.get("/docs").status_code == 404


def test_public_bodies_never_carry_mock_only_fields(public: TestClient) -> None:
    bodies: list[object] = [public.get(STATUS_PATH).json()]
    requests: list[tuple[str, dict[str, object]]] = [
        (DEALLOCATIONS_PATH, deallocation_body("m1")),
        (ALLOCATIONS_PATH, allocation_body("m2")),
        (DEALLOCATIONS_PATH, deallocation_body("m3")),
        (DEALLOCATIONS_PATH, deallocation_body("m4", device_path=DAX196_BOOT)),
        (DEALLOCATIONS_PATH, deallocation_body("m5", target_node="10.0.0.1")),
        (DEALLOCATIONS_PATH, deallocation_body("m6", target_node=NODE_197)),
        (ALLOCATIONS_PATH, allocation_body("m7", request_size_gib=1)),
        (ALLOCATIONS_PATH, allocation_body("m2")),
        (ALLOCATIONS_PATH, _without(allocation_body("m8"), "access")),
    ]
    for path, body in requests:
        bodies.append(public.post(path, json=body).json())
    bodies.append(public.get("/__test/state").json())
    for public_body in bodies:
        assert_no_mock_only_fields(public_body)


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #


def test_audit_records_request_mutation_response_with_monotonic_seq(
    public: TestClient, admin: TestClient
) -> None:
    assert admin.get("/__test/health").json()["seq"] == 0
    public.get(STATUS_PATH)
    response = public.post(DEALLOCATIONS_PATH, json=deallocation_body("audit-1"))
    assert response.status_code == 200
    records = admin.get("/__test/audit", params={"after_seq": 0}).json()["records"]
    seqs = [record["seq"] for record in records]
    assert seqs == list(range(1, len(records) + 1))
    assert set(records[0]) == {
        "seq",
        "kind",
        "operation",
        "request_id",
        "target_node",
        "device_path",
        "status_code",
        "body",
        "timestamp",
    }
    dealloc = [r for r in records if r["operation"] == "deallocate"]
    assert [r["kind"] for r in dealloc] == ["request", "mutation", "response"]
    assert dealloc[0]["body"] == deallocation_body("audit-1")
    assert dealloc[0]["request_id"] == "audit-1"
    assert dealloc[1]["body"] == {
        "path": DAX196_1,
        "from_state": "assigned",
        "to_state": "free",
        "node": NODE_196,
    }
    assert dealloc[2]["status_code"] == 200
    assert dealloc[2]["body"] == response.json()
    assert [r["kind"] for r in records[:2]] == ["request", "response"]
    assert records[0]["operation"] == "status" and records[0]["request_id"] == ""
    health = admin.get("/__test/health").json()
    assert health == {"status": "ok", "fixture": str(FIXTURE_PATH), "seq": seqs[-1]}
    later = admin.get("/__test/audit", params={"after_seq": seqs[-2]}).json()
    assert [r["seq"] for r in later["records"]] == [seqs[-1]]


# --------------------------------------------------------------------------- #
# Import boundary and direct execution
# --------------------------------------------------------------------------- #


def test_production_code_never_imports_the_mock() -> None:
    offenders: list[str] = []
    for source in (REPO_ROOT / "lmcache").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            if any("mock_memory_allocation_service" in name for name in names):
                offenders.append(str(source))
    assert offenders == []


def test_direct_execution_serves_public_and_admin_listeners() -> None:
    public_port, admin_port = free_port(), free_port()
    command = [
        sys.executable,
        "-m",
        MODULE,
        "--fixture",
        str(FIXTURE_PATH),
        "--public-host",
        "127.0.0.1",
        "--public-port",
        str(public_port),
        "--admin-host",
        "127.0.0.1",
        "--admin-port",
        str(admin_port),
    ]
    process = subprocess.Popen(
        command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        deadline = time.monotonic() + 20
        health: httpx.Response | None = None
        while time.monotonic() < deadline:
            try:
                health = httpx.get(
                    f"http://127.0.0.1:{admin_port}/__test/health", timeout=1
                )
                break
            except httpx.HTTPError:
                assert process.poll() is None, "mock process exited early"
                time.sleep(0.1)
        assert health is not None, "admin listener never became healthy"
        assert health.json()["status"] == "ok"
        status = httpx.get(f"http://127.0.0.1:{public_port}{STATUS_PATH}", timeout=5)
        assert status.json() == INITIAL_STATUS
        assert (
            httpx.get(f"http://127.0.0.1:{public_port}/__test/health").status_code
            == 404
        )
        assert (
            httpx.get(f"http://127.0.0.1:{admin_port}{STATUS_PATH}").status_code == 404
        )
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            raise AssertionError("mock did not stop on SIGTERM") from None
