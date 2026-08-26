# SPDX-License-Identifier: Apache-2.0
"""Conformance test for any implementation of the frozen outside API.

Run it against a real (or mock) Memory Allocation service by URL::

    uv run pytest -q tests/e2e/mp_memory_coordinator/test_outside_api_conformance.py \\
        -m outside_api --outside-api-url http://127.0.0.1:18080

It uses plain ``requests`` and the golden key sets of PLAN.md section 2 --
no production client, no mock -- so an implementation is judged against the
contract alone. The mutating round trip deallocates one currently assigned
runtime device and allocates the same size back to the same node, restoring
the pre-test assignment; the service must therefore expose at least one
assigned runtime path when the test starts. Requested but unreachable is a
failure, never a skip.
"""

# Standard
import uuid

# Third Party
import pytest
import requests

pytestmark = pytest.mark.outside_api

STATUS_PATH = "/v2/apps/lmcache"
DEALLOCATIONS_PATH = "/v2/apps/lmcache/deallocations"
ALLOCATIONS_PATH = "/v2/apps/lmcache/allocations"

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


@pytest.fixture(scope="module")
def outside_url(request: pytest.FixtureRequest) -> str:
    url = str(request.config.getoption("--outside-api-url")).rstrip("/")
    assert url, "conformance requested without --outside-api-url"
    try:
        response = requests.get(f"{url}{STATUS_PATH}", timeout=10)
    except requests.RequestException as exc:
        pytest.fail(f"outside service unreachable at {url}: {exc}")
    assert response.status_code == 200, response.text
    return url


def _status(url: str) -> dict:
    response = requests.get(f"{url}{STATUS_PATH}", timeout=10)
    assert response.status_code == 200, response.text
    return response.json()


def test_status_is_bare_target_node_to_paths_object(outside_url: str) -> None:
    body = _status(outside_url)
    assert isinstance(body, dict), "status must be a JSON object, not a list"
    assert body, "status lists no nodes"
    for node, paths in body.items():
        assert isinstance(node, str) and node, "target_node keys must be strings"
        assert isinstance(paths, list), f"{node}: value must be a list (no wrapper)"
        for path in paths:
            assert isinstance(path, str) and path.startswith("/"), (node, path)
    # No wrapper or metadata keys such as "nodes", "status", "devices".
    assert not any(isinstance(v, dict) for v in body.values())


def test_deallocate_then_allocate_round_trip_restores_assignment(
    outside_url: str,
) -> None:
    before = _status(outside_url)
    candidates = [(node, paths[0]) for node, paths in before.items() if paths]
    assert candidates, "no assigned runtime path to exercise; assign one first"
    node, path = sorted(candidates)[0]

    shrink_id = f"conformance-shrink-{uuid.uuid4().hex[:8]}"
    dealloc = requests.post(
        f"{outside_url}{DEALLOCATIONS_PATH}",
        json={"request_id": shrink_id, "target_node": node, "device_path": path},
        timeout=30,
    )
    assert dealloc.status_code == 200, dealloc.text
    released = dealloc.json()
    assert set(released) >= DEALLOCATION_RESPONSE_KEYS, set(released)
    assert released["status"] == "DONE"
    assert released["request_id"] == shrink_id
    assert released["target_node"] == node
    assert released["device_path"] == path
    assert isinstance(released["released_size_gib"], int)
    assert not isinstance(released["released_size_gib"], bool)
    assert released["released_size_gib"] > 0
    after_dealloc = _status(outside_url)
    assert path not in after_dealloc.get(node, [])
    assert all(path not in paths for paths in after_dealloc.values())

    grow_id = f"conformance-grow-{uuid.uuid4().hex[:8]}"
    alloc = requests.post(
        f"{outside_url}{ALLOCATIONS_PATH}",
        json={
            "request_id": grow_id,
            "target_node": node,
            "request_size_gib": released["released_size_gib"],
            "mode": "devdax",
            "purpose": "lmcache-dax",
            "access": "exclusive",
        },
        timeout=30,
    )
    assert alloc.status_code == 200, alloc.text
    granted = alloc.json()
    assert set(granted) >= ALLOCATION_RESPONSE_KEYS, set(granted)
    assert "request_size_gib" not in granted, "response spelling is requested_size_gib"
    assert granted["status"] == "DONE"
    assert granted["request_id"] == grow_id
    assert granted["target_node"] == node
    assert isinstance(granted["device_path"], str) and granted["device_path"]
    assert granted["device_path"].startswith("/")
    assert (
        granted["requested_size_gib"]
        == granted["granted_size_gib"]
        == released["released_size_gib"]
    )
    after_alloc = _status(outside_url)
    assert granted["device_path"] in after_alloc.get(node, [])
    assert [n for n, ps in after_alloc.items() if granted["device_path"] in ps] == [
        node
    ]
    # The pre-existing path returned to the same node keeps the inventory
    # constant; a service that allocates a different same-size local path is
    # still conformant, so only the count is compared.
    assert {n: len(ps) for n, ps in after_alloc.items()} == {
        n: len(ps) for n, ps in before.items()
    }


@pytest.mark.parametrize(
    "body",
    [
        {"target_node": "x", "device_path": "/p"},  # missing request_id
        {"request_id": "r", "target_node": "x", "path": "/p"},  # renamed field
        {"request_id": "r", "target_node": "x", "device_path": "/p", "extra": 1},
    ],
    ids=["missing", "renamed", "extra"],
)
def test_deallocation_rejects_malformed_requests_without_mutation(
    outside_url: str, body: dict
) -> None:
    before = _status(outside_url)
    response = requests.post(
        f"{outside_url}{DEALLOCATIONS_PATH}", json=body, timeout=30
    )
    assert 400 <= response.status_code < 500, response.text
    assert _status(outside_url) == before


@pytest.mark.parametrize(
    "body",
    [
        {  # requested_size_gib is the response spelling, not the request one
            "request_id": "r",
            "target_node": "x",
            "requested_size_gib": 64,
            "mode": "devdax",
            "purpose": "lmcache-dax",
            "access": "exclusive",
        },
        {  # wrong literal
            "request_id": "r",
            "target_node": "x",
            "request_size_gib": 64,
            "mode": "fsdax",
            "purpose": "lmcache-dax",
            "access": "exclusive",
        },
        {  # missing literal fields
            "request_id": "r",
            "target_node": "x",
            "request_size_gib": 64,
        },
    ],
    ids=["response_spelling", "wrong_literal", "missing_literals"],
)
def test_allocation_rejects_malformed_requests_without_mutation(
    outside_url: str, body: dict
) -> None:
    before = _status(outside_url)
    response = requests.post(f"{outside_url}{ALLOCATIONS_PATH}", json=body, timeout=30)
    assert 400 <= response.status_code < 500, response.text
    assert _status(outside_url) == before


def test_unknown_node_is_rejected_without_mutation(outside_url: str) -> None:
    before = _status(outside_url)
    node = "203.0.113.254"
    assert node not in before
    response = requests.post(
        f"{outside_url}{ALLOCATIONS_PATH}",
        json={
            "request_id": f"conformance-unknown-{uuid.uuid4().hex[:8]}",
            "target_node": node,
            "request_size_gib": 1,
            "mode": "devdax",
            "purpose": "lmcache-dax",
            "access": "exclusive",
        },
        timeout=30,
    )
    assert 400 <= response.status_code < 500, response.text
    assert _status(outside_url) == before
