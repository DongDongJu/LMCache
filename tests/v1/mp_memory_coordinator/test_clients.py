# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the three HTTP clients.

Driven with ``httpx.MockTransport`` (no sockets). The outside-service tests
assert the *complete* frozen contract from the plan: exact method and path,
every documented request field and nothing else, the
``request_size_gib``/``requested_size_gib`` spelling difference, fixed
literal values, exact echo validation, timeout/malformed handling, and the
absence of POST retries. The coordinator/MP tests validate against the golden
fixtures captured from the real services.
"""

# Standard
from pathlib import Path
import asyncio
import json

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.clients import (
    AmbiguousMutationError,
    ClientConnectionError,
    ClientHTTPError,
    ClientResponseError,
    ClientTimeoutError,
)
from lmcache.v1.mp_memory_coordinator.clients.memory_allocation_client import (
    MemoryAllocationClient,
    OutsideContractError,
    OutsideExplicitFailure,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_server_client import (
    MPServerClient,
    format_gib,
)
from lmcache.v1.mp_memory_coordinator.models import (
    AllocationRequest,
    DaxDeviceNotFound,
    DaxRemoveBlocked,
    DaxRemoveMode,
    DaxRemoveResponse,
    DeallocationRequest,
)

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "e2e"
    / "mp_memory_coordinator"
    / "fixtures"
    / "golden"
)

# Section 2 of the plan, verbatim: the frozen outside API.
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

DONOR_IP = "192.0.2.40"
RECEIVER_IP = "192.0.2.41"
DONOR_PATH = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
RECEIVER_PATH = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"


def _golden(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text())


def _run(coro):
    return asyncio.run(coro)


class _Recorder:
    """Records every request the mock transport sees."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.content]


# -- MP Coordinator client ------------------------------------------------------


def test_coordinator_client_reads_instances_and_usage_from_golden() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        if request.url.path == "/instances":
            return httpx.Response(200, json=_golden("coordinator_instances.json"))
        if request.url.path == "/instances/usage":
            return httpx.Response(200, json=_golden("coordinator_instances_usage.json"))
        return httpx.Response(404)

    client = MPCoordinatorClient(
        "http://coord:9300/",
        timeout_seconds=1.0,
        attempts=2,
        transport=httpx.MockTransport(handler),
    )

    async def run():
        instances = await client.get_instances()
        usage = await client.get_fleet_usage()
        await client.aclose()
        return instances, usage

    instances, usage = _run(run())
    assert [r.method for r in rec.requests] == ["GET", "GET"]
    assert [str(r.url) for r in rec.requests] == [
        "http://coord:9300/instances",
        "http://coord:9300/instances/usage",
    ]
    donor = instances.instances[0]
    assert donor.instance_id == "mp-donor"
    assert donor.endpoint == "10.0.0.11:8080"
    assert donor.worker_ip == DONOR_IP
    assert usage.instances[1].private_dax() is not None
    assert usage.instances[1].private_dax().usage_ratio == 0.875
    assert usage.shared_modules == []


def test_coordinator_client_rejects_missing_required_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"instances": [{"instance_id": "x"}]})

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=1,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ClientResponseError):
        _run(client.get_instances())


def test_coordinator_client_allows_unknown_fields() -> None:
    body = _golden("coordinator_instances.json")
    body["instances"][0]["future_field"] = {"nested": True}
    body["extra_top_level"] = 1

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    )
    assert _run(client.get_instances()).instances[0].instance_id == "mp-donor"


def test_get_retries_5xx_then_succeeds_and_4xx_is_immediate() -> None:
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=_golden("coordinator_instances.json"))

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=3,
        transport=httpx.MockTransport(flaky),
    )
    assert len(_run(client.get_instances()).instances) == 2
    assert calls["n"] == 3

    calls["n"] = 0

    def not_found(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, json={"error": "nope"})

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=3,
        transport=httpx.MockTransport(not_found),
    )
    with pytest.raises(ClientHTTPError) as excinfo:
        _run(client.get_instances())
    assert excinfo.value.status_code == 404
    assert calls["n"] == 1


def test_get_timeout_is_bounded_and_typed() -> None:
    calls = {"n": 0}

    def timeout(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=0.1,
        attempts=2,
        transport=httpx.MockTransport(timeout),
    )
    with pytest.raises(ClientTimeoutError):
        _run(client.get_fleet_usage())
    assert calls["n"] == 2


def test_get_connection_failure_is_typed() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=0.1,
        attempts=1,
        transport=httpx.MockTransport(down),
    )
    with pytest.raises(ClientConnectionError):
        _run(client.get_instances())


def test_get_non_json_body_is_response_error() -> None:
    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=0.1,
        attempts=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")),
    )
    with pytest.raises(ClientResponseError):
        _run(client.get_instances())


# -- MP server client -------------------------------------------------------------


def _mp_client(handler) -> MPServerClient:
    return MPServerClient(
        timeout_seconds=1.0, attempts=1, transport=httpx.MockTransport(handler)
    )


def test_mp_server_status_and_dax_status_from_golden() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/status":
            return httpx.Response(200, json=_golden("mp_status.json"))
        if request.url.path == "/reconfigure/dax/status":
            return httpx.Response(200, json=_golden("mp_reconfigure_dax_status.json"))
        if request.url.path == "/healthcheck":
            return httpx.Response(200, json={"status": "healthy"})
        return httpx.Response(404)

    client = _mp_client(handler)

    async def run():
        healthy = await client.is_healthy("http://10.0.0.11:8080")
        status = await client.get_status("http://10.0.0.11:8080")
        dax = await client.get_dax_status("http://10.0.0.11:8080")
        await client.aclose()
        return healthy, status, dax

    healthy, status, dax = _run(run())
    assert healthy is True
    assert status.is_healthy is True
    assert status.storage_manager.num_l2_adapters == 1
    adapter = status.storage_manager.l2_adapters[0]
    assert adapter.type == "dax" and adapter.hotplug_enabled and not adapter.closing
    assert dax.enabled and dax.num_adapters == 1 and dax.backend == "dax"
    assert dax.adapters[0].adapter_index == 0
    assert dax.adapters[0].supported_operations == ["status", "add", "remove", "resize"]
    devices = dax.adapters[0].status.devices
    assert [d.index for d in devices] == [0, 1]
    assert (
        devices[1].slot_capacity_bytes == devices[1].max_slots * devices[1].slot_bytes
    )
    assert devices[1].busy_references == 0
    assert not devices[1].is_terminal


def test_mp_server_dax_status_tombstone_is_terminal_and_excluded_from_live() -> None:
    body = _golden("mp_reconfigure_dax_status_after_evict.json")
    client = _mp_client(lambda r: httpx.Response(200, json=body))
    dax = _run(client.get_dax_status("http://mp:8080"))
    hotplug = dax.adapters[0].status
    assert [d.state for d in hotplug.devices] == ["active", "removed"]
    assert hotplug.devices[1].is_terminal
    assert [d.index for d in hotplug.live_devices()] == [0]
    assert hotplug.find_live(hotplug.devices[1].device_path) is None


def test_mp_server_remove_drain_sends_exact_body_and_parses_response() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(
            200, json=_golden("mp_reconfigure_dax_remove_drain_response.json")
        )

    client = _mp_client(handler)
    result = _run(
        client.remove_dax_device(
            "http://mp:8080",
            adapter_index=0,
            device_path=DONOR_PATH,
            mode=DaxRemoveMode.DRAIN,
        )
    )
    assert isinstance(result, DaxRemoveResponse)
    assert result.state == "draining" and result.operation == "drain"
    assert rec.requests[0].method == "POST"
    assert rec.requests[0].url.path == "/reconfigure/dax/remove"
    assert rec.bodies() == [
        {
            "adapter_index": 0,
            "device_path": DONOR_PATH,
            "mode": "drain",
            "force": False,
        }
    ]


def test_mp_server_remove_evict_409_and_404_are_typed_outcomes() -> None:
    def busy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "status": "blocked",
                "reason": "device has externally locked or borrowed slots",
                "locked_key_count": 2,
                "borrowed_slot_count": 1,
            },
        )

    blocked = _run(
        _mp_client(busy).remove_dax_device(
            "http://mp:8080",
            adapter_index=0,
            device_path=DONOR_PATH,
            mode=DaxRemoveMode.EVICT,
        )
    )
    assert isinstance(blocked, DaxRemoveBlocked)
    assert blocked.locked_key_count == 2 and blocked.borrowed_slot_count == 1

    missing = _run(
        _mp_client(
            lambda r: httpx.Response(
                404, json=_golden("mp_reconfigure_dax_remove_404_response.json")
            )
        ).remove_dax_device(
            "http://mp:8080",
            adapter_index=0,
            device_path=DONOR_PATH,
            mode=DaxRemoveMode.EVICT,
        )
    )
    assert isinstance(missing, DaxDeviceNotFound)

    with pytest.raises(ClientHTTPError) as excinfo:
        _run(
            _mp_client(
                lambda r: httpx.Response(403, json={"error": "hotplug disabled"})
            ).remove_dax_device(
                "http://mp:8080",
                adapter_index=0,
                device_path=DONOR_PATH,
                mode=DaxRemoveMode.EVICT,
            )
        )
    assert excinfo.value.status_code == 403


def test_mp_server_add_sends_gib_string_and_parses_device() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(200, json=_golden("mp_reconfigure_dax_add_response.json"))

    result = _run(
        _mp_client(handler).add_dax_device(
            "http://mp:8080",
            adapter_index=0,
            device_path=RECEIVER_PATH,
            size=format_gib(64 * 1024**3),
        )
    )
    assert result.device.state == "active"
    assert rec.requests[0].url.path == "/reconfigure/dax/add"
    assert rec.bodies() == [
        {"adapter_index": 0, "device_path": RECEIVER_PATH, "size": "64GiB"}
    ]


def test_mp_server_post_is_not_retried_and_drop_is_ambiguous() -> None:
    calls = {"n": 0}

    def drop(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadError("connection dropped")

    with pytest.raises(AmbiguousMutationError):
        _run(
            _mp_client(drop).add_dax_device(
                "http://mp:8080",
                adapter_index=0,
                device_path=RECEIVER_PATH,
                size="64GiB",
            )
        )
    assert calls["n"] == 1


def test_format_gib_rejects_non_whole_gib() -> None:
    assert format_gib(2 * 1024**3) == "2GiB"
    with pytest.raises(ValueError):
        format_gib(1024**3 + 1)
    with pytest.raises(ValueError):
        format_gib(0)


# -- Outside Memory Allocation client ----------------------------------------------


def _alloc_client(handler, attempts: int = 1) -> MemoryAllocationClient:
    return MemoryAllocationClient(
        "http://alloc:8080/",
        timeout_seconds=1.0,
        attempts=attempts,
        transport=httpx.MockTransport(handler),
    )


def _deallocation_response(request_id: str, **overrides) -> dict:
    body = {
        "status": "DONE",
        "request_id": request_id,
        "target_node": DONOR_IP,
        "device_path": DONOR_PATH,
        "released_size_gib": 64,
    }
    body.update(overrides)
    return body


def _allocation_response(request_id: str, **overrides) -> dict:
    body = {
        "status": "DONE",
        "request_id": request_id,
        "target_node": RECEIVER_IP,
        "device_path": RECEIVER_PATH,
        "requested_size_gib": 64,
        "granted_size_gib": 64,
    }
    body.update(overrides)
    return body


def test_outside_status_is_bare_mapping() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        return httpx.Response(200, json={DONOR_IP: [DONOR_PATH], RECEIVER_IP: []})

    status = _run(_alloc_client(handler).get_status())
    assert status == {DONOR_IP: [DONOR_PATH], RECEIVER_IP: []}
    assert rec.requests[0].method == "GET"
    assert str(rec.requests[0].url) == "http://alloc:8080/v2/apps/lmcache"


@pytest.mark.parametrize(
    "body",
    [
        {"nodes": {DONOR_IP: [DONOR_PATH]}},  # wrapper
        [DONOR_IP],  # list
        {DONOR_IP: DONOR_PATH},  # not a list
        {DONOR_IP: [{"path": DONOR_PATH}]},  # nested objects
        {DONOR_IP: [1]},
    ],
)
def test_outside_status_rejects_wrapped_or_malformed(body: object) -> None:
    with pytest.raises(OutsideContractError):
        _run(_alloc_client(lambda r: httpx.Response(200, json=body)).get_status())


def test_deallocation_request_is_exact_and_response_fully_validated() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        sent = json.loads(request.content)
        return httpx.Response(200, json=_deallocation_response(sent["request_id"]))

    request = DeallocationRequest(
        request_id="lmcache-node-a-shrink-0002",
        target_node=DONOR_IP,
        device_path=DONOR_PATH,
    )
    response = _run(_alloc_client(handler).deallocate(request))

    sent = rec.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://alloc:8080/v2/apps/lmcache/deallocations"
    assert sent.headers["content-type"] == "application/json"
    body = json.loads(sent.content)
    assert set(body) == DEALLOCATION_REQUEST_KEYS
    assert body == {
        "request_id": "lmcache-node-a-shrink-0002",
        "target_node": DONOR_IP,
        "device_path": DONOR_PATH,
    }
    assert set(response.model_dump()) >= DEALLOCATION_RESPONSE_KEYS
    assert response.released_size_gib == 64
    assert response.status == "DONE"


def test_allocation_request_is_exact_with_literal_values() -> None:
    rec = _Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        rec.requests.append(request)
        sent = json.loads(request.content)
        return httpx.Response(200, json=_allocation_response(sent["request_id"]))

    request = AllocationRequest(
        request_id="lmcache-node-a-grow-0002",
        target_node=RECEIVER_IP,
        request_size_gib=64,
    )
    response = _run(_alloc_client(handler).allocate(request))

    sent = rec.requests[0]
    assert sent.method == "POST"
    assert str(sent.url) == "http://alloc:8080/v2/apps/lmcache/allocations"
    body = json.loads(sent.content)
    assert set(body) == ALLOCATION_REQUEST_KEYS
    assert body == {
        "request_id": "lmcache-node-a-grow-0002",
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    # The request spelling is request_size_gib; the response spelling is
    # requested_size_gib. Both are asserted, never conflated.
    assert "requested_size_gib" not in body
    assert "request_size_gib" not in response.model_dump()
    assert set(response.model_dump()) >= ALLOCATION_RESPONSE_KEYS
    assert response.requested_size_gib == response.granted_size_gib == 64
    assert response.device_path == RECEIVER_PATH


def test_outside_requests_forbid_internal_fields() -> None:
    for extra in ("worker_ip", "instance_id", "registration_time", "adapter_index"):
        with pytest.raises(ValueError):
            DeallocationRequest.model_validate(
                {
                    "request_id": "r",
                    "target_node": DONOR_IP,
                    "device_path": DONOR_PATH,
                    extra: "x",
                }
            )
        with pytest.raises(ValueError):
            AllocationRequest.model_validate(
                {
                    "request_id": "r",
                    "target_node": RECEIVER_IP,
                    "request_size_gib": 64,
                    extra: "x",
                }
            )
    with pytest.raises(ValueError):
        AllocationRequest(
            request_id="r",
            target_node=RECEIVER_IP,
            request_size_gib=64,
            mode="fsdax",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": "other"},
        {"target_node": RECEIVER_IP},
        {"device_path": RECEIVER_PATH},
        {"status": "PENDING"},
        {"released_size_gib": 0},
        {"released_size_gib": "64"},
        {"released_size_gib": None},
    ],
)
def test_deallocation_echo_and_field_validation(overrides: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _deallocation_response("rid")
        body.update(overrides)
        return httpx.Response(200, json=body)

    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(handler).deallocate(
                DeallocationRequest(
                    request_id="rid", target_node=DONOR_IP, device_path=DONOR_PATH
                )
            )
        )


@pytest.mark.parametrize("missing", sorted(DEALLOCATION_RESPONSE_KEYS))
def test_deallocation_missing_documented_field_is_rejected(missing: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _deallocation_response("rid")
        del body[missing]
        return httpx.Response(200, json=body)

    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(handler).deallocate(
                DeallocationRequest(
                    request_id="rid", target_node=DONOR_IP, device_path=DONOR_PATH
                )
            )
        )


@pytest.mark.parametrize("missing", sorted(ALLOCATION_RESPONSE_KEYS))
def test_allocation_missing_documented_field_is_rejected(missing: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _allocation_response("rid")
        del body[missing]
        return httpx.Response(200, json=body)

    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(handler).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )


def test_allocation_renamed_field_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _allocation_response("rid")
        body["request_size_gib"] = body.pop("requested_size_gib")
        return httpx.Response(200, json=body)

    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(handler).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": "other"},
        {"target_node": DONOR_IP},
        {"status": "FAILED"},
        {"device_path": ""},
        {"requested_size_gib": 32},
        {"granted_size_gib": 32},
        {"requested_size_gib": 128, "granted_size_gib": 128},
        {"granted_size_gib": 64.0},
    ],
)
def test_allocation_echo_and_size_validation(overrides: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _allocation_response("rid")
        body.update(overrides)
        return httpx.Response(200, json=body)

    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(handler).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )


def test_outside_explicit_failure_is_typed_and_not_retried() -> None:
    calls = {"n": 0}

    def refuse(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(409, json={"error": "no free device"})

    with pytest.raises(OutsideExplicitFailure) as excinfo:
        _run(
            _alloc_client(refuse, attempts=3).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )
    assert excinfo.value.status_code == 409
    assert calls["n"] == 1


def test_outside_post_timeout_is_ambiguous_and_never_retried() -> None:
    calls = {"n": 0}

    def slow(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("no response")

    with pytest.raises(AmbiguousMutationError):
        _run(
            _alloc_client(slow, attempts=3).deallocate(
                DeallocationRequest(
                    request_id="rid", target_node=DONOR_IP, device_path=DONOR_PATH
                )
            )
        )
    assert calls["n"] == 1


def test_outside_post_connect_failure_is_not_ambiguous() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ClientConnectionError):
        _run(
            _alloc_client(refused).deallocate(
                DeallocationRequest(
                    request_id="rid", target_node=DONOR_IP, device_path=DONOR_PATH
                )
            )
        )


def test_outside_post_malformed_body_is_contract_error() -> None:
    with pytest.raises(ClientResponseError):
        _run(
            _alloc_client(lambda r: httpx.Response(200, text="not json")).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )
    with pytest.raises(OutsideContractError):
        _run(
            _alloc_client(lambda r: httpx.Response(200, json=[1, 2])).allocate(
                AllocationRequest(
                    request_id="rid", target_node=RECEIVER_IP, request_size_gib=64
                )
            )
        )
