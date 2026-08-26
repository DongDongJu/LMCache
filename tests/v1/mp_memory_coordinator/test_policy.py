# SPDX-License-Identifier: Apache-2.0
"""Tests for observation (sandwich join) and the dry-run policy.

Builders start from the golden fixtures so the documents under test carry
the complete current schemas.
"""

# Standard
from pathlib import Path
import asyncio
import json

# Third Party
import httpx
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    GIB,
    AllocationOrigin,
    CoordinatorInstance,
    DaxReconfigureStatus,
    InstanceIdentity,
    InstanceSample,
    InstanceUsage,
    ManagedAllocation,
    MPStatus,
)
from lmcache.v1.mp_memory_coordinator.policy import (
    DeviceChoice,
    LivePreflight,
    MembershipSnapshot,
    MoveProposal,
    PressureHistory,
    PressureLevel,
    Rejection,
    RejectionReason,
    choose_donor_device,
    classify,
    evaluate_pair,
    join_sandwich,
    preflight_problems,
    rank_candidates,
    read_sandwich,
)

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "e2e"
    / "mp_memory_coordinator"
    / "fixtures"
    / "golden"
)
DONOR_IP = "192.0.2.40"
RECEIVER_IP = "192.0.2.41"
DONOR_PATH = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
BOOTSTRAP_PATH = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
CONFIG = MPMemoryCoordinatorConfig(stable_samples=3)


def _golden(name: str) -> dict:
    return json.loads((GOLDEN / name).read_text())


def _instances() -> list[CoordinatorInstance]:
    return [
        CoordinatorInstance.model_validate(i)
        for i in _golden("coordinator_instances.json")["instances"]
    ]


def _usage() -> list[InstanceUsage]:
    return [
        InstanceUsage.model_validate(i)
        for i in _golden("coordinator_instances_usage.json")["instances"]
    ]


def _identity(instance_id: str, worker_ip: str, epoch: float = 1.0) -> InstanceIdentity:
    return InstanceIdentity(
        instance_id=instance_id,
        registration_time=epoch,
        endpoint="10.0.0.1:8080",
        worker_ip=worker_ip,
    )


def _sample(
    instance_id: str,
    ratio: float,
    worker_ip: str = DONOR_IP,
    epoch: float = 1.0,
    capacity: int = 128 * GIB,
) -> InstanceSample:
    return InstanceSample(
        identity=_identity(instance_id, worker_ip, epoch),
        registered=True,
        declared_capacity=True,
        used_bytes=int(ratio * capacity),
        capacity_bytes=capacity,
        usage_ratio=ratio,
        sampled_at=0.0,
    )


def _snapshot(*samples: InstanceSample) -> MembershipSnapshot:
    return MembershipSnapshot(
        coordinator_reachable=True,
        samples={s.identity.instance_id: s for s in samples},
        sampled_at=0.0,
    )


def _dax(used_gib: dict[int, int] | None = None, **overrides) -> DaxReconfigureStatus:
    """Golden DAX status with 64 GiB devices at index 0 (bootstrap) and 1."""
    body = _golden("mp_reconfigure_dax_status.json")
    for device in body["adapters"][0]["status"]["devices"]:
        device["slot_bytes"] = 1 << 20
        device["max_dax_size_bytes"] = 64 * GIB
        device["max_slots"] = 64 * 1024
        device["live_slot_count"] = (used_gib or {4: 4}).get(device["index"], 4) * 1024
    status = body["adapters"][0]["status"]
    status["slot_bytes"] = 1 << 20
    status["total_capacity_bytes"] = 128 * GIB
    status["total_used_bytes"] = sum(
        d["live_slot_count"] * d["slot_bytes"] for d in status["devices"]
    )
    status.update(overrides)
    return DaxReconfigureStatus.model_validate(body)


def _status(**overrides) -> MPStatus:
    body = _golden("mp_status.json")
    body.update(overrides)
    return MPStatus.model_validate(body)


def _preflight(dax: DaxReconfigureStatus | None = None, status: MPStatus | None = None):
    return LivePreflight(status=status or _status(), dax=dax or _dax())


def _allocation(path: str = DONOR_PATH, worker_ip: str = DONOR_IP) -> ManagedAllocation:
    return ManagedAllocation(
        worker_ip=worker_ip,
        instance_id="mp-donor",
        device_path=path,
        allocation_size_gib=64,
        device_map_size_bytes=64 * GIB,
        slot_capacity_bytes=64 * GIB,
        adapter_index=0,
        origin=AllocationOrigin.ADOPTED,
        last_confirmed_state="active",
        last_confirmed_at=0.0,
    )


# -- sandwich join ---------------------------------------------------------------


def test_join_accepts_golden_fleet() -> None:
    snapshot = join_sandwich(_instances(), _usage(), _instances(), 5.0)
    assert snapshot.coordinator_reachable
    assert set(snapshot.samples) == {"mp-donor", "mp-receiver"}
    donor = snapshot.samples["mp-donor"]
    assert donor.identity.worker_ip == DONOR_IP
    assert donor.identity.endpoint == "10.0.0.11:8080"
    assert donor.usage_ratio == 0.0625
    assert donor.capacity_bytes == 128 * GIB
    assert snapshot.rejections == []
    assert snapshot.still_matches(donor.identity)


@pytest.mark.parametrize(
    ("field", "value"),
    [("registration_time", 123.0), ("ip", "10.9.9.9"), ("http_port", 9999)],
)
def test_join_rejects_identity_change_between_reads(field: str, value: object) -> None:
    second = _instances()
    second[0] = second[0].model_copy(update={field: value})
    snapshot = join_sandwich(_instances(), _usage(), second, 0.0)
    assert "mp-donor" not in snapshot.samples
    assert [r.reason for r in snapshot.rejections] == [RejectionReason.IDENTITY_CHANGED]


def test_join_rejects_worker_ip_change_between_reads() -> None:
    second = _instances()
    second[0].metadata["worker_ip"] = "192.0.2.99"
    snapshot = join_sandwich(_instances(), _usage(), second, 0.0)
    assert "mp-donor" not in snapshot.samples
    assert snapshot.rejections[0].reason is RejectionReason.IDENTITY_CHANGED


def test_join_rejects_instance_missing_from_second_read() -> None:
    snapshot = join_sandwich(_instances(), _usage(), _instances()[1:], 0.0)
    assert set(snapshot.samples) == {"mp-receiver"}
    assert snapshot.rejections[0].reason is RejectionReason.NOT_IN_BOTH_READS


def test_join_rejects_missing_and_duplicate_worker_ip() -> None:
    instances = _instances()
    instances[0].metadata.clear()
    snapshot = join_sandwich(instances, _usage(), instances, 0.0)
    assert "mp-donor" not in snapshot.samples
    assert snapshot.rejections[0].reason is RejectionReason.MISSING_WORKER_IP

    instances = _instances()
    instances[1].metadata["worker_ip"] = DONOR_IP
    snapshot = join_sandwich(instances, _usage(), instances, 0.0)
    assert snapshot.samples == {}
    assert {r.reason for r in snapshot.rejections} == {
        RejectionReason.DUPLICATE_WORKER_IP
    }


def test_join_rejects_usage_conditions() -> None:
    usage = _usage()
    usage[0] = usage[0].model_copy(update={"registered": False})
    snapshot = join_sandwich(_instances(), usage, _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.UNREGISTERED

    usage = _usage()
    usage[0] = usage[0].model_copy(update={"declared_capacity": False})
    snapshot = join_sandwich(_instances(), usage, _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.UNDECLARED_CAPACITY

    usage = _usage()
    usage[0].modules[1].shared = True
    snapshot = join_sandwich(_instances(), usage, _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.NO_PRIVATE_DAX

    usage = _usage()
    usage[0].modules[1].backend = "s3"
    snapshot = join_sandwich(_instances(), usage, _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.NO_PRIVATE_DAX

    usage = _usage()
    usage[0].modules[1].usage_ratio = None
    snapshot = join_sandwich(_instances(), usage, _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.NULL_RATIO

    snapshot = join_sandwich(_instances(), _usage()[1:], _instances(), 0.0)
    assert snapshot.rejections[0].reason is RejectionReason.MISSING_USAGE


def test_read_sandwich_issues_three_reads_in_order_and_handles_outage() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/instances":
            return httpx.Response(200, json=_golden("coordinator_instances.json"))
        return httpx.Response(200, json=_golden("coordinator_instances_usage.json"))

    client = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=1,
        transport=httpx.MockTransport(handler),
    )
    snapshot = asyncio.run(read_sandwich(client, lambda: 7.0))
    assert paths == ["/instances", "/instances/usage", "/instances"]
    assert snapshot.coordinator_reachable and snapshot.sampled_at == 7.0
    assert set(snapshot.samples) == {"mp-donor", "mp-receiver"}

    down = MPCoordinatorClient(
        "http://coord:9300",
        timeout_seconds=1.0,
        attempts=1,
        transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
    )
    snapshot = asyncio.run(read_sandwich(down, lambda: 8.0))
    assert not snapshot.coordinator_reachable
    assert snapshot.samples == {}


# -- classification and history ---------------------------------------------------


def test_classify_thresholds() -> None:
    assert classify(0.75, CONFIG) is PressureLevel.HIGH
    assert classify(0.9, CONFIG) is PressureLevel.HIGH
    assert classify(0.40, CONFIG) is PressureLevel.LOW
    assert classify(0.0, CONFIG) is PressureLevel.LOW
    assert classify(0.5, CONFIG) is PressureLevel.NORMAL


def test_history_needs_three_consecutive_same_level_samples() -> None:
    history = PressureHistory(stable_samples=3)
    donor = _sample("mp-donor", 0.1)
    for expected in (None, None, PressureLevel.LOW):
        history.observe(_snapshot(donor), CONFIG)
        assert history.stable_level(donor.identity) is expected
    assert history.count(donor.identity) == 3
    assert history.snapshot() == {"mp-donor": {"level": "LOW", "count": 3}}


def test_history_resets_on_identity_change_level_change_or_gap() -> None:
    history = PressureHistory(stable_samples=3)
    donor = _sample("mp-donor", 0.1)
    history.observe(_snapshot(donor), CONFIG)
    history.observe(_snapshot(donor), CONFIG)

    # Registration epoch change: same instance_id, new identity.
    bumped = _sample("mp-donor", 0.1, epoch=2.0)
    history.observe(_snapshot(bumped), CONFIG)
    assert history.count(bumped.identity) == 1
    assert history.count(donor.identity) == 0

    # Level change resets.
    history.observe(_snapshot(bumped), CONFIG)
    history.observe(_snapshot(_sample("mp-donor", 0.9, epoch=2.0)), CONFIG)
    assert history.count(bumped.identity) == 1

    # A cycle without the instance (e.g. transient undeclared capacity)
    # drops its history entirely.
    history.observe(_snapshot(), CONFIG)
    assert history.count(bumped.identity) == 0

    # Coordinator outage resets everything.
    history.observe(_snapshot(bumped), CONFIG)
    history.observe(MembershipSnapshot(coordinator_reachable=False), CONFIG)
    assert history.snapshot() == {}


# -- ranking ---------------------------------------------------------------------------


def _stable_history(*samples: InstanceSample) -> PressureHistory:
    history = PressureHistory(stable_samples=3)
    for _ in range(3):
        history.observe(_snapshot(*samples), CONFIG)
    return history


def test_rank_candidates_orders_deterministically() -> None:
    low_a = _sample("mp-b", 0.2, worker_ip="192.0.2.2")
    low_b = _sample("mp-a", 0.2, worker_ip="192.0.2.1")
    lowest = _sample("mp-z", 0.1, worker_ip="192.0.2.3")
    high_a = _sample("mp-y", 0.8, worker_ip="192.0.2.4")
    high_b = _sample("mp-x", 0.8, worker_ip="192.0.2.5")
    highest = _sample("mp-w", 0.95, worker_ip="192.0.2.6")
    normal = _sample("mp-n", 0.5, worker_ip="192.0.2.7")
    samples = (low_a, low_b, lowest, high_a, high_b, highest, normal)
    history = _stable_history(*samples)

    candidates = rank_candidates(_snapshot(*samples), history, {}, now=0.0)

    assert [s.identity.instance_id for s in candidates.donors] == [
        "mp-z",
        "mp-a",
        "mp-b",
    ]
    assert [s.identity.instance_id for s in candidates.receivers] == [
        "mp-w",
        "mp-x",
        "mp-y",
    ]
    assert candidates.rejections == []


def test_rank_candidates_rejects_unstable_and_cooling_instances() -> None:
    donor = _sample("mp-donor", 0.1)
    receiver = _sample("mp-receiver", 0.9, worker_ip=RECEIVER_IP)
    history = PressureHistory(stable_samples=3)
    history.observe(_snapshot(donor, receiver), CONFIG)
    candidates = rank_candidates(_snapshot(donor, receiver), history, {}, now=0.0)
    assert candidates.donors == [] and candidates.receivers == []
    assert {r.reason for r in candidates.rejections} == {
        RejectionReason.HISTORY_NOT_STABLE
    }

    history = _stable_history(donor, receiver)
    cooldowns = {donor.identity.key: 100.0}
    candidates = rank_candidates(
        _snapshot(donor, receiver), history, cooldowns, now=50.0
    )
    assert candidates.donors == []
    assert [s.identity.instance_id for s in candidates.receivers] == ["mp-receiver"]
    assert candidates.rejections[0].reason is RejectionReason.COOLDOWN
    # Expired cooldown no longer blocks.
    candidates = rank_candidates(
        _snapshot(donor, receiver), history, cooldowns, now=150.0
    )
    assert [s.identity.instance_id for s in candidates.donors] == ["mp-donor"]


# -- preflight -------------------------------------------------------------------------


def test_preflight_passes_on_golden_documents() -> None:
    assert preflight_problems(_preflight(), CONFIG) == []


def test_preflight_rejects_engine_and_adapter_conditions() -> None:
    assert preflight_problems(_preflight(status=_status(is_healthy=False)), CONFIG)

    body = _golden("mp_status.json")
    body["storage_manager"]["is_healthy"] = False
    assert preflight_problems(_preflight(status=MPStatus.model_validate(body)), CONFIG)

    for key, value in (
        ("is_healthy", False),
        ("closing", True),
        ("hotplug_enabled", False),
    ):
        body = _golden("mp_status.json")
        body["storage_manager"]["l2_adapters"][0][key] = value
        problems = preflight_problems(
            _preflight(status=MPStatus.model_validate(body)), CONFIG
        )
        assert problems, key

    # Zero and two DAX adapters.
    body = _golden("mp_status.json")
    body["storage_manager"]["l2_adapters"] = []
    body["storage_manager"]["num_l2_adapters"] = 0
    assert preflight_problems(_preflight(status=MPStatus.model_validate(body)), CONFIG)
    body = _golden("mp_status.json")
    body["storage_manager"]["l2_adapters"].append(
        dict(body["storage_manager"]["l2_adapters"][0])
    )
    assert preflight_problems(_preflight(status=MPStatus.model_validate(body)), CONFIG)


def test_preflight_rejects_dax_status_conditions() -> None:
    body = _golden("mp_reconfigure_dax_status.json")
    body["adapters"].append(dict(body["adapters"][0]))
    body["num_adapters"] = 2
    dax = DaxReconfigureStatus.model_validate(body)
    assert preflight_problems(_preflight(dax=dax), CONFIG)

    body = _golden("mp_reconfigure_dax_status.json")
    body["adapters"] = []
    body["num_adapters"] = 0
    body["enabled"] = False
    assert preflight_problems(
        _preflight(dax=DaxReconfigureStatus.model_validate(body)), CONFIG
    )

    body = _golden("mp_reconfigure_dax_status.json")
    body["adapters"][0]["supported_operations"] = ["status"]
    assert preflight_problems(
        _preflight(dax=DaxReconfigureStatus.model_validate(body)), CONFIG
    )

    body = _golden("mp_reconfigure_dax_status.json")
    body["adapters"][0]["status"]["devices"][1]["state"] = "draining"
    assert preflight_problems(
        _preflight(dax=DaxReconfigureStatus.model_validate(body)), CONFIG
    )


def test_preflight_ignores_removed_tombstones() -> None:
    body = _golden("mp_reconfigure_dax_status_after_evict.json")
    dax = DaxReconfigureStatus.model_validate(body)
    assert dax.adapters[0].status.devices[1].state == "removed"
    assert preflight_problems(_preflight(dax=dax), CONFIG) == []


# -- device choice ---------------------------------------------------------------------


def test_choose_donor_device_prefers_least_used_managed_runtime_device() -> None:
    donor = _identity("mp-donor", DONOR_IP)
    choice = choose_donor_device(
        donor, _dax({0: 4, 1: 4}).adapters[0].status, [_allocation()], CONFIG
    )
    assert isinstance(choice, DeviceChoice)
    assert choice.device.device_path == DONOR_PATH
    assert choice.device.index == 1
    assert choice.allocation.allocation_size_gib == 64


def test_choose_donor_device_rejects_bootstrap_unmanaged_and_wrong_size() -> None:
    donor = _identity("mp-donor", DONOR_IP)
    dax = _dax().adapters[0].status

    # Only the bootstrap path is managed (index 0): never movable.
    rejection = choose_donor_device(donor, dax, [_allocation(BOOTSTRAP_PATH)], CONFIG)
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectionReason.NO_MANAGED_DEVICE

    # No inventory at all.
    rejection = choose_donor_device(donor, dax, [], CONFIG)
    assert isinstance(rejection, Rejection)

    # Managed under another worker.
    rejection = choose_donor_device(
        donor, dax, [_allocation(worker_ip=RECEIVER_IP)], CONFIG
    )
    assert isinstance(rejection, Rejection)

    # Size-inconsistent allocation.
    bad = _allocation().model_copy(update={"allocation_size_gib": 32})
    rejection = choose_donor_device(donor, dax, [bad], CONFIG)
    assert isinstance(rejection, Rejection)

    # Map size not matching the live device.
    bad = _allocation().model_copy(
        update={"device_map_size_bytes": 32 * GIB, "allocation_size_gib": 32}
    )
    rejection = choose_donor_device(donor, dax, [bad], CONFIG)
    assert isinstance(rejection, Rejection)

    # Wrong prefix.
    strict = MPMemoryCoordinatorConfig(allowed_device_path_prefix="/dev/other/")
    rejection = choose_donor_device(donor, dax, [_allocation()], strict)
    assert isinstance(rejection, Rejection)


def test_choose_donor_device_respects_min_devices() -> None:
    donor = _identity("mp-donor", DONOR_IP)
    dax = _dax().adapters[0].status
    strict = MPMemoryCoordinatorConfig(min_devices_per_instance=2)
    rejection = choose_donor_device(donor, dax, [_allocation()], strict)
    assert isinstance(rejection, Rejection)
    assert rejection.reason is RejectionReason.MIN_DEVICES


# -- pair evaluation ---------------------------------------------------------------


def _pair():
    donor = _sample("mp-donor", 0.0625)
    receiver = _sample("mp-receiver", 0.875, worker_ip=RECEIVER_IP, capacity=64 * GIB)
    receiver_dax = _dax({0: 56, 1: 0})
    # Receiver has one 64 GiB device at index 0 holding 56 GiB.
    body = receiver_dax.model_dump()
    body["adapters"][0]["status"]["devices"] = body["adapters"][0]["status"]["devices"][
        :1
    ]
    body["adapters"][0]["status"]["total_capacity_bytes"] = 64 * GIB
    body["adapters"][0]["status"]["total_used_bytes"] = 56 * GIB
    return donor, receiver, DaxReconfigureStatus.model_validate(body)


def test_evaluate_pair_proposes_the_golden_move() -> None:
    donor, receiver, receiver_dax = _pair()
    result = evaluate_pair(
        donor,
        receiver,
        _preflight(dax=_dax({0: 4, 1: 4})),
        _preflight(dax=receiver_dax),
        [_allocation()],
        CONFIG,
    )
    assert isinstance(result, MoveProposal)
    assert result.choice.device.device_path == DONOR_PATH
    assert result.projected_donor_ratio == pytest.approx(8 / 64)
    assert result.donor_live_capacity_bytes == 128 * GIB
    assert result.receiver_live_capacity_bytes == 64 * GIB
    assert result.as_dict()["allocation_size_gib"] == 64


def test_evaluate_pair_rejects_live_ratio_mismatch() -> None:
    donor, receiver, receiver_dax = _pair()
    # Coordinator says LOW, but live DAX says 90% full.
    result = evaluate_pair(
        donor,
        receiver,
        _preflight(dax=_dax({0: 60, 1: 55})),
        _preflight(dax=receiver_dax),
        [_allocation()],
        CONFIG,
    )
    assert isinstance(result, list)
    assert result[0].reason is RejectionReason.LIVE_RATIO_MISMATCH


def test_evaluate_pair_rejects_projected_donor_ratio_and_gap() -> None:
    donor, receiver, receiver_dax = _pair()
    # 100 GiB used across 128 GiB (LOW by threshold? 0.78 -> not LOW; use a
    # config with a high low_ratio to isolate the projection rule).
    config = MPMemoryCoordinatorConfig(
        low_ratio=0.8, high_ratio=0.85, minimum_ratio_gap=0.0
    )
    result = evaluate_pair(
        donor.model_copy(update={"usage_ratio": 0.78}),
        receiver,
        _preflight(dax=_dax({0: 50, 1: 50})),
        _preflight(dax=receiver_dax),
        [_allocation()],
        config,
    )
    assert isinstance(result, list)
    assert result[0].reason is RejectionReason.PROJECTED_DONOR_RATIO

    tight = MPMemoryCoordinatorConfig(minimum_ratio_gap=0.9)
    result = evaluate_pair(
        donor,
        receiver,
        _preflight(dax=_dax({0: 4, 1: 4})),
        _preflight(dax=receiver_dax),
        [_allocation()],
        tight,
    )
    assert isinstance(result, list)
    assert result[0].reason is RejectionReason.INSUFFICIENT_GAP


def test_evaluate_pair_reports_preflight_failure_before_anything_else() -> None:
    donor, receiver, receiver_dax = _pair()
    result = evaluate_pair(
        donor,
        receiver,
        _preflight(status=_status(is_healthy=False)),
        _preflight(dax=receiver_dax),
        [],
        CONFIG,
    )
    assert isinstance(result, list)
    assert result[0].reason is RejectionReason.PREFLIGHT_FAILED
    assert result[0].instance_id == "mp-donor"
