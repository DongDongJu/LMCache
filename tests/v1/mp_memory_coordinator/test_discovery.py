# SPDX-License-Identifier: Apache-2.0
"""Unit tests for outside-status device discovery."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.config import (
    MPMemoryCoordinatorConfig,
    config_from_mapping,
)
from lmcache.v1.mp_memory_coordinator.discovery import discover, owners_of
from lmcache.v1.mp_memory_coordinator.models import (
    GIB,
    AllocationOrigin,
    DaxDeviceStatus,
    DaxHotplugStatus,
    InstanceIdentity,
    InstanceSample,
    JournalDocument,
    ManagedAllocation,
    OutsideStatus,
)

WORKER = "192.0.2.40"
OTHER_WORKER = "192.0.2.41"
INSTANCE = "mp-donor"
BOOT = "/dev/dax-cxl/ns_pod-a/dax0.0"
RUNTIME = "/dev/dax-cxl/ns_pod-a/dax0.3"
SLOT = 1 << 20


def _config(**overrides: object) -> MPMemoryCoordinatorConfig:
    fields: dict[object, object] = dict(state_directory="/tmp/unused")
    fields.update(overrides)
    return config_from_mapping(fields)


def _device(
    path: str,
    index: int,
    *,
    size_bytes: int = 64 * GIB,
    state: str = "active",
    is_healthy: bool = True,
    closing: bool = False,
) -> DaxDeviceStatus:
    return DaxDeviceStatus(
        index=index,
        device_id=index,
        device_path=path,
        state=state,
        is_healthy=is_healthy,
        closing=closing,
        max_dax_size_bytes=size_bytes,
        slot_bytes=SLOT,
        max_slots=size_bytes // SLOT,
        live_slot_count=0,
        locked_key_count=0,
        borrowed_slot_count=0,
        active_read_count=0,
        active_write_count=0,
        inflight_store_tasks=0,
        inflight_lookup_tasks=0,
        inflight_load_tasks=0,
    )


def _dax(*devices: DaxDeviceStatus) -> DaxHotplugStatus:
    return DaxHotplugStatus(
        hotplug_enabled=True,
        slot_bytes=SLOT,
        total_capacity_bytes=sum(d.slot_capacity_bytes for d in devices),
        total_used_bytes=0,
        devices=list(devices),
    )


def _samples(worker_ip: str = WORKER) -> dict[str, InstanceSample]:
    identity = InstanceIdentity(
        instance_id=INSTANCE,
        registration_time=1.0,
        endpoint="10.0.0.11:9000",
        worker_ip=worker_ip,
    )
    return {
        INSTANCE: InstanceSample(
            identity=identity,
            registered=True,
            declared_capacity=True,
            used_bytes=8 * GIB,
            capacity_bytes=128 * GIB,
            usage_ratio=0.0625,
            sampled_at=1.0,
        )
    }


def _outside(**nodes: list[str]) -> OutsideStatus:
    return dict(nodes)


def _run(
    outside: OutsideStatus,
    dax: DaxHotplugStatus,
    document: JournalDocument | None = None,
    config: MPMemoryCoordinatorConfig | None = None,
    worker_ip: str = WORKER,
):
    doc = document if document is not None else JournalDocument()
    result = discover(
        _samples(worker_ip),
        {INSTANCE: dax},
        outside,
        doc,
        config or _config(),
        123.0,
    )
    return result, doc


def test_discovers_a_device_the_outside_service_confirms() -> None:
    result, doc = _run(
        _outside(**{WORKER: [RUNTIME]}),
        _dax(_device(BOOT, 0), _device(RUNTIME, 1)),
    )
    assert [a.device_path for a in result.discovered] == [RUNTIME]
    allocation = doc.inventory[0]
    assert allocation.origin is AllocationOrigin.DISCOVERED
    assert allocation.worker_ip == WORKER
    assert allocation.instance_id == INSTANCE
    assert allocation.allocation_size_gib == 64
    assert allocation.device_map_size_bytes == 64 * GIB
    assert allocation.is_size_consistent()
    assert allocation.last_confirmed_state == "active"
    assert allocation.last_confirmed_at == 123.0


def test_bootstrap_device_is_never_discovered() -> None:
    result, doc = _run(_outside(**{WORKER: [BOOT]}), _dax(_device(BOOT, 0)))
    assert doc.inventory == []
    assert result.skipped[BOOT] == "DAX index 0 (bootstrap)"


def test_path_absent_from_outside_status_is_never_claimed() -> None:
    result, doc = _run(_outside(**{WORKER: []}), _dax(_device(RUNTIME, 1)))
    assert doc.inventory == []
    assert "not [192.0.2.40]" in result.skipped[RUNTIME]


def test_path_owned_by_another_node_is_never_claimed() -> None:
    result, doc = _run(
        _outside(**{WORKER: [], OTHER_WORKER: [RUNTIME]}),
        _dax(_device(RUNTIME, 1)),
    )
    assert doc.inventory == []
    assert OTHER_WORKER in result.skipped[RUNTIME]


def test_path_listed_under_two_nodes_is_never_claimed() -> None:
    # An ambiguous owner means the single-writer contract is broken; the
    # coordinator must not pick a side.
    result, doc = _run(
        _outside(**{WORKER: [RUNTIME], OTHER_WORKER: [RUNTIME]}),
        _dax(_device(RUNTIME, 1)),
    )
    assert doc.inventory == []
    assert result.skipped[RUNTIME].startswith("outside status lists the path under")


@pytest.mark.parametrize(
    "device,expected",
    [
        (_device(RUNTIME, 1, state="draining"), "state is draining"),
        (_device(RUNTIME, 1, is_healthy=False), "unhealthy or closing"),
        (_device(RUNTIME, 1, closing=True), "unhealthy or closing"),
    ],
)
def test_unusable_devices_are_skipped(device: DaxDeviceStatus, expected: str) -> None:
    result, doc = _run(_outside(**{WORKER: [RUNTIME]}), _dax(device))
    assert doc.inventory == []
    assert result.skipped[RUNTIME] == expected


@pytest.mark.parametrize("size_bytes", [0, 64 * GIB + 1, GIB // 2])
def test_non_whole_gib_devices_are_skipped(size_bytes: int) -> None:
    result, doc = _run(
        _outside(**{WORKER: [RUNTIME]}),
        _dax(_device(RUNTIME, 1, size_bytes=size_bytes)),
    )
    assert doc.inventory == []
    assert "whole number of GiB" in result.skipped[RUNTIME]


def test_path_outside_the_allowed_prefix_is_skipped() -> None:
    stray = "/dev/dax0.3"
    result, doc = _run(_outside(**{WORKER: [stray]}), _dax(_device(stray, 1)))
    assert doc.inventory == []
    assert result.skipped[stray] == "outside /dev/dax-cxl/"


def test_an_already_owned_path_is_not_duplicated() -> None:
    document = JournalDocument(
        inventory=[
            ManagedAllocation(
                worker_ip=WORKER,
                instance_id=INSTANCE,
                device_path=RUNTIME,
                allocation_size_gib=64,
                device_map_size_bytes=64 * GIB,
                slot_capacity_bytes=64 * GIB,
                adapter_index=0,
                origin=AllocationOrigin.ADOPTED,
                last_confirmed_state="active",
                last_confirmed_at=0.0,
            )
        ]
    )
    result, doc = _run(
        _outside(**{WORKER: [RUNTIME]}), _dax(_device(RUNTIME, 1)), document
    )
    assert result.discovered == []
    assert result.skipped[RUNTIME] == "already owned"
    assert len(doc.inventory) == 1
    assert doc.inventory[0].origin is AllocationOrigin.ADOPTED


def test_an_instance_without_a_dax_status_is_skipped() -> None:
    document = JournalDocument()
    result = discover(
        _samples(),
        {},
        _outside(**{WORKER: [RUNTIME]}),
        document,
        _config(),
        123.0,
    )
    assert result.discovered == []
    assert document.inventory == []


def test_repeated_passes_are_idempotent() -> None:
    document = JournalDocument()
    outside = _outside(**{WORKER: [RUNTIME]})
    dax = _dax(_device(BOOT, 0), _device(RUNTIME, 1))
    for _ in range(3):
        discover(_samples(), {INSTANCE: dax}, outside, document, _config(), 123.0)
    assert [a.device_path for a in document.inventory] == [RUNTIME]


def test_owners_of_returns_every_listing_node_sorted() -> None:
    outside = _outside(**{OTHER_WORKER: [RUNTIME], WORKER: [RUNTIME]})
    assert owners_of(outside, RUNTIME) == [WORKER, OTHER_WORKER]
    assert owners_of(outside, BOOT) == []
