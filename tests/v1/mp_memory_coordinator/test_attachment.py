# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the pure attach planner."""

# Third Party
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.attachment import (
    AttachmentReport,
    plan_attachments,
)
from lmcache.v1.mp_memory_coordinator.config import (
    MPMemoryCoordinatorConfig,
    config_from_mapping,
)
from lmcache.v1.mp_memory_coordinator.models import (
    GIB,
    DaxDeviceStatus,
    DaxHotplugStatus,
    DaxPhysicalStatus,
    DaxWatcherStatus,
    InstanceIdentity,
    InstanceSample,
    OutsideStatus,
)

WORKER = "192.0.2.40"
OTHER_WORKER = "192.0.2.41"
INSTANCE = "mp-donor"
OTHER_INSTANCE = "mp-receiver"
DIRECTORY = "/dev/dax-cxl/ns_pod-a"
BOOT = f"{DIRECTORY}/dax0.0"
ATTACHED = f"{DIRECTORY}/dax0.1"
SPARE = f"{DIRECTORY}/dax0.2"
SPARE_B = f"{DIRECTORY}/dax0.3"
SLOT = 1 << 20
NOW = 5000.0


def _config(**overrides: object) -> MPMemoryCoordinatorConfig:
    fields: dict[object, object] = dict(state_directory="/tmp/unused")
    fields.update(overrides)
    return config_from_mapping(fields)


def _device(path: str, index: int, *, state: str = "active") -> DaxDeviceStatus:
    return DaxDeviceStatus(
        index=index,
        device_id=index,
        device_path=path,
        state=state,
        is_healthy=state not in ("closed", "removed"),
        closing=False,
        max_dax_size_bytes=64 * GIB,
        slot_bytes=SLOT,
        max_slots=64 * 1024,
        live_slot_count=0,
        locked_key_count=0,
        borrowed_slot_count=0,
        active_read_count=0,
        active_write_count=0,
        inflight_store_tasks=0,
        inflight_lookup_tasks=0,
        inflight_load_tasks=0,
    )


def _physical(
    path: str, *, mode: str = "devdax", size_bytes: int = 64 * GIB
) -> DaxPhysicalStatus:
    return DaxPhysicalStatus(
        device_path=path,
        mode=mode,
        present=mode != "absent",
        major=249,
        minor=3,
        kernel_name="dax2.3",
        driver="device_dax" if mode == "devdax" else "",
        size_bytes=size_bytes,
        align_bytes=2 << 20,
        probed_at=NOW - 1.0,
        detail="",
    )


def _dax(
    devices: list[DaxDeviceStatus],
    present: list[DaxPhysicalStatus],
    *,
    enabled: bool = True,
    hotplug_enabled: bool = True,
) -> DaxHotplugStatus:
    watcher = (
        DaxWatcherStatus(
            enabled=True,
            directory=DIRECTORY,
            interval_seconds=1.0,
            last_scan_at=NOW - 0.5,
            present_devices=present,
        )
        if enabled
        else DaxWatcherStatus(enabled=False)
    )
    return DaxHotplugStatus(
        hotplug_enabled=hotplug_enabled,
        slot_bytes=SLOT,
        total_capacity_bytes=sum(d.slot_capacity_bytes for d in devices),
        total_used_bytes=0,
        devices=devices,
        watcher=watcher,
    )


def _sample(instance_id: str, worker_ip: str, port: int) -> InstanceSample:
    return InstanceSample(
        identity=InstanceIdentity(
            instance_id=instance_id,
            registration_time=1.0,
            endpoint=f"10.0.0.1:{port}",
            worker_ip=worker_ip,
        ),
        registered=True,
        declared_capacity=True,
        used_bytes=8 * GIB,
        capacity_bytes=128 * GIB,
        usage_ratio=0.0625,
        sampled_at=1.0,
    )


def _outside(**nodes: list[str]) -> OutsideStatus:
    return dict(nodes)


def _plan(
    dax: DaxHotplugStatus,
    outside: OutsideStatus,
    *,
    failures: dict[str, float] | None = None,
    config: MPMemoryCoordinatorConfig | None = None,
) -> AttachmentReport:
    return plan_attachments(
        {INSTANCE: _sample(INSTANCE, WORKER, 9000)},
        {INSTANCE: dax},
        outside,
        config or _config(),
        failures=failures or {},
        now=NOW,
    )


def _default_dax(*present: DaxPhysicalStatus) -> DaxHotplugStatus:
    return _dax(
        [_device(BOOT, 0), _device(ATTACHED, 1)],
        [_physical(BOOT), _physical(ATTACHED), *present],
    )


def test_present_owned_unattached_device_is_planned() -> None:
    report = _plan(_default_dax(_physical(SPARE)), _outside(**{WORKER: [SPARE]}))
    assert [p.device_path for p in report.planned] == [SPARE]
    plan = report.planned[0]
    assert plan.identity.instance_id == INSTANCE
    assert plan.identity.worker_ip == WORKER
    assert plan.size_bytes == 64 * GIB
    assert report.skipped == {BOOT: "already attached", ATTACHED: "already attached"}
    assert report.as_dict() == {
        "planned": [
            {
                "instance_id": INSTANCE,
                "worker_ip": WORKER,
                "device_path": SPARE,
                "size_bytes": 64 * GIB,
            }
        ],
        "skipped": {BOOT: "already attached", ATTACHED: "already attached"},
    }


@pytest.mark.parametrize(
    "mode", ["system-ram", "unbound", "not-a-device", "absent", "unknown"]
)
def test_non_devdax_modes_are_never_planned(mode: str) -> None:
    report = _plan(
        _default_dax(_physical(SPARE, mode=mode)), _outside(**{WORKER: [SPARE]})
    )
    assert report.planned == []
    assert report.skipped[SPARE] == f"mode is {mode}"


def test_path_outside_the_allowed_prefix_is_skipped() -> None:
    stray = "/dev/dax2.3"
    report = _plan(_default_dax(_physical(stray)), _outside(**{WORKER: [stray]}))
    assert report.planned == []
    assert report.skipped[stray] == "outside /dev/dax-cxl/"


@pytest.mark.parametrize("state", ["active", "draining"])
def test_a_live_entry_of_any_non_terminal_state_is_already_attached(
    state: str,
) -> None:
    dax = _dax(
        [_device(BOOT, 0), _device(SPARE, 1, state=state)],
        [_physical(BOOT), _physical(SPARE)],
    )
    report = _plan(dax, _outside(**{WORKER: [SPARE]}))
    assert report.planned == []
    assert report.skipped[SPARE] == "already attached"


def test_hotplug_disabled_adapter_is_never_planned() -> None:
    # An adapter that refuses hotplug would answer every add with 403; the
    # planner skips it instead of retrying once per cooldown forever.
    dax = _dax(
        [_device(BOOT, 0), _device(ATTACHED, 1)],
        [_physical(BOOT), _physical(ATTACHED), _physical(SPARE)],
        hotplug_enabled=False,
    )
    report = _plan(dax, _outside(**{WORKER: [SPARE]}))
    assert report.planned == []
    assert report.skipped[SPARE] == "hotplug disabled"
    assert report.skipped[ATTACHED] == "already attached"


def test_a_removed_tombstone_does_not_count_as_attached() -> None:
    # After a completed move the donor path stays present and its adapter
    # entry is a tombstone; it is re-attached only if the outside service
    # assigns it to this worker again.
    dax = _dax(
        [_device(BOOT, 0), _device(SPARE, 1, state="removed")],
        [_physical(BOOT), _physical(SPARE)],
    )
    assert _plan(dax, _outside(**{WORKER: []})).planned == []
    report = _plan(dax, _outside(**{WORKER: [SPARE]}))
    assert [p.device_path for p in report.planned] == [SPARE]


@pytest.mark.parametrize(
    "outside,expected_owners",
    [
        (_outside(**{WORKER: []}), "[]"),
        (_outside(**{WORKER: [], OTHER_WORKER: [SPARE]}), f"['{OTHER_WORKER}']"),
        (
            _outside(**{WORKER: [SPARE], OTHER_WORKER: [SPARE]}),
            f"['{WORKER}', '{OTHER_WORKER}']",
        ),
    ],
)
def test_unproven_ownership_is_never_attached(
    outside: OutsideStatus, expected_owners: str
) -> None:
    report = _plan(_default_dax(_physical(SPARE)), outside)
    assert report.planned == []
    assert report.skipped[SPARE] == (
        f"outside status lists the path under {expected_owners}, not [{WORKER}]"
    )


@pytest.mark.parametrize("size_bytes", [0, 64 * GIB + 1, GIB // 2])
def test_non_whole_gib_sizes_are_skipped(size_bytes: int) -> None:
    report = _plan(
        _default_dax(_physical(SPARE, size_bytes=size_bytes)),
        _outside(**{WORKER: [SPARE]}),
    )
    assert report.planned == []
    assert report.skipped[SPARE] == (
        f"size {size_bytes} is not a positive whole number of GiB"
    )


def test_recent_failure_is_skipped_until_the_cooldown_expires() -> None:
    config = _config(cooldown_seconds=30.0)
    outside = _outside(**{WORKER: [SPARE]})
    dax = _default_dax(_physical(SPARE))
    recent = _plan(dax, outside, failures={SPARE: NOW - 29.0}, config=config)
    assert recent.planned == []
    assert recent.skipped[SPARE] == "recent attach failure"
    expired = _plan(dax, outside, failures={SPARE: NOW - 30.0}, config=config)
    assert [p.device_path for p in expired.planned] == [SPARE]


def test_watcher_disabled_contributes_nothing() -> None:
    dax = _dax([_device(BOOT, 0)], [_physical(SPARE)], enabled=False)
    report = _plan(dax, _outside(**{WORKER: [SPARE]}))
    assert report.planned == [] and report.skipped == {}


def test_instance_without_a_dax_status_contributes_nothing() -> None:
    report = plan_attachments(
        {INSTANCE: _sample(INSTANCE, WORKER, 9000)},
        {},
        _outside(**{WORKER: [SPARE]}),
        _config(),
        failures={},
        now=NOW,
    )
    assert report.planned == [] and report.skipped == {}


def test_plans_are_ordered_by_instance_id_then_path() -> None:
    other_dir = "/dev/dax-cxl/ns_pod-b"
    other_spare = f"{other_dir}/dax0.1"
    samples = {
        OTHER_INSTANCE: _sample(OTHER_INSTANCE, OTHER_WORKER, 9001),
        INSTANCE: _sample(INSTANCE, WORKER, 9000),
    }
    statuses = {
        OTHER_INSTANCE: _dax(
            [_device(f"{other_dir}/dax0.0", 0)], [_physical(other_spare)]
        ),
        INSTANCE: _dax([_device(BOOT, 0)], [_physical(SPARE_B), _physical(SPARE)]),
    }
    outside = _outside(**{WORKER: [SPARE, SPARE_B], OTHER_WORKER: [other_spare]})
    report = plan_attachments(
        samples, statuses, outside, _config(), failures={}, now=NOW
    )
    assert [(p.identity.instance_id, p.device_path) for p in report.planned] == [
        (INSTANCE, SPARE),
        (INSTANCE, SPARE_B),
        (OTHER_INSTANCE, other_spare),
    ]
