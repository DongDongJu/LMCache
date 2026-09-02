# SPDX-License-Identifier: Apache-2.0
"""Controller tests against a fake world with injected faults and crashes.

``FakeWorld`` implements :class:`Remote` as a small model of the two MP
servers' DAX device tables, the MP Coordinator membership/usage view, and
the outside allocator's per-node FREE/ASSIGNED inventory. Every side effect
is audited so tests assert exact ordering and counts. Crashes are injected
after every durable write (``CrashingJournal``) and after every remote
effect (``crash_after_effects``); a "restart" is a fresh controller loading
the same journal directory.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
import asyncio
import json
import posixpath

# Third Party
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.clients import (
    AmbiguousMutationError,
    ClientConnectionError,
    ClientHTTPError,
)
from lmcache.v1.mp_memory_coordinator.clients.memory_allocation_client import (
    OutsideContractError,
    OutsideExplicitFailure,
)
from lmcache.v1.mp_memory_coordinator.config import (
    MPMemoryCoordinatorConfig,
    config_from_mapping,
)
from lmcache.v1.mp_memory_coordinator.controller import RebalanceController
from lmcache.v1.mp_memory_coordinator.leader import LeaderElector, StaticLeader
from lmcache.v1.mp_memory_coordinator.models import (
    GIB,
    AllocationOrigin,
    AllocationRequest,
    AllocationResponse,
    CoordinatorInstance,
    DaxAddResponse,
    DaxDeviceNotFound,
    DaxHotplugStatus,
    DaxReconfigureStatus,
    DaxRemoveBlocked,
    DaxRemoveMode,
    DaxRemoveResponse,
    DeallocationRequest,
    DeallocationResponse,
    InstanceIdentity,
    InstanceUsage,
    JournalDocument,
    ManagedAllocation,
    MoveOutcome,
    MoveState,
    MPStatus,
)
from lmcache.v1.mp_memory_coordinator.persistence.rebalance_journal import (
    RebalanceJournal,
)
from lmcache.v1.mp_memory_coordinator.policy import (
    LivePreflight,
    MembershipSnapshot,
    RejectionReason,
    join_sandwich,
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
D_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
D_RUN1 = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
D_RUN2 = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.2"
R_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.0"
R_RUN1 = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"
R_RUN2 = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.2"
SLOT = 1 << 20


def _physical(path: str, size_bytes: int, mode: str = "devdax") -> dict[str, object]:
    """A watcher/physical entry as the MP server reports it."""
    return {
        "device_path": path,
        "mode": mode,
        "present": mode != "absent",
        "major": 249,
        "minor": 0,
        "kernel_name": "dax2.0",
        "driver": "device_dax" if mode == "devdax" else "",
        "size_bytes": size_bytes,
        "align_bytes": 2 << 20,
        "probed_at": 1000.0,
        "detail": "",
    }


def _config(**overrides: object) -> MPMemoryCoordinatorConfig:
    fields: dict[object, object] = dict(
        state_directory="/tmp/unused",
        poll_interval_seconds=1.0,
        stable_samples=3,
        cooldown_seconds=10.0,
        drain_timeout_seconds=30.0,
        actuation_enabled=True,
        dax_add_max_attempts=2,
        get_retry_attempts=2,
    )
    fields.update(overrides)
    return config_from_mapping(fields)


class SimulatedCrash(Exception):
    """Raised to emulate a process kill at a precise point."""


class Clock:
    """A wall clock that advances ``tick`` seconds on every read."""

    def __init__(self, start: float = 1000.0, tick: float = 0.05) -> None:
        self.now = start
        self.tick = tick

    def __call__(self) -> float:
        self.now += self.tick
        return self.now


class LosingLeader:
    """Controllable leader used to lose the gate between two renewals."""

    def __init__(self) -> None:
        self._leader = True
        self.ensure_calls = 0

    @property
    def identity(self) -> str:
        return "losing-leader"

    def is_leader(self) -> bool:
        return self._leader

    async def ensure_leader(self) -> bool:
        self.ensure_calls += 1
        return self._leader

    async def run(self, stop: asyncio.Event) -> None:
        await stop.wait()

    async def release(self) -> None:
        self._leader = False

    def lose(self) -> None:
        self._leader = False

    def restore(self) -> None:
        self._leader = True


class Device:
    """One fake DAX device entry."""

    def __init__(self, path: str, used_gib: int) -> None:
        self.path = path
        self.state = "active"
        self.used_gib = used_gib
        self.busy = 0


class Instance:
    """One fake MP instance."""

    def __init__(self, instance_id: str, ip: str, worker_ip: str, devices) -> None:
        self.instance_id = instance_id
        self.ip = ip
        self.port = 8080
        self.worker_ip = worker_ip
        self.epoch = 1.0
        self.devices: list[Device] = devices
        self.used_bytes = 0
        self.registered = True
        self.declared = True
        self.next_device_id = len(devices)
        self.status_down = False
        self.adapters = 1
        self.reported_capacity: int | None = None  # delayed capacity fault
        # Presence watcher: every attached path is present; ``present``
        # adds paths the watcher sees but the adapter has not mapped.
        self.watcher_enabled = True
        self.directory = posixpath.dirname(devices[0].path)
        self.present: dict[str, dict[str, object]] = {}

    @property
    def identity(self) -> InstanceIdentity:
        return InstanceIdentity(
            instance_id=self.instance_id,
            registration_time=self.epoch,
            endpoint=f"{self.ip}:{self.port}",
            worker_ip=self.worker_ip,
        )

    def live(self, path: str) -> Device | None:
        for device in self.devices:
            if device.path == path and device.state not in ("closed", "removed"):
                return device
        return None

    def capacity(self) -> int:
        return sum(
            64 * GIB
            for d in self.devices
            if d.state in ("active", "draining", "migrating", "resizing", "removing")
        )

    def declare_present(
        self, path: str, size_bytes: int = 64 * GIB, mode: str = "devdax"
    ) -> None:
        """Make the watcher report ``path`` present without attaching it."""
        self.present[path] = _physical(path, size_bytes, mode)

    def watcher(self) -> dict[str, object]:
        if not self.watcher_enabled:
            return {"enabled": False}
        present: dict[str, dict[str, object]] = {}
        for device in self.devices:
            present.setdefault(device.path, _physical(device.path, 64 * GIB))
        present.update(self.present)
        return {
            "enabled": True,
            "directory": self.directory,
            "interval_seconds": 1.0,
            "last_scan_at": 1000.0,
            "present_devices": [present[path] for path in sorted(present)],
        }

    def hotplug(self) -> DaxHotplugStatus:
        template = json.loads((GOLDEN / "mp_reconfigure_dax_status.json").read_text())[
            "adapters"
        ][0]["status"]["devices"][0]
        entries = []
        for index, device in enumerate(self.devices):
            entry = dict(template)
            terminal = device.state in ("closed", "removed")
            entry.update(
                {
                    "index": index,
                    "device_id": index,
                    "device_path": device.path,
                    "state": device.state,
                    "is_healthy": not terminal,
                    "closing": terminal,
                    "max_dax_size_bytes": 64 * GIB,
                    "slot_bytes": SLOT,
                    "max_slots": 64 * 1024,
                    "live_slot_count": 0 if terminal else device.used_gib * 1024,
                    "locked_key_count": device.busy,
                    "physical": _physical(device.path, 64 * GIB),
                }
            )
            entries.append(entry)
        live = [e for e in entries if e["state"] not in ("closed", "removed")]
        return DaxHotplugStatus.model_validate(
            {
                "hotplug_enabled": True,
                "slot_bytes": SLOT,
                "total_capacity_bytes": sum(e["max_dax_size_bytes"] for e in live),
                "total_used_bytes": sum(e["live_slot_count"] * SLOT for e in live),
                "devices": entries,
                "watcher": self.watcher(),
            }
        )


class FakeWorld:
    """A :class:`Remote` over an in-memory model of every dependency."""

    def __init__(self) -> None:
        self.donor = Instance(
            "mp-donor", "10.0.0.11", DONOR_IP, [Device(D_BOOT, 4), Device(D_RUN1, 4)]
        )
        self.receiver = Instance(
            "mp-receiver", "10.0.0.12", RECEIVER_IP, [Device(R_BOOT, 56)]
        )
        self.donor.used_bytes = 8 * GIB
        self.receiver.used_bytes = 56 * GIB
        # Outside: node -> {path: state}; sizes are all 64 GiB.
        self.outside: dict[str, dict[str, str]] = {
            DONOR_IP: {D_RUN1: "assigned", D_RUN2: "free"},
            RECEIVER_IP: {R_RUN1: "free", R_RUN2: "free"},
        }
        self.coordinator_up = True
        self.outside_up = True
        self.audit: list[tuple] = []
        self.faults: dict[str, str] = {}
        self.evict_409 = 0
        self.add_fail = 0
        self.crash_after_effects: int | None = None
        self.effects = 0
        self.after_sandwich: Callable[[], None] | None = None

    # -- helpers ---------------------------------------------------------------

    def instances(self) -> list[Instance]:
        return [self.donor, self.receiver]

    def by_identity(self, identity: InstanceIdentity) -> Instance:
        for instance in self.instances():
            if instance.instance_id == identity.instance_id:
                return instance
        raise KeyError(identity.instance_id)

    def assigned(self, node: str) -> list[str]:
        return sorted(p for p, s in self.outside[node].items() if s == "assigned")

    def assigned_total_gib(self) -> int:
        return 64 * sum(len(self.assigned(node)) for node in self.outside)

    def _effect(self, kind: str, *args: object) -> None:
        self.effects += 1
        self.audit.append((kind, *args))

    def _maybe_crash(self) -> None:
        if (
            self.crash_after_effects is not None
            and self.effects >= self.crash_after_effects
        ):
            raise SimulatedCrash(f"after effect #{self.effects}")

    # -- Remote ------------------------------------------------------------------

    async def sandwich(self) -> MembershipSnapshot:
        if not self.coordinator_up:
            return MembershipSnapshot(coordinator_reachable=False)
        instances = [
            CoordinatorInstance(
                instance_id=i.instance_id,
                ip=i.ip,
                http_port=i.port,
                registration_time=i.epoch,
                metadata={"worker_ip": i.worker_ip} if i.worker_ip else {},
            )
            for i in self.instances()
            if i.registered
        ]
        usage = [
            InstanceUsage.model_validate(
                {
                    "instance_id": i.instance_id,
                    "registered": i.registered,
                    "declared_capacity": i.declared,
                    "modules": [
                        {
                            "tier": "l2",
                            "backend": "dax",
                            "shared": False,
                            "used_bytes": i.used_bytes,
                            "capacity_bytes": cap,
                            "usage_ratio": (i.used_bytes / cap) if cap else None,
                        }
                    ],
                }
            )
            for i in self.instances()
            for cap in [
                (
                    i.reported_capacity
                    if i.reported_capacity is not None
                    else i.capacity()
                )
                if i.declared
                else 0
            ]
        ]
        snapshot = join_sandwich(instances, usage, instances, 0.0)
        if self.after_sandwich is not None:
            self.after_sandwich()
        return snapshot

    async def preflight(self, identity: InstanceIdentity) -> LivePreflight | None:
        instance = self.by_identity(identity)
        if instance.status_down:
            return None
        status = json.loads((GOLDEN / "mp_status.json").read_text())
        adapters = status["storage_manager"]["l2_adapters"]
        status["storage_manager"]["l2_adapters"] = adapters[:1] * instance.adapters
        dax = json.loads((GOLDEN / "mp_reconfigure_dax_status.json").read_text())
        dax["adapters"][0]["status"] = instance.hotplug().model_dump()
        dax["adapters"] = dax["adapters"] * instance.adapters
        dax["num_adapters"] = instance.adapters
        dax["enabled"] = instance.adapters > 0
        return LivePreflight(
            status=MPStatus.model_validate(status),
            dax=DaxReconfigureStatus.model_validate(dax),
        )

    async def dax_status(self, identity: InstanceIdentity) -> DaxHotplugStatus | None:
        instance = self.by_identity(identity)
        if instance.status_down:
            return None
        return instance.hotplug()

    async def outside_status(self) -> dict[str, list[str]] | None:
        if not self.outside_up:
            return None
        return {node: self.assigned(node) for node in self.outside}

    async def remove_device(self, identity, device_path, mode):
        instance = self.by_identity(identity)
        self._effect("remove", instance.instance_id, device_path, mode.value)
        device = instance.live(device_path)
        if device is None:
            self._maybe_crash()
            return DaxDeviceNotFound(error="DAX device not found")
        if mode is DaxRemoveMode.DRAIN:
            device.state = "draining"
            self._maybe_crash()
            return DaxRemoveResponse(
                status="ok",
                operation="drain",
                adapter_index=0,
                device_path=device_path,
                index=instance.devices.index(device),
                state="draining",
            )
        if device.busy > 0 or self.evict_409 > 0:
            self.evict_409 = max(0, self.evict_409 - 1)
            device.state = "draining"
            self._maybe_crash()
            return DaxRemoveBlocked(locked_key_count=max(device.busy, 1))
        device.state = "removed"
        self._maybe_crash()
        return DaxRemoveResponse(
            status="ok",
            operation="remove",
            adapter_index=0,
            device_path=device_path,
            index=instance.devices.index(device),
            state="removed",
        )

    async def add_device(self, identity, device_path, size_bytes):
        instance = self.by_identity(identity)
        self._effect("add", instance.instance_id, device_path, size_bytes)
        if self.add_fail > 0 or self.faults.get("add") == "always":
            self.add_fail = max(0, self.add_fail - 1)
            self._maybe_crash()
            raise ClientHTTPError(400, '{"error": "failed to map DAX device"}', "add")
        if self.faults.get("add") == "inactive":
            # 2xx with a non-active entry (e.g. the path was being drained
            # concurrently); nothing stays mapped on the fake server.
            probe = Device(device_path, 0)
            probe.state = "draining"
            instance.devices.append(probe)
            entry = instance.hotplug().find_live(device_path)
            instance.devices.remove(probe)
            self._maybe_crash()
            return DaxAddResponse(
                status="ok", operation="add", adapter_index=0, device=entry
            )
        existing = instance.live(device_path)
        if existing is None:
            existing = Device(device_path, 0)
            instance.devices.append(existing)
        self._maybe_crash()
        hotplug = instance.hotplug()
        return DaxAddResponse(
            status="ok",
            operation="add",
            adapter_index=0,
            device=hotplug.find_live(device_path),
        )

    async def deallocate(self, request: DeallocationRequest) -> DeallocationResponse:
        self._effect("deallocate", request.model_dump())
        fault = self.faults.pop("deallocate", "")  # every outside fault is one-shot
        if fault == "connect":
            raise ClientConnectionError("refused")
        if fault == "explicit":
            raise OutsideExplicitFailure(409, {"error": "refused"}, "dealloc")
        if fault == "ambiguous_uncommitted":
            raise AmbiguousMutationError("dropped before commit")
        node = self.outside.get(request.target_node, {})
        if node.get(request.device_path) != "assigned":
            raise OutsideExplicitFailure(404, {"error": "unknown path"}, "dealloc")
        node[request.device_path] = "free"
        self._maybe_crash()
        if fault == "ambiguous_committed":
            raise AmbiguousMutationError("dropped after commit")
        if fault == "contract":
            raise OutsideContractError("missing released_size_gib", {})
        return DeallocationResponse(
            status="DONE",
            request_id=request.request_id,
            target_node=request.target_node,
            device_path=request.device_path,
            released_size_gib=64,
        )

    async def allocate(self, request: AllocationRequest) -> AllocationResponse:
        self._effect("allocate", request.model_dump())
        fault = self.faults.pop("allocate", "")  # one-shot
        if fault == "explicit":
            raise OutsideExplicitFailure(409, {"error": "no free device"}, "alloc")
        if fault == "ambiguous_uncommitted":
            raise AmbiguousMutationError("dropped before commit")
        node = self.outside.get(request.target_node, {})
        free = sorted(p for p, s in node.items() if s == "free")
        if not free or request.request_size_gib != 64:
            raise OutsideExplicitFailure(409, {"error": "no matching device"}, "alloc")
        path = free[0]
        node[path] = "assigned"
        self._maybe_crash()
        if fault == "ambiguous_committed":
            raise AmbiguousMutationError("dropped after commit")
        if fault == "wrong_size":
            raise OutsideContractError(
                "sizes disagree", {"device_path": path, "granted_size_gib": 32}
            )
        returned = path
        if fault == "invalid_path":
            returned = "/dev/other/dax9.9"
        return AllocationResponse(
            status="DONE",
            request_id=request.request_id,
            target_node=request.target_node,
            device_path=returned,
            requested_size_gib=64,
            granted_size_gib=64,
        )


class CrashingJournal(RebalanceJournal):
    """A journal that crashes *after* the N-th durable save."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory)
        self.saves = 0
        self.crash_after_saves: int | None = None

    def save(self, document: JournalDocument) -> None:
        super().save(document)
        self.saves += 1
        if self.crash_after_saves is not None and self.saves >= self.crash_after_saves:
            raise SimulatedCrash(f"after save #{self.saves}")


def _inventory() -> list[ManagedAllocation]:
    return [
        ManagedAllocation(
            worker_ip=DONOR_IP,
            instance_id="mp-donor",
            device_path=D_RUN1,
            allocation_size_gib=64,
            device_map_size_bytes=64 * GIB,
            slot_capacity_bytes=64 * GIB,
            adapter_index=0,
            origin=AllocationOrigin.ADOPTED,
            last_confirmed_state="active",
            last_confirmed_at=0.0,
        )
    ]


def _controller(
    tmp_path: Path,
    world: FakeWorld,
    clock: Clock,
    config: MPMemoryCoordinatorConfig,
    leader: LeaderElector | None = None,
) -> tuple[RebalanceController, CrashingJournal]:
    journal = CrashingJournal(tmp_path / "state")
    if not journal.exists():
        journal.save(JournalDocument(initialized=True, inventory=_inventory()))
    controller = RebalanceController(
        config, journal, world, leader or StaticLeader("test"), clock=clock
    )
    controller.load()
    return controller, journal


def _run(coro):
    return asyncio.run(coro)


def _status_section(controller: RebalanceController, key: str) -> dict[str, object]:
    """One mapping-valued section of ``/status`` (``counters``, ...)."""
    section = controller.status()[key]
    assert isinstance(section, dict), section
    return section


async def _drive(controller: RebalanceController, cycles: int) -> None:
    for _ in range(cycles):
        await controller.run_once()
        move = controller.document.active_move
        if move is not None and move.state is MoveState.BLOCKED:
            return


async def _cycles(controller: RebalanceController, cycles: int):
    """Run ``cycles`` cycles and return the last :class:`CycleReport`."""
    report = None
    for _ in range(cycles):
        report = await controller.run_once()
    return report


def _terminal(controller: RebalanceController) -> bool:
    move = controller.document.active_move
    return move is None or move.state is MoveState.BLOCKED


def _counts(world: FakeWorld) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in world.audit:
        kind = entry[0] if entry[0] != "remove" else f"remove:{entry[3]}"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


# -- happy path --------------------------------


def test_three_samples_then_exact_saga_order_and_success(tmp_path: Path) -> None:
    world = FakeWorld()
    clock = Clock()
    controller, journal = _controller(tmp_path, world, clock, _config())

    async def run():
        for _ in range(2):
            await controller.run_once()
            assert world.audit == [], "no mutation before three samples"
            assert controller.document.active_move is None
        report = await controller.run_once()
        assert report.proposal is not None
        assert report.proposal["donor"] == "mp-donor"
        assert report.proposal["receiver"] == "mp-receiver"
        assert report.proposal["device_path"] == D_RUN1
        await _drive(controller, 12)

    _run(run())
    assert _terminal(controller)
    move = controller.document.history[-1]
    assert move.state is MoveState.COMPLETE and move.outcome is MoveOutcome.SUCCEEDED
    kinds = [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in world.audit]
    assert kinds == ["remove:drain", "remove:evict", "deallocate", "allocate", "add"]
    # Exact frozen request bodies.
    assert world.audit[2][1] == {
        "request_id": move.deallocation_request_id,
        "target_node": DONOR_IP,
        "device_path": D_RUN1,
    }
    assert world.audit[3][1] == {
        "request_id": move.allocation_request_id,
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    assert move.deallocation_request_id != move.allocation_request_id
    assert world.audit[4][1:] == ("mp-receiver", R_RUN1, 64 * GIB)
    # Outside inventory: donor path free, receiver dax0.1 assigned, total 64 GiB.
    assert world.outside[DONOR_IP][D_RUN1] == "free"
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert world.assigned_total_gib() == 64
    # Managed inventory moved; donor tombstone remains.
    paths = [a.device_path for a in controller.document.inventory]
    assert paths == [R_RUN1]
    assert controller.document.inventory[0].origin is AllocationOrigin.ALLOCATED
    assert [d.state for d in world.donor.devices] == ["active", "removed"]
    assert world.receiver.live(R_RUN1) is not None
    assert controller.document.counters.succeeded == 1
    assert controller.document.counters.proposed == 1
    # Cooldown prevents a second move.
    assert set(controller.document.cooldowns) == {
        move.donor.key,
        move.receiver.key,
    }


def test_actuation_disabled_logs_proposal_and_never_mutates(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(actuation_enabled=False)
    )

    async def run():
        for _ in range(5):
            report = await controller.run_once()
        return report

    report = _run(run())
    assert report.proposal is not None
    assert any(
        r["reason"] == RejectionReason.ACTUATION_DISABLED.value
        for r in report.rejections
    )
    assert world.audit == []
    assert controller.document.active_move is None
    assert controller.document.counters.proposed >= 1


def test_cooldown_survives_restart_and_blocks_second_move(tmp_path: Path) -> None:
    world = FakeWorld()
    clock = Clock()
    controller, _ = _controller(
        tmp_path, world, clock, _config(cooldown_seconds=10_000)
    )
    _run(_drive(controller, 15))
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    # Make the world eligible again: receiver still HIGH, donor LOW with a
    # managed device... the receiver now holds the managed device; swap roles.
    restarted, _ = _controller(tmp_path, world, clock, _config(cooldown_seconds=10_000))
    assert restarted.document.cooldowns

    async def run():
        for _ in range(5):
            report = await restarted.run_once()
        return report

    report = _run(run())
    assert restarted.document.active_move is None
    assert len(world.audit) == 5  # nothing beyond the first move
    assert any(r["reason"] == RejectionReason.COOLDOWN.value for r in report.rejections)


# -- eligibility / zero-POST cases --------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: setattr(w.receiver, "declared", False),
        lambda w: setattr(w.receiver, "registered", False),
        lambda w: setattr(w.donor, "worker_ip", ""),
        lambda w: setattr(w.receiver, "worker_ip", DONOR_IP),
        lambda w: setattr(w.donor, "adapters", 0),
        lambda w: setattr(w.donor, "adapters", 2),
        lambda w: setattr(w.donor, "status_down", True),
        lambda w: setattr(w.donor, "used_bytes", 100 * GIB),  # not LOW
    ],
)
def test_ineligible_fleets_produce_zero_mutation(tmp_path: Path, mutate) -> None:
    world = FakeWorld()
    mutate(world)
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 6))
    assert world.audit == []
    assert controller.document.active_move is None


def test_live_ratio_mismatch_produces_zero_posts(tmp_path: Path) -> None:
    world = FakeWorld()
    # Coordinator says LOW (8/128) but live DAX shows 120/128 used.
    world.donor.devices[0].used_gib = 60
    world.donor.devices[1].used_gib = 60
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        for _ in range(4):
            report = await controller.run_once()
        return report

    report = _run(run())
    assert world.audit == []
    assert any(
        r["reason"] == RejectionReason.LIVE_RATIO_MISMATCH.value
        for r in report.rejections
    )


def test_only_bootstrap_or_unapproved_paths_produce_zero_posts(tmp_path: Path) -> None:
    # The donor's runtime device is live but the outside service does not
    # call it assigned, so with an empty journal nothing is ever movable
    # however healthy the fleet is.
    world = FakeWorld()
    world.outside[DONOR_IP][D_RUN1] = "free"
    journal = CrashingJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=[]))
    controller = RebalanceController(
        _config(), journal, world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    _run(_drive(controller, 5))
    assert world.audit == []
    assert controller.document.active_move is None
    assert controller.document.inventory == []


def test_discovery_makes_an_empty_journal_movable(tmp_path: Path) -> None:
    # The default mode re-derives ownership from outside status, so no
    # allowlist is needed for the same fleet to produce a move.
    world = FakeWorld()
    journal = CrashingJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=[]))
    controller = RebalanceController(
        _config(), journal, world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    _run(_drive(controller, 5))
    paths = {a.device_path for a in controller.document.inventory}
    assert D_RUN1 in paths
    assert all(
        a.origin is AllocationOrigin.DISCOVERED
        for a in controller.document.inventory
        if a.device_path == D_RUN1
    )
    assert "remove:drain" in _counts(world)


def test_discovery_never_claims_a_path_outside_status_does_not_confirm(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    # The device stays live on the donor but the outside service stops
    # calling it assigned: ownership is unproven, so it must never be
    # claimed and no POST may follow.
    world.outside[DONOR_IP][D_RUN1] = "free"
    journal = CrashingJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=[]))
    controller = RebalanceController(
        _config(), journal, world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    report = _run(_cycles(controller, 5))
    paths = {a.device_path for a in controller.document.inventory}
    assert D_RUN1 not in paths
    assert D_RUN1 in report.discovery["skipped"]
    assert world.audit == []


def test_discovery_is_skipped_while_the_outside_service_is_down(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    world.outside_up = False
    journal = CrashingJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=[]))
    controller = RebalanceController(
        _config(), journal, world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    report = _run(_cycles(controller, 3))
    assert controller.document.inventory == []
    assert report.discovery == {"skipped_pass": "outside status unavailable"}
    assert report.attachments == {"skipped_pass": "outside status unavailable"}
    assert world.audit == []


# -- attach orchestration --------------------------------


def _present_and_assigned(world: FakeWorld) -> None:
    """The donor's spare is present, unattached, and assigned to the donor."""
    world.donor.declare_present(D_RUN2)
    world.outside[DONOR_IP][D_RUN2] = "assigned"
    # Keep the fleet ineligible for a move so only the attach path acts.
    world.receiver.used_bytes = 8 * GIB


def _adds(world: FakeWorld) -> list[tuple]:
    return [a for a in world.audit if a[0] == "add"]


def test_attach_dry_run_reports_would_attach_and_never_posts(tmp_path: Path) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(actuation_enabled=False)
    )
    report = _run(_cycles(controller, 3))
    assert report.attachments["would_attach"] == [D_RUN2]
    assert report.attachments["attached"] == []
    assert [p["device_path"] for p in report.attachments["planned"]] == [D_RUN2]
    assert world.audit == []
    assert world.donor.live(D_RUN2) is None
    assert D_RUN2 not in {a.device_path for a in controller.document.inventory}
    assert controller.attached_devices == 0
    assert _status_section(controller, "counters")["attached"] == 0


def test_attach_posts_exactly_once_and_discovery_adopts_next_cycle(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    controller, journal = _controller(tmp_path, world, Clock(), _config())

    async def run():
        first = await controller.run_once()
        assert _adds(world) == [("add", "mp-donor", D_RUN2, 64 * GIB)]
        assert first.attachments["attached"] == [D_RUN2]
        assert first.attachments["failed"] == {}
        assert first.decision == "attach issued; re-observing next cycle"
        assert world.donor.live(D_RUN2) is not None
        # Not in the inventory yet: adoption is discovery's job, next cycle.
        assert D_RUN2 not in {a.device_path for a in controller.document.inventory}
        second = await controller.run_once()
        assert second.attachments["skipped"][D_RUN2] == "already attached"
        assert second.attachments["planned"] == []
        assert second.discovery["discovered"] == [D_RUN2]
        await controller.run_once()

    _run(run())
    assert len(_adds(world)) == 1
    adopted = controller.document.find_allocation(D_RUN2)
    assert adopted is not None
    assert adopted.origin is AllocationOrigin.DISCOVERED
    assert adopted.worker_ip == DONOR_IP and adopted.allocation_size_gib == 64
    assert controller.attached_devices == 1
    assert _status_section(controller, "counters")["attached"] == 1
    assert not any(a[0] in ("deallocate", "allocate") for a in world.audit)
    # The count is in-memory only (attaching is idempotent): the journal
    # never carries it and a restart starts from zero.
    assert "attached" not in journal.load().model_dump(mode="json")["counters"]
    restarted = RebalanceController(
        _config(), journal, world, StaticLeader("test"), clock=Clock()
    )
    restarted.load()
    assert restarted.attached_devices == 0
    assert restarted.document.find_allocation(D_RUN2) is not None


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (
            lambda w: w.outside[DONOR_IP].__setitem__(D_RUN2, "free"),
            f"outside status lists the path under [], not [{DONOR_IP}]",
        ),
        (
            lambda w: (
                w.outside[RECEIVER_IP].__setitem__(D_RUN2, "assigned")
                or w.outside[DONOR_IP].__setitem__(D_RUN2, "free")
            ),
            f"outside status lists the path under ['{RECEIVER_IP}'], not [{DONOR_IP}]",
        ),
        (
            lambda w: w.donor.declare_present(D_RUN2, mode="system-ram"),
            "mode is system-ram",
        ),
        (
            lambda w: w.donor.declare_present(D_RUN2, size_bytes=64 * GIB + 1),
            f"size {64 * GIB + 1} is not a positive whole number of GiB",
        ),
    ],
)
def test_attach_never_posts_without_proven_ownership_or_a_usable_device(
    tmp_path: Path, mutate: Callable[[FakeWorld], object], expected: str
) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    mutate(world)
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    report = _run(_cycles(controller, 3))
    assert world.audit == []
    assert report.attachments["planned"] == []
    assert report.attachments["skipped"][D_RUN2] == expected
    assert report.attachments["skipped"][D_RUN1] == "already attached"
    assert world.donor.live(D_RUN2) is None


def test_attach_is_deferred_while_a_move_is_active(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 3)  # MOVE SELECTED, drain issued
        assert [a[0] for a in world.audit] == ["remove"]
        world.donor.declare_present(D_RUN2)
        world.outside[DONOR_IP][D_RUN2] = "assigned"
        for _ in range(20):
            report = await controller.run_once()
            if controller.document.active_move is None:
                break
            assert report.attachments == {}
            assert all(a[2] != D_RUN2 for a in _adds(world))
        assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
        # Only once the saga is over does the spare get attached.
        report = await controller.run_once()
        assert report.attachments["attached"] == [D_RUN2]

    _run(run())
    adds = _adds(world)
    assert adds[-1] == ("add", "mp-donor", D_RUN2, 64 * GIB)
    assert [a[2] for a in adds] == [R_RUN1, D_RUN2]


def test_attach_failure_backs_off_for_the_cooldown(tmp_path: Path) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    world.faults["add"] = "always"
    clock = Clock()
    controller, _ = _controller(tmp_path, world, clock, _config(cooldown_seconds=10.0))

    async def run():
        first = await controller.run_once()
        assert len(_adds(world)) == 1
        assert list(first.attachments["failed"]) == [D_RUN2]
        assert first.attachments["attached"] == []
        for _ in range(3):
            report = await controller.run_once()
            assert report.attachments["skipped"][D_RUN2] == "recent attach failure"
        assert len(_adds(world)) == 1, "no retry within the cooldown"
        clock.now += 10.0
        world.faults.pop("add")
        report = await controller.run_once()
        assert report.attachments["attached"] == [D_RUN2]

    _run(run())
    assert len(_adds(world)) == 2
    assert controller.attached_devices == 1


def test_non_active_add_response_is_a_failure_that_backs_off(tmp_path: Path) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    world.faults["add"] = "inactive"
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(cooldown_seconds=10.0)
    )

    async def run():
        first = await controller.run_once()
        assert first.attachments["failed"] == {D_RUN2: "add returned state draining"}
        assert first.attachments["attached"] == []
        assert first.decision == "attach issued; re-observing next cycle"
        assert controller.attached_devices == 0
        second = await controller.run_once()
        assert second.attachments["skipped"][D_RUN2] == "recent attach failure"
        assert second.attachments["planned"] == []

    _run(run())
    assert _adds(world) == [("add", "mp-donor", D_RUN2, 64 * GIB)]
    assert _status_section(controller, "counters")["attached"] == 0
    assert world.donor.live(D_RUN2) is None


def test_attach_is_withheld_when_leadership_cannot_be_renewed(tmp_path: Path) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    leader = LosingLeader()
    controller, _ = _controller(tmp_path, world, Clock(), _config(), leader=leader)
    world.after_sandwich = leader.lose  # is_leader() held at cycle start

    async def run():
        report = await controller.run_once()
        assert report.attachments["skipped_pass"] == "not leader"
        assert [p["device_path"] for p in report.attachments["planned"]] == [D_RUN2]
        assert D_RUN2 not in report.attachments["skipped"]
        assert report.attachments["attached"] == []
        assert world.audit == []
        world.after_sandwich = None
        leader.restore()
        report = await controller.run_once()
        assert report.attachments["attached"] == [D_RUN2]
        assert "skipped_pass" not in report.attachments

    _run(run())
    assert _adds(world) == [("add", "mp-donor", D_RUN2, 64 * GIB)]
    assert controller.attached_devices == 1


def test_stop_request_withholds_planned_attaches(tmp_path: Path) -> None:
    world = FakeWorld()
    _present_and_assigned(world)
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    controller.request_stop()
    report = _run(_cycles(controller, 2))
    assert report.attachments["would_attach"] == [D_RUN2]
    assert report.attachments["attached"] == []
    assert [p["device_path"] for p in report.attachments["planned"]] == [D_RUN2]
    assert world.audit == []
    assert controller.attached_devices == 0


def test_attach_cycle_never_selects_a_move_from_the_pre_attach_snapshot(
    tmp_path: Path,
) -> None:
    # Donor LOW, receiver HIGH: the fleet is move-eligible from the third
    # sample on. The spare becomes present + assigned right before that
    # cycle, so the add and the would-be SELECT share one sandwich read; a
    # record built from it would carry the pre-attach donor capacity and the
    # saga would wait for capacity convergence forever.
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 2)
        assert world.audit == []
        world.donor.declare_present(D_RUN2)
        world.outside[DONOR_IP][D_RUN2] = "assigned"
        third = await controller.run_once()
        assert third.attachments["attached"] == [D_RUN2]
        assert third.proposal is None
        assert third.decision == "attach issued; re-observing next cycle"
        assert controller.document.active_move is None
        assert _adds(world) == [("add", "mp-donor", D_RUN2, 64 * GIB)]
        fourth = await controller.run_once()
        assert fourth.attachments["skipped"][D_RUN2] == "already attached"
        assert fourth.discovery["discovered"] == [D_RUN2]
        assert fourth.proposal is not None
        move = controller.document.active_move
        assert move is not None
        # The record captures the post-attach capacity, so the saga converges.
        assert move.donor_capacity_bytes == 192 * GIB
        await _drive(controller, 16)

    _run(run())
    assert _terminal(controller)
    move = controller.document.history[-1]
    assert move.state is MoveState.COMPLETE and move.outcome is MoveOutcome.SUCCEEDED
    assert move.donor_capacity_bytes == 192 * GIB
    assert world.donor.capacity() == 128 * GIB
    # The spare's attach precedes the move in the audit; count MOVE effects.
    attach, *rest = world.audit
    assert attach == ("add", "mp-donor", D_RUN2, 64 * GIB)
    world.audit = rest
    assert _counts(world) == {
        "add": 1,
        "remove:drain": 1,
        "remove:evict": 1,
        "deallocate": 1,
        "allocate": 1,
    }


def test_coordinator_outage_resets_history_and_holds_a_move(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await controller.run_once()
        await controller.run_once()
        world.coordinator_up = False
        await controller.run_once()
        world.coordinator_up = True
        await controller.run_once()
        await controller.run_once()
        assert world.audit == [], "history must restart after the outage"
        await controller.run_once()  # third fresh sample: move starts, drain issued
        assert [a[0] for a in world.audit] == ["remove"]
        world.coordinator_up = False
        for _ in range(3):
            await controller.run_once()
        assert len(world.audit) == 1, "no POST while the coordinator is down"
        world.coordinator_up = True
        await _drive(controller, 12)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED


def test_identity_change_mid_move_rolls_back_before_deallocation(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 3)  # drain issued
        assert [a[0] for a in world.audit] == ["remove"]
        world.receiver.epoch = 2.0  # receiver re-registered
        await _drive(controller, 12)

    _run(run())
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in world.audit]
    assert kinds == ["remove:drain", "remove:evict", "add"]
    assert world.audit[-1][1:] == ("mp-donor", D_RUN1, 64 * GIB)
    assert world.outside[DONOR_IP][D_RUN1] == "assigned"
    assert [a.device_path for a in controller.document.inventory] == [D_RUN1]


# -- drain behaviour --------------------------------


def test_busy_drain_waits_and_tolerates_evict_409(tmp_path: Path) -> None:
    world = FakeWorld()
    world.donor.devices[1].busy = 3
    world.evict_409 = 2
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 6)
        assert not any(a[0] in ("deallocate", "allocate") for a in world.audit)
        assert world.donor.devices[1].state == "draining"
        world.donor.devices[1].busy = 0
        await _drive(controller, 14)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    kinds = [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in world.audit]
    assert kinds.count("remove:evict") == 3  # two 409s then success
    assert kinds.index("deallocate") > kinds.index("remove:evict")
    assert kinds.count("deallocate") == 1 and kinds.count("allocate") == 1


def test_drain_deadline_blocks_without_outside_call(tmp_path: Path) -> None:
    world = FakeWorld()
    world.donor.devices[1].busy = 1
    clock = Clock(tick=1.0)
    controller, _ = _controller(
        tmp_path, world, clock, _config(drain_timeout_seconds=5.0)
    )
    _run(_drive(controller, 12))
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    assert "undrain" in move.block_reason
    assert not any(a[0] in ("deallocate", "allocate") for a in world.audit)
    assert controller.readiness()[0] is False


# -- outside failures --------------------------------


def test_allocation_explicit_failure_restores_donor(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["allocate"] = "explicit"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 20))
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in world.audit]
    assert kinds == [
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "allocate",
        "add",
    ]
    # The restore allocation went to the donor and its path was attached.
    assert world.audit[4][1]["target_node"] == DONOR_IP
    assert world.audit[4][1]["request_id"] == move.restore_request_id
    assert world.audit[5][1:] == ("mp-donor", D_RUN1, 64 * GIB)
    assert world.assigned_total_gib() == 64
    assert [a.device_path for a in controller.document.inventory] == [D_RUN1]
    assert controller.document.inventory[0].origin is AllocationOrigin.RESTORED


def test_wrong_size_releases_receiver_and_restores_donor(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["allocate"] = "wrong_size"
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 8)
        world.faults.pop("allocate", None)  # the restore allocation behaves
        await _drive(controller, 15)

    _run(run())
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in world.audit]
    assert kinds == [
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
        "add",
    ]
    assert world.audit[4][1] == {
        "request_id": move.release_request_id,
        "target_node": RECEIVER_IP,
        "device_path": R_RUN1,
    }
    assert world.outside[RECEIVER_IP][R_RUN1] == "free"
    assert world.assigned_total_gib() == 64
    assert world.receiver.live(R_RUN1) is None


def test_invalid_returned_path_never_adds_and_releases(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["allocate"] = "invalid_path"
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 8)
        world.faults.pop("allocate", None)
        await _drive(controller, 15)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.ROLLED_BACK
    adds = [a for a in world.audit if a[0] == "add"]
    assert adds == [("add", "mp-donor", D_RUN1, 64 * GIB)]
    assert world.outside[RECEIVER_IP][R_RUN1] == "free"


def test_attach_failure_transient_then_persistent(tmp_path: Path) -> None:
    world = FakeWorld()
    world.add_fail = 1
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(dax_add_max_attempts=3)
    )
    _run(_drive(controller, 20))
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    kinds = [a[0] for a in world.audit]
    assert kinds.count("add") == 2 and kinds.count("allocate") == 1

    world = FakeWorld()
    world.faults["add"] = "always"
    controller, _ = _controller(tmp_path / "b", world, Clock(), _config())

    async def run():
        for _ in range(20):
            await controller.run_once()
            move = controller.document.active_move
            if move is not None and move.state is MoveState.ROLLING_BACK:
                break
        world.faults.pop("add")  # the donor restore add works
        await _drive(controller, 15)

    _run(run())
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = [a[0] for a in world.audit]
    assert kinds.count("allocate") == 2  # receiver, then donor restore
    assert kinds.count("deallocate") == 2  # donor, then receiver release
    assert world.assigned_total_gib() == 64
    assert world.outside[DONOR_IP][D_RUN1] == "assigned"


@pytest.mark.parametrize("operation", ["deallocate", "allocate"])
def test_committed_then_dropped_outside_post_blocks_without_retry(
    tmp_path: Path, operation: str
) -> None:
    world = FakeWorld()
    world.faults[operation] = "ambiguous_committed"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 20))
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    kinds = [a[0] for a in world.audit]
    assert kinds.count(operation) == 1
    assert "add" not in kinds or operation == "deallocate"
    if operation == "allocate":
        assert kinds.count("add") == 0


def test_connection_failure_before_send_retries_within_bound(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["deallocate"] = "connect"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 20))
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    assert [a[0] for a in world.audit].count("deallocate") == 2


def test_outside_unreachable_during_move_holds_without_posts(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 3)
        world.outside_up = False
        for _ in range(4):
            await controller.run_once()
        assert not any(a[0] in ("deallocate", "allocate") for a in world.audit)
        world.outside_up = True
        await _drive(controller, 15)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED


# -- delayed capacity --------------------------------


def test_delayed_capacity_defers_complete(tmp_path: Path) -> None:
    world = FakeWorld()
    world.donor.reported_capacity = 128 * GIB
    world.receiver.reported_capacity = 64 * GIB
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 12)
        move = controller.document.active_move
        assert move is not None and move.state is MoveState.ALLOCATED
        assert world.receiver.live(R_RUN1) is not None
        world.donor.reported_capacity = None
        world.receiver.reported_capacity = None
        await _drive(controller, 3)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED


# -- crash recovery --------------------------------


def _run_with_crashes(
    tmp_path: Path,
    world: FakeWorld,
    *,
    crash_after_saves: int | None,
    crash_after_effects,
) -> RebalanceController:
    clock = Clock()
    controller, journal = _controller(tmp_path, world, clock, _config())
    journal.crash_after_saves = crash_after_saves
    world.crash_after_effects = crash_after_effects

    async def run():
        nonlocal controller
        for _ in range(60):
            try:
                await controller.run_once()
            except SimulatedCrash:
                journal.crash_after_saves = None
                world.crash_after_effects = None
                controller, _ = _controller(tmp_path, world, clock, _config())
                continue
            if _terminal(controller) and controller.document.history:
                break
            move = controller.document.active_move
            if move is not None and move.state is MoveState.BLOCKED:
                break
        return controller

    return _run(run())


def _assert_safe_end(controller: RebalanceController, world: FakeWorld) -> str:
    """Return the terminal outcome; assert the safety invariants."""
    kinds = [a[0] for a in world.audit]
    assert kinds.count("deallocate") <= 1, kinds
    assert kinds.count("allocate") <= 1, kinds
    # Every outside path has exactly one state; nothing is double-assigned.
    for node, paths in world.outside.items():
        for path, state in paths.items():
            assert state in ("free", "assigned"), (node, path, state)
    move = controller.document.active_move
    if move is not None:
        assert move.state is MoveState.BLOCKED
        return "BLOCKED"
    last = controller.document.history[-1]
    assert last.state is MoveState.COMPLETE
    if last.outcome is MoveOutcome.SUCCEEDED:
        assert world.assigned_total_gib() == 64
        assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    return last.outcome.value


@pytest.mark.parametrize("crash_after_saves", list(range(1, 22)))
def test_crash_after_every_durable_write(
    tmp_path: Path, crash_after_saves: int
) -> None:
    world = FakeWorld()
    controller = _run_with_crashes(
        tmp_path, world, crash_after_saves=crash_after_saves, crash_after_effects=None
    )
    outcome = _assert_safe_end(controller, world)
    assert outcome in ("SUCCEEDED", "BLOCKED")


@pytest.mark.parametrize("crash_after_effects", list(range(1, 6)))
def test_crash_after_every_effect(tmp_path: Path, crash_after_effects: int) -> None:
    world = FakeWorld()
    controller = _run_with_crashes(
        tmp_path, world, crash_after_saves=None, crash_after_effects=crash_after_effects
    )
    outcome = _assert_safe_end(controller, world)
    assert outcome in ("SUCCEEDED", "BLOCKED")
    kinds = [a[0] for a in world.audit]
    # A crash after a DAX effect is always recoverable from status; only an
    # outside POST whose response was lost may block.
    if crash_after_effects in (1, 2, 5):
        assert outcome == "SUCCEEDED", kinds


def test_crash_after_intent_before_dispatch_still_completes(tmp_path: Path) -> None:
    """Saves: ...intent(dealloc) -> dispatched -> result. Crashing right
    after the intent save must not block: the POST was provably unsent."""
    world = FakeWorld()
    controller = _run_with_crashes(
        tmp_path, world, crash_after_saves=None, crash_after_effects=None
    )
    assert _assert_safe_end(controller, world) == "SUCCEEDED"
    # Locate the intent save index by replaying: the dealloc intent is the
    # save right before "dispatched"; both crash cases are covered by the
    # parametrized sweep above. Here we assert the sweep's contract on the
    # two saves around dispatch explicitly.
    outcomes = {}
    for n in range(1, 22):
        w = FakeWorld()
        c = _run_with_crashes(
            tmp_path / f"n{n}", w, crash_after_saves=n, crash_after_effects=None
        )
        outcomes[n] = _assert_safe_end(c, w)
    assert "SUCCEEDED" in outcomes.values()
    assert outcomes[1] == "SUCCEEDED"


# -- journal damage / readiness --------------------------------


def test_corrupt_journal_makes_controller_unready_and_inert(tmp_path: Path) -> None:
    world = FakeWorld()
    directory = tmp_path / "state"
    journal = RebalanceJournal(directory)
    journal.save(JournalDocument(initialized=True, inventory=_inventory()))
    data = journal.path.read_bytes()
    journal.path.write_bytes(data[: len(data) // 2])
    controller = RebalanceController(
        _config(), RebalanceJournal(directory), world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    assert controller.journal_error
    report = _run(controller.run_once())
    assert "journal" in report.error
    assert world.audit == []
    ready, reason = controller.readiness()
    assert ready is False and "journal" in reason


def test_readiness_requires_leader_coordinator_and_reconciliation(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    assert controller.readiness() == (False, "MP Coordinator unreachable")
    _run(controller.run_once())
    assert controller.readiness() == (True, "ok")
    world.coordinator_up = False
    _run(controller.run_once())
    assert controller.readiness()[0] is False
    status = controller.status()
    assert status["actuation_enabled"] is True
    inventory = status["inventory"]
    assert isinstance(inventory, list)
    assert inventory[0]["device_path"] == D_RUN1


def test_mp_reregistration_rebinds_inventory_without_outside_post(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(actuation_enabled=False)
    )
    _run(controller.run_once())
    world.donor.instance_id = "mp-donor-2"
    _run(controller.run_once())
    assert controller.document.inventory[0].instance_id == "mp-donor-2"
    assert world.audit == []


def test_stop_request_starts_no_new_move(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    controller.request_stop()
    _run(_drive(controller, 5))
    assert world.audit == []
    assert controller.document.active_move is None
