# SPDX-License-Identifier: Apache-2.0
"""Controller tests against a fake world with injected faults and crashes.

``FakeWorld`` implements :class:`Remote` as a small model of the two MP
servers' DAX device tables, the MP Coordinator membership/usage view, and
the outside allocator's per-node FREE/ASSIGNED inventory behind a global
pool budget. Every side effect is audited so tests assert exact ordering and
counts. Crashes are injected after every durable write (``CrashingJournal``)
and after every remote effect (``crash_after_effects``); a "restart" is a
fresh controller loading the same journal directory.

Grow before move: the coordinator first asks the allocator for new receiver
capacity. The default ``pool_budget_gib`` equals the initially assigned
total (64 GiB), so in every MOVE scenario that GROW probe is refused with an
explicit failure (``NOT_SERVED``, nothing changed) and the MOVE saga then
runs exactly as it always did; ``_move_audit`` strips that leading refused
probe so MOVE assertions read unchanged. GROW scenarios raise the budget.
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
    NO_DONOR,
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
    EffectFailure,
    EffectName,
    InstanceIdentity,
    InstanceUsage,
    JournalDocument,
    ManagedAllocation,
    MoveKind,
    MoveOutcome,
    MoveRecord,
    MoveState,
    MPStatus,
    RollbackStep,
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
from lmcache.v1.mp_memory_coordinator.recovery import GROW_MAX_RECEIVER_REBINDS

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
    for key, value in overrides.items():
        fields[key] = value
    return config_from_mapping(fields)


class SimulatedCrash(BaseException):
    """Raised to emulate a process kill at a precise point.

    A ``BaseException`` on purpose: the controller reports any ``Exception``
    escaping a cycle instead of propagating it (readiness must drop), while
    a kill is not something a cycle can report.
    """


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
        # Further MP instances a test may register (extra receivers).
        self.others: list[Instance] = []
        # Outside: node -> {path: state}; sizes are all 64 GiB.
        self.outside: dict[str, dict[str, str]] = {
            DONOR_IP: {D_RUN1: "assigned", D_RUN2: "free"},
            RECEIVER_IP: {R_RUN1: "free", R_RUN2: "free"},
        }
        # Global pool admission: an allocation is refused (explicit 409,
        # nothing changes) when it would push the assigned total above this.
        # The default equals the initially assigned total, so the pool is
        # exhausted until a deallocation frees room: every MOVE scenario sees
        # the GROW probe refused and then runs the donor saga unchanged.
        self.pool_budget_gib = 64
        # Request ids of allocations that assigned a path (served).
        self.served_allocations: list[str] = []
        # Request ids of outside POSTs that reached the allocator (anything
        # past a connect failure), whatever it then answered.
        self.delivered_posts: list[str] = []
        self.coordinator_up = True
        self.outside_up = True
        self.audit: list[tuple] = []
        self.faults: dict[str, str] = {}
        self.allocate_connect_failures = 0
        self.evict_409 = 0
        self.add_fail = 0
        self.crash_after_effects: int | None = None
        self.effects = 0
        self.after_sandwich: Callable[[], None] | None = None

    # -- helpers ---------------------------------------------------------------

    def instances(self) -> list[Instance]:
        return [self.donor, self.receiver, *self.others]

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

    def find(self, identity: InstanceIdentity) -> Instance | None:
        """The instance behind ``identity``; ``None`` once it is gone."""
        for instance in self.instances():
            if instance.instance_id == identity.instance_id:
                return instance
        return None

    async def preflight(self, identity: InstanceIdentity) -> LivePreflight | None:
        instance = self.find(identity)
        if instance is None or instance.status_down:
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
        instance = self.find(identity)
        if instance is None or instance.status_down:
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
        self.delivered_posts.append(request.request_id)
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
        # Pool admission precedes injected faults, so a one-shot fault always
        # hits a request the pool would serve (a MOVE's allocation, or a
        # GROW's once the budget is raised), never the refused probe.
        if self.assigned_total_gib() + request.request_size_gib > self.pool_budget_gib:
            self.delivered_posts.append(request.request_id)
            self._maybe_crash()
            raise OutsideExplicitFailure(409, {"error": "pool exhausted"}, "alloc")
        if self.allocate_connect_failures > 0:
            self.allocate_connect_failures -= 1
            raise ClientConnectionError("refused")
        fault = self.faults.pop("allocate", "")  # one-shot
        if fault == "connect":
            raise ClientConnectionError("refused")
        self.delivered_posts.append(request.request_id)
        if fault == "explicit":
            raise OutsideExplicitFailure(409, {"error": "no free device"}, "alloc")
        if fault == "ambiguous_uncommitted":
            raise AmbiguousMutationError("dropped before commit")
        if fault == "contract":
            raise OutsideContractError("missing device_path", {})
        node = self.outside.get(request.target_node, {})
        free = sorted(p for p, s in node.items() if s == "free")
        if not free or request.request_size_gib != 64:
            raise OutsideExplicitFailure(409, {"error": "no matching device"}, "alloc")
        path = free[0]
        node[path] = "assigned"
        self.served_allocations.append(request.request_id)
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


def _kinds(entries: list[tuple]) -> list[str]:
    return [a[0] if a[0] != "remove" else f"remove:{a[3]}" for a in entries]


def _counts(world: FakeWorld) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in _kinds(_move_audit(world)):
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _grow_probe(world: FakeWorld) -> tuple:
    """Return the refused GROW probe that leads the audit in a MOVE scenario."""
    assert world.audit, "no GROW probe was issued"
    probe = world.audit[0]
    assert probe[0] == "allocate", probe
    assert probe[1]["target_node"] == RECEIVER_IP and probe[1]["request_size_gib"] == 64
    assert probe[1]["request_id"] not in world.served_allocations, "probe was served"
    assert probe[1]["request_id"].startswith("grow-")
    return probe


def _move_audit(world: FakeWorld) -> list[tuple]:
    """The audit after the refused GROW probe: the MOVE saga's own effects."""
    _grow_probe(world)
    return world.audit[1:]


def _move_history(controller: RebalanceController) -> list[MoveRecord]:
    return [m for m in controller.document.history if m.kind is MoveKind.MOVE]


def _assert_probe_not_served(controller: RebalanceController) -> MoveRecord:
    """The first archived saga is the refused GROW; nothing changed for it."""
    probe = controller.document.history[0]
    assert probe.kind is MoveKind.GROW
    assert probe.state is MoveState.COMPLETE
    assert probe.outcome is MoveOutcome.NOT_SERVED
    assert probe.donor == NO_DONOR
    ledger = probe.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.failure is EffectFailure.EXPLICIT
    assert ledger.error == "explicit failure 409"
    assert list(probe.effects) == ["allocate"]
    assert probe.new_path == ""
    return probe


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
        # Grow before move: the third sample proposes a GROW for the HIGH
        # receiver; the exhausted pool refuses it (NOT_SERVED) and the very
        # next proposal is the MOVE.
        report = await controller.run_once()
        assert report.proposal is not None
        assert report.proposal["kind"] == "grow"
        assert report.proposal["receiver"] == "mp-receiver"
        await controller.run_once()
        _assert_probe_not_served(controller)
        assert controller.document.cooldowns == {}
        report = await controller.run_once()
        assert report.proposal is not None
        assert report.proposal["kind"] == "move"
        assert report.proposal["donor"] == "mp-donor"
        assert report.proposal["receiver"] == "mp-receiver"
        assert report.proposal["device_path"] == D_RUN1
        assert any(
            r["reason"] == RejectionReason.GROW_BACKOFF.value for r in report.rejections
        )
        await _drive(controller, 12)

    _run(run())
    assert _terminal(controller)
    move = controller.document.history[-1]
    assert move.kind is MoveKind.MOVE
    assert move.state is MoveState.COMPLETE and move.outcome is MoveOutcome.SUCCEEDED
    audit = _move_audit(world)
    kinds = _kinds(audit)
    assert kinds == ["remove:drain", "remove:evict", "deallocate", "allocate", "add"]
    # Exact frozen request bodies.
    assert audit[2][1] == {
        "request_id": move.deallocation_request_id,
        "target_node": DONOR_IP,
        "device_path": D_RUN1,
    }
    assert audit[3][1] == {
        "request_id": move.allocation_request_id,
        "target_node": RECEIVER_IP,
        "request_size_gib": 64,
        "mode": "devdax",
        "purpose": "lmcache-dax",
        "access": "exclusive",
    }
    assert move.deallocation_request_id != move.allocation_request_id
    assert audit[4][1:] == ("mp-receiver", R_RUN1, 64 * GIB)
    assert world.served_allocations == [move.allocation_request_id]
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
    assert controller.document.counters.proposed == 2  # the GROW, then the MOVE
    assert controller.document.counters.not_served == 1
    assert controller.document.counters.grown == 0
    assert [m.outcome for m in controller.document.history] == [
        MoveOutcome.NOT_SERVED,
        MoveOutcome.SUCCEEDED,
    ]
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
    # A dry run proposes the GROW every cycle: nothing is probed, so the
    # MOVE alternative is never reached.
    assert report.proposal["kind"] == "grow"
    assert any(
        r["reason"] == RejectionReason.ACTUATION_DISABLED.value
        for r in report.rejections
    )
    assert world.audit == []
    assert controller.document.active_move is None
    assert controller.document.grow_backoffs == {}
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
    assert len(_move_audit(world)) == 5  # nothing beyond the first move
    assert any(r["reason"] == RejectionReason.COOLDOWN.value for r in report.rejections)


# -- eligibility / zero-POST cases --------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: setattr(w.receiver, "declared", False),
        lambda w: setattr(w.receiver, "registered", False),
        lambda w: setattr(w.receiver, "worker_ip", DONOR_IP),
    ],
)
def test_ineligible_receivers_produce_zero_posts(tmp_path: Path, mutate) -> None:
    world = FakeWorld()
    mutate(world)
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 6))
    assert world.audit == []
    assert controller.document.active_move is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda w: setattr(w.donor, "worker_ip", ""),
        lambda w: setattr(w.donor, "adapters", 0),
        lambda w: setattr(w.donor, "adapters", 2),
        lambda w: setattr(w.donor, "status_down", True),
        lambda w: setattr(w.donor, "used_bytes", 100 * GIB),  # not LOW
    ],
)
def test_ineligible_donors_produce_only_the_refused_grow_probe(
    tmp_path: Path, mutate
) -> None:
    # The receiver is still stable HIGH, so the coordinator asks the pool for
    # new capacity first; the exhausted pool refuses, nothing changes, and
    # without a donor no MOVE follows.
    world = FakeWorld()
    mutate(world)
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 6))
    assert _move_audit(world) == []
    assert world.served_allocations == []
    assert world.assigned_total_gib() == 64
    assert controller.document.active_move is None
    _assert_probe_not_served(controller)
    assert controller.document.cooldowns == {}


def test_live_ratio_mismatch_produces_zero_posts(tmp_path: Path) -> None:
    world = FakeWorld()
    # Coordinator says LOW (8/128) but live DAX shows 120/128 used.
    world.donor.devices[0].used_gib = 60
    world.donor.devices[1].used_gib = 60
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        for _ in range(6):  # two unstable, the refused GROW, then the MOVE pass
            report = await controller.run_once()
        return report

    report = _run(run())
    assert _move_audit(world) == []
    assert any(
        r["reason"] == RejectionReason.LIVE_RATIO_MISMATCH.value
        for r in report.rejections
    )


def test_only_bootstrap_or_unapproved_paths_produce_zero_posts(tmp_path: Path) -> None:
    # The donor's runtime device is live but the outside service does not
    # call it assigned, so with an empty journal nothing is ever movable
    # however healthy the fleet is. The pool is exhausted (nothing is
    # assigned, so the budget is zero) so a GROW cannot be served either.
    world = FakeWorld()
    world.outside[DONOR_IP][D_RUN1] = "free"
    world.pool_budget_gib = 0
    journal = CrashingJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=[]))
    controller = RebalanceController(
        _config(), journal, world, StaticLeader("t"), clock=Clock()
    )
    controller.load()
    _run(_drive(controller, 5))
    assert _move_audit(world) == []
    assert world.served_allocations == []
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
    _run(_drive(controller, 7))
    paths = {a.device_path for a in controller.document.inventory}
    assert D_RUN1 in paths
    assert all(
        a.origin is AllocationOrigin.DISCOVERED
        for a in controller.document.inventory
        if a.device_path == D_RUN1
    )
    assert "remove:drain" in _kinds(_move_audit(world))


def test_discovery_never_claims_a_path_outside_status_does_not_confirm(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    # The device stays live on the donor but the outside service stops
    # calling it assigned: ownership is unproven, so it must never be
    # claimed and no MOVE POST may follow. The pool is exhausted (budget
    # zero) so the receiver's GROW probe is refused without effect.
    world.outside[DONOR_IP][D_RUN1] = "free"
    world.pool_budget_gib = 0
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
    assert _move_audit(world) == []
    assert world.served_allocations == []


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
        await _drive(controller, 5)  # GROW refused, MOVE SELECTED, drain issued
        assert [a[0] for a in _move_audit(world)] == ["remove"]
        world.donor.declare_present(D_RUN2)
        world.outside[DONOR_IP][D_RUN2] = "assigned"
        world.pool_budget_gib = 128  # the pool admits what it already assigned
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
        world.pool_budget_gib = 128  # the pool admits what it already assigned
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
        # Grow before move: the first proposal is the receiver's GROW, which
        # the exhausted pool refuses; the MOVE follows two cycles later.
        assert fourth.proposal["kind"] == "grow"
        await controller.run_once()
        _assert_probe_not_served(controller)
        sixth = await controller.run_once()
        assert sixth.proposal is not None and sixth.proposal["kind"] == "move"
        move = controller.document.active_move
        assert move is not None and move.kind is MoveKind.MOVE
        # The record captures the post-attach capacity, so the saga converges.
        assert move.donor_capacity_bytes == 192 * GIB
        await _drive(controller, 16)

    _run(run())
    assert _terminal(controller)
    move = controller.document.history[-1]
    assert move.state is MoveState.COMPLETE and move.outcome is MoveOutcome.SUCCEEDED
    assert move.donor_capacity_bytes == 192 * GIB
    assert world.donor.capacity() == 128 * GIB
    # The spare's attach precedes the probe in the audit; count MOVE effects.
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
        await controller.run_once()  # third fresh sample: the GROW probe
        await controller.run_once()  # refused -> NOT_SERVED
        await controller.run_once()  # move starts, drain issued
        assert [a[0] for a in _move_audit(world)] == ["remove"]
        world.coordinator_up = False
        for _ in range(3):
            await controller.run_once()
        assert len(_move_audit(world)) == 1, "no POST while the coordinator is down"
        world.coordinator_up = True
        await _drive(controller, 12)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED


def test_lost_leadership_after_sandwich_does_not_dispatch_effect(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    leader = LosingLeader()
    controller, journal = _controller(
        tmp_path, world, Clock(), _config(), leader=leader
    )

    def lose_once_a_move_has_been_selected() -> None:
        move = controller.document.active_move
        if move is not None and move.kind is MoveKind.MOVE:
            leader.lose()

    world.after_sandwich = lose_once_a_move_has_been_selected
    _run(_drive(controller, 5))

    # The refused GROW probe took three renewals (select, then two around
    # its POST); the MOVE's select and pre-POST renewals make six.
    assert leader.ensure_calls == 6
    assert _move_audit(world) == []
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.DONOR_DRAINING
    drain = move.effect(EffectName.DONOR_DRAIN)
    assert drain is not None
    assert drain.attempts == 0
    assert drain.dispatched is False
    assert "lost leadership after identity check" in move.last_error

    persisted = journal.load().active_move
    assert persisted is not None
    persisted_drain = persisted.effect(EffectName.DONOR_DRAIN)
    assert persisted_drain is not None
    assert persisted_drain.attempts == 0
    assert persisted_drain.dispatched is False

    world.after_sandwich = None
    leader.restore()
    _run(_drive(controller, 16))

    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    assert _counts(world) == {
        "remove:drain": 1,
        "remove:evict": 1,
        "deallocate": 1,
        "allocate": 1,
        "add": 1,
    }


def test_identity_change_mid_move_rolls_back_before_deallocation(
    tmp_path: Path,
) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 5)  # GROW refused, then the drain issued
        assert [a[0] for a in _move_audit(world)] == ["remove"]
        world.receiver.epoch = 2.0  # receiver re-registered
        await _drive(controller, 12)

    _run(run())
    move = controller.document.history[-1]
    assert move.kind is MoveKind.MOVE
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = _kinds(_move_audit(world))
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
        await _drive(controller, 8)
        assert not any(a[0] in ("deallocate", "allocate") for a in _move_audit(world))
        assert world.donor.devices[1].state == "draining"
        world.donor.devices[1].busy = 0
        await _drive(controller, 14)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    kinds = _kinds(_move_audit(world))
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
    _run(_drive(controller, 14))
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    assert "undrain" in move.block_reason
    assert not any(a[0] in ("deallocate", "allocate") for a in _move_audit(world))
    assert controller.readiness()[0] is False


# -- outside failures --------------------------------


def test_allocation_explicit_failure_restores_donor(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["allocate"] = "explicit"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 22))
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    audit = _move_audit(world)
    assert _kinds(audit) == [
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "allocate",
        "add",
    ]
    # The restore allocation went to the donor and its path was attached.
    assert audit[4][1]["target_node"] == DONOR_IP
    assert audit[4][1]["request_id"] == move.restore_request_id
    assert audit[5][1:] == ("mp-donor", D_RUN1, 64 * GIB)
    assert world.assigned_total_gib() == 64
    assert [a.device_path for a in controller.document.inventory] == [D_RUN1]
    assert controller.document.inventory[0].origin is AllocationOrigin.RESTORED


def test_wrong_size_releases_receiver_and_restores_donor(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["allocate"] = "wrong_size"
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 10)
        world.faults.pop("allocate", None)  # the restore allocation behaves
        await _drive(controller, 15)

    _run(run())
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    audit = _move_audit(world)
    assert _kinds(audit) == [
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "deallocate",
        "allocate",
        "add",
    ]
    assert audit[4][1] == {
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
        await _drive(controller, 10)
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
    _run(_drive(controller, 22))
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    kinds = [a[0] for a in _move_audit(world)]
    assert kinds.count("add") == 2 and kinds.count("allocate") == 1

    world = FakeWorld()
    world.faults["add"] = "always"
    controller, _ = _controller(tmp_path / "b", world, Clock(), _config())

    async def run():
        for _ in range(22):
            await controller.run_once()
            move = controller.document.active_move
            if move is not None and move.state is MoveState.ROLLING_BACK:
                break
        world.faults.pop("add")  # the donor restore add works
        await _drive(controller, 15)

    _run(run())
    move = controller.document.history[-1]
    assert move.outcome is MoveOutcome.ROLLED_BACK
    kinds = [a[0] for a in _move_audit(world)]
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
    _run(_drive(controller, 22))
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    assert move.kind is MoveKind.MOVE
    kinds = [a[0] for a in _move_audit(world)]
    assert kinds.count(operation) == 1
    assert "add" not in kinds or operation == "deallocate"
    if operation == "allocate":
        assert kinds.count("add") == 0


def test_connection_failure_before_send_retries_within_bound(tmp_path: Path) -> None:
    world = FakeWorld()
    world.faults["deallocate"] = "connect"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 22))
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    assert [a[0] for a in world.audit].count("deallocate") == 2


def test_outside_unreachable_during_move_holds_without_posts(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _drive(controller, 5)
        world.outside_up = False
        for _ in range(4):
            await controller.run_once()
        assert not any(a[0] in ("deallocate", "allocate") for a in _move_audit(world))
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
        await _drive(controller, 14)
        move = controller.document.active_move
        assert move is not None and move.state is MoveState.ALLOCATED
        assert world.receiver.live(R_RUN1) is not None
        world.donor.reported_capacity = None
        world.receiver.reported_capacity = None
        await _drive(controller, 4)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED


# -- crash recovery --------------------------------


def _move_done(controller: RebalanceController) -> bool:
    """A MOVE reached a terminal state (the refused GROW alone is not done)."""
    move = controller.document.active_move
    if move is not None:
        return move.state is MoveState.BLOCKED
    return bool(_move_history(controller))


def _any_done(controller: RebalanceController) -> bool:
    """Any saga reached a terminal state."""
    return _terminal(controller) and (
        controller.document.active_move is not None or bool(controller.document.history)
    )


def _run_with_crashes(
    tmp_path: Path,
    world: FakeWorld,
    *,
    crash_after_saves: int | None,
    crash_after_effects: int | None,
    done: Callable[[RebalanceController], bool] = _move_done,
) -> RebalanceController:
    clock = Clock()
    controller, journal = _controller(tmp_path, world, clock, _config())
    journal.crash_after_saves = crash_after_saves
    world.crash_after_effects = crash_after_effects

    async def run():
        nonlocal controller
        for _ in range(80):
            try:
                await controller.run_once()
            except SimulatedCrash:
                journal.crash_after_saves = None
                world.crash_after_effects = None
                controller, _ = _controller(tmp_path, world, clock, _config())
                continue
            if done(controller):
                break
        return controller

    return _run(run())


def _assert_safe_end(controller: RebalanceController, world: FakeWorld) -> str:
    """Return the terminal outcome; assert the safety invariants."""
    kinds = [a[0] for a in world.audit]
    assert kinds.count("deallocate") <= 1, kinds
    # At most one allocation ever assigned a path; refused probes are not
    # effects on the pool.
    assert len(world.served_allocations) <= 1, world.served_allocations
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
        assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
        if last.kind is MoveKind.MOVE:
            assert world.assigned_total_gib() == 64
        else:
            assert world.assigned_total_gib() == 128
    return last.outcome.value


# Durable writes of one clean GROW-refused-then-MOVE run; pinned by
# ``test_clean_move_run_save_count_matches_the_sweep_bound`` so the sweep
# below crashes after every one of them.
_MOVE_RUN_SAVES = 26


def test_clean_move_run_save_count_matches_the_sweep_bound(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, journal = _controller(tmp_path, world, Clock(), _config())
    saves_before = journal.saves
    _run(_drive(controller, 20))
    assert _assert_safe_end(controller, world) == "SUCCEEDED"
    assert journal.saves - saves_before == _MOVE_RUN_SAVES


@pytest.mark.parametrize("crash_after_saves", list(range(1, _MOVE_RUN_SAVES + 1)))
def test_crash_after_every_durable_write(
    tmp_path: Path, crash_after_saves: int
) -> None:
    world = FakeWorld()
    controller = _run_with_crashes(
        tmp_path, world, crash_after_saves=crash_after_saves, crash_after_effects=None
    )
    outcome = _assert_safe_end(controller, world)
    assert outcome in ("SUCCEEDED", "BLOCKED")


@pytest.mark.parametrize("crash_after_effects", list(range(1, 7)))
def test_crash_after_every_effect(tmp_path: Path, crash_after_effects: int) -> None:
    world = FakeWorld()
    controller = _run_with_crashes(
        tmp_path, world, crash_after_saves=None, crash_after_effects=crash_after_effects
    )
    outcome = _assert_safe_end(controller, world)
    assert outcome in ("SUCCEEDED", "BLOCKED")
    kinds = [a[0] for a in world.audit]
    # Effects: 1 refused GROW probe, 2 drain, 3 evict, 4 deallocate,
    # 5 allocate, 6 add. A crash after a DAX effect is always recoverable
    # from status; only an outside POST whose response was lost may block.
    if crash_after_effects in (2, 3, 6):
        assert outcome == "SUCCEEDED", kinds
    if crash_after_effects == 1:
        # The probe's answer was lost: dispatched-unknown, never re-sent.
        assert outcome == "BLOCKED", kinds
        blocked = controller.document.active_move
        assert blocked is not None and blocked.kind is MoveKind.GROW
        assert kinds.count("allocate") == 1


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
    for n in range(1, _MOVE_RUN_SAVES + 1):
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


# -- GROW: allocate for the receiver, add on the receiver, no donor -------------

GROW_ALLOCATION = {
    "target_node": RECEIVER_IP,
    "request_size_gib": 64,
    "mode": "devdax",
    "purpose": "lmcache-dax",
    "access": "exclusive",
}


def _grow_world() -> FakeWorld:
    """The default fleet with a pool that can still serve one 64 GiB device."""
    world = FakeWorld()
    world.pool_budget_gib = 128
    return world


def _allocates(world: FakeWorld) -> list[tuple]:
    return [a for a in world.audit if a[0] == "allocate"]


def test_grow_succeeds_without_touching_a_donor(tmp_path: Path) -> None:
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        for _ in range(2):
            await controller.run_once()
            assert world.audit == [], "no mutation before three samples"
        report = await controller.run_once()
        assert report.proposal == {
            "kind": "grow",
            "receiver": "mp-receiver",
            "receiver_worker_ip": RECEIVER_IP,
            "request_size_gib": 64,
            "size_source": "fleet_inventory",
            "receiver_live_capacity_bytes": 64 * GIB,
            "receiver_live_ratio": 0.875,
        }
        assert report.move_id.startswith("grow-")
        await _drive(controller, 10)

    _run(run())
    assert _terminal(controller)
    move = controller.document.history[-1]
    assert move.kind is MoveKind.GROW and not move.has_donor
    assert move.donor == NO_DONOR
    assert move.state is MoveState.COMPLETE and move.outcome is MoveOutcome.SUCCEEDED
    # Exactly one allocation POST and one receiver add; no donor effect.
    assert world.audit == [
        ("allocate", {"request_id": move.allocation_request_id, **GROW_ALLOCATION}),
        ("add", "mp-receiver", R_RUN1, 64 * GIB),
    ]
    assert world.served_allocations == [move.allocation_request_id]
    assert list(move.effects) == ["allocate", "receiver_add"]
    assert all(e.confirmed for e in move.effects.values())
    assert move.new_path == R_RUN1 and move.granted_size_gib == 64
    # The pool grew: the donor path stays assigned, the receiver gained one.
    assert world.outside[DONOR_IP][D_RUN1] == "assigned"
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert world.assigned_total_gib() == 128
    assert [d.state for d in world.donor.devices] == ["active", "active"]
    assert world.receiver.live(R_RUN1) is not None
    # Inventory: nothing removed, the new path appended with origin ALLOCATED.
    assert [a.device_path for a in controller.document.inventory] == [D_RUN1, R_RUN1]
    grown = controller.document.inventory[1]
    assert grown.origin is AllocationOrigin.ALLOCATED
    assert grown.worker_ip == RECEIVER_IP and grown.instance_id == "mp-receiver"
    assert grown.allocation_size_gib == 64
    assert grown.device_map_size_bytes == 64 * GIB
    assert grown.slot_capacity_bytes == 64 * GIB
    # Cooldown on the receiver only; no backoff (the pool served).
    assert set(controller.document.cooldowns) == {move.receiver.key}
    assert controller.document.grow_backoffs == {}
    counters = controller.document.counters
    assert (counters.proposed, counters.succeeded, counters.grown) == (1, 1, 1)
    assert counters.not_served == 0 and counters.rolled_back == 0


def test_grow_not_served_then_move_next_cycle(tmp_path: Path) -> None:
    world = FakeWorld()  # pool exhausted: budget == assigned total
    clock = Clock()
    controller, _ = _controller(tmp_path, world, clock, _config())

    async def run():
        await _cycles(controller, 3)
        move = controller.document.active_move
        assert move is not None and move.kind is MoveKind.GROW
        assert move.state is MoveState.ALLOCATING
        ledger = move.effect(EffectName.ALLOCATE)
        assert ledger is not None and ledger.dispatched
        assert ledger.error == "explicit failure 409"
        assert ledger.failure is EffectFailure.EXPLICIT
        assert ledger.before_paths == []
        fourth = await controller.run_once()
        assert fourth.decision.startswith("finish NOT_SERVED")
        probe = _assert_probe_not_served(controller)
        assert controller.document.active_move is None
        assert controller.document.cooldowns == {}
        assert controller.document.grow_backoffs == {
            RECEIVER_IP: pytest.approx(probe.updated_at + 10.0)
        }
        assert [(a.device_path, a.origin) for a in controller.document.inventory] == [
            (D_RUN1, AllocationOrigin.ADOPTED)
        ]
        fifth = await controller.run_once()
        assert any(
            r["reason"] == RejectionReason.GROW_BACKOFF.value
            and r["instance_id"] == "mp-receiver"
            for r in fifth.rejections
        )
        assert fifth.proposal is not None and fifth.proposal["kind"] == "move"
        await _drive(controller, 14)

    _run(run())
    assert [m.outcome for m in controller.document.history] == [
        MoveOutcome.NOT_SERVED,
        MoveOutcome.SUCCEEDED,
    ]
    assert [m.kind for m in controller.document.history] == [
        MoveKind.GROW,
        MoveKind.MOVE,
    ]
    assert _kinds(world.audit) == [
        "allocate",
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "add",
    ]
    assert len(world.served_allocations) == 1
    probe, move = world.audit[0][1], world.audit[4][1]
    assert probe["request_id"] != move["request_id"]
    assert probe["request_id"] == controller.document.history[0].allocation_request_id
    counters = controller.document.counters
    assert (counters.not_served, counters.succeeded, counters.grown) == (1, 1, 0)
    assert counters.proposed == 2
    done = controller.document.history[-1]
    assert set(controller.document.cooldowns) == {done.donor.key, done.receiver.key}


@pytest.mark.parametrize(
    ("cooldown", "poll"),
    [(5.0, 10.0), (10.0, 10.0), (10.5, 10.0), (300.0, 10.0)],
)
def test_move_fallback_is_reached_under_the_real_poll_schedule(
    tmp_path: Path, cooldown: float, poll: float
) -> None:
    """The grow backoff outlives the idle poll that follows a ``NOT_SERVED``
    finish whatever ``cooldown_seconds`` is (it lasts at least two idle
    polls), so the next idle cycle runs the donor search instead of probing
    the exhausted pool again forever. Cycles are driven with the sleeps
    ``run_forever`` would take and a clock that charges every read."""
    world = FakeWorld()  # pool exhausted: every GROW probe is refused
    clock = Clock(tick=1.0)
    config = _config(
        cooldown_seconds=cooldown,
        poll_interval_seconds=poll,
        dax_poll_interval_seconds=2.0,
    )
    controller, _ = _controller(tmp_path, world, clock, config)

    def kinds_done() -> list[MoveKind]:
        return [m.kind for m in controller.document.history]

    async def run_until(kind: MoveKind, budget: int) -> None:
        for _ in range(budget):
            await controller.run_once()
            # Sleep exactly what run_forever sleeps before the next cycle.
            clock.now += (
                config.dax_poll_interval_seconds
                if controller.document.active_move is not None
                else config.poll_interval_seconds
            )
            if kind in kinds_done():
                return
        raise AssertionError(
            f"no {kind.value} within {budget} cycles; history={kinds_done()} "
            f"audit={_kinds(world.audit)}"
        )

    async def run() -> None:
        await run_until(MoveKind.GROW, 10)
        probe = _assert_probe_not_served(controller)
        assert controller.document.grow_backoffs == {
            RECEIVER_IP: pytest.approx(probe.updated_at + max(cooldown, 2 * poll))
        }
        await run_until(MoveKind.MOVE, 30)

    _run(run())
    assert [(m.kind, m.outcome) for m in controller.document.history] == [
        (MoveKind.GROW, MoveOutcome.NOT_SERVED),
        (MoveKind.MOVE, MoveOutcome.SUCCEEDED),
    ]
    assert _kinds(world.audit) == [
        "allocate",
        "remove:drain",
        "remove:evict",
        "deallocate",
        "allocate",
        "add",
    ]
    counters = controller.document.counters
    assert (counters.not_served, counters.succeeded, counters.proposed) == (1, 1, 2)


def test_grow_backoff_expires_and_grow_is_retried(tmp_path: Path) -> None:
    world = FakeWorld()
    world.donor.used_bytes = 100 * GIB  # no donor: nothing to move
    clock = Clock()
    controller, _ = _controller(tmp_path, world, clock, _config())

    async def run():
        await _cycles(controller, 4)
        _assert_probe_not_served(controller)
        for _ in range(3):
            report = await controller.run_once()
            assert any(
                r["reason"] == RejectionReason.GROW_BACKOFF.value
                for r in report.rejections
            ), report.rejections
            assert report.proposal is None
        assert len(world.audit) == 1, "no POST while the backoff holds"
        clock.now += 10.0  # cooldown_seconds
        world.pool_budget_gib = 128
        report = await controller.run_once()
        assert report.proposal is not None and report.proposal["kind"] == "grow"
        assert controller.document.grow_backoffs == {}
        await _drive(controller, 10)

    _run(run())
    assert controller.document.history[-1].outcome is MoveOutcome.SUCCEEDED
    allocates = _allocates(world)
    assert len(allocates) == 2
    assert allocates[0][1]["request_id"] != allocates[1][1]["request_id"]
    assert len(world.served_allocations) == 1
    assert [a[0] for a in world.audit] == ["allocate", "allocate", "add"]


def test_grow_backoff_survives_restart(tmp_path: Path) -> None:
    world = FakeWorld()
    world.donor.used_bytes = 100 * GIB
    clock = Clock()
    controller, journal = _controller(tmp_path, world, clock, _config())
    _run(_cycles(controller, 4))
    _assert_probe_not_served(controller)
    assert journal.load().grow_backoffs == controller.document.grow_backoffs

    restarted, _ = _controller(tmp_path, world, clock, _config())
    report = _run(_cycles(restarted, 4))
    assert report is not None
    assert any(
        r["reason"] == RejectionReason.GROW_BACKOFF.value for r in report.rejections
    )
    assert report.proposal is None
    assert len(world.audit) == 1, "no re-probe after a restart within the backoff"


def test_grow_dry_run_logs_proposal_without_posts(tmp_path: Path) -> None:
    world = _grow_world()
    controller, _ = _controller(
        tmp_path, world, Clock(), _config(actuation_enabled=False)
    )
    report = _run(_cycles(controller, 5))
    assert report is not None and report.proposal is not None
    assert report.proposal["kind"] == "grow"
    assert report.proposal["receiver"] == "mp-receiver"
    assert "donor" not in report.proposal
    assert any(
        r["reason"] == RejectionReason.ACTUATION_DISABLED.value
        for r in report.rejections
    )
    assert world.audit == []
    assert controller.document.active_move is None
    assert controller.document.grow_backoffs == {}
    assert controller.document.counters.proposed >= 1


def _assert_grow_rolled_back_by_release(
    controller: RebalanceController, world: FakeWorld
) -> MoveRecord:
    move = controller.document.history[-1]
    assert move.kind is MoveKind.GROW
    assert move.outcome is MoveOutcome.ROLLED_BACK
    release = [a for a in world.audit if a[0] == "deallocate"]
    assert release == [
        (
            "deallocate",
            {
                "request_id": move.release_request_id,
                "target_node": RECEIVER_IP,
                "device_path": R_RUN1,
            },
        )
    ]
    assert not any(a[0] == "remove" for a in world.audit)
    assert world.outside[RECEIVER_IP][R_RUN1] == "free"
    assert world.assigned_total_gib() == 64
    assert world.receiver.live(R_RUN1) is None
    assert [d.state for d in world.donor.devices] == ["active", "active"]
    assert [a.device_path for a in controller.document.inventory] == [D_RUN1]
    assert set(controller.document.cooldowns) == {move.receiver.key}
    assert controller.document.grow_backoffs == {}
    assert controller.document.counters.rolled_back == 1
    assert controller.document.counters.not_served == 0
    return move


def test_grow_wrong_size_releases_receiver_without_donor_effects(
    tmp_path: Path,
) -> None:
    world = _grow_world()
    world.faults["allocate"] = "wrong_size"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 10))
    move = _assert_grow_rolled_back_by_release(controller, world)
    assert _kinds(world.audit) == ["allocate", "deallocate"]
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.failure is EffectFailure.CONTRACT


def test_grow_invalid_returned_path_never_adds_and_releases(tmp_path: Path) -> None:
    world = _grow_world()
    world.faults["allocate"] = "invalid_path"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 10))
    _assert_grow_rolled_back_by_release(controller, world)
    assert _adds(world) == []
    assert _kinds(world.audit) == ["allocate", "deallocate"]


def test_grow_persistent_add_failure_releases_receiver(tmp_path: Path) -> None:
    world = _grow_world()
    world.faults["add"] = "always"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 12))
    _assert_grow_rolled_back_by_release(controller, world)
    assert _kinds(world.audit) == ["allocate", "add", "add", "deallocate"]


@pytest.mark.parametrize("fault", ["ambiguous_committed", "ambiguous_uncommitted"])
def test_grow_committed_then_dropped_allocation_blocks_without_retry(
    tmp_path: Path, fault: str
) -> None:
    world = _grow_world()
    world.faults["allocate"] = fault
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 10))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert "no retry" in move.block_reason
    assert _kinds(world.audit) == ["allocate"]
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.dispatched
    assert not ledger.response and not ledger.error
    ready, reason = controller.readiness()
    assert ready is False and "BLOCKED" in reason and move.move_id in reason
    # Whatever the pool did stays for the operator to reconcile.
    expected = "assigned" if fault == "ambiguous_committed" else "free"
    assert world.outside[RECEIVER_IP][R_RUN1] == expected
    assert controller.document.grow_backoffs == {}
    assert controller.document.counters.blocked == 1


def test_grow_contract_violation_without_effect_blocks(tmp_path: Path) -> None:
    world = _grow_world()
    world.faults["allocate"] = "contract"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 10))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert "contract" in move.block_reason
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.failure is EffectFailure.CONTRACT
    assert _kinds(world.audit) == ["allocate"]
    assert controller.document.grow_backoffs == {}
    assert controller.document.counters.not_served == 0
    assert controller.document.counters.blocked == 1


def test_grow_connection_failure_before_send_retries_within_bound(
    tmp_path: Path,
) -> None:
    world = _grow_world()
    world.faults["allocate"] = "connect"
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 12))
    move = controller.document.history[-1]
    assert move.kind is MoveKind.GROW and move.outcome is MoveOutcome.SUCCEEDED
    allocates = _allocates(world)
    assert len(allocates) == 2
    assert {a[1]["request_id"] for a in allocates} == {move.allocation_request_id}
    assert world.served_allocations == [move.allocation_request_id]
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.attempts == 2


def _the_grow(controller: RebalanceController) -> MoveRecord:
    """The GROW saga of a scenario, active or archived (a MOVE may follow)."""
    move = controller.document.active_move
    if move is not None and move.kind is MoveKind.GROW:
        return move
    grows = [m for m in controller.document.history if m.kind is MoveKind.GROW]
    assert grows, "no GROW saga ran"
    return grows[0]


@pytest.mark.parametrize(
    ("fault", "issued", "delivered", "end"),
    [
        ("connect", 2, 1, MoveOutcome.SUCCEEDED),
        ("unreachable", 2, 0, MoveState.BLOCKED),
        ("explicit", 1, 1, MoveOutcome.NOT_SERVED),
        ("contract", 1, 1, MoveState.BLOCKED),
        ("ambiguous_uncommitted", 1, 1, MoveState.BLOCKED),
        ("ambiguous_committed", 1, 1, MoveState.BLOCKED),
    ],
)
def test_grow_at_most_one_allocation_post_may_reach_the_allocator(
    tmp_path: Path,
    fault: str,
    issued: int,
    delivered: int,
    end: MoveOutcome | MoveState,
) -> None:
    """The at-most-once rule, stated precisely: a POST whose connection was
    never established (it provably delivered nothing) is re-issued with the
    same request id and ``before_paths`` up to ``get_retry_attempts`` times;
    once a POST may have reached the allocator -- answered, contract
    violation, or lost mid-flight -- no second one is ever issued for that
    request id, so the allocator sees each id at most once."""
    world = _grow_world()
    if fault == "unreachable":
        world.allocate_connect_failures = 2  # == get_retry_attempts
    else:
        world.faults["allocate"] = fault
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 12))
    move = _the_grow(controller)
    if isinstance(end, MoveOutcome):
        assert move.state is MoveState.COMPLETE and move.outcome is end
    else:
        assert move.state is end
    request_id = move.allocation_request_id
    posts = [a for a in _allocates(world) if a[1]["request_id"] == request_id]
    assert len(posts) == issued <= _config().get_retry_attempts
    assert {tuple(a[1]["request_id"] for a in posts)} == {(request_id,) * issued}
    assert world.delivered_posts.count(request_id) == delivered
    assert delivered <= 1
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.attempts == issued
    assert ledger.request_id == request_id
    assert ledger.dispatched is (delivered == 1)


def test_grow_outside_unreachable_exhausts_then_blocks(tmp_path: Path) -> None:
    world = _grow_world()
    world.allocate_connect_failures = 2  # == get_retry_attempts
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_drive(controller, 10))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert "unreachable after 2 attempts" in move.block_reason
    allocates = _allocates(world)
    assert len(allocates) == 2
    assert {a[1]["request_id"] for a in allocates} == {move.allocation_request_id}
    assert world.served_allocations == []
    assert world.outside[RECEIVER_IP][R_RUN1] == "free"
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None and ledger.dispatched is False


def test_grow_lost_leadership_before_post_reissues_next_cycle(tmp_path: Path) -> None:
    world = _grow_world()
    leader = LosingLeader()
    controller, journal = _controller(
        tmp_path, world, Clock(), _config(), leader=leader
    )

    def lose_once_selected() -> None:
        if controller.document.active_move is not None:
            leader.lose()

    world.after_sandwich = lose_once_selected
    _run(_drive(controller, 3))
    assert leader.ensure_calls == 3
    assert world.audit == []
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.ALLOCATING
    ledger = move.effect(EffectName.ALLOCATE)
    assert ledger is not None
    assert ledger.attempts == 0 and ledger.dispatched is False
    assert "lost leadership after identity check" in move.last_error
    persisted = journal.load().active_move
    assert persisted is not None
    persisted_ledger = persisted.effect(EffectName.ALLOCATE)
    assert persisted_ledger is not None and persisted_ledger.dispatched is False

    world.after_sandwich = None
    leader.restore()
    _run(_drive(controller, 12))
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    allocates = _allocates(world)
    assert len(allocates) == 1
    assert allocates[0][1]["request_id"] == ledger.request_id
    assert _kinds(world.audit) == ["allocate", "add"]


def _drive_to_allocated(controller: RebalanceController, world: FakeWorld) -> None:
    _run(_cycles(controller, 4))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.ALLOCATED
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert _adds(world) == []


@pytest.mark.parametrize("change", ["epoch", "instance_id"])
def test_grow_receiver_replaced_after_allocation_never_holds_forever(
    tmp_path: Path, change: str
) -> None:
    """The receiver re-registers on the same worker after the allocation:
    the saga rebinds to it and completes the add against the new identity."""
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _drive_to_allocated(controller, world)
    old_identity = world.receiver.identity
    if change == "epoch":
        world.receiver.epoch = 2.0
    else:
        world.receiver.instance_id = "mp-receiver-2"
    new_identity = world.receiver.identity
    assert new_identity != old_identity

    async def run():
        report = await controller.run_once()
        assert report.decision.startswith("persist ALLOCATED/NONE: receiver re-reg")
        move = controller.document.active_move
        assert move is not None and move.receiver == new_identity
        await _drive(controller, 10)

    _run(run())
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    assert done.receiver == new_identity
    assert _kinds(world.audit) == ["allocate", "add"]
    assert world.audit[1][1:] == (new_identity.instance_id, R_RUN1, 64 * GIB)
    assert len(world.served_allocations) == 1
    assert set(controller.document.cooldowns) == {new_identity.key}
    grown = controller.document.find_allocation(R_RUN1)
    assert grown is not None and grown.instance_id == new_identity.instance_id


@pytest.mark.parametrize("status_down", [True, False])
def test_grow_receiver_vanished_after_allocation_blocks_after_grace_without_release(
    tmp_path: Path, status_down: bool
) -> None:
    """No instance comes back on the worker: after ``drain_timeout_seconds``
    the saga stops holding; the path is never released underneath a
    possibly attached mapping and never while the receiver's status cannot
    be read, so it ends BLOCKED for the operator."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    controller, _ = _controller(tmp_path, world, clock, _config())
    _drive_to_allocated(controller, world)
    world.receiver.registered = False
    world.receiver.status_down = status_down

    async def run():
        for _ in range(40):
            report = await controller.run_once()
            if _terminal(controller):
                return report
            assert not any(a[0] == "deallocate" for a in world.audit)
        raise AssertionError("the saga held forever")

    _run(run())
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert "receiver vanished" in move.block_reason
    assert not any(a[0] in ("deallocate", "add") for a in world.audit)
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert controller.readiness()[0] is False


def _drive_to_add_confirmed(controller: RebalanceController, world: FakeWorld) -> None:
    """Reach ALLOCATED with the add confirmed and the usage view lagging."""
    world.receiver.reported_capacity = 64 * GIB  # the usage view lags the add
    _drive_to_allocated(controller, world)
    _run(_cycles(controller, 3))  # add, confirm, wait for capacity
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.ALLOCATED
    add = move.effect(EffectName.RECEIVER_ADD)
    assert add is not None and add.confirmed and add.attempts == 1
    assert world.receiver.live(R_RUN1) is not None
    assert _kinds(world.audit) == ["allocate", "add"]


@pytest.mark.parametrize("change", ["epoch", "instance_id"])
def test_grow_receiver_restart_drops_the_add_and_it_is_re_driven(
    tmp_path: Path, change: str
) -> None:
    """A2: the receiver restarts after its add was confirmed and comes back
    without the hot-added path (a restarted MP server keeps nothing): the
    rebind drops the confirmation, the idempotent add is issued once more
    against the new identity, and the saga still ends SUCCEEDED."""
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _drive_to_add_confirmed(controller, world)
    old_identity = world.receiver.identity
    if change == "epoch":
        world.receiver.epoch = 2.0
    else:
        world.receiver.instance_id = "mp-receiver-2"
    world.receiver.devices = [Device(R_BOOT, 56)]  # only the bootstrap device
    world.receiver.reported_capacity = None
    new_identity = world.receiver.identity
    assert new_identity != old_identity

    async def run():
        report = await controller.run_once()
        assert report.decision.startswith("persist ALLOCATED/NONE: receiver re-reg")
        move = controller.document.active_move
        assert move is not None and move.receiver == new_identity
        add = move.effect(EffectName.RECEIVER_ADD)
        assert add is not None and not add.confirmed and add.confirmed_at == 0.0
        assert add.attempts == 1  # the attempt budget is kept
        await _drive(controller, 10)

    _run(run())
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    assert done.receiver == new_identity
    assert _kinds(world.audit) == ["allocate", "add", "add"]
    assert world.audit[1][1:] == (old_identity.instance_id, R_RUN1, 64 * GIB)
    assert world.audit[2][1:] == (new_identity.instance_id, R_RUN1, 64 * GIB)
    add = done.effect(EffectName.RECEIVER_ADD)
    assert add is not None and add.confirmed and add.attempts == 2
    assert world.receiver.live(R_RUN1) is not None
    assert len(world.served_allocations) == 1
    assert not any(a[0] == "deallocate" for a in world.audit)
    grown = controller.document.find_allocation(R_RUN1)
    assert grown is not None and grown.instance_id == new_identity.instance_id


def _restart_receiver(world: FakeWorld) -> None:
    """The receiver MP server restarted again: a fresh registration epoch and
    only its bootstrap device (a restarted server keeps nothing)."""
    world.receiver.epoch += 1.0
    world.receiver.devices = [Device(R_BOOT, 56)]
    world.receiver.reported_capacity = None


async def _run_until_terminal(
    controller: RebalanceController, cycles: int, before_each: Callable[[], None]
) -> None:
    for _ in range(cycles):
        before_each()
        await controller.run_once()
        if _terminal(controller):
            return
    move = controller.document.active_move
    raise AssertionError(
        f"still non-terminal after {cycles} cycles: "
        f"{move.state.value if move else None}; readiness={controller.readiness()}"
    )


def test_grow_receiver_restart_loop_after_allocation_is_bounded(
    tmp_path: Path,
) -> None:
    """A2: a receiver that keeps re-registering with a fresh identity faster
    than ``drain_timeout_seconds`` is rebound at most
    ``GROW_MAX_RECEIVER_REBINDS`` times; the next loss blocks instead of
    resetting the grace forever with ``/readyz`` green."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    config = _config()
    controller, _ = _controller(tmp_path, world, clock, config)
    _drive_to_add_confirmed(controller, world)
    start = clock.now

    _run(
        _run_until_terminal(
            controller, GROW_MAX_RECEIVER_REBINDS + 5, lambda: _restart_receiver(world)
        )
    )
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert move.receiver_rebinds == GROW_MAX_RECEIVER_REBINDS
    assert "re-registered" in move.block_reason and RECEIVER_IP in move.block_reason
    assert clock.now - start <= 2 * config.drain_timeout_seconds
    assert _kinds(world.audit) == ["allocate", "add"]  # no release, no more adds
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert controller.readiness()[0] is False


def test_grow_release_step_receiver_restart_loop_is_bounded(tmp_path: Path) -> None:
    """A2 in ROLLING_BACK/RELEASE_RECEIVER: the same restart loop cannot keep
    the release step alive forever either; it blocks after the rebind cap
    without ever POSTing a release (the gate needs a stable identity)."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    config = _config()
    controller, _ = _controller(tmp_path, world, clock, config)
    _drive_to_allocated(controller, world)
    # Nobody on the worker and no readable status: after the grace the path
    # is provably unattached and the saga enters the release step.
    world.receiver.registered = False
    world.receiver.status_down = True

    async def to_release_step() -> None:
        for _ in range(60):
            await controller.run_once()
            move = controller.document.active_move
            if move is not None and move.rollback_step is RollbackStep.RELEASE_RECEIVER:
                return
        raise AssertionError("never reached RELEASE_RECEIVER")

    _run(to_release_step())
    world.receiver.registered = True
    world.receiver.status_down = False
    start = clock.now
    _run(
        _run_until_terminal(
            controller, GROW_MAX_RECEIVER_REBINDS + 5, lambda: _restart_receiver(world)
        )
    )
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    assert move.rollback_step is RollbackStep.RELEASE_RECEIVER
    assert move.receiver_rebinds == GROW_MAX_RECEIVER_REBINDS
    assert "re-registered" in move.block_reason
    assert clock.now - start <= 2 * config.drain_timeout_seconds
    assert not any(a[0] == "deallocate" for a in world.audit)
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert controller.readiness()[0] is False


def test_grow_allocator_outage_after_the_add_is_confirmed_is_bounded(
    tmp_path: Path,
) -> None:
    """A4: with the add confirmed and the usage view lagging, an allocator
    that stays unreadable cannot keep the GROW waiting forever: once
    ``drain_timeout_seconds`` pass beyond the convergence timeout without
    the allocation being re-verifiable, the saga blocks and readiness
    drops. The receiver keeps serving the path; nothing is released."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    config = _config(capacity_convergence_timeout_seconds=5.0)
    controller, _ = _controller(tmp_path, world, clock, config)
    _drive_to_add_confirmed(controller, world)
    world.outside_up = False
    start = clock.now

    _run(_run_until_terminal(controller, 80, lambda: None))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert "allocator" in move.block_reason and "unreadable" in move.block_reason
    bound = config.capacity_convergence_timeout_seconds + config.drain_timeout_seconds
    assert clock.now - start <= bound + 10
    assert _kinds(world.audit) == ["allocate", "add"]
    assert world.receiver.live(R_RUN1) is not None
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert controller.document.find_allocation(R_RUN1) is None
    assert controller.readiness()[0] is False


def test_grow_allocator_back_before_the_outage_bound_finishes_with_warning(
    tmp_path: Path,
) -> None:
    """The outage bound is a bound, not a deadline: an allocator that comes
    back within it and still lists the path lets the bounded convergence
    wait finish ``SUCCEEDED`` (with its warning)."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    config = _config(capacity_convergence_timeout_seconds=5.0)
    controller, _ = _controller(tmp_path, world, clock, config)
    _drive_to_add_confirmed(controller, world)
    world.outside_up = False

    async def run() -> None:
        for _ in range(4):  # well past the convergence timeout
            report = await controller.run_once()
            assert report.decision.startswith("hold: waiting for capacity"), report
        world.outside_up = True
        report = await controller.run_once()
        assert report.decision.startswith("finish SUCCEEDED"), report

    _run(run())
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    assert "did not converge" in done.last_error
    assert controller.document.find_allocation(R_RUN1) is not None


def test_grow_path_dropped_by_the_allocator_after_the_add_blocks(
    tmp_path: Path,
) -> None:
    """The allocator is readable but no longer lists the path under the
    receiver once the convergence timeout elapsed: that contradiction is
    terminal (never a silent hold, never a success)."""
    world = _grow_world()
    clock = Clock(tick=1.0)
    config = _config(capacity_convergence_timeout_seconds=5.0)
    controller, _ = _controller(tmp_path, world, clock, config)
    _drive_to_add_confirmed(controller, world)
    world.outside[RECEIVER_IP][R_RUN1] = "free"  # dropped behind our back

    _run(_run_until_terminal(controller, 20, lambda: None))
    move = controller.document.active_move
    assert move is not None and move.state is MoveState.BLOCKED
    assert "lists it under []" in move.block_reason
    assert _kinds(world.audit) == ["allocate", "add"]
    assert controller.document.find_allocation(R_RUN1) is None


def test_grow_receiver_rejected_by_the_sandwich_then_accepted_again_succeeds(
    tmp_path: Path,
) -> None:
    """A receiver the sandwich rejects for a while (still registered on its
    worker, still readable) is not vanished-and-unattached: the saga holds
    without any effect and completes once the receiver is accepted again."""
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _drive_to_allocated(controller, world)
    world.receiver.declared = False  # rejected: undeclared_capacity

    async def run():
        for _ in range(3):
            report = await controller.run_once()
            assert report.decision.startswith("hold: receiver vanished"), report
        move = controller.document.active_move
        assert move is not None and move.state is MoveState.ALLOCATED
        assert move.rollback_step is RollbackStep.NONE
        assert _adds(world) == []
        world.receiver.declared = True
        await _drive(controller, 10)

    _run(run())
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    assert done.receiver == world.receiver.identity
    assert _kinds(world.audit) == ["allocate", "add"]


def test_grow_receiver_rejected_past_the_grace_with_the_path_attached_blocks(
    tmp_path: Path,
) -> None:
    """The receiver stays registered but rejected for longer than the grace
    while the added path is live on it: the saga blocks in ALLOCATED as
    attached -- never "provably unattached", never ROLLING_BACK, and the
    path is neither released nor detached."""
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(tick=1.0), _config())
    _drive_to_add_confirmed(controller, world)
    world.receiver.declared = False

    async def run():
        for _ in range(40):
            await controller.run_once()
            if _terminal(controller):
                return
        raise AssertionError("the saga held forever")

    _run(run())
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.GROW
    assert move.state is MoveState.BLOCKED
    assert move.rollback_step is RollbackStep.NONE
    assert "is attached" in move.block_reason
    assert "provably unattached" not in move.block_reason
    assert _kinds(world.audit) == ["allocate", "add"]
    assert world.outside[RECEIVER_IP][R_RUN1] == "assigned"
    assert world.receiver.live(R_RUN1) is not None


def test_grow_deferred_by_an_unreachable_fresh_sandwich_names_the_coordinator(
    tmp_path: Path,
) -> None:
    """The pre-POST re-check failing on the MP Coordinator is reported as
    such (a GROW has no donor to blame); the intent stays undispatched and
    the single POST follows once the coordinator is back."""
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_cycles(controller, 2))

    def coordinator_down() -> None:
        world.coordinator_up = False

    world.after_sandwich = coordinator_down  # the cycle's own read succeeds

    async def run():
        report = await controller.run_once()
        assert report.decision == "effect allocate"
        move = controller.document.active_move
        assert move is not None and move.kind is MoveKind.GROW
        assert move.state is MoveState.ALLOCATING
        assert move.last_error == "allocate: MP Coordinator unreachable before POST"
        ledger = move.effect(EffectName.ALLOCATE)
        assert ledger is not None and not ledger.dispatched and ledger.attempts == 0
        assert world.audit == []
        world.after_sandwich = None
        world.coordinator_up = True
        await _drive(controller, 10)

    _run(run())
    done = controller.document.history[-1]
    assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
    assert _kinds(world.audit) == ["allocate", "add"]
    assert world.audit[0][1]["request_id"] == done.allocation_request_id


def test_status_never_shows_an_expired_grow_backoff(tmp_path: Path) -> None:
    """An expired backoff is filtered out of /status at once, even while a
    saga occupies every cycle or before any cycle ran after a restart; the
    journal drops it in the next idle cycle."""
    world = FakeWorld()  # pool exhausted: the GROW is refused, then a MOVE
    clock = Clock()
    controller, journal = _controller(tmp_path, world, clock, _config())
    _run(_cycles(controller, 5))
    move = controller.document.active_move
    assert move is not None and move.kind is MoveKind.MOVE
    assert set(controller.document.grow_backoffs) == {RECEIVER_IP}
    assert set(_status_section(controller, "grow_backoffs")) == {RECEIVER_IP}
    clock.now += 10.0  # cooldown_seconds: the backoff expires
    assert _status_section(controller, "grow_backoffs") == {}
    _run(_cycles(controller, 1))
    assert controller.document.active_move is not None  # the MOVE goes on
    assert set(controller.document.grow_backoffs) == {RECEIVER_IP}  # durable
    assert _status_section(controller, "grow_backoffs") == {}
    # A restart sees the same: filtered view, durable entry until idle.
    restarted = RebalanceController(
        _config(), journal, world, StaticLeader("test"), clock=clock
    )
    restarted.load()
    assert set(restarted.document.grow_backoffs) == {RECEIVER_IP}
    assert _status_section(restarted, "grow_backoffs") == {}
    _run(_drive(controller, 14))  # the MOVE completes; the next idle cycle prunes
    assert controller.document.active_move is None
    assert controller.document.grow_backoffs == {}


def test_a_receiver_rejection_is_reported_once_across_both_passes(
    tmp_path: Path,
) -> None:
    """The GROW pass and the MOVE pass both reject an unreadable receiver;
    the cycle report names that rejection once."""
    world = FakeWorld()
    world.receiver.status_down = True
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    report = _run(_cycles(controller, 3))
    assert report is not None and report.proposal is None
    receiver_rejections = [
        r["reason"] for r in report.rejections if r["instance_id"] == "mp-receiver"
    ]
    assert receiver_rejections == [RejectionReason.PREFLIGHT_UNAVAILABLE.value]
    assert world.audit == []


def test_proposal_deferred_in_the_cycle_that_attached(tmp_path: Path) -> None:
    """A3: an attach cycle never proposes -- not even a GROW."""
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _cycles(controller, 2)
        world.donor.declare_present(D_RUN2)
        world.outside[DONOR_IP][D_RUN2] = "assigned"
        third = await controller.run_once()
        assert third.attachments["attached"] == [D_RUN2]
        assert third.proposal is None
        assert third.decision == "attach issued; re-observing next cycle"
        assert controller.document.active_move is None
        assert world.audit == [("add", "mp-donor", D_RUN2, 64 * GIB)]
        fourth = await controller.run_once()
        assert fourth.proposal is not None and fourth.proposal["kind"] == "grow"
        move = controller.document.active_move
        assert move is not None and move.kind is MoveKind.GROW
        # The record captures the post-attach receiver capacity.
        assert move.receiver_capacity_bytes == 64 * GIB

    _run(run())


def test_attach_is_deferred_while_a_grow_is_active(tmp_path: Path) -> None:
    world = _grow_world()
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        await _cycles(controller, 3)  # GROW selected, allocation issued
        assert _kinds(world.audit) == ["allocate"]
        world.donor.declare_present(D_RUN2)
        world.outside[DONOR_IP][D_RUN2] = "assigned"
        for _ in range(20):
            report = await controller.run_once()
            if controller.document.active_move is None:
                break
            assert report.attachments == {}
            assert all(a[2] != D_RUN2 for a in _adds(world))
        done = controller.document.history[-1]
        assert done.kind is MoveKind.GROW and done.outcome is MoveOutcome.SUCCEEDED
        report = await controller.run_once()
        assert report.attachments["attached"] == [D_RUN2]

    _run(run())
    assert [a[2] for a in _adds(world)] == [R_RUN1, D_RUN2]


def test_grow_prefers_the_highest_pressure_receiver_then_the_next(
    tmp_path: Path,
) -> None:
    """Several stable-HIGH receivers: the best (highest ratio) grows first;
    once it cools, the next one does."""
    world = FakeWorld()
    other_ip = "192.0.2.42"
    other_boot = "/dev/dax-cxl/lmcache-e2e--mp-198/dax0.0"
    other_run = "/dev/dax-cxl/lmcache-e2e--mp-198/dax0.1"
    other = Instance("mp-receiver-b", "10.0.0.13", other_ip, [Device(other_boot, 60)])
    other.used_bytes = 60 * GIB  # 0.9375 > the default receiver's 0.875
    world.others.append(other)
    world.outside[other_ip] = {other_run: "free"}
    world.pool_budget_gib = 192
    controller, _ = _controller(tmp_path, world, Clock(), _config())

    async def run():
        third = await controller.run_once()
        third = await controller.run_once()
        third = await controller.run_once()
        assert third.proposal is not None
        assert third.proposal["kind"] == "grow"
        assert third.proposal["receiver"] == "mp-receiver-b"
        for _ in range(20):
            if len(controller.document.history) == 2:
                break
            await controller.run_once()
        first, second = controller.document.history
        assert first.receiver.instance_id == "mp-receiver-b"
        assert second.receiver.instance_id == "mp-receiver"
        assert second.kind is MoveKind.GROW

    _run(run())
    targets = [a[1]["target_node"] for a in _allocates(world)]
    assert targets == [other_ip, RECEIVER_IP]
    assert [m.outcome for m in controller.document.history] == [
        MoveOutcome.SUCCEEDED,
        MoveOutcome.SUCCEEDED,
    ]
    assert controller.document.counters.grown == 2


def test_status_exposes_grow_backoffs_and_kind(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(_cycles(controller, 3))
    status = controller.status()
    active = status["active_move"]
    assert isinstance(active, dict) and active["kind"] == "grow"
    assert status["grow_backoffs"] == {}
    _run(_cycles(controller, 1))
    status = controller.status()
    assert status["active_move"] is None
    backoffs = status["grow_backoffs"]
    assert isinstance(backoffs, dict) and set(backoffs) == {RECEIVER_IP}
    counters = status["counters"]
    assert isinstance(counters, dict)
    assert counters["not_served"] == 1 and counters["grown"] == 0
    assert _status_section(controller, "last_cycle")["proposal"] is None


# -- GROW crash recovery --------------------------------


# Durable writes of one clean GROW run (select, allocate intent/dispatched/
# result, ALLOCATED, add intent/dispatched/result, confirm, finish); pinned
# by ``test_clean_grow_run_save_count_matches_the_sweep_bound``.
_GROW_RUN_SAVES = 10


def test_clean_grow_run_save_count_matches_the_sweep_bound(tmp_path: Path) -> None:
    world = _grow_world()
    controller, journal = _controller(tmp_path, world, Clock(), _config())
    saves_before = journal.saves
    _run(_drive(controller, 10))
    assert _assert_safe_end(controller, world) == "SUCCEEDED"
    assert journal.saves - saves_before == _GROW_RUN_SAVES


@pytest.mark.parametrize("crash_after_saves", list(range(1, _GROW_RUN_SAVES + 1)))
def test_grow_crash_after_every_durable_write(
    tmp_path: Path, crash_after_saves: int
) -> None:
    world = _grow_world()
    controller = _run_with_crashes(
        tmp_path,
        world,
        crash_after_saves=crash_after_saves,
        crash_after_effects=None,
        done=_any_done,
    )
    outcome = _assert_safe_end(controller, world)
    assert outcome in ("SUCCEEDED", "BLOCKED")
    last = controller.document.active_move or controller.document.history[-1]
    assert last.kind is MoveKind.GROW
    if outcome == "BLOCKED":
        # Only a lost allocation answer may block: dispatched, no outcome.
        ledger = last.effect(EffectName.ALLOCATE)
        assert ledger is not None and ledger.dispatched
        assert not ledger.response and not ledger.error
        assert _adds(world) == []


@pytest.mark.parametrize("crash_after_effects", [1, 2])
def test_grow_crash_after_every_effect(
    tmp_path: Path, crash_after_effects: int
) -> None:
    world = _grow_world()
    controller = _run_with_crashes(
        tmp_path,
        world,
        crash_after_saves=None,
        crash_after_effects=crash_after_effects,
        done=_any_done,
    )
    outcome = _assert_safe_end(controller, world)
    if crash_after_effects == 1:
        # The allocation's answer was lost: BLOCKED, never re-sent.
        assert outcome == "BLOCKED"
        assert _kinds(world.audit) == ["allocate"]
    else:
        # The add is re-driven from status.
        assert outcome == "SUCCEEDED"
        assert _kinds(world.audit) == ["allocate", "add"]


# -- cycle hardening --------------------------------


def test_cycle_exception_is_reported_and_readiness_drops(tmp_path: Path) -> None:
    """A5: a failing cycle body never leaves a stale healthy report behind."""
    world = FakeWorld()
    controller, _ = _controller(tmp_path, world, Clock(), _config())
    _run(controller.run_once())
    assert controller.readiness() == (True, "ok")

    def boom() -> None:
        raise RuntimeError("boom")

    world.after_sandwich = boom
    report = _run(controller.run_once())
    assert report.error == "cycle failed: boom"
    assert controller.last_report is report
    ready, reason = controller.readiness()
    assert ready is False and "boom" in reason
    assert world.audit == []
    world.after_sandwich = None
    _run(controller.run_once())
    assert controller.readiness() == (True, "ok")
