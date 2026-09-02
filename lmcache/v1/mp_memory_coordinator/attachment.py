# SPDX-License-Identifier: Apache-2.0
"""Attach orchestration: plan which present, owned devices to attach.

An MP server with a presence watcher reports every Device-DAX path found in
its watched directory (``watcher.present_devices``), attached or not. The
server never attaches one on its own, because presence is not ownership:
FREE runtime devices are visible on every worker and a deallocated donor
path stays present after its removal. Only the coordinator can prove
ownership, through the same rule discovery uses -- the outside Memory
Allocation service lists the exact path under exactly the one worker IP the
MP instance registered -- so only the coordinator issues the add.

:func:`plan_attachments` is pure: it turns the accepted samples, their DAX
statuses, and the outside status into an :class:`AttachmentReport`. The
controller decides whether to act on it (dry run or actuation) and records
failures so a path that cannot be mapped is retried only after
``cooldown_seconds``.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
import math

# First Party
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.discovery import owners_of
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_PHYSICAL_DEVDAX,
    GIB,
    DaxHotplugStatus,
    DaxPhysicalStatus,
    InstanceIdentity,
    InstanceSample,
    OutsideStatus,
)


@dataclass(frozen=True)
class AttachPlan:
    """One device the coordinator may attach this cycle.

    Attributes:
        identity: The MP instance that reported the device present.
        device_path: The exact path to map.
        size_bytes: The whole device size from sysfs; becomes the map size.
    """

    identity: InstanceIdentity
    device_path: str
    size_bytes: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly form for the cycle report."""
        return {
            "instance_id": self.identity.instance_id,
            "worker_ip": self.identity.worker_ip,
            "device_path": self.device_path,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class AttachmentReport:
    """What one planning pass decided.

    Attributes:
        planned: Devices eligible for an add, in deterministic order
            (instance id, then path).
        skipped: ``device_path -> reason`` for every present device that
            was considered and not planned, including already attached ones.
            Keyed by path alone: the ownership rule lets at most one
            instance plan a given path, so plans and failure cooldowns never
            cross instances; when several instances report the same path
            present (a FREE runtime device is visible on every worker) the
            last instance in id order supplies the reason shown here.
    """

    planned: list[AttachPlan] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly form for the cycle report."""
        return {
            "planned": [plan.as_dict() for plan in self.planned],
            "skipped": dict(self.skipped),
        }


def plan_attachments(
    samples: Mapping[str, InstanceSample],
    dax_statuses: Mapping[str, DaxHotplugStatus],
    outside: OutsideStatus,
    config: MPMemoryCoordinatorConfig,
    *,
    failures: Mapping[str, float],
    now: float,
) -> AttachmentReport:
    """Plan the adds for every present, owned, unattached device.

    Pure; nothing is posted here. For each accepted sample whose DAX status
    reports an enabled watcher, every present device is planned when all of
    the following hold:

    * its physical mode is ``devdax`` (never ``system-ram``, ``unbound``,
      or anything the server could not classify);
    * its path starts with ``allowed_device_path_prefix``;
    * no non-tombstone adapter entry already has the path;
    * the adapter accepts hotplug (``hotplug_enabled``);
    * outside status lists that exact path under exactly the one worker IP
      the instance registered (the discovery ownership rule);
    * its sysfs size is a positive whole number of GiB; and
    * no attach of it failed within the last ``cooldown_seconds``.

    Args:
        samples: Accepted samples of the current sandwich read, keyed by
            instance id.
        dax_statuses: The single DAX adapter's hotplug status per instance
            id. An instance missing here, or one whose watcher is disabled,
            contributes nothing.
        outside: The outside ``target_node -> device_path[]`` document.
        config: Path prefix and failure cooldown.
        failures: ``device_path -> wall-clock time`` of the last failed add.
        now: Wall-clock time the cooldown is measured against.

    Returns:
        The report; ``planned`` is sorted by instance id then path.
    """
    report = AttachmentReport()
    for instance_id, sample in sorted(samples.items()):
        dax = dax_statuses.get(instance_id)
        if dax is None or not dax.watcher.enabled:
            continue
        present = sorted(dax.watcher.present_devices, key=lambda p: p.device_path)
        for physical in present:
            reason = _skip_reason(
                physical, sample.identity.worker_ip, dax, outside, config, failures, now
            )
            if reason:
                report.skipped[physical.device_path] = reason
                continue
            report.planned.append(
                AttachPlan(
                    identity=sample.identity,
                    device_path=physical.device_path,
                    size_bytes=physical.size_bytes,
                )
            )
    return report


def _skip_reason(
    physical: DaxPhysicalStatus,
    worker_ip: str,
    dax: DaxHotplugStatus,
    outside: OutsideStatus,
    config: MPMemoryCoordinatorConfig,
    failures: Mapping[str, float],
    now: float,
) -> str:
    """Return why ``physical`` may not be attached (``""`` if it may).

    Args:
        physical: One present device of the instance's watcher.
        worker_ip: The registered worker IP of the reporting instance.
        dax: The instance's hotplug status, for the already-attached and
            hotplug-enabled checks.
        outside: The outside status document.
        config: Path prefix and failure cooldown.
        failures: ``device_path -> time`` of the last failed add.
        now: Wall-clock time.

    Returns:
        The first violated rule, or ``""`` when the device is eligible.
    """
    path = physical.device_path
    if physical.mode != DAX_PHYSICAL_DEVDAX:
        return f"mode is {physical.mode}"
    if not path.startswith(config.allowed_device_path_prefix):
        return f"outside {config.allowed_device_path_prefix}"
    if dax.find_live(path) is not None:
        return "already attached"
    if not dax.hotplug_enabled:
        return "hotplug disabled"
    owners = owners_of(outside, path)
    if owners != [worker_ip]:
        return f"outside status lists the path under {owners}, not [{worker_ip}]"
    if physical.size_bytes <= 0 or physical.size_bytes % GIB != 0:
        return f"size {physical.size_bytes} is not a positive whole number of GiB"
    if failures.get(path, -math.inf) + config.cooldown_seconds > now:
        return "recent attach failure"
    return ""
