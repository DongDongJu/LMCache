# SPDX-License-Identifier: Apache-2.0
"""Ownership-proving discovery of runtime devices from outside status.

Adoption (:mod:`lmcache.v1.mp_memory_coordinator.adoption`) requires an
operator to name every path in a file. Discovery replaces the file with the
outside Memory Allocation service, which is already the authoritative
owner of every managed path: a live DAX device is adopted only while that
service lists its *exact* path under the *same* worker IP that the MP
instance registered. The coordinator therefore still never manages a device
the outside service does not say belongs to that node -- the property that
makes the donor's ``POST /deallocations`` safe -- but no path has to be
transcribed by hand.

Unlike adoption this runs on every cycle, so a path that changes with its
Pod name is re-derived instead of going stale. It is purely additive: an
entry already in the inventory is left alone, and nothing is ever removed
here (a device that disappears from outside status is reported by the
controller's reconciliation, which never deletes either). Removal stays a
consequence of a completed move.

Every live device that is *not* adopted is reported with a reason, so an
empty inventory is always explained rather than silent.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_ACTIVE_STATE,
    GIB,
    AllocationOrigin,
    DaxDeviceStatus,
    DaxHotplugStatus,
    InstanceSample,
    JournalDocument,
    ManagedAllocation,
    OutsideStatus,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class DiscoveryResult:
    """What one discovery pass decided.

    Attributes:
        discovered: Allocations newly appended to the inventory.
        skipped: ``device_path -> reason`` for every live device that was
            considered and not adopted, including ones already owned.
    """

    discovered: list[ManagedAllocation] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly form for the cycle report."""
        return {
            "discovered": [a.device_path for a in self.discovered],
            "skipped": self.skipped,
        }


def owners_of(outside: OutsideStatus, device_path: str) -> list[str]:
    """Return every outside node listing ``device_path``, sorted.

    Args:
        outside: The outside status document.
        device_path: The exact path to look up.

    Returns:
        The node keys whose path list contains ``device_path``.
    """
    return sorted(node for node, paths in outside.items() if device_path in paths)


def discover(
    samples: Mapping[str, InstanceSample],
    dax_statuses: Mapping[str, DaxHotplugStatus],
    outside: OutsideStatus,
    document: JournalDocument,
    config: MPMemoryCoordinatorConfig,
    now: float,
) -> DiscoveryResult:
    """Adopt every live device the outside service proves the worker owns.

    Pure; the caller persists ``document`` when ``discovered`` is non-empty.
    A device is adopted when all of the following hold:

    * its DAX index is ``> 0`` (index ``0`` is the bootstrap device, which
      is never movable and so never managed);
    * its state is ``active`` and it is healthy and not closing;
    * its path starts with ``allowed_device_path_prefix``;
    * outside status lists that exact path under exactly the one worker IP
      the instance registered;
    * its DAX map size is a positive whole number of GiB; and
    * no inventory entry already claims the path.

    Args:
        samples: Accepted samples of the current sandwich read, keyed by
            instance id.
        dax_statuses: The single DAX adapter's hotplug status per instance
            id. An instance missing here is skipped entirely.
        outside: The outside ``target_node -> device_path[]`` document.
        document: The journal document; ``inventory`` is appended in place.
        config: Path prefix and adapter index.
        now: Wall-clock time stamped on each new allocation.

    Returns:
        The result.
    """
    result = DiscoveryResult()
    for instance_id, sample in sorted(samples.items()):
        dax = dax_statuses.get(instance_id)
        if dax is None:
            continue
        worker_ip = sample.identity.worker_ip
        for device in dax.live_devices():
            reason = _skip_reason(device, worker_ip, outside, document, config)
            if reason:
                result.skipped[device.device_path] = reason
                continue
            allocation = ManagedAllocation(
                worker_ip=worker_ip,
                instance_id=instance_id,
                device_path=device.device_path,
                allocation_size_gib=device.max_dax_size_bytes // GIB,
                device_map_size_bytes=device.max_dax_size_bytes,
                slot_capacity_bytes=device.slot_capacity_bytes,
                adapter_index=config.adapter_index,
                origin=AllocationOrigin.DISCOVERED,
                last_confirmed_state=device.state,
                last_confirmed_at=now,
            )
            document.inventory.append(allocation)
            result.discovered.append(allocation)
            logger.info(
                "discovered %s on %s (%d GiB; outside status confirms ownership)",
                allocation.device_path,
                worker_ip,
                allocation.allocation_size_gib,
            )
    return result


def _skip_reason(
    device: DaxDeviceStatus,
    worker_ip: str,
    outside: OutsideStatus,
    document: JournalDocument,
    config: MPMemoryCoordinatorConfig,
) -> str:
    """Return why ``device`` may not be discovered (``""`` if it may).

    Args:
        device: One live device of the instance's DAX adapter.
        worker_ip: The registered worker IP of the owning instance.
        outside: The outside status document.
        document: The journal document, for the already-owned check.
        config: Path prefix.

    Returns:
        The first violated rule, or ``""`` when the device is eligible.
    """
    path = device.device_path
    if document.find_allocation(path) is not None:
        return "already owned"
    if device.index <= 0:
        return "DAX index 0 (bootstrap)"
    if device.state != DAX_ACTIVE_STATE:
        return f"state is {device.state}"
    if not device.is_healthy or device.closing:
        return "unhealthy or closing"
    if not path.startswith(config.allowed_device_path_prefix):
        return f"outside {config.allowed_device_path_prefix}"
    owners = owners_of(outside, path)
    if owners != [worker_ip]:
        return f"outside status lists the path under {owners}, not [{worker_ip}]"
    if device.max_dax_size_bytes <= 0 or device.max_dax_size_bytes % GIB != 0:
        return (
            f"map size {device.max_dax_size_bytes} is not a positive "
            f"whole number of GiB"
        )
    return ""
