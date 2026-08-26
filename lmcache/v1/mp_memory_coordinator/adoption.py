# SPDX-License-Identifier: Apache-2.0
"""One-time adoption of pre-existing runtime devices from an allowlist.

Normal startup never discovers devices. An operator lists every allocation
explicitly (worker IP, exact path, allocation GiB, DAX map bytes); each is
adopted only when the path is active at DAX index ``> 0`` on the MP instance
registered for that worker, is listed under the same worker in outside
status, matches the approved size, and is not already owned. The journal's
``initialized`` marker is set afterwards so a restart never repeats it.

Allowlist file::

    allocations:
      - worker_ip: 192.168.0.40
        device_path: /dev/dax-cxl/NAMESPACE_POD_NAME/dax0.x
        allocation_size_gib: 64
        device_map_size_bytes: 68719476736
"""

# Standard
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

# Third Party
from pydantic import BaseModel, ConfigDict, ValidationError
import yaml

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.clients import ClientError
from lmcache.v1.mp_memory_coordinator.clients.memory_allocation_client import (
    MemoryAllocationClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_server_client import MPServerClient
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.models import (
    DAX_ACTIVE_STATE,
    GIB,
    AllocationOrigin,
    InstanceSample,
    JournalDocument,
    ManagedAllocation,
)
from lmcache.v1.mp_memory_coordinator.policy import (
    fetch_preflight,
    preflight_problems,
    read_sandwich,
)

logger = init_logger(__name__)


class AdoptionEntry(BaseModel):
    """One allowlisted allocation."""

    model_config = ConfigDict(extra="forbid")

    worker_ip: str
    device_path: str
    allocation_size_gib: int
    device_map_size_bytes: int


class AdoptionFile(BaseModel):
    """The allowlist document."""

    model_config = ConfigDict(extra="forbid")

    allocations: list[AdoptionEntry]


def load_adoption_file(path: Path) -> list[AdoptionEntry]:
    """Parse and validate an allowlist file.

    Args:
        path: The YAML file.

    Returns:
        The entries in file order.

    Raises:
        ValueError: If the document is not the documented mapping, has
            unknown keys, or an entry is size-inconsistent (map bytes must be
            a whole number of GiB equal to ``allocation_size_gib``).
    """
    raw = yaml.safe_load(path.read_text())
    try:
        document = AdoptionFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid adoption file: {exc}") from exc
    for entry in document.allocations:
        if entry.allocation_size_gib <= 0:
            raise ValueError(f"{path}: {entry.device_path}: allocation_size_gib <= 0")
        if (
            entry.device_map_size_bytes % GIB != 0
            or entry.device_map_size_bytes // GIB != entry.allocation_size_gib
        ):
            raise ValueError(
                f"{path}: {entry.device_path}: device_map_size_bytes must equal "
                f"allocation_size_gib GiB"
            )
        if not entry.device_path.startswith("/"):
            raise ValueError(f"{path}: {entry.device_path}: path must be absolute")
    return document.allocations


@dataclass(frozen=True)
class AdoptionResult:
    """What adoption did.

    Attributes:
        adopted: Newly managed allocations.
        rejected: ``device_path -> reason`` for entries not adopted.
    """

    adopted: list[ManagedAllocation] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)


async def adopt(
    entries: list[AdoptionEntry],
    document: JournalDocument,
    *,
    coordinator: MPCoordinatorClient,
    mp_client: MPServerClient,
    allocator: MemoryAllocationClient,
    config: MPMemoryCoordinatorConfig,
    clock: Callable[[], float],
) -> AdoptionResult:
    """Verify each allowlisted entry against live state and adopt it.

    Mutates ``document.inventory`` (appending adopted entries) and sets
    ``document.initialized``; the caller persists the document.

    Args:
        entries: The allowlist.
        document: The journal document to update.
        coordinator: For the sandwich membership read.
        mp_client: For the owning instance's DAX status.
        allocator: For the outside status.
        config: Path prefix and adapter index.
        clock: Wall-clock source.

    Returns:
        The adoption result.

    Raises:
        ClientError: If the MP Coordinator or outside status is unreachable;
            nothing is adopted then.
    """
    snapshot = await read_sandwich(coordinator, clock)
    if not snapshot.coordinator_reachable:
        raise ClientError("MP Coordinator unreachable; adoption aborted")
    outside = await allocator.get_status()
    by_worker = {s.identity.worker_ip: s for s in snapshot.samples.values()}
    result = AdoptionResult()
    for entry in entries:
        reason = await _check_entry(
            entry, document, by_worker, outside, mp_client, config
        )
        if reason:
            result.rejected[entry.device_path] = reason
            logger.warning("adoption rejected %s: %s", entry.device_path, reason)
            continue
        sample = by_worker[entry.worker_ip]
        preflight = await fetch_preflight(mp_client, sample.identity)
        if preflight is None:
            result.rejected[entry.device_path] = "MP status unavailable"
            continue
        device = preflight.dax.adapters[0].status.find_live(entry.device_path)
        if device is None:
            result.rejected[entry.device_path] = "not live"
            continue
        allocation = ManagedAllocation(
            worker_ip=entry.worker_ip,
            instance_id=sample.identity.instance_id,
            device_path=entry.device_path,
            allocation_size_gib=entry.allocation_size_gib,
            device_map_size_bytes=entry.device_map_size_bytes,
            slot_capacity_bytes=device.slot_capacity_bytes,
            adapter_index=config.adapter_index,
            origin=AllocationOrigin.ADOPTED,
            last_confirmed_state=device.state,
            last_confirmed_at=clock(),
        )
        document.inventory.append(allocation)
        result.adopted.append(allocation)
        logger.info("adopted %s on %s", entry.device_path, entry.worker_ip)
    document.initialized = True
    return result


async def _check_entry(
    entry: AdoptionEntry,
    document: JournalDocument,
    by_worker: Mapping[str, InstanceSample],
    outside: dict[str, list[str]],
    mp_client: MPServerClient,
    config: MPMemoryCoordinatorConfig,
) -> str:
    """Return why ``entry`` may not be adopted (``""`` if it may)."""
    if document.find_allocation(entry.device_path) is not None:
        return "already owned"
    if not entry.device_path.startswith(config.allowed_device_path_prefix):
        return f"path is outside {config.allowed_device_path_prefix}"
    owners = [node for node, paths in outside.items() if entry.device_path in paths]
    if owners != [entry.worker_ip]:
        return f"outside status lists the path under {owners}"
    sample = by_worker.get(entry.worker_ip)
    if sample is None:
        return "no accepted MP instance registered for this worker_ip"
    preflight = await fetch_preflight(mp_client, sample.identity)
    if preflight is None:
        return "MP status unavailable"
    problems = preflight_problems(preflight, config)
    if problems:
        return "; ".join(problems)
    device = preflight.dax.adapters[0].status.find_live(entry.device_path)
    if device is None:
        return "path is not a live DAX device on the instance"
    if device.index <= 0:
        return "path is at DAX index 0 (bootstrap)"
    if device.state != DAX_ACTIVE_STATE:
        return f"device state is {device.state}"
    if device.max_dax_size_bytes != entry.device_map_size_bytes:
        return (
            f"live map size {device.max_dax_size_bytes} != approved "
            f"{entry.device_map_size_bytes}"
        )
    return ""
