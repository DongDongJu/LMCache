# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures: a real memory orchestrator and L1s mapping one region file."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.memory_orchestrator.api import RegionUsage
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.shared_l1.utils import (
    CAPACITY,
    region_file,
    region_usage,
    shared_config,
)
from tests.v1.memory_orchestrator.harness import Orchestrator, running_orchestrator
import lmcache.v1.memory_management as memory_management


@pytest.fixture(autouse=True)
def cpu_mapping_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace GPU registration and pinned allocation, keeping real CPU views."""
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: True)
    monkeypatch.setattr(current_device_spec, "unpin_memory", lambda *args: True)
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        memory_management, "_free_cpu_memory", lambda *args, **kwargs: None
    )


@pytest.fixture
def capacity() -> int:
    """Region capacity in bytes; override with ``pytest.mark.parametrize``."""
    return CAPACITY


@pytest.fixture
def visibility_mode() -> str:
    """Region visibility mode; override with ``pytest.mark.parametrize``."""
    return "coherent"


@pytest.fixture
def orchestrator(
    tmp_path: Path, capacity: int, visibility_mode: str
) -> Iterator[Orchestrator]:
    with running_orchestrator(
        tmp_path / "state", capacity_bytes=capacity, visibility_mode=visibility_mode
    ) as orch:
        yield orch


@pytest.fixture
def region(tmp_path: Path, capacity: int) -> Path:
    """The file every L1 of a test maps as the shared region."""
    return region_file(tmp_path / "region", capacity)


@pytest.fixture
def make_manager(
    orchestrator: Orchestrator, region: Path
) -> Iterator[Callable[..., L1Manager]]:
    """Build ordinary L1Managers, each a separate client of one region.

    ``create(client_id, path=region, **kwargs)`` passes ``kwargs`` to
    ``shared_config``.
    """
    with ExitStack() as stack:

        def create(client_id: str, path: Path = region, **kwargs: Any) -> L1Manager:
            config = shared_config(path, orchestrator.endpoint, client_id, **kwargs)
            manager = L1Manager(config)
            stack.callback(manager.close)
            return manager

        yield create


@pytest.fixture
def usage(orchestrator: Orchestrator) -> Iterator[Callable[[], RegionUsage]]:
    """Read the region's counters through a separate probe client."""
    with region_usage(orchestrator.endpoint) as read:
        yield read
