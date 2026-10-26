# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for shared Device-DAX L1 tests: GPU registration is stubbed."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform import current_device_spec
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
