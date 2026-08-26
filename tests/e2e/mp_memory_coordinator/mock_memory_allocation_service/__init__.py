# SPDX-License-Identifier: Apache-2.0
"""Strict development mock of the outside Memory Allocation service.

Public entry points: :func:`build_apps` and :func:`create_state` for
in-process tests, and ``python -m ...mock_memory_allocation_service`` for a
standalone process.  This package never imports production code from
``lmcache``.
"""

# Local
from .app import build_apps, create_state
from .faults import BarrierRegistry, BarrierSpec, FaultRegistry, FaultSpec
from .state import MockAllocatorState

__all__ = [
    "BarrierRegistry",
    "BarrierSpec",
    "FaultRegistry",
    "FaultSpec",
    "MockAllocatorState",
    "build_apps",
    "create_state",
]
