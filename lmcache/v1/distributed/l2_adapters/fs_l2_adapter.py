# SPDX-License-Identifier: Apache-2.0
"""Compatibility bridge for the filesystem L2 adapter in lmcache-l2-plugins.

Install the companion package to keep using ``type: "fs"`` and these imports.
The implementation and configuration live in ``lmcache_l2_plugins.fs``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING

# Third Party
from lmcache_l2_plugins.fs import FSL2Adapter as FSL2Adapter
from lmcache_l2_plugins.fs import FSL2AdapterConfig as FSL2AdapterConfig
from lmcache_l2_plugins.fs import _filename_to_object_key as _filename_to_object_key
from lmcache_l2_plugins.fs import _object_key_to_filename as _object_key_to_filename

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import register_l2_adapter_factory

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: L1MemoryDesc | None = None,
) -> L2AdapterInterface:
    """Create the external filesystem adapter for a legacy ``fs`` config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_type("fs", FSL2AdapterConfig)
register_l2_adapter_factory("fs", _create_fs_adapter)

__all__ = ["FSL2Adapter", "FSL2AdapterConfig"]
