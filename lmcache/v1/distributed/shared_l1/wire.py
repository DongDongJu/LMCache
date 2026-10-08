# SPDX-License-Identifier: Apache-2.0
"""Conversions between L1 types and memory-orchestrator wire types."""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.memory_orchestrator.api import WireLayout, WireObjectKey


def _dtype_name(dtype: torch.dtype) -> str:
    """Return the wire name of a torch dtype (``torch.bfloat16`` -> ``bfloat16``)."""
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    """Resolve a wire dtype name written by another server."""
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype {name!r} in a shared-L1 layout")
    return dtype


def key_to_wire(key: ObjectKey) -> WireObjectKey:
    """Project an object key onto the orchestrator's key message.

    Args:
        key: The L1 object key.

    Returns:
        The field-for-field wire key; equal keys map to equal wire keys.
    """
    return WireObjectKey(
        chunk_hash=key.chunk_hash,
        model_name=key.model_name,
        kv_rank=key.kv_rank,
        object_group_id=key.object_group_id,
        cache_salt=key.cache_salt,
    )


def layout_to_wire(layout: MemoryLayoutDesc) -> WireLayout:
    """Encode the layout a writer used so readers can rebuild the object.

    Args:
        layout: Shapes and dtypes of the object's tensors.

    Returns:
        The wire layout with plain integer dims and dtype names.
    """
    return WireLayout(
        shapes=tuple(tuple(int(dim) for dim in shape) for shape in layout.shapes),
        dtypes=tuple(_dtype_name(dtype) for dtype in layout.dtypes),
    )


def layout_from_wire(layout: WireLayout) -> MemoryLayoutDesc:
    """Decode a layout recorded by the writer of a shared object.

    Args:
        layout: The wire layout returned with a read grant.

    Returns:
        The equivalent memory layout description.

    Raises:
        ValueError: A dtype name is unknown to this torch build, or shapes and
            dtypes differ in length.
    """
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in layout.shapes],
        dtypes=[_dtype_from_name(name) for name in layout.dtypes],
    )
