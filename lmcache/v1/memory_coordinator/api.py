# SPDX-License-Identifier: Apache-2.0
"""Immutable metadata contracts; keys and offsets never carry payload or pointers."""

# Standard
from dataclasses import asdict
from enum import Enum
import json
import math

# Third Party
from pydantic import BaseModel, ConfigDict, Field, StrictInt

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey

_WIRE_DTYPE_ITEMSIZE = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "int32": 4,
    "float32": 4,
    "int64": 8,
    "float64": 8,
}


def wire_dtype_itemsize(name: str) -> int:
    """Return bytes per element for canonical dtype ``name``; reject unknown names."""
    try:
        return _WIRE_DTYPE_ITEMSIZE[name]
    except KeyError:
        raise ValueError(f"unknown wire dtype {name!r}") from None


def canonical_key(key: EncodedObjectKey) -> str:
    """Return deterministic JSON for ``key``; raise ValueError for invalid encoding."""
    if type(key.kv_rank) is not int or type(key.object_group_id) is not int:
        raise ValueError("kv_rank and object_group_id must be integers")
    canonical = key.to_object_key().to_encoded_object_key()
    if canonical != key:
        raise ValueError("encoded object key must use its canonical representation")
    return json.dumps(asdict(canonical), sort_keys=True, separators=(",", ":"))


class MemoryCoordinatorError(RuntimeError):
    """Base exception for invalid ownership or a fenced pool."""


class OutOfSpaceError(MemoryCoordinatorError):
    """The complete write batch cannot fit; no allocation was performed."""


class InvalidReservationError(MemoryCoordinatorError):
    """A token does not own the requested operation, or a batch repeats a token."""


class StaleHandleError(MemoryCoordinatorError):
    """A supplied handle differs from the reservation's authoritative handle."""


class StaleEpochError(MemoryCoordinatorError):
    """The caller names a different pool incarnation."""


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class WireLayout(_WireModel):
    """Tensor shapes and canonical dtype names; nested tuples prevent alias mutation.

    JSON encoding uses arrays. Scalar shapes are valid; a reserved object's total
    size must be positive. Shape/dtype compatibility is checked by ``size_bytes``.
    """

    shapes: tuple[tuple[StrictInt, ...], ...]
    dtypes: tuple[str, ...]

    def size_bytes(self) -> int:
        """Return payload bytes; raise ValueError for mismatched or invalid layouts."""
        if len(self.shapes) != len(self.dtypes):
            raise ValueError("shapes and dtypes must align")
        total = 0
        for shape, dtype in zip(self.shapes, self.dtypes, strict=True):
            if any(dim < 0 for dim in shape):
                raise ValueError("dimensions must be non-negative")
            total += math.prod(shape) * wire_dtype_itemsize(dtype)
        return total


class RegionContract(_WireModel):
    """Immutable logical region identity, byte geometry, layout ID and fresh epoch."""

    region_id: str
    capacity_bytes: int
    alignment_bytes: int
    layout_id: str
    region_epoch: str


class ObjectState(str, Enum):
    """An absent key becomes WRITING, then immutable VALID; abort removes WRITING."""

    WRITING = "WRITING"
    VALID = "VALID"


class SharedObjectHandle(_WireModel):
    """Region-relative byte offset/length and a monotonic uint64 object generation."""

    region_id: str
    offset: int = Field(ge=0)
    length: int = Field(gt=0)
    generation: int = Field(gt=0, le=(1 << 64) - 1)


class WriteReserveItem(_WireModel):
    """Object key and tensor layout requested in one capacity-atomic write batch."""

    key: EncodedObjectKey
    layout: WireLayout


class ReservationRef(_WireModel):
    """Exact key, handle and opaque ownership token required to finish or abort."""

    key: EncodedObjectKey
    handle: SharedObjectHandle
    token: str


class WriteGrant(ReservationRef):
    """Exclusive authority to initialize a WRITING object with this layout."""

    layout: WireLayout


class ReadGrant(ReservationRef):
    """One independent read reservation over an immutable VALID object."""

    layout: WireLayout


class SnapshotObject(_WireModel):
    """Token-free object location, visibility state and active-reader count."""

    handle: SharedObjectHandle
    state: ObjectState
    active_readers: int


class SnapshotResponse(_WireModel):
    """Detached diagnostics; used_bytes includes alignment gaps and aborted extents."""

    region: RegionContract
    used_bytes: int
    object_count: int
    objects: dict[str, SnapshotObject]
