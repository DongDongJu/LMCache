# SPDX-License-Identifier: Apache-2.0
"""One process's ephemeral allocation and object-lifetime authority.

Extents are never reused. This library has no physical-memory authority: durable
state and exclusive writer fencing are prerequisites for shared DAX integration.
"""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import threading
import uuid

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey

# Local
from .api import (
    InvalidReservationError,
    MemoryCoordinatorError,
    ObjectState,
    OutOfSpaceError,
    ReadGrant,
    RegionContract,
    ReservationRef,
    SharedObjectHandle,
    SnapshotObject,
    SnapshotResponse,
    StaleEpochError,
    StaleHandleError,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
    canonical_key,
)


@dataclass
class _ObjectRecord:
    handle: SharedObjectHandle
    layout: WireLayout
    write_token: str | None
    state: ObjectState = ObjectState.WRITING
    read_tokens: set[str] = field(default_factory=set)


class MemoryPool:
    """Serialize metadata reservations for one fixed logical region.

    Args:
        region_id: Stable logical region identity, never a device-path assertion.
        capacity_bytes: Positive allocation limit in bytes.
        alignment_bytes: Positive power-of-two allocation alignment.
        layout_id: Immutable layout-profile identity supplied by the caller.

    Raises:
        ValueError: An identity is empty or byte geometry is invalid.

    A new instance mints a new epoch. There is no restart recovery, expiry,
    eviction or extent reuse. Every operation except ``is_healthy`` and repeated
    ``close`` raises MemoryCoordinatorError after close; inherited pools cannot
    be used after fork. Grants authorize metadata only within this instance.
    """

    def __init__(
        self,
        region_id: str,
        capacity_bytes: int,
        alignment_bytes: int,
        layout_id: str,
    ) -> None:
        if not region_id.strip() or not layout_id.strip():
            raise ValueError("region_id and layout_id must not be empty")
        if type(capacity_bytes) is not int or capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be a positive integer")
        if (
            type(alignment_bytes) is not int
            or alignment_bytes <= 0
            or alignment_bytes & (alignment_bytes - 1)
        ):
            raise ValueError("alignment_bytes must be a positive power of two")
        self._contract = RegionContract(
            region_id=region_id,
            capacity_bytes=capacity_bytes,
            alignment_bytes=alignment_bytes,
            layout_id=layout_id,
            region_epoch=uuid.uuid4().hex,
        )
        self._lock = threading.RLock()
        self._owner_pid = os.getpid()
        self._closed = False
        self._next_offset = 0
        self._next_generation = 1
        self._objects: dict[str, _ObjectRecord] = {}

    def region_contract(self) -> RegionContract:
        """Return this instance's immutable geometry, identities and fresh epoch."""
        with self._guard():
            return self._contract

    def is_healthy(self) -> bool:
        """Return whether this process owns the pool and it has not been closed."""
        return os.getpid() == self._owner_pid and not self._closed

    def close(self) -> None:
        """Fence later operations; repeated close is harmless in the owning process."""
        self._ensure_owner()
        with self._lock:
            self._closed = True

    def check_epoch(self, region_epoch: str) -> None:
        """Validate the caller's epoch; raise StaleEpochError on another incarnation."""
        with self._guard():
            if region_epoch != self._contract.region_epoch:
                raise StaleEpochError("request epoch does not match the pool epoch")

    def reserve_writes(self, items: list[WriteReserveItem]) -> list[WriteGrant | None]:
        """Return a grant per absent key, or None for an existing WRITING/VALID key.

        Args:
            items: Unique keys with positive-sized layouts, in result order.

        Raises:
            ValueError: Keys repeat, are noncanonical, or layouts are invalid.
            OutOfSpaceError: The entire absent-key batch cannot fit.

        Failure changes no state; concurrent requests for a key have one winner.
        """
        with self._guard():
            keys = self._unique_keys([item.key for item in items])
            lengths = [item.layout.size_bytes() for item in items]
            if any(length <= 0 for length in lengths):
                raise ValueError("write layouts must describe a positive size")
            cursor, generation = self._next_offset, self._next_generation
            result: list[WriteGrant | None] = []
            planned: dict[str, _ObjectRecord] = {}
            for item, key, length in zip(items, keys, lengths, strict=True):
                if key in self._objects:
                    result.append(None)
                    continue
                alignment = self._contract.alignment_bytes
                offset = (cursor + alignment - 1) // alignment * alignment
                if (
                    length > self._contract.capacity_bytes - offset
                    or generation > (1 << 64) - 1
                ):
                    raise OutOfSpaceError("write batch exceeds capacity or generations")
                grant = WriteGrant(
                    key=item.key,
                    handle=SharedObjectHandle(
                        region_id=self._contract.region_id,
                        offset=offset,
                        length=length,
                        generation=generation,
                    ),
                    token=uuid.uuid4().hex,
                    layout=item.layout,
                )
                result.append(grant)
                planned[key] = _ObjectRecord(grant.handle, grant.layout, grant.token)
                cursor, generation = offset + length, generation + 1
            self._objects.update(planned)
            self._next_offset, self._next_generation = cursor, generation
            return result

    def finish_writes(self, reservations: list[ReservationRef]) -> None:
        """Publish all supplied writes as VALID after the caller completes payload I/O.

        Raises InvalidReservationError for repeated or unowned tokens and
        StaleHandleError for an altered handle. Invalid batches change no state.
        """
        with self._guard():
            records = self._validate_reservations(reservations, writing=True)
            for record in records:
                record.state, record.write_token = ObjectState.VALID, None

    def abort_writes(self, reservations: list[ReservationRef]) -> None:
        """Remove supplied WRITING reservations; their allocated extents stay consumed.

        Raises InvalidReservationError for repeated or unowned tokens and
        StaleHandleError for an altered handle. Invalid batches change no state.
        """
        with self._guard():
            self._validate_reservations(reservations, writing=True)
            for reservation in reservations:
                del self._objects[canonical_key(reservation.key)]

    def reserve_reads(self, keys: list[EncodedObjectKey]) -> list[ReadGrant | None]:
        """Return independent read grants for VALID keys, or None for each miss.

        Args:
            keys: Unique object keys, in result order.

        Raises:
            ValueError: Keys repeat or are noncanonical; no pins are acquired.
        """
        with self._guard():
            canonical_keys = self._unique_keys(keys)
            result: list[ReadGrant | None] = []
            for key, canonical in zip(keys, canonical_keys, strict=True):
                record = self._objects.get(canonical)
                if record is None or record.state is not ObjectState.VALID:
                    result.append(None)
                    continue
                token = uuid.uuid4().hex
                record.read_tokens.add(token)
                result.append(
                    ReadGrant(
                        key=key, handle=record.handle, token=token, layout=record.layout
                    )
                )
            return result

    def finish_reads(self, reservations: list[ReservationRef]) -> None:
        """Release supplied reads after payload consumption completes.

        Raises InvalidReservationError for repeated or unowned tokens and
        StaleHandleError for an altered handle. Invalid batches change no state.
        """
        self._release_reads(reservations)

    def abort_reads(self, reservations: list[ReservationRef]) -> None:
        """Release cancelled reads using the same validation as finish_reads."""
        self._release_reads(reservations)

    def snapshot(self) -> SnapshotResponse:
        """Return detached token-free allocation and reservation diagnostics."""
        with self._guard():
            return SnapshotResponse(
                region=self._contract,
                used_bytes=self._next_offset,
                object_count=len(self._objects),
                objects={
                    key: SnapshotObject(
                        handle=record.handle,
                        state=record.state,
                        active_readers=len(record.read_tokens),
                    )
                    for key, record in self._objects.items()
                },
            )

    def _release_reads(self, reservations: list[ReservationRef]) -> None:
        with self._guard():
            records = self._validate_reservations(reservations, writing=False)
            for reservation, record in zip(reservations, records, strict=True):
                record.read_tokens.remove(reservation.token)

    def _validate_reservations(
        self, reservations: list[ReservationRef], *, writing: bool
    ) -> list[_ObjectRecord]:
        tokens = [reservation.token for reservation in reservations]
        if len(tokens) != len(set(tokens)):
            raise InvalidReservationError("duplicate reservation token")
        records = []
        for reservation in reservations:
            record = self._objects.get(canonical_key(reservation.key))
            if record is None:
                raise InvalidReservationError("reservation does not own the operation")
            if writing:
                owns_operation = (
                    record.state is ObjectState.WRITING
                    and record.write_token == reservation.token
                )
            else:
                owns_operation = reservation.token in record.read_tokens
            if not owns_operation:
                raise InvalidReservationError("reservation does not own the operation")
            if record.handle != reservation.handle:
                raise StaleHandleError("reservation has a stale handle")
            records.append(record)
        return records

    @staticmethod
    def _unique_keys(keys: list[EncodedObjectKey]) -> list[str]:
        canonical = [canonical_key(key) for key in keys]
        if len(canonical) != len(set(canonical)):
            raise ValueError("a batch must not contain duplicate keys")
        return canonical

    def _ensure_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise MemoryCoordinatorError("memory pool cannot be used after fork")

    @contextmanager
    def _guard(self) -> Iterator[None]:
        self._ensure_owner()  # Never acquire a lock inherited from another process.
        with self._lock:
            if self._closed:
                raise MemoryCoordinatorError("memory pool is closed")
            yield
