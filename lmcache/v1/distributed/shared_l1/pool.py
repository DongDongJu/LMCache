# SPDX-License-Identifier: Apache-2.0
"""Minimal process-safe state for a shared L1 memory pool.

The coordinator-owned pool keeps metadata in memory. The KV payload remains
in a separately mapped regular file or character device.
"""

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol
import ctypes
import mmap
import os
import stat
import threading
import uuid
import warnings

_GENERATION_SEQUENCE_BITS = 64


class SharedL1Error(RuntimeError):
    """Base error for the minimal shared-L1 implementation."""


class ObjectAlreadyExistsError(SharedL1Error):
    """Raised when a write would replace a WRITING or VALID object."""


class OutOfSpaceError(SharedL1Error):
    """Raised when the monotonic allocator cannot fit an object."""


class InvalidReservationError(SharedL1Error):
    """Raised when a reservation token does not own an operation."""


class ObjectBusyError(SharedL1Error):
    """Raised when an object is being written or pinned by readers."""


class RegionContractMismatchError(SharedL1Error):
    """Raised when local region expectations differ from the pool contract."""


class StaleHandleError(SharedL1Error):
    """Raised when a handle does not identify the current object extent."""


class SharedMemoryVisibility(Protocol):
    """Apply platform visibility to exact shared-memory object ranges."""

    @property
    def granularity(self) -> int:
        """Return the required byte alignment of independently visible objects."""
        ...

    def publish(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        """Make completed writer bytes visible before metadata commit.

        Args:
            device_fd: Live read-write descriptor for the mapped device.
            mapped_address: Process-local address of the first object byte.
            device_offset: Device-relative offset of the first object byte.
            length: Number of object bytes to publish.
            generation: Full generation of the object being published.
        """
        ...

    def acquire(
        self,
        *,
        device_fd: int,
        mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        """Invalidate stale local bytes before a reader consumes them.

        Args:
            device_fd: Live read-write descriptor for the mapped device.
            mapped_address: Process-local address of the first object byte.
            device_offset: Device-relative offset of the first object byte.
            length: Number of object bytes to acquire.
            generation: Full generation of the object being acquired.
        """
        ...


@dataclass(frozen=True)
class SharedRegionContract:
    """Immutable identity and geometry of one shared payload layout.

    ``layout_id`` is an operator-supplied fingerprint or version for the
    physical byte layout. It deliberately does not infer inode or device
    identity.
    """

    region_id: str
    capacity: int
    alignment: int
    layout_id: str
    generation_epoch: int

    def __post_init__(self) -> None:
        """Validate all advertised layout fields."""
        if not self.region_id:
            raise ValueError("region_id must not be empty")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise ValueError("alignment must be a positive power of two")
        if not self.layout_id:
            raise ValueError("layout_id must not be empty")
        if self.generation_epoch <= 0:
            raise ValueError("generation_epoch must be positive")


@dataclass(frozen=True)
class SharedObjectHandle:
    """Location of an object in a shared physical region.

    The handle contains no process-local virtual address. ``offset`` is
    relative to the start of the logical region, independent of where each
    process maps that region.
    """

    region_id: str
    offset: int
    length: int
    generation: int

    def __post_init__(self) -> None:
        """Validate the stable location fields."""
        if not self.region_id:
            raise ValueError("region_id must not be empty")
        if self.offset < 0:
            raise ValueError("offset must be non-negative")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.generation <= 0:
            raise ValueError("generation must be positive")


@dataclass(frozen=True)
class WriteReservation:
    """Exclusive authority to initialize one shared object."""

    object_key: str
    handle: SharedObjectHandle
    token: str


@dataclass(frozen=True)
class ReadReservation:
    """Authority for one active read of a VALID shared object."""

    object_key: str
    handle: SharedObjectHandle
    token: str


@dataclass
class _ObjectRecord:
    handle: SharedObjectHandle
    state: str
    write_token: str | None
    active_readers: int = 0


@dataclass(frozen=True)
class _ReadLease:
    object_key: str
    handle: SharedObjectHandle


class InMemorySharedL1Pool:
    """Own monotonic shared-region state in one process.

    The object is safe for concurrent callers and may be hosted behind a local
    RPC mechanism such as :class:`multiprocessing.managers.BaseManager`.
    It is functional-test pool state, not a durable or highly available
    service. Restarting it loses all allocator, object, and read-lease metadata.
    Extents are aligned and never reused; aborting a write intentionally leaks
    the reserved extent. Each generation embeds a fresh child-process boot epoch
    so handles from a prior process cannot name objects after restart.
    Restart is a coordinated reset: no reader may survive while continuing to
    use the prior pool contract.

    Args:
        region_id: Stable identity shared by every mapping of the pool.
        capacity: Logical region size in bytes.
        alignment: Required allocation-offset alignment in bytes.
        layout_id: Nonempty operator-supplied physical-layout fingerprint.

    Raises:
        ValueError: If the configuration is invalid.
    """

    def __init__(
        self,
        region_id: str,
        capacity: int,
        alignment: int,
        layout_id: str,
    ) -> None:
        boot_epoch = uuid.uuid4().int
        self._contract = SharedRegionContract(
            region_id=region_id,
            capacity=capacity,
            alignment=alignment,
            layout_id=layout_id,
            generation_epoch=boot_epoch,
        )
        self._next_offset = 0
        self._next_generation = (boot_epoch << _GENERATION_SEQUENCE_BITS) | 1
        self._objects: dict[str, _ObjectRecord] = {}
        self._read_leases: dict[str, _ReadLease] = {}
        self._lock = threading.RLock()

    def region_contract(self) -> SharedRegionContract:
        """Return the immutable payload-region contract advertised to clients.

        Returns:
            Region identity, capacity, alignment, physical layout ID, and
            pool generation epoch.
        """
        return self._contract

    def reserve_write(self, object_key: str, length: int) -> WriteReservation:
        """Reserve a new aligned extent in the WRITING state.

        Args:
            object_key: Stable application key for the object.
            length: Payload size in bytes.

        Returns:
            A write reservation containing its opaque ownership token.

        Raises:
            ValueError: If the key or length is invalid.
            ObjectAlreadyExistsError: If the key is already WRITING or VALID.
            OutOfSpaceError: If the monotonic allocator is exhausted.
        """
        self._validate_object_key(object_key)
        if length <= 0:
            raise ValueError("length must be positive")

        with self._lock:
            existing = self._objects.get(object_key)
            if existing is not None:
                raise ObjectAlreadyExistsError(
                    f"object {object_key!r} is already {existing.state}"
                )

            offset = self._align_up(
                self._next_offset,
                self._contract.alignment,
            )
            if (
                length > self._contract.capacity
                or offset > self._contract.capacity - length
            ):
                raise OutOfSpaceError(
                    f"cannot allocate {length} bytes from offset {offset} "
                    f"in a {self._contract.capacity}-byte region"
                )

            handle = SharedObjectHandle(
                region_id=self._contract.region_id,
                offset=offset,
                length=length,
                generation=self._next_generation,
            )
            token = uuid.uuid4().hex
            self._objects[object_key] = _ObjectRecord(
                handle=handle,
                state="WRITING",
                write_token=token,
            )
            self._next_offset = offset + length
            self._next_generation += 1
            return WriteReservation(object_key, handle, token)

    def finish_write(self, reservation: WriteReservation) -> SharedObjectHandle:
        """Commit a successfully published WRITING object as immutable VALID.

        Args:
            reservation: Reservation returned by :meth:`reserve_write`.

        Returns:
            The published object's stable handle.

        Raises:
            InvalidReservationError: If the token does not own the write.
            StaleHandleError: If the reservation handle does not match.

        Notes:
            The caller must complete the mapped region's publish operation
            before invoking this metadata transition.
        """
        with self._lock:
            record = self._validate_write_reservation(reservation)
            record.state = "VALID"
            record.write_token = None
            return reservation.handle

    def abort_write(self, reservation: WriteReservation) -> None:
        """Abort a write without making its reserved extent reusable.

        Args:
            reservation: Reservation returned by :meth:`reserve_write`.

        Raises:
            InvalidReservationError: If the token does not own the write.
            StaleHandleError: If the reservation handle does not match.

        Notes:
            Object metadata is deleted so the key may be retried, but the
            allocator cursor is not rewound. The abandoned extent is leaked.
        """
        with self._lock:
            self._validate_write_reservation(reservation)
            del self._objects[reservation.object_key]

    def reserve_read(
        self,
        object_key: str,
        expected_handle: SharedObjectHandle | None = None,
    ) -> ReadReservation | None:
        """Reserve a read if the named object is VALID.

        Args:
            object_key: Stable application key for the object.
            expected_handle: Optional location expected by the caller. A
                mismatched region, extent, or generation is rejected.

        Returns:
            A read reservation, or ``None`` when the object is absent or still
            WRITING.

        Raises:
            ValueError: If ``object_key`` is invalid.
            StaleHandleError: If ``expected_handle`` does not match.
        """
        self._validate_object_key(object_key)
        if (
            expected_handle is not None
            and expected_handle.region_id != self._contract.region_id
        ):
            raise StaleHandleError(
                f"expected region {expected_handle.region_id!r}, "
                f"pool uses {self._contract.region_id!r}"
            )

        with self._lock:
            record = self._objects.get(object_key)
            if record is None:
                return None
            if expected_handle is not None and record.handle != expected_handle:
                raise StaleHandleError(
                    f"expected handle {expected_handle!r}, current handle is "
                    f"{record.handle!r}"
                )
            if record.state != "VALID":
                return None

            token = uuid.uuid4().hex
            self._read_leases[token] = _ReadLease(object_key, record.handle)
            record.active_readers += 1
            return ReadReservation(object_key, record.handle, token)

    def delete(
        self,
        object_key: str,
        expected_handle: SharedObjectHandle | None = None,
    ) -> bool:
        """Delete unpinned VALID object metadata without reusing its extent.

        Args:
            object_key: Stable application key for the object.
            expected_handle: Optional location expected by the caller.

        Returns:
            ``True`` when metadata was removed, or ``False`` when the key was
            absent.

        Raises:
            ValueError: If ``object_key`` is invalid.
            StaleHandleError: If ``expected_handle`` does not match.
            ObjectBusyError: If the object is WRITING or has active readers.
        """
        self._validate_object_key(object_key)
        with self._lock:
            record = self._objects.get(object_key)
            if record is None:
                return False
            if expected_handle is not None and record.handle != expected_handle:
                raise StaleHandleError(
                    f"expected handle {expected_handle!r}, current handle is "
                    f"{record.handle!r}"
                )
            if record.state == "WRITING":
                raise ObjectBusyError(f"object {object_key!r} is still WRITING")
            if record.active_readers:
                raise ObjectBusyError(
                    f"object {object_key!r} is pinned by "
                    f"{record.active_readers} active reader(s)"
                )
            if record.state != "VALID":
                raise SharedL1Error(
                    f"object {object_key!r} has invalid state {record.state!r}"
                )
            del self._objects[object_key]
            return True

    def finish_read(self, reservation: ReadReservation) -> None:
        """Release a successfully completed read reservation.

        Args:
            reservation: Reservation returned by :meth:`reserve_read`.

        Raises:
            InvalidReservationError: If the token does not own an active read.
            StaleHandleError: If the reservation metadata does not match.
        """
        self._release_read(reservation)

    def abort_read(self, reservation: ReadReservation) -> None:
        """Release an incomplete read reservation.

        Args:
            reservation: Reservation returned by :meth:`reserve_read`.

        Raises:
            InvalidReservationError: If the token does not own an active read.
            StaleHandleError: If the reservation metadata does not match.
        """
        self._release_read(reservation)

    def snapshot(self) -> dict[str, object]:
        """Return a token-free snapshot of pool-owned metadata.

        Returns:
            Region configuration, allocator cursors, and each object's handle,
            state, and active-reader count. Reservation tokens are omitted.
        """
        with self._lock:
            objects = {
                object_key: {
                    "handle": record.handle,
                    "state": record.state,
                    "active_readers": record.active_readers,
                }
                for object_key, record in self._objects.items()
            }
            return {
                "region_id": self._contract.region_id,
                "capacity": self._contract.capacity,
                "alignment": self._contract.alignment,
                "layout_id": self._contract.layout_id,
                "generation_epoch": self._contract.generation_epoch,
                "next_offset": self._next_offset,
                "next_generation": self._next_generation,
                "objects": objects,
            }

    def _validate_write_reservation(
        self,
        reservation: WriteReservation,
    ) -> _ObjectRecord:
        record = self._objects.get(reservation.object_key)
        if (
            record is None
            or record.state != "WRITING"
            or record.write_token != reservation.token
        ):
            raise InvalidReservationError(
                f"token does not own WRITING object {reservation.object_key!r}"
            )
        if record.handle != reservation.handle:
            raise StaleHandleError(
                f"reservation handle {reservation.handle!r} does not match "
                f"{record.handle!r}"
            )
        return record

    def _release_read(self, reservation: ReadReservation) -> None:
        with self._lock:
            lease = self._read_leases.get(reservation.token)
            if lease is None or lease.object_key != reservation.object_key:
                raise InvalidReservationError("token does not own an active read")
            if lease.handle != reservation.handle:
                raise StaleHandleError(
                    f"reservation handle {reservation.handle!r} does not "
                    f"match {lease.handle!r}"
                )

            record = self._objects.get(reservation.object_key)
            if record is None or record.active_readers <= 0:
                raise SharedL1Error("active reader accounting is inconsistent")
            del self._read_leases[reservation.token]
            record.active_readers -= 1

    @staticmethod
    def _validate_object_key(object_key: str) -> None:
        if not isinstance(object_key, str) or not object_key:
            raise ValueError("object_key must be a non-empty string")

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return ((value + alignment - 1) // alignment) * alignment


class SharedMemoryRegion:
    """Map an existing file or character device with ``MAP_SHARED``.

    Args:
        path: Existing regular file or Device-DAX character device.
        contract: Immutable region contract advertised for the pool.
        mapping_offset: Host-local byte offset at which to map the pool.
        expected_contract: Optional local expectation to validate before
            mapping.
        visibility: Platform-qualified exact-range visibility primitive.

    Raises:
        RegionContractMismatchError: If the advertised and locally expected
            contracts differ.
        ValueError: If arguments, file type, or file size are invalid.
        SharedL1Error: If a character device has no visibility primitive or
            the contract cannot isolate objects at its visibility granularity.
        OSError: If opening or mapping the pool fails.

    Notes:
        The mapping's virtual address and ``mapping_offset`` are local details;
        neither appears in :class:`SharedObjectHandle`.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        contract: SharedRegionContract,
        mapping_offset: int = 0,
        expected_contract: SharedRegionContract | None = None,
        *,
        visibility: SharedMemoryVisibility | None = None,
    ) -> None:
        if expected_contract is not None and contract != expected_contract:
            raise RegionContractMismatchError(
                f"advertised contract {contract!r} does not match local "
                f"expectation {expected_contract!r}"
            )
        if mapping_offset < 0:
            raise ValueError("mapping_offset must be non-negative")
        if mapping_offset % mmap.PAGESIZE != 0:
            raise ValueError(f"mapping_offset must be aligned to {mmap.PAGESIZE} bytes")
        if visibility is not None:
            visibility_granularity = visibility.granularity
            if (
                isinstance(visibility_granularity, bool)
                or not isinstance(visibility_granularity, int)
                or visibility_granularity <= 0
                or visibility_granularity & (visibility_granularity - 1)
            ):
                raise ValueError(
                    "visibility granularity must be a positive power of two"
                )
            if (
                contract.alignment < visibility_granularity
                or contract.alignment % visibility_granularity != 0
            ):
                raise SharedL1Error(
                    f"region alignment {contract.alignment} cannot isolate "
                    f"{visibility_granularity}-byte visibility ranges"
                )
            if mapping_offset % visibility_granularity != 0:
                raise SharedL1Error(
                    f"mapping offset {mapping_offset} is not aligned to the "
                    f"{visibility_granularity}-byte visibility granularity"
                )

        region_path = os.fspath(path)
        file_descriptor = os.open(region_path, os.O_RDWR)
        mapping: mmap.mmap | None = None
        mapping_marker: ctypes.c_ubyte | None = None
        try:
            file_stat = os.fstat(file_descriptor)
            is_regular = stat.S_ISREG(file_stat.st_mode)
            if not is_regular and not stat.S_ISCHR(file_stat.st_mode):
                raise ValueError(
                    "shared region must be a regular file or character device"
                )
            if not is_regular and visibility is None:
                raise SharedL1Error(
                    "shared character-device mappings require an explicit "
                    "visibility primitive"
                )
            if is_regular and mapping_offset + contract.capacity > file_stat.st_size:
                raise ValueError(
                    "regular file is smaller than mapping_offset + contract capacity"
                )
            mapping = mmap.mmap(
                file_descriptor,
                contract.capacity,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=mapping_offset,
            )
            if visibility is not None:
                mapping_marker = ctypes.c_ubyte.from_buffer(mapping)
                mapped_address = ctypes.addressof(mapping_marker)
                if mapped_address % visibility_granularity != 0:
                    raise SharedL1Error(
                        f"mapped base address is not aligned to the "
                        f"{visibility_granularity}-byte visibility granularity"
                    )
            else:
                mapped_address = None
        except BaseException:
            if mapping_marker is not None:
                del mapping_marker
            if mapping is not None:
                mapping.close()
            os.close(file_descriptor)
            raise

        self.contract = contract
        self.mapping_offset = mapping_offset
        self._mapping: mmap.mmap | None = mapping
        self._mapping_marker: ctypes.c_ubyte | None = mapping_marker
        self._mapped_address: int | None = mapped_address
        self._file_descriptor: int | None = file_descriptor
        self._is_regular = is_regular
        self._visibility = visibility
        self._active_views = 0
        self._operation_lock = threading.RLock()

    def write(
        self,
        handle: SharedObjectHandle,
        payload: bytes | bytearray | memoryview,
    ) -> None:
        """Copy one complete payload into its reserved logical extent.

        Args:
            handle: Handle returned in an active write reservation.
            payload: Exactly ``handle.length`` bytes.

        Raises:
            StaleHandleError: If the handle belongs to another region.
            ValueError: If the handle is out of bounds or length differs.
            SharedL1Error: If the region is closed.

        Notes:
            The pool child may publish the object only after this method
            returns. Callers must not write again after ``finish_write``.
        """
        with self._operation_lock:
            mapping, _, _ = self._ensure_open()
            start, end = self._bounds(handle)
            payload_view = memoryview(payload)
            if payload_view.nbytes != handle.length:
                raise ValueError(
                    f"payload has {payload_view.nbytes} bytes, expected {handle.length}"
                )
            if not payload_view.contiguous:
                raise ValueError("payload must be contiguous")
            mapping[start:end] = payload_view.cast("B")

    def publish(self, handle: SharedObjectHandle) -> None:
        """Publish one completed write before committing it as ``VALID``.

        Args:
            handle: Handle returned in the active write reservation.

        Raises:
            StaleHandleError: If the handle belongs to another region epoch.
            ValueError: If the handle lies outside this mapping.
            SharedL1Error: If a character-device visibility primitive is
                unavailable or the region is closed.

        Notes:
            Call this only after every CPU or DMA write has completed, and call
            ``finish_write`` only after this method succeeds.
            Exceptions raised by the visibility primitive propagate unchanged.
        """
        with self._operation_lock:
            mapping, file_descriptor, mapped_address = self._ensure_open()
            start, _ = self._bounds(handle)
            if self._visibility is None:
                if not self._is_regular:
                    raise SharedL1Error(
                        "Device-DAX publish requires a visibility primitive"
                    )
                mapping.flush()
                return
            assert mapped_address is not None
            self._visibility.publish(
                device_fd=file_descriptor,
                mapped_address=mapped_address + start,
                device_offset=self.mapping_offset + start,
                length=handle.length,
                generation=handle.generation,
            )

    def acquire(self, handle: SharedObjectHandle) -> None:
        """Acquire one published object before the first payload load.

        Args:
            handle: Handle returned in an active read reservation.

        Raises:
            StaleHandleError: If the handle belongs to another region epoch.
            ValueError: If the handle lies outside this mapping.
            SharedL1Error: If a character-device visibility primitive is
                unavailable or the region is closed.

        Notes:
            ``read`` and ``read_view`` call this automatically. Call it
            directly only before another consumer, such as an H2D transfer,
            reads bytes from the mapped range.
            Exceptions raised by the visibility primitive propagate unchanged.
        """
        with self._operation_lock:
            _, file_descriptor, mapped_address = self._ensure_open()
            start, _ = self._bounds(handle)
            self._acquire_locked(
                handle,
                start,
                file_descriptor=file_descriptor,
                mapped_address=mapped_address,
            )

    @contextmanager
    def read_view(
        self,
        handle: SharedObjectHandle,
    ) -> Iterator[memoryview]:
        """Yield a zero-copy, read-only view of one object.

        Args:
            handle: Handle from an active read reservation.

        Yields:
            A read-only memoryview valid only inside the context.

        Raises:
            StaleHandleError: If the handle belongs to another region.
            ValueError: If the handle is outside this mapping.
            SharedL1Error: If the region is closed or visibility is unavailable.

        Notes:
            The visibility acquire completes before the view is constructed.
            The view is released on context exit so this region can then be
            closed without retaining an exported mmap buffer.
            Exceptions raised by the visibility primitive propagate unchanged.
        """
        with self._operation_lock:
            mapping, file_descriptor, mapped_address = self._ensure_open()
            start, end = self._bounds(handle)
            self._acquire_locked(
                handle,
                start,
                file_descriptor=file_descriptor,
                mapped_address=mapped_address,
            )
            mapping_view = memoryview(mapping)
            object_view = mapping_view[start:end]
            read_only_view = object_view.toreadonly()
            self._active_views += 1
        try:
            yield read_only_view
        finally:
            read_only_view.release()
            object_view.release()
            mapping_view.release()
            with self._operation_lock:
                self._active_views -= 1

    def read(self, handle: SharedObjectHandle) -> bytes:
        """Copy a complete object from its logical extent.

        Args:
            handle: Handle from an active read reservation.

        Returns:
            The object's bytes.

        Raises:
            StaleHandleError: If the handle belongs to another region.
            ValueError: If the handle is outside this mapping.
            SharedL1Error: If the region is closed or visibility is unavailable.

        Notes:
            Exceptions raised by the visibility primitive propagate unchanged.
        """
        with self._operation_lock:
            mapping, file_descriptor, mapped_address = self._ensure_open()
            start, end = self._bounds(handle)
            self._acquire_locked(
                handle,
                start,
                file_descriptor=file_descriptor,
                mapped_address=mapped_address,
            )
            return mapping[start:end]

    def flush(self) -> None:
        """Flush a regular-file mapping.

        Raises:
            SharedL1Error: If called for a character-device mapping or the
                region is closed.
            OSError: If the operating system cannot flush a regular file.

        Notes:
            This method is not a Device-DAX visibility boundary. Device-DAX
            writers must call :meth:`publish` with the exact object handle.
        """
        with self._operation_lock:
            mapping, _, _ = self._ensure_open()
            if not self._is_regular:
                raise SharedL1Error(
                    "flush is not a Device-DAX visibility boundary; "
                    "call publish(handle)"
                )
            mapping.flush()

    def close(self) -> None:
        """Close this process's local mapping and retained descriptor.

        Raises:
            SharedL1Error: If a zero-copy view is still active.

        Notes:
            Repeated calls are safe. Closing is serialized with reads, writes,
            and visibility operations.
        """
        with self._operation_lock:
            if self._file_descriptor is None:
                return
            if self._active_views:
                raise SharedL1Error(
                    f"cannot close shared region with {self._active_views} "
                    "active zero-copy view(s)"
                )
            mapping = self._mapping
            mapping_marker = self._mapping_marker
            file_descriptor = self._file_descriptor
            assert mapping is not None

            self._mapping_marker = None
            self._mapped_address = None
            if mapping_marker is not None:
                del mapping_marker
            try:
                mapping.close()
            except BaseException:
                if self._visibility is not None:
                    mapping_marker = ctypes.c_ubyte.from_buffer(mapping)
                    self._mapping_marker = mapping_marker
                    self._mapped_address = ctypes.addressof(mapping_marker)
                raise
            self._mapping = None
            self._file_descriptor = None
            os.close(file_descriptor)

    def __del__(self) -> None:
        """Release an unclosed mapping during garbage collection."""
        if getattr(self, "_file_descriptor", None) is None:
            return
        try:
            warnings.warn(
                "unclosed SharedMemoryRegion; releasing it during garbage collection",
                ResourceWarning,
                stacklevel=2,
                source=self,
            )
        except BaseException:
            pass
        finally:
            try:
                self.close()
            except BaseException:
                pass

    def __enter__(self) -> "SharedMemoryRegion":
        """Return this mapping for use as a context manager."""
        return self

    def __exit__(
        self,
        exception_type: object,
        exception: object,
        traceback: object,
    ) -> None:
        """Close the local mapping when leaving a context."""
        self.close()

    def _bounds(self, handle: SharedObjectHandle) -> tuple[int, int]:
        if handle.region_id != self.contract.region_id:
            raise StaleHandleError(
                f"handle belongs to {handle.region_id!r}, contract belongs to "
                f"{self.contract.region_id!r}"
            )
        generation_epoch = handle.generation >> _GENERATION_SEQUENCE_BITS
        if generation_epoch != self.contract.generation_epoch:
            raise StaleHandleError(
                f"handle generation epoch {generation_epoch} does not match "
                f"contract epoch {self.contract.generation_epoch}"
            )
        if handle.offset > self.contract.capacity - handle.length:
            raise ValueError("handle lies outside the mapped logical region")
        if handle.offset % self.contract.alignment != 0:
            raise ValueError(
                f"handle offset must be aligned to {self.contract.alignment} bytes"
            )
        return handle.offset, handle.offset + handle.length

    def _acquire_locked(
        self,
        handle: SharedObjectHandle,
        start: int,
        *,
        file_descriptor: int,
        mapped_address: int | None,
    ) -> None:
        if self._visibility is None:
            if not self._is_regular:
                raise SharedL1Error(
                    "Device-DAX acquire requires a visibility primitive"
                )
            return
        assert mapped_address is not None
        self._visibility.acquire(
            device_fd=file_descriptor,
            mapped_address=mapped_address + start,
            device_offset=self.mapping_offset + start,
            length=handle.length,
            generation=handle.generation,
        )

    def _ensure_open(self) -> tuple[mmap.mmap, int, int | None]:
        mapping = self._mapping
        file_descriptor = self._file_descriptor
        if mapping is None or file_descriptor is None:
            raise SharedL1Error("shared memory region is closed")
        return mapping, file_descriptor, self._mapped_address
