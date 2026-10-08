# SPDX-License-Identifier: Apache-2.0
"""Remote binding of the L1 lifecycle: a region a memory orchestrator owns."""

# Standard
from dataclasses import dataclass, field
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, SharedL1Config
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerListener,
    L1MemoryDesc,
    L1ObjectMeta,
    L1OperationResult,
)
from lmcache.v1.distributed.l1_manager import next_l1_manager_id, validate_read_locks
from lmcache.v1.distributed.shared_l1.media import RegionMedium, open_medium
from lmcache.v1.distributed.shared_l1.visibility import Visibility, create_visibility
from lmcache.v1.distributed.shared_l1.wire import (
    key_to_wire,
    layout_from_wire,
    layout_to_wire,
)
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    Handle,
    OrchestratorError,
    OrchestratorUnavailableError,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionFencedError,
    RegionUsage,
    TokenStatus,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

logger = init_logger(__name__)

# The eviction controller polls usage every second and status endpoints poll
# too; one Usage RPC per this many seconds serves all of them.
_USAGE_TTL_SECONDS = 1.0

# After the orchestrator was unreachable, lookups, reservations and usage
# polls skip the RPC for this long instead of each paying the client's
# retries; they miss or overflow at once.
_UNREACHABLE_COOLDOWN_SECONDS = 1.0


@dataclass
class _WriteGrant:
    """A write extent this server reserved and has not committed or aborted."""

    token: bytes
    handle: Handle
    payload_bytes: int
    memory_obj: TensorMemoryObj


@dataclass
class _ReadHold:
    """Read leases this server holds on one object, and its view."""

    memory_obj: TensorMemoryObj
    leases: list[bytes] = field(default_factory=list)


class SharedL1Manager:
    """An L1 whose objects live in a region a memory orchestrator owns.

    Several MP servers map the same region, each through a medium on its
    host: a Device-DAX device for CXL memory several hosts attach, or a file
    for several servers on one host. The orchestrator (``lmcache memory``)
    owns extents, the key index and object state; this manager maps the
    medium and moves KV bytes. It keeps only what this
    process holds: write extents until they are committed or aborted, and
    read leases until they are released. Commits and reads are bracketed by
    the region's visibility hook, so another host never reads a stale line.

    Nothing here crashes the server: an unreachable orchestrator turns reads
    into misses and writes into ``OUT_OF_MEMORY`` (ordered overflow then moves
    them to the next L1). An orchestrator that restarted or wants a reset
    fences the manager for good; it serves nothing until the server restarts.

    Args:
        config: An L1 configuration with a ``shared`` section and a
            ``client_id`` (the MP server fills it); its type picks the medium.

    Raises:
        ValueError: The configuration is incomplete or does not match the
            orchestrator's region.
        RuntimeError: The orchestrator wants an offline reset, or visibility
            is unsupported on this host.
        OrchestratorError: The orchestrator refused or could not be reached.
        OSError: The medium cannot be mapped.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        shared = config.shared
        if shared is None or shared.client_id is None:
            raise ValueError("A shared L1 needs a shared section with a client_id")
        self._config = config
        self._shared: SharedL1Config = shared
        self._l1_manager_id = next_l1_manager_id()
        self._lock = threading.Lock()
        self._listeners: list[L1ManagerListener] = []
        self._event_bus = get_event_bus()

        self._client = MemoryOrchestratorClient(
            shared.orchestrator,
            shared.region_id,
            shared.client_id,
            rpc_timeout_s=shared.rpc_timeout_seconds,
        )
        medium: RegionMedium | None = None
        visibility: Visibility | None = None
        try:
            contract = self._client.describe_region()
            if contract.reset_required:
                raise RuntimeError(
                    f"Memory orchestrator for region {shared.region_id!r} needs "
                    "an offline reset: stop every server using the region, "
                    "stop the orchestrator, delete its startup marker, restart"
                )
            if contract.capacity_bytes > config.memory_config.size_in_bytes:
                raise ValueError(
                    f"Region {shared.region_id!r} holds {contract.capacity_bytes} "
                    "bytes, more than this L1's size_gb allows "
                    f"({config.memory_config.size_in_bytes} bytes)"
                )
            visibility = create_visibility(contract.visibility_mode)
            medium = open_medium(
                config, contract.capacity_bytes, contract.alignment_bytes
            )
            self._client.register(
                layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
                mapped_bytes=medium.capacity_bytes,
                visibility_mode=visibility.mode,
            )
        except BaseException:
            if medium is not None:
                medium.close()
            if visibility is not None:
                visibility.close()
            self._client.close()
            raise
        self._contract: RegionContract = contract
        self._medium: RegionMedium = medium
        self._visibility = visibility

        self._writes: dict[tuple[ObjectKey, str], _WriteGrant] = {}
        self._reads: dict[ObjectKey, _ReadHold] = {}
        self._staging_bytes = 0
        self._reachable = True
        self._retry_after = 0.0
        self._fenced_reason: str | None = None
        self._usage: RegionUsage | None = None
        self._usage_time = 0.0
        logger.info(
            "Shared L1 %r mapped region %r (%d bytes, epoch %d, %s) through %s "
            "%s (%s) as client %r",
            config.tag,
            shared.region_id,
            contract.capacity_bytes,
            self._client.region_epoch,
            visibility.mode,
            medium.backend_type.value,
            medium.path,
            medium.data_path,
            shared.client_id,
        )

    # Identity and description

    @property
    def l1_manager_id(self) -> int:
        """Return this manager's process-local owner tag."""
        return self._l1_manager_id

    @property
    def config(self) -> L1ManagerConfig:
        """Return this manager's configuration."""
        return self._config

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for this L1's lifecycle notifications.

        Args:
            listener: The listener to register.
        """
        with self._lock:
            self._listeners.append(listener)

    # Write path

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Ask the orchestrator for one extent per key.

        Args:
            keys: Keys to write.
            is_temporary: Must be all False; shared objects are never
                temporary.
            layout_desc: Layout of every object in the batch.
            tag: The writer's identity; ``finish_write`` must pass the same.

        Returns:
            Per key ``(SUCCESS, view)`` for a granted extent,
            ``(KEY_NOT_WRITABLE, None)`` when the key is already committed or
            being written by any server (including this tag), and
            ``(OUT_OF_MEMORY, None)`` when the region is full or the
            orchestrator cannot be used, so ordered overflow tries the next L1.

        Raises:
            ValueError: ``keys`` and ``is_temporary`` differ in length, or a
                temporary object is requested.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )
        if any(is_temporary):
            raise ValueError("A shared L1 holds no temporary objects")
        ret: dict[ObjectKey, L1OperationResult] = {}
        with self._lock:
            pending = []
            for key in keys:
                if (key, tag) in self._writes:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                else:
                    pending.append(key)
        if not pending:
            return ret
        if self._skip_request_rpc():
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret

        payload_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        wire_layout = layout_to_wire(layout_desc)
        try:
            grants = self._client.reserve_write(
                [
                    WriteRequest(key_to_wire(key), payload_bytes, wire_layout)
                    for key in pending
                ]
            )
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveWrite", exc)
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret
        self._on_rpc_ok()

        granted: list[ObjectKey] = []
        bad_tokens: list[bytes] = []
        with self._lock:
            for key, grant in zip(pending, grants, strict=True):
                if grant.status is WriteStatus.OUT_OF_SPACE:
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue
                if grant.status is not WriteStatus.WRITE_GRANTED:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                assert grant.handle is not None and grant.token is not None
                try:
                    memory_obj = self._medium.view(
                        grant.handle.offset, grant.handle.length, layout_desc
                    )
                except ValueError:
                    logger.exception("Shared L1: orchestrator granted a bad extent")
                    bad_tokens.append(grant.token)
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                memory_obj.set_l1_manager(self._l1_manager_id)
                self._writes[(key, tag)] = _WriteGrant(
                    grant.token, grant.handle, payload_bytes, memory_obj
                )
                self._staging_bytes += grant.handle.length
                ret[key] = (L1Error.SUCCESS, memory_obj)
                granted.append(key)
            listeners = list(self._listeners)
        if bad_tokens:
            self._abort_tokens(bad_tokens)

        for listener in listeners:
            listener.on_l1_keys_reserved_write(granted)
        self._publish(EventType.L1_WRITE_RESERVED, {"keys": granted, "tag": tag})
        return ret

    def finish_write(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Publish ``tag``'s written extents and commit them.

        The caller must have waited for the device-to-host copies.

        Args:
            keys: Keys whose extents were written.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS`` once committed, ``KEY_NOT_EXIST`` when ``tag``
            reserved nothing for the key, ``KEY_IN_WRONG_STATE`` when the
            orchestrator refused the commit or could not be asked (the store
            fails; a later writer may find the key committed after all).
        """
        ret, committed = self._commit(keys, tag)
        with self._lock:
            listeners = list(self._listeners)
        committed_keys = [key for key, _ in committed]
        if committed_keys:
            for listener in listeners:
                listener.on_l1_keys_write_finished(committed_keys)
            self._publish(
                EventType.L1_WRITE_FINISHED,
                {
                    "keys": committed_keys,
                    "meta": [self._object_meta(grant) for _, grant in committed],
                },
            )
        return ret

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1, tag: str = ""
    ) -> dict[ObjectKey, L1OperationResult]:
        """Commit ``tag``'s extents, then take read leases on them.

        Two round trips, not one atomic step. That is safe here: a committed
        object stays readable until the region is reset.

        Args:
            keys: Keys whose extents were written.
            read_locks: Read leases to take per committed key.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``(SUCCESS, view)`` with leases taken, else the commit or
            read error and None.
        """
        commit_errors, committed = self._commit(keys, tag)
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (error, None)
            for key, error in commit_errors.items()
            if error is not L1Error.SUCCESS
        }
        committed_keys = [key for key, _ in committed]
        if committed_keys:
            ret.update(self._reserve_read(committed_keys, read_locks, notify=False))
            with self._lock:
                listeners = list(self._listeners)
            for listener in listeners:
                listener.on_l1_keys_finish_write_and_reserve_read(committed_keys)
            self._publish(
                EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                {
                    "keys": committed_keys,
                    "meta": [self._object_meta(grant) for _, grant in committed],
                },
            )
        return ret

    def finish_write_and_delete(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Abort ``tag``'s write reservations.

        The orchestrator retires the extents (they are never reused here)
        and the keys become writable again.

        Args:
            keys: Keys whose reservations to abort.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when ``tag`` reserved
            nothing for the key, or ``KEY_IN_WRONG_STATE`` when the
            orchestrator refused or could not be asked.
        """
        ret: dict[ObjectKey, L1Error] = {}
        taken: list[tuple[ObjectKey, _WriteGrant]] = []
        with self._lock:
            for key in keys:
                grant = self._writes.pop((key, tag), None)
                if grant is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                self._staging_bytes -= grant.handle.length
                taken.append((key, grant))
            listeners = list(self._listeners)
        if not taken:
            return ret
        statuses = self._abort_tokens([grant.token for _, grant in taken])
        for (key, _), status in zip(taken, statuses, strict=True):
            ret[key] = (
                L1Error.SUCCESS
                if status is TokenStatus.OK
                else L1Error.KEY_IN_WRONG_STATE
            )
        gone = [key for key, _ in taken]
        for listener in listeners:
            listener.on_l1_keys_deleted_by_manager(gone)
        return ret

    # Read path

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Take ``read_locks`` read leases per committed key.

        Args:
            keys: Keys to read.
            read_locks: Leases per key, one per reader of the object.

        Returns:
            Per key ``(SUCCESS, view)`` with the leases held, else
            ``(KEY_NOT_EXIST, None)``: missing, still being written, or the
            orchestrator cannot be used.
        """
        return self._reserve_read(keys, read_locks, notify=True)

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return views of keys this server holds read leases on.

        Args:
            keys: Keys previously returned by ``reserve_read``.

        Returns:
            Per key ``(SUCCESS, view)``, or ``(KEY_NOT_EXIST, None)`` when no
            lease is held.
        """
        with self._lock:
            ret: dict[ObjectKey, L1OperationResult] = {}
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                else:
                    ret[key] = (L1Error.SUCCESS, hold.memory_obj)
            return ret

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        """Release ``read_locks`` leases per key.

        Args:
            keys: Keys to release.
            read_locks: Leases to release per key.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when no lease is held, or
            ``KEY_IN_WRONG_STATE`` when the orchestrator refused a lease or
            could not be asked (the lease is dropped locally either way).
        """
        count = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1Error] = {}
        released: list[tuple[ObjectKey, list[bytes]]] = []
        with self._lock:
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                tokens = hold.leases[-count:]
                del hold.leases[-count:]
                if not hold.leases:
                    del self._reads[key]
                released.append((key, tokens))
            listeners = list(self._listeners)
        if not released:
            return ret
        tokens = [token for _, key_tokens in released for token in key_tokens]
        statuses: list[TokenStatus]
        try:
            statuses = self._client.finish_read(tokens)
            self._on_rpc_ok()
        except OrchestratorError as exc:
            self._on_rpc_error("FinishRead", exc)
            statuses = [TokenStatus.STALE_TOKEN] * len(tokens)
        position = 0
        done: list[ObjectKey] = []
        for key, key_tokens in released:
            key_statuses = statuses[position : position + len(key_tokens)]
            position += len(key_tokens)
            if all(status is TokenStatus.OK for status in key_statuses):
                ret[key] = L1Error.SUCCESS
                done.append(key)
            else:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        for listener in listeners:
            listener.on_l1_keys_read_finished(done)
        self._publish(EventType.L1_READ_FINISHED, {"keys": done})
        return ret

    # Operations the shared region does not support in this milestone

    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Refuse: shared objects are not deleted in this milestone.

        Args:
            keys: Keys to delete.
            force: Ignored; nothing can force a shared delete.

        Returns:
            ``KEY_IS_LOCKED`` for every key.
        """
        return {key: L1Error.KEY_IS_LOCKED for key in keys}

    def clear(self, force: bool = False) -> None:
        """Do nothing: shared objects are not dropped in this milestone.

        Args:
            force: Ignored.
        """
        logger.info(
            "Shared L1 %r: clear leaves shared objects in place", self._config.tag
        )

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Tell listeners the keys were accessed; no lock, no RPC.

        Args:
            keys: Keys that were retrieved or stored.
        """
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            listener.on_l1_keys_accessed(keys)

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Return False: shared objects are never evicted here.

        Args:
            key: Any key.

        Returns:
            False.
        """
        return False

    # Capacity and status

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(allocated_bytes, capacity_bytes)`` of the whole region.

        Allocation is monotonic, so allocated bytes include extents of
        aborted writes. The value is at most one second old.

        Returns:
            Region-wide usage; the last known usage, or zero used bytes, when
            the orchestrator cannot be asked.
        """
        usage = self._region_usage()
        used = usage.allocated_bytes if usage is not None else 0
        return used, self._contract.capacity_bytes

    def get_staging_memory_usage(self) -> int:
        """Return bytes of extents this server reserved and has not committed."""
        with self._lock:
            return self._staging_bytes

    def get_capacity_bytes_by_backend(self) -> dict[L1BackendType, int]:
        """Return the region capacity under the medium's backend type."""
        return {self._medium.backend_type: self._contract.capacity_bytes}

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None: L2 adapters never register the shared region."""
        return None

    def owns_device(self, device_path: str) -> bool:
        """Whether ``device_path`` names the mapped medium.

        Args:
            device_path: Candidate device path or alias.

        Returns:
            True when it resolves to the mapped device or file.
        """
        return self._medium.owns_device(device_path)

    def memory_region_count(self) -> int:
        """Return 1: one mapped region."""
        return 1

    def report_status(self) -> dict:
        """Return the L1 status fields plus a ``shared`` section.

        ``is_healthy`` is False while the orchestrator is unreachable and for
        good once the manager is fenced.
        """
        usage = self._region_usage()
        used, total = self.get_memory_usage()
        with self._lock:
            write_count = len(self._writes)
            read_count = len(self._reads)
            staging_bytes = self._staging_bytes
        return {
            "is_healthy": self.memcheck(),
            "total_object_count": usage.valid + usage.writing if usage else 0,
            "write_locked_count": write_count,
            "read_locked_count": read_count,
            "temporary_count": 0,
            "staging_object_count": write_count,
            "staging_bytes": staging_bytes,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_configured_bytes": total,
            "capacity_bytes_by_backend": {self._medium.backend_type.value: total},
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            "write_ttl_seconds": self._config.write_ttl_seconds,
            "read_ttl_seconds": self._config.read_ttl_seconds,
            "shared": {
                "orchestrator": self._shared.orchestrator,
                "region_id": self._shared.region_id,
                "client_id": self._shared.client_id,
                "region_epoch": self._client.region_epoch,
                "medium": self._medium.backend_type.value,
                "path": self._medium.path,
                "visibility_mode": self._visibility.mode,
                "data_path": self._medium.data_path,
                "orchestrator_reachable": self._reachable,
                "fenced": self._fenced_reason,
                "region_usage": None
                if usage is None
                else {
                    "allocated_bytes": usage.allocated_bytes,
                    "valid_bytes": usage.valid_bytes,
                    "writing": usage.writing,
                    "valid": usage.valid,
                    "consumed": usage.consumed,
                    "read_leases": usage.read_leases,
                    "clients": usage.clients,
                },
            },
        }

    def memcheck(self) -> bool:
        """Whether the orchestrator was reachable on the last call and the
        manager is not fenced."""
        return self._reachable and self._fenced_reason is None

    def close(self) -> None:
        """Abort unfinished writes, release leases, leave the region.

        The orchestrator retires this client's remaining state on
        ``CloseClient`` as well, so a failed RPC here leaks nothing.
        """
        with self._lock:
            writes = list(self._writes.values())
            leases = [token for hold in self._reads.values() for token in hold.leases]
            self._writes.clear()
            self._reads.clear()
            self._staging_bytes = 0
        if self._fenced_reason is None:
            if writes:
                self._abort_tokens([grant.token for grant in writes])
            if leases:
                try:
                    self._client.finish_read(leases)
                except OrchestratorError:
                    logger.warning("Shared L1: releasing read leases at close failed")
        for grant in writes:
            grant.memory_obj.invalidate()
        self._client.close()
        self._visibility.close()
        self._medium.close()

    # Private helpers

    def _commit(
        self, keys: list[ObjectKey], tag: str
    ) -> tuple[dict[ObjectKey, L1Error], list[tuple[ObjectKey, _WriteGrant]]]:
        """Publish and commit ``tag``'s extents; notify nobody."""
        ret: dict[ObjectKey, L1Error] = {}
        taken: list[tuple[ObjectKey, _WriteGrant]] = []
        with self._lock:
            for key in keys:
                grant = self._writes.pop((key, tag), None)
                if grant is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                self._staging_bytes -= grant.handle.length
                taken.append((key, grant))
        if not taken:
            return ret, []
        if self._fenced_reason is not None:
            ret.update((key, L1Error.KEY_IN_WRONG_STATE) for key, _ in taken)
            return ret, []
        # The copies into these extents are complete; push the bytes out of
        # this host's caches before anyone else may read them.
        self._visibility.publish(
            [
                (self._medium.address(grant.handle.offset), grant.payload_bytes)
                for _, grant in taken
            ]
        )
        try:
            statuses = self._client.finish_write([grant.token for _, grant in taken])
        except OrchestratorError as exc:
            self._on_rpc_error("FinishWrite", exc)
            ret.update((key, L1Error.KEY_IN_WRONG_STATE) for key, _ in taken)
            return ret, []
        self._on_rpc_ok()
        committed: list[tuple[ObjectKey, _WriteGrant]] = []
        for (key, grant), status in zip(taken, statuses, strict=True):
            if status is TokenStatus.OK:
                ret[key] = L1Error.SUCCESS
                committed.append((key, grant))
            else:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        return ret, committed

    def _reserve_read(
        self, keys: list[ObjectKey], read_locks: int, notify: bool
    ) -> dict[ObjectKey, L1OperationResult]:
        """Lease committed objects, invalidate their ranges, return views."""
        count = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_EXIST, None) for key in keys
        }
        if not keys or self._skip_request_rpc():
            return ret
        try:
            grants = self._client.reserve_read(
                [ReadRequest(key_to_wire(key), count) for key in keys]
            )
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveRead", exc)
            return ret
        self._on_rpc_ok()

        granted: list[tuple[ObjectKey, ReadGrantResult]] = [
            (key, grant)
            for key, grant in zip(keys, grants, strict=True)
            if grant.status is ReadStatus.READ_GRANTED
        ]
        if not granted:
            return ret
        # Drop any line of these extents this host cached before the writer
        # published them; the host-to-device copy must read the device.
        self._visibility.acquire(
            [
                (self._medium.address(grant.handle.offset), grant.payload_bytes)
                for _, grant in granted
                if grant.handle is not None
            ]
        )
        bad_leases: list[bytes] = []
        leased: list[ObjectKey] = []
        with self._lock:
            for key, grant in granted:
                assert grant.handle is not None and grant.layout is not None
                hold = self._reads.get(key)
                if hold is None:
                    try:
                        memory_obj = self._medium.view(
                            grant.handle.offset,
                            grant.handle.length,
                            layout_from_wire(grant.layout),
                        )
                    except ValueError:
                        logger.exception("Shared L1: orchestrator granted a bad read")
                        bad_leases.extend(grant.leases)
                        continue
                    memory_obj.set_l1_manager(self._l1_manager_id)
                    hold = _ReadHold(memory_obj)
                    self._reads[key] = hold
                hold.leases.extend(grant.leases)
                ret[key] = (L1Error.SUCCESS, hold.memory_obj)
                leased.append(key)
            listeners = list(self._listeners)
        if bad_leases:
            try:
                self._client.finish_read(bad_leases)
            except OrchestratorError as exc:
                self._on_rpc_error("FinishRead", exc)
        if notify:
            for listener in listeners:
                listener.on_l1_keys_reserved_read(leased)
            self._publish(EventType.L1_READ_RESERVED, {"keys": leased})
        return ret

    def _abort_tokens(self, tokens: list[bytes]) -> list[TokenStatus]:
        """AbortWrite; on failure report every token stale."""
        if self._fenced_reason is not None:
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        try:
            statuses = self._client.abort_write(tokens)
        except OrchestratorError as exc:
            self._on_rpc_error("AbortWrite", exc)
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        self._on_rpc_ok()
        return statuses

    def _region_usage(self) -> RegionUsage | None:
        """Return region usage at most ``_USAGE_TTL_SECONDS`` old."""
        now = time.monotonic()
        if (
            not self._skip_request_rpc()
            and now - self._usage_time >= _USAGE_TTL_SECONDS
        ):
            self._usage_time = now
            try:
                self._usage = self._client.usage()
                self._on_rpc_ok()
            except OrchestratorError as exc:
                self._on_rpc_error("Usage", exc)
        return self._usage

    def _on_rpc_ok(self) -> None:
        if not self._reachable:
            logger.info("Shared L1 %r: orchestrator reachable again", self._config.tag)
        self._reachable = True

    def _on_rpc_error(self, rpc: str, exc: OrchestratorError) -> None:
        """Record a failed RPC; a fencing error disables the manager for good."""
        if isinstance(exc, RegionFencedError):
            if self._fenced_reason is None:
                self._fenced_reason = f"{rpc}: {exc}"
                logger.error(
                    "Shared L1 %r is fenced (%s). It serves nothing until this "
                    "server restarts after the region is reset.",
                    self._config.tag,
                    self._fenced_reason,
                )
            return
        if isinstance(exc, OrchestratorUnavailableError):
            self._retry_after = time.monotonic() + _UNREACHABLE_COOLDOWN_SECONDS
        if self._reachable:
            logger.warning(
                "Shared L1 %r: %s failed (%s); reads miss and writes overflow",
                self._config.tag,
                rpc,
                exc,
            )
        self._reachable = False

    def _skip_request_rpc(self) -> bool:
        """Whether a lookup, reservation or usage poll should not ask now.

        Fenced managers never ask again; an unreachable orchestrator is
        asked again after the cooldown. Commits, aborts and releases always
        ask, because they change state the orchestrator holds.
        """
        return self._fenced_reason is not None or time.monotonic() < self._retry_after

    def _publish(self, event_type: EventType, metadata: dict) -> None:
        # ``shared`` keeps these placements out of per-server fleet cache
        # events; the region, not this server, owns them.
        self._event_bus.publish(
            Event(
                event_type=event_type,
                metadata={"l1_tag": self._config.tag, "shared": True, **metadata},
            )
        )

    def _object_meta(self, grant: _WriteGrant) -> L1ObjectMeta:
        return L1ObjectMeta(
            size_bytes=grant.payload_bytes, backend=self._medium.backend_type
        )
