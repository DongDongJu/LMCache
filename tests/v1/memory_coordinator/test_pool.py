# SPDX-License-Identifier: Apache-2.0
"""Public contracts for the ephemeral shared-pool metadata authority."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from multiprocessing.connection import Connection
import multiprocessing
import threading

# Third Party
from pydantic import ValidationError
import pytest

# First Party
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.memory_coordinator.api import (
    InvalidReservationError,
    MemoryCoordinatorError,
    ObjectState,
    OutOfSpaceError,
    ReservationRef,
    StaleEpochError,
    StaleHandleError,
    WireLayout,
    WriteGrant,
    WriteReserveItem,
    canonical_key,
)
from lmcache.v1.memory_coordinator.pool import MemoryPool


def _key(seed: int) -> EncodedObjectKey:
    return EncodedObjectKey(f"{seed:08x}", "model", 0)


def _item(seed: int, length: int = 16) -> WriteReserveItem:
    return WriteReserveItem(
        key=_key(seed), layout=WireLayout(shapes=((length,),), dtypes=("uint8",))
    )


def _pool(capacity: int = 8192) -> MemoryPool:
    return MemoryPool("region", capacity, 64, "layout")


def _write(pool: MemoryPool, seed: int) -> WriteGrant:
    grant = pool.reserve_writes([_item(seed)])[0]
    assert grant is not None
    return grant


def _ref(grant: ReservationRef) -> ReservationRef:
    return ReservationRef(key=grant.key, handle=grant.handle, token=grant.token)


def _check_inherited_pool(pool: MemoryPool, connection: Connection) -> None:
    outcomes = [not pool.is_healthy()]
    for operation in (pool.snapshot, pool.region_contract, pool.close):
        try:
            operation()
            outcomes.append(False)
        except MemoryCoordinatorError:
            outcomes.append(True)
    connection.send(outcomes)
    connection.close()


def test_lifecycle_partial_reads_and_independent_pins() -> None:
    pool = _pool()
    writer = _write(pool, 1)
    assert pool.reserve_reads([_key(1), _key(2)]) == [None, None]
    pool.finish_writes([_ref(writer)])
    assert pool.reserve_writes([_item(1, 128)]) == [None]
    first, miss = pool.reserve_reads([_key(1), _key(2)])
    second = pool.reserve_reads([_key(1)])[0]
    assert first is not None and second is not None and miss is None
    assert first.handle == second.handle == writer.handle
    assert first.layout == second.layout == writer.layout
    assert first.token != second.token
    pool.finish_reads([_ref(first)])
    record = pool.snapshot().objects[canonical_key(_key(1))]
    assert record.state is ObjectState.VALID and record.active_readers == 1
    with pytest.raises(InvalidReservationError):
        pool.finish_reads([_ref(first)])
    pool.abort_reads([_ref(second)])
    assert pool.snapshot().objects[canonical_key(_key(1))].active_readers == 0
    with pytest.raises(InvalidReservationError):
        pool.finish_writes([_ref(writer)])


def test_alignment_atomic_capacity_and_nonreuse() -> None:
    pool = _pool(capacity=192)
    first = _write(pool, 1)
    assert first.handle.offset == 0
    before = pool.snapshot()
    with pytest.raises(OutOfSpaceError):
        pool.reserve_writes([_item(2), _item(3, 65)])
    assert pool.snapshot() == before
    pool.abort_writes([_ref(first)])
    second = _write(pool, 1)
    assert second.handle.offset == 64
    assert second.handle.generation == first.handle.generation + 1
    last = pool.reserve_writes([_item(3, 64)])[0]
    assert last is not None and last.handle.offset == 128
    assert pool.snapshot().used_bytes == 192
    with pytest.raises(OutOfSpaceError):
        pool.reserve_writes([_item(4)])


def test_concurrent_writers_have_unique_disjoint_grants() -> None:
    pool = _pool()
    barrier = threading.Barrier(16)

    def reserve(items: list[WriteReserveItem]) -> list[WriteGrant | None]:
        barrier.wait(timeout=5)
        return pool.reserve_writes(items)

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(reserve, [[_item(i)] for i in range(16)]))
        results += list(executor.map(reserve, [[_item(99)]] * 16))
    grants = [batch[0] for batch in results if batch[0] is not None]
    assert len(grants) == 17
    assert len({grant.token for grant in grants}) == 17
    handles = sorted(
        (grant.handle for grant in grants), key=lambda handle: handle.offset
    )
    assert len({handle.generation for handle in handles}) == 17
    assert all(handle.offset % 64 == 0 for handle in handles)
    assert all(
        a.offset + a.length <= b.offset
        for a, b in zip(handles, handles[1:], strict=False)
    )


@pytest.mark.parametrize(
    "operation", ["finish_writes", "abort_writes", "finish_reads", "abort_reads"]
)
@pytest.mark.parametrize(
    "defect",
    ["token", "key", "duplicate", "region_id", "offset", "length", "generation"],
)
def test_invalid_batches_do_not_partially_change_state(
    operation: str, defect: str
) -> None:
    pool = _pool()
    grants: list[ReservationRef] = [_ref(_write(pool, seed)) for seed in (1, 2)]
    if operation.endswith("reads"):
        pool.finish_writes(grants)
        grants = [
            _ref(grant) for grant in pool.reserve_reads([_key(1), _key(2)]) if grant
        ]
    bad = grants[1]
    expected: type[MemoryCoordinatorError] = InvalidReservationError
    if defect == "token":
        bad = bad.model_copy(update={"token": "wrong"})
    elif defect == "key":
        bad = bad.model_copy(update={"key": _key(99)})
    elif defect == "duplicate":
        bad = grants[0]
    else:
        value = (
            "other-region" if defect == "region_id" else getattr(bad.handle, defect) + 1
        )
        bad = bad.model_copy(
            update={"handle": bad.handle.model_copy(update={defect: value})}
        )
        expected = StaleHandleError
    before = pool.snapshot()
    with pytest.raises(expected):
        getattr(pool, operation)([grants[0], bad])
    assert pool.snapshot() == before
    getattr(pool, operation)(grants)
    with pytest.raises(InvalidReservationError):
        getattr(pool, operation)(grants)


def test_empty_batches_and_duplicate_keys() -> None:
    pool = _pool()
    before = pool.snapshot()
    assert pool.reserve_writes([]) == pool.reserve_reads([]) == []
    for operation in (
        pool.finish_reads,
        pool.abort_reads,
        pool.finish_writes,
        pool.abort_writes,
    ):
        operation([])
    with pytest.raises(ValueError, match="duplicate"):
        pool.reserve_writes([_item(1), _item(1)])
    with pytest.raises(ValueError, match="duplicate"):
        pool.reserve_reads([_key(1), _key(1)])
    assert pool.snapshot() == before


@pytest.mark.parametrize(
    "shapes,dtypes",
    [
        ([], []),
        ([[0]], ["uint8"]),
        ([[-1]], ["uint8"]),
        ([[1]], []),
        ([[1]], ["object"]),
    ],
)
def test_invalid_layouts_fail_before_allocating(
    shapes: list, dtypes: list[str]
) -> None:
    pool = _pool()
    layout = WireLayout.model_validate({"shapes": shapes, "dtypes": dtypes})
    before = pool.snapshot()
    with pytest.raises(ValueError):
        pool.reserve_writes([_item(1), WriteReserveItem(key=_key(2), layout=layout)])
    assert pool.snapshot() == before


def test_layout_json_arrays_and_immutable_aliases() -> None:
    shapes = [[2, 3], []]
    dtypes = ["float16", "int32"]
    source = {"shapes": shapes, "dtypes": dtypes}
    layout = WireLayout.model_validate(source)
    assert layout.size_bytes() == 16
    pool = _pool()
    grant = pool.reserve_writes([WriteReserveItem(key=_key(1), layout=layout)])[0]
    assert grant is not None
    shapes[0][0] = 99
    dtypes[0] = "uint8"
    assert layout.shapes == ((2, 3), ()) and grant.layout.size_bytes() == 16
    with pytest.raises(TypeError):
        grant.layout.shapes[0][0] = 99  # type: ignore[index]
    with pytest.raises(ValidationError):
        grant.handle.offset = 99  # type: ignore[misc]
    assert WireLayout.model_validate_json(layout.model_dump_json()) == layout
    assert layout.model_dump(mode="json")["shapes"] == [[2, 3], []]
    pool.finish_writes([_ref(grant)])
    snapshot = pool.snapshot()
    assert "token" not in snapshot.model_dump_json()
    snapshot.objects.clear()
    assert pool.snapshot().object_count == 1


def test_metadata_models_reject_unknown_fields() -> None:
    grant = _write(_pool(), 1)
    for model in (grant, grant.handle, grant.layout, _ref(grant), _item(1)):
        with pytest.raises(ValidationError):
            type(model).model_validate({**model.model_dump(), "payload": [1]})
    for dimension in (True, 1.5, "1"):
        with pytest.raises(ValidationError):
            WireLayout.model_validate({"shapes": [[dimension]], "dtypes": ["uint8"]})


@pytest.mark.parametrize(
    "changes",
    [
        {"chunk_hash_hex": "AABB"},
        {"chunk_hash_hex": "aa bb"},
        {"chunk_hash_hex": "invalid"},
        {"model_name": "bad@model"},
        {"cache_salt": "bad/salt"},
    ]
    + [
        {field: value}
        for field in ("kv_rank", "object_group_id")
        for value in (True, 1.0, "1")
    ],
)
def test_invalid_or_noncanonical_keys_fail_closed(changes: dict[str, object]) -> None:
    pool = _pool()
    key = replace(_key(1), **changes)  # type: ignore[arg-type]
    before = pool.snapshot()
    with pytest.raises(ValueError):
        pool.reserve_writes([WriteReserveItem(key=key, layout=_item(1).layout)])
    with pytest.raises(ValueError):
        pool.reserve_reads([key])
    assert pool.snapshot() == before


def test_object_groups_and_salts_are_distinct_keys() -> None:
    pool = _pool()
    keys = [
        _key(1),
        replace(_key(1), object_group_id=1),
        replace(_key(1), cache_salt="tenant"),
    ]
    grants = pool.reserve_writes(
        [WriteReserveItem(key=key, layout=_item(1).layout) for key in keys]
    )
    assert all(grants) and pool.snapshot().object_count == 3


@pytest.mark.parametrize(
    "capacity,alignment",
    [(0, 64), (-1, 64), (True, 64), (1.5, 64), (64, 0), (64, -1), (64, 3), (64, True)],
)
def test_invalid_geometry_is_rejected(capacity: int, alignment: int) -> None:
    with pytest.raises(ValueError):
        MemoryPool("region", capacity, alignment, "layout")


def test_epoch_contract_and_close_fences() -> None:
    pool = _pool()
    contract = pool.region_contract()
    pool.check_epoch(contract.region_epoch)
    with pytest.raises(StaleEpochError):
        _pool().check_epoch(contract.region_epoch)
    with pytest.raises(ValidationError):
        contract.capacity_bytes = 1  # type: ignore[misc]
    pool.close()
    pool.close()
    assert not pool.is_healthy()
    for operation in (
        pool.snapshot,
        pool.region_contract,
        lambda: pool.check_epoch(contract.region_epoch),
        lambda: pool.reserve_writes([]),
        lambda: pool.reserve_reads([]),
        lambda: pool.finish_writes([]),
        lambda: pool.abort_writes([]),
        lambda: pool.finish_reads([]),
        lambda: pool.abort_reads([]),
    ):
        with pytest.raises(MemoryCoordinatorError, match="closed"):
            operation()


def test_fork_fences_inherited_pool_and_preserves_parent() -> None:
    pool = _pool()
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_check_inherited_pool, args=(pool, sender))
    child.start()
    sender.close()
    try:
        assert receiver.poll(5), "inherited pool blocked instead of failing closed"
        assert receiver.recv() == [True] * 4
        child.join(5)
        assert child.exitcode == 0
    finally:
        if child.is_alive():
            child.terminate()
            child.join(5)
        receiver.close()
    assert pool.is_healthy() and _write(pool, 1).handle.offset == 0
