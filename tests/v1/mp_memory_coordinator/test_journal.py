# SPDX-License-Identifier: Apache-2.0
"""Tests for the atomic, checksummed rebalance journal."""

# Standard
from pathlib import Path
import json
import os

# Third Party
import pytest

# First Party
from lmcache.v1.mp_memory_coordinator.models import (
    JOURNAL_SCHEMA_VERSION,
    NO_DONOR,
    AllocationOrigin,
    EffectFailure,
    EffectName,
    EffectRecord,
    InstanceIdentity,
    JournalDocument,
    ManagedAllocation,
    MoveKind,
    MoveOutcome,
    MoveRecord,
    MoveState,
)
from lmcache.v1.mp_memory_coordinator.persistence.rebalance_journal import (
    JOURNAL_FILE_NAME,
    JournalCorruptError,
    JournalVersionError,
    RebalanceJournal,
    compute_checksum,
)


def _identity(instance_id: str, worker_ip: str) -> InstanceIdentity:
    return InstanceIdentity(
        instance_id=instance_id,
        registration_time=1.0,
        endpoint="10.0.0.1:8080",
        worker_ip=worker_ip,
    )


def _record() -> MoveRecord:
    record = MoveRecord(
        move_id="move-1",
        state=MoveState.DEALLOCATING,
        donor=_identity("mp-donor", "192.0.2.40"),
        receiver=_identity("mp-receiver", "192.0.2.41"),
        donor_capacity_bytes=128 << 30,
        receiver_capacity_bytes=64 << 30,
        old_path="/dev/dax-cxl/x/dax0.1",
        old_device_index=1,
        old_map_size_bytes=64 << 30,
        old_slot_capacity_bytes=64 << 30,
        allocation_size_gib=64,
        deallocation_request_id="move-1-deallocate",
        allocation_request_id="move-1-allocate",
        release_request_id="move-1-release",
        restore_request_id="move-1-restore",
        created_at=1.0,
        updated_at=2.0,
    )
    record.effects[EffectName.DEALLOCATE.value] = EffectRecord(
        name=EffectName.DEALLOCATE,
        request_id="move-1-deallocate",
        intent_at=2.0,
        before_paths=["/dev/dax-cxl/x/dax0.1"],
        dispatched=True,
    )
    return record


def _document() -> JournalDocument:
    return JournalDocument(
        initialized=True,
        inventory=[
            ManagedAllocation(
                worker_ip="192.0.2.40",
                instance_id="mp-donor",
                device_path="/dev/dax-cxl/x/dax0.1",
                allocation_size_gib=64,
                device_map_size_bytes=64 << 30,
                slot_capacity_bytes=64 << 30,
                adapter_index=0,
                origin=AllocationOrigin.ADOPTED,
                last_confirmed_state="active",
                last_confirmed_at=1.0,
            )
        ],
        cooldowns={"k": 100.0},
        active_move=_record(),
    )


def test_missing_journal_loads_fresh_uninitialized_document(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path / "state")
    assert not journal.exists()
    document = journal.load()
    assert document == JournalDocument()
    assert document.initialized is False
    assert document.active_move is None


def test_save_then_load_round_trips_every_field(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path / "state")
    document = _document()
    journal.save(document)
    assert journal.exists()
    loaded = journal.load()
    assert loaded == document
    assert loaded.active_move is not None
    effect = loaded.active_move.effect(EffectName.DEALLOCATE)
    assert effect is not None and effect.dispatched and not effect.confirmed
    assert loaded.find_allocation("/dev/dax-cxl/x/dax0.1") is not None
    assert loaded.allocations_for("192.0.2.41") == []


def test_save_is_atomic_same_directory_replacement(tmp_path: Path) -> None:
    directory = tmp_path / "state"
    journal = RebalanceJournal(directory)
    journal.save(JournalDocument())
    journal.save(_document())
    # No temporary file survives a clean save; only the journal itself.
    assert sorted(p.name for p in directory.iterdir()) == [JOURNAL_FILE_NAME]
    envelope = json.loads((directory / JOURNAL_FILE_NAME).read_text())
    assert envelope["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert envelope["checksum"] == compute_checksum(envelope["payload"])
    assert oct(os.stat(directory / JOURNAL_FILE_NAME).st_mode & 0o777) == "0o600"


def test_truncated_journal_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    journal.save(_document())
    data = journal.path.read_bytes()
    journal.path.write_bytes(data[: len(data) // 2])
    with pytest.raises(JournalCorruptError):
        journal.load()


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    journal.save(_document())
    envelope = json.loads(journal.path.read_text())
    envelope["payload"]["cooldowns"]["k"] = 999.0
    journal.path.write_text(json.dumps(envelope))
    with pytest.raises(JournalCorruptError, match="checksum"):
        journal.load()


def test_unknown_schema_version_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    journal.save(_document())
    envelope = json.loads(journal.path.read_text())
    envelope["schema_version"] = 99
    journal.path.write_text(json.dumps(envelope))
    with pytest.raises(JournalVersionError):
        journal.load()


def test_envelope_without_payload_or_non_object_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    journal.path.write_text('{"schema_version": 1}')
    with pytest.raises(JournalCorruptError):
        journal.load()
    journal.path.write_text("[1, 2, 3]")
    with pytest.raises(JournalCorruptError):
        journal.load()
    journal.path.write_text("")
    with pytest.raises(JournalCorruptError):
        journal.load()


def test_invalid_payload_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    payload = {"schema_version": 1, "initialized": "maybe", "unknown_key": 1}
    envelope = {
        "schema_version": 1,
        "checksum": compute_checksum(payload),
        "payload": payload,
    }
    journal.path.write_text(json.dumps(envelope))
    with pytest.raises(JournalCorruptError, match="invalid payload"):
        journal.load()


def test_move_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        JournalDocument.model_validate({"schema_version": 1, "extra": True})


# -- GROW compatibility --------------------------------------------------------------


def _rewrite_payload(journal: RebalanceJournal, payload: dict[str, object]) -> None:
    envelope = json.loads(journal.path.read_text())
    envelope["payload"] = payload
    envelope["checksum"] = compute_checksum(payload)
    journal.path.write_text(json.dumps(envelope))


def test_journal_written_before_grow_loads_as_move(tmp_path: Path) -> None:
    """A journal from the previous build (no ``kind``, no ``failure``, no
    ``receiver_rebinds``, no ``grow_backoffs``, no GROW counters) loads and
    behaves as a MOVE."""
    journal = RebalanceJournal(tmp_path)
    journal.save(_document())
    payload = json.loads(journal.path.read_text())["payload"]
    del payload["active_move"]["kind"]
    del payload["active_move"]["receiver_rebinds"]
    del payload["active_move"]["effects"]["deallocate"]["failure"]
    del payload["grow_backoffs"]
    del payload["counters"]["not_served"]
    del payload["counters"]["grown"]
    _rewrite_payload(journal, payload)
    loaded = journal.load()
    assert loaded.active_move is not None
    assert loaded.active_move.kind is MoveKind.MOVE
    assert loaded.active_move.has_donor
    assert loaded.active_move.receiver_rebinds == 0
    effect = loaded.active_move.effect(EffectName.DEALLOCATE)
    assert effect is not None and effect.failure is EffectFailure.NONE
    assert loaded.grow_backoffs == {}
    assert loaded.counters.not_served == 0 and loaded.counters.grown == 0
    assert loaded == _document()


def _grow_record() -> MoveRecord:
    record = MoveRecord(
        move_id="grow-1",
        kind=MoveKind.GROW,
        state=MoveState.COMPLETE,
        outcome=MoveOutcome.NOT_SERVED,
        donor=NO_DONOR,
        receiver=_identity("mp-receiver", "192.0.2.41"),
        donor_capacity_bytes=0,
        receiver_capacity_bytes=64 << 30,
        old_path="",
        old_device_index=-1,
        old_map_size_bytes=64 << 30,
        old_slot_capacity_bytes=0,
        allocation_size_gib=64,
        deallocation_request_id="",
        allocation_request_id="grow-1-allocate",
        release_request_id="grow-1-release",
        restore_request_id="",
        receiver_rebinds=1,
        created_at=1.0,
        updated_at=2.0,
    )
    record.effects[EffectName.ALLOCATE.value] = EffectRecord(
        name=EffectName.ALLOCATE,
        request_id="grow-1-allocate",
        intent_at=2.0,
        dispatched=True,
        attempts=1,
        error="explicit failure 409",
        failure=EffectFailure.EXPLICIT,
    )
    return record


def test_grow_record_round_trips_every_field(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    document = JournalDocument(
        initialized=True,
        history=[_grow_record()],
        grow_backoffs={"192.0.2.41": 123.0},
    )
    journal.save(document)
    loaded = journal.load()
    assert loaded == document
    assert not loaded.history[0].has_donor
    payload = json.loads(journal.path.read_text())["payload"]
    assert payload["history"][0]["kind"] == "grow"
    assert payload["history"][0]["outcome"] == "NOT_SERVED"
    assert payload["history"][0]["effects"]["allocate"]["failure"] == "explicit"
    assert payload["history"][0]["receiver_rebinds"] == 1
    assert payload["grow_backoffs"] == {"192.0.2.41": 123.0}


def test_unknown_kind_or_outcome_fails_closed(tmp_path: Path) -> None:
    journal = RebalanceJournal(tmp_path)
    journal.save(_document())
    payload = json.loads(journal.path.read_text())["payload"]
    payload["active_move"]["kind"] = "shrink"
    _rewrite_payload(journal, payload)
    with pytest.raises(JournalCorruptError, match="invalid payload"):
        journal.load()
    payload["active_move"]["kind"] = "move"
    payload["active_move"]["outcome"] = "MAYBE"
    _rewrite_payload(journal, payload)
    with pytest.raises(JournalCorruptError, match="invalid payload"):
        journal.load()
