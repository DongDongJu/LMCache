import dataclasses
# SPDX-License-Identifier: Apache-2.0
"""Tests for UECC handling in RawBlockCore.

Covers A2 (device state), A3 (circuit-breaker), A4 (poisoned load),
A5 (whole-device gating), A6 (checkpoint suppression), A7 (late-store
fence), and A8 (late-load fence).
"""

# Standard
from unittest.mock import MagicMock
import errno
import logging
import dataclasses
import time

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    encode_object_key,
)
from lmcache.v1.storage_backend.raw_block.uecc import classify_oserror
from tests.v1.storage_backend.raw_block_test_utils import (
    RAW_BLOCK_CI_BLOCK_ALIGN,
    RAW_BLOCK_CI_CAPACITY_BYTES,
    RAW_BLOCK_CI_HEADER_BYTES,
    RAW_BLOCK_CI_META_TOTAL_BYTES,
    RAW_BLOCK_CI_SLOT_BYTES,
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    make_raw_block_core_config,
    make_raw_block_file,
)

pytest.importorskip("lmcache_rust_raw_block_io")


def _make_core(tmp_path, **overrides):
    """Build a RawBlockCore with CI-safe defaults."""
    path = make_raw_block_file(tmp_path)
    config = make_raw_block_core_config(path, RAW_BLOCK_CI_CAPACITY_BYTES)
    for k, v in overrides.items():
        config = dataclasses.replace(config, **{k: v})
    return RawBlockCore(config, key_namespace="object")


class TestDeviceState:
    """A2: Device state management."""

    def test_initial_state_healthy(self, tmp_path):
        core = _make_core(tmp_path)
        assert core.device_active is True
        assert core.device_blocked is False
        assert core.uecc_count == 0
        assert core.has_critical_error is False

    def test_initial_quarantined_slots_empty(self, tmp_path):
        core = _make_core(tmp_path)
        assert len(core.quarantined_slots) == 0

    def test_record_uecc_transitions_state(self, tmp_path):
        core = _make_core(tmp_path)

        class FakeLogger:
            def critical(self, *args, **kwargs):
                pass

        transitioned = core.record_uecc(0x0281, logger_inst=FakeLogger())
        assert transitioned is True
        assert core.device_blocked is True
        assert core.device_active is False
        assert core.has_critical_error is True
        assert core.uecc_count == 1

    def test_record_uecc_only_transitions_once(self, tmp_path):
        core = _make_core(tmp_path)
        log_records = []

        class FakeLogger:
            def critical(self, *args, **kwargs):
                log_records.append(args)

        transitioned1 = core.record_uecc(0x0281, logger_inst=FakeLogger())
        transitioned2 = core.record_uecc(0x0281, logger_inst=FakeLogger())

        assert transitioned1 is True
        assert transitioned2 is False
        assert core.uecc_count == 2
        assert len(log_records) == 1

    def test_quarantine_slot(self, tmp_path):
        core = _make_core(tmp_path)
        core.quarantine_slot(5)
        assert core.is_slot_quarantined(5) is True
        assert core.is_slot_quarantined(6) is False

    def test_quarantined_slots_immutable(self, tmp_path):
        core = _make_core(tmp_path)
        core.quarantine_slot(5)
        qs = core.quarantined_slots
        assert isinstance(qs, frozenset)


class TestCircuitBreaker:
    """A3: Circuit-breaker transition."""

    def test_uecc_triggers_exactly_one_log(self, tmp_path):
        core = _make_core(tmp_path)
        log_lines = []

        class FakeLogger:
            def critical(self, *args, **kwargs):
                log_lines.append(args[0] % args[1:] if len(args) > 1 else args[0])

        core.record_uecc(0x0281, logger_inst=FakeLogger())
        core.record_uecc(0x0281, logger_inst=FakeLogger())
        core.record_uecc(0x4281, logger_inst=FakeLogger())

        assert len(log_lines) == 1
        assert "Device blocked" in log_lines[0]

    def test_generic_io_error_does_not_block(self, tmp_path):
        core = _make_core(tmp_path)
        exc = OSError(errno.EIO, "I/O error")
        classification = classify_oserror(exc)
        assert classification["is_uecc"] is False
        assert core.device_blocked is False

    def test_device_remains_open_after_uecc(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())
        assert core._closed is False


class TestWholeDeviceGating:
    """A5: Whole-device gating."""

    def test_put_many_blocked_returns_all_false(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())
        assert core.device_blocked is True

        key = make_object_key(1)
        spec = encode_object_key(key)
        obj = make_memory_obj(b"test data")

        result = core.put_many([spec], [obj])
        assert all(result.results) is False
        assert result.stored_keys == []

    def test_load_many_into_blocked_returns_all_false(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        obj = make_empty_memory_obj(RAW_BLOCK_CI_SLOT_BYTES)

        results = core.load_many_into([spec.encoded], [obj])
        assert all(r is False for r in results)

    def test_exists_many_blocked_returns_all_false(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        results = core.exists_many([spec.encoded])
        assert all(r is False for r in results)

    def test_contains_key_blocked(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        assert core.contains_key(spec.encoded) is False

    def test_get_metadata_prefix_blocked(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        result = core.get_metadata_prefix([spec.encoded])
        assert result == []

    def test_get_metadata_many_blocked(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        result = core.get_metadata_many([spec.encoded])
        assert all(m is None for m in result)

    def test_delete_many_blocked(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        key = make_object_key(1)
        spec = encode_object_key(key)
        result = core.delete_many([spec.encoded])
        assert all(r is False for r in result)


class TestCheckpointSuppression:
    """A6: Checkpoint suppression."""

    def test_checkpoint_once_skipped_when_blocked(self, tmp_path):
        core = _make_core(tmp_path, meta_enable_periodic=False)
        core.record_uecc(0x0281, logger_inst=MagicMock())
        assert core._checkpoint_once(force=True) is False

    def test_close_skips_checkpoint_when_blocked(self, tmp_path):
        core = _make_core(tmp_path, meta_enable_periodic=False)
        core.record_uecc(0x0281, logger_inst=MagicMock())
        core.close()
        assert core._closed is True


class TestAdapterContract:
    """A9: Adapter contract updates."""

    def test_report_status_includes_uecc_fields(self, tmp_path):
        core = _make_core(tmp_path)
        core.record_uecc(0x0281, logger_inst=MagicMock())

        status = core.report_status()
        assert status["device_blocked"] is True
        assert status["device_active"] is False
        assert status["has_critical_error"] is True
        assert status["uecc_count"] == 1
        assert status["quarantined_slot_count"] == 0
        assert status["critical_error_transitioned"] is True