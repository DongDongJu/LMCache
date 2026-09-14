# SPDX-License-Identifier: Apache-2.0
"""Tests for the UECC decoder module."""

# Standard
import errno
import types

# Third Party
import pytest

# First Party
from lmcache.v1.storage_backend.raw_block.uecc import (
    UECC_STATUS_CODE,
    UECC_STATUS_MASK,
    classify_oserror,
    decode_uecc_status,
    is_uecc_errno,
)


class TestIsUeccErrno:
    """Test the is_uecc_errno function."""

    def test_exact_uecc_match(self):
        assert is_uecc_errno(0x0281) is True

    def test_uecc_with_dnr_bit(self):
        assert is_uecc_errno(0x4281) is True

    def test_uecc_with_sct(self):
        assert is_uecc_errno(0x2281) is True

    def test_generic_io_error(self):
        assert is_uecc_errno(errno.EIO) is False

    def test_timeout_error(self):
        assert is_uecc_errno(errno.ETIMEDOUT) is False

    def test_success(self):
        assert is_uecc_errno(0x0000) is False

    def test_off_by_one_below(self):
        assert is_uecc_errno(0x0280) is False

    def test_off_by_one_above(self):
        assert is_uecc_errno(0x0282) is False

    def test_max_masked_value(self):
        assert is_uecc_errno(0x07ff) is False

    def test_non_integer_input(self):
        assert is_uecc_errno("not an int") is False

    def test_none_input(self):
        assert is_uecc_errno(None) is False

    def test_negative_input(self):
        assert is_uecc_errno(-1) is False


class TestDecodeUeccStatus:
    """Test the decode_uecc_status function."""

    def test_decode_0x0281(self):
        result = decode_uecc_status(0x0281)
        assert result["is_uecc"] is True
        assert result["status_code"] == 0x0281
        assert result["dnr"] is False
        assert result["sct"] == 0
        assert result["sc"] == 0x81

    def test_decode_0x4281_dnr_set(self):
        result = decode_uecc_status(0x4281)
        assert result["is_uecc"] is True
        assert result["dnr"] is True
        assert result["sct"] == 1

    def test_decode_non_uecc(self):
        result = decode_uecc_status(0x0002)
        assert result["is_uecc"] is False
        assert result["status_code"] == 0x0002


class TestClassifyOSError:
    """Test the classify_oserror function."""

    def test_uecc_oserror(self):
        exc = OSError(0x0281, "UECC error")
        result = classify_oserror(exc)
        assert result["is_uecc"] is True
        assert result["error_type"] == "uecc"
        assert result["errno"] == 0x0281
        assert result["decoded"]["is_uecc"] is True

    def test_uecc_with_dnr(self):
        exc = OSError(0x4281, "UECC with DNR")
        result = classify_oserror(exc)
        assert result["is_uecc"] is True
        assert result["decoded"]["dnr"] is True

    def test_generic_io_error(self):
        exc = OSError(errno.EIO, "I/O error")
        result = classify_oserror(exc)
        assert result["is_uecc"] is False
        assert result["error_type"] == "io_error"

    def test_timeout_error(self):
        exc = OSError(errno.ETIMEDOUT, "Timeout")
        result = classify_oserror(exc)
        assert result["is_uecc"] is False
        assert result["error_type"] == "timeout"

    def test_non_oserror_exception(self):
        exc = ValueError("not an oserror")
        result = classify_oserror(exc)
        assert result["is_uecc"] is False
        assert result["error_type"] == "other"

    def test_oserror_with_none_errno(self):
        exc = OSError("no errno")
        exc.errno = None
        result = classify_oserror(exc)
        assert result["is_uecc"] is False
        assert result["errno"] is None

    def test_oserror_with_non_int_errno(self):
        exc = OSError("str errno")
        exc.errno = "not an int"
        result = classify_oserror(exc)
        assert result["is_uecc"] is False