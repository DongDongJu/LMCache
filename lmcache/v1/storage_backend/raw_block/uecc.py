# SPDX-License-Identifier: Apache-2.0
"""
UECC (Uncorrectable ECC) detection and status decoding for NVMe devices.

This module provides pure-Python utilities for classifying OSError exceptions
raised by the Rust raw-block device as UECC errors. It implements the NVMe
status code matching pattern from the NVMe specification.

NVMe Status Layout (bits 0-15 of the 16-bit status word):
  - Bits 0-10: Status Code (SQ Command Status or Completion Code)
  - Bit 11: DNR (Do Not Retry)
  - Bits 11-13: SCT (Status Code Type)
  - Bits 0-7: SC (Status Code within SCT)

The UECC error is identified by status code 0x0281 in the NVMe specification.
"""

# Standard
from typing import Any

# UECC status constants (NVMe spec)
UECC_STATUS_CODE = 0x0281
"""NVMe status code for Unrecoverable ECC Error (bits 0-10)."""

UECC_STATUS_MASK = 0x07FF
"""Bit mask for the status code field (bits 0-10)."""


def is_uecc_errno(errno_value: int) -> bool:
    """Return True if the errno/exit_status matches the UECC pattern.

    The NVMe spec identifies UECC errors by status code 0x0281. The Rust
    raw-block device exposes this as the OSError.errno field of the exception.

    Args:
        errno_value: The raw integer errno or exit status from the
            Rust raw-block device OSError.

    Returns:
        True when ``(errno_value & 0x07ff) == 0x0281``.
    """
    if not isinstance(errno_value, int):
        return False
    return (errno_value & UECC_STATUS_MASK) == UECC_STATUS_CODE


def decode_uecc_status(status: int) -> dict[str, Any]:
    """Decode an NVMe status word into its component fields.

    Args:
        status: Raw 16-bit NVMe completion status word.

    Returns:
        Dict with keys:
            ``is_uecc`` (bool): Whether this status represents a UECC error.
            ``sct`` (int): Status Code Type (bits 11-13).
            ``sc`` (int): Status Code within SCT (bits 0-7).
            ``dnr`` (bool): Do Not Retry flag (bit 11).
            ``status_code`` (int): Raw status code value (bits 0-10).
    """
    status_code = int(status) & UECC_STATUS_MASK
    dnr = bool(status & (1 << 11))
    sct = (int(status) >> 11) & 0x07
    sc = int(status) & 0x7F

    return {
        "is_uecc": status_code == UECC_STATUS_CODE,
        "sct": sct,
        "sc": sc,
        "dnr": dnr,
        "status_code": status_code,
    }


def classify_oserror(exc: BaseException) -> dict[str, Any]:
    """Classify an OSError raised by raw-block I/O.

    Args:
        exc: The exception to inspect. Must be an OSError for meaningful
            classification; non-OSError exceptions are classified as
            non-UECC.

    Returns:
        Dict with classification result including:
            ``is_uecc`` (bool): True if this is a UECC error.
            ``errno`` (int | None): The original errno if available.
            ``decoded`` (dict | None): Decoded NVMe status fields if errno
                is an integer; None otherwise.
            ``error_type`` (str): Human-readable classification ("uecc",
                "io_error", "timeout", or "other").
    """
    if not isinstance(exc, OSError):
        return {
            "is_uecc": False,
            "errno": None,
            "decoded": None,
            "error_type": "other",
        }

    errno_value = getattr(exc, "errno", None)
    if errno_value is None or not isinstance(errno_value, int):
        return {
            "is_uecc": False,
            "errno": errno_value,
            "decoded": None,
            "error_type": "other",
        }

    if is_uecc_errno(errno_value):
        return {
            "is_uecc": True,
            "errno": errno_value,
            "decoded": decode_uecc_status(errno_value),
            "error_type": "uecc",
        }

    # Check for other common NVMe error patterns
    import errno as _errno

    if errno_value == _errno.EIO:
        return {
            "is_uecc": False,
            "errno": errno_value,
            "decoded": None,
            "error_type": "io_error",
        }

    if errno_value == _errno.ETIMEDOUT:
        return {
            "is_uecc": False,
            "errno": errno_value,
            "decoded": None,
            "error_type": "timeout",
        }

    # Generic non-UECC error
    return {
        "is_uecc": False,
        "errno": errno_value,
        "decoded": None,
        "error_type": "other",
    }