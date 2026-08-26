# SPDX-License-Identifier: Apache-2.0
"""Strict request models and response builders for the frozen public API.

The three public routes, their request fields, and their response fields are
frozen by PLAN.md Section 2.  These models are deliberately written from that
text and never imported from ``lmcache.v1.mp_memory_coordinator`` so that the
mock and the production client cannot drift together and still pass.

Both request models use ``extra="forbid"`` and ``strict=True``: a missing,
renamed, or extra field, or a value of the wrong JSON type (for example a
boolean or a string where an integer is required), is rejected with 422 before
any handler runs, so an invalid request can never mutate state.
"""

# Standard
from typing import Literal

# Third Party
from pydantic import BaseModel, ConfigDict, Field

DEALLOCATION_RESPONSE_STATUS: str = "DONE"
"""Literal ``status`` value of every successful deallocation response."""

ALLOCATION_RESPONSE_STATUS: str = "DONE"
"""Literal ``status`` value of every successful allocation response."""


class DeallocationRequest(BaseModel):
    """Exact request body of ``POST /v2/apps/lmcache/deallocations``.

    Attributes:
        request_id: Caller-chosen identifier echoed back verbatim.
        target_node: Worker IP that owns ``device_path``.
        device_path: Absolute path of the assigned runtime device to release.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1)
    target_node: str = Field(min_length=1)
    device_path: str = Field(min_length=1)


class AllocationRequest(BaseModel):
    """Exact request body of ``POST /v2/apps/lmcache/allocations``.

    Attributes:
        request_id: Caller-chosen identifier echoed back verbatim.
        target_node: Worker IP that must receive the device.
        request_size_gib: Exact size of the device to select, in GiB.  The
            response spells this ``requested_size_gib``; the two names are
            intentionally different in the frozen contract.
        mode: Literal ``"devdax"``.
        purpose: Literal ``"lmcache-dax"``.
        access: Literal ``"exclusive"``.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    request_id: str = Field(min_length=1)
    target_node: str = Field(min_length=1)
    request_size_gib: int = Field(gt=0)
    mode: Literal["devdax"]
    purpose: Literal["lmcache-dax"]
    access: Literal["exclusive"]


def build_deallocation_response(
    request_id: str, target_node: str, device_path: str, released_size_gib: int
) -> dict[str, str | int]:
    """Build the exact successful deallocation response body.

    Args:
        request_id: Value echoed from the request.
        target_node: Value echoed from the request.
        device_path: Value echoed from the request.
        released_size_gib: Recorded size of the released device in GiB.

    Returns:
        A JSON object with exactly the keys ``status``, ``request_id``,
        ``target_node``, ``device_path`` and ``released_size_gib``.
    """
    return {
        "status": DEALLOCATION_RESPONSE_STATUS,
        "request_id": request_id,
        "target_node": target_node,
        "device_path": device_path,
        "released_size_gib": released_size_gib,
    }


def build_allocation_response(
    request_id: str,
    target_node: str,
    device_path: str,
    requested_size_gib: int,
    granted_size_gib: int,
) -> dict[str, str | int]:
    """Build the exact successful allocation response body.

    Args:
        request_id: Value echoed from the request.
        target_node: Value echoed from the request.
        device_path: Pre-existing path of the selected runtime device.
        requested_size_gib: The request's ``request_size_gib`` value.
        granted_size_gib: Recorded size of the selected device in GiB.

    Returns:
        A JSON object with exactly the keys ``status``, ``request_id``,
        ``target_node``, ``device_path``, ``requested_size_gib`` and
        ``granted_size_gib``.
    """
    return {
        "status": ALLOCATION_RESPONSE_STATUS,
        "request_id": request_id,
        "target_node": target_node,
        "device_path": device_path,
        "requested_size_gib": requested_size_gib,
        "granted_size_gib": granted_size_gib,
    }
