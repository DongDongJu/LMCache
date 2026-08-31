# SPDX-License-Identifier: Apache-2.0
"""Client of the frozen outside Memory Allocation API.

    GET  /api/v2/lmcache
    POST /api/v2/lmcache/deallocations
    POST /api/v2/lmcache/allocations

The request bodies are exactly the documented fields; every documented
response field is required and echo fields are validated against the sent
request. A POST is issued once. Any transport failure after the request may
have been sent is :class:`AmbiguousMutationError`; an explicit non-2xx or
non-``DONE`` answer is :class:`OutsideExplicitFailure`; a 2xx body that is
missing a field or echoes a different value is :class:`OutsideContractError`.
"""

# Standard
from typing import TypeVar

# Third Party
from pydantic import ValidationError
import httpx

# First Party
from lmcache.v1.mp_memory_coordinator.clients import (
    ClientError,
    ClientResponseError,
    JSONBody,
    get_json,
    post_json_once,
)
from lmcache.v1.mp_memory_coordinator.models import (
    OUTSIDE_ALLOCATIONS_PATH,
    OUTSIDE_DEALLOCATIONS_PATH,
    OUTSIDE_STATUS_DONE,
    OUTSIDE_STATUS_PATH,
    AllocationRequest,
    AllocationResponse,
    DeallocationRequest,
    DeallocationResponse,
    OutsideStatus,
    parse_outside_status,
)

ResponseT = TypeVar("ResponseT", DeallocationResponse, AllocationResponse)


class OutsideExplicitFailure(ClientError):
    """The service explicitly rejected the request (non-2xx or non-DONE).

    Attributes:
        status_code: HTTP status code of the answer.
        body: The decoded body, for logs.
    """

    def __init__(self, status_code: int, body: object, url: str) -> None:
        """Args:
        status_code: HTTP status code.
        body: Decoded response body.
        url: Request URL.
        """
        super().__init__(f"outside service refused POST {url}: {status_code} {body}")
        self.status_code = status_code
        self.body = body


class OutsideContractError(ClientResponseError):
    """A 2xx response violated the frozen contract (missing/mismatched field).

    The effect may have been applied; the caller must reconcile with status.

    Attributes:
        fields: Documented scalar fields that *were* present in the body
            (e.g. a returned ``device_path`` beside a wrong size), so the
            caller can record what the service claimed. Empty when the body
            was not an object.
    """

    def __init__(self, message: str, fields: dict[str, str | int] | None = None):
        """Args:
        message: Human-readable violation.
        fields: Documented scalar fields present in the body.
        """
        super().__init__(message)
        self.fields: dict[str, str | int] = dict(fields or {})


def _scalar_fields(body: JSONBody) -> dict[str, str | int]:
    """Return the ``str``/``int`` members of a JSON object body (else empty)."""
    if not isinstance(body.value, dict):
        return {}
    return {
        str(k): v
        for k, v in body.value.items()
        if isinstance(v, (str, int)) and not isinstance(v, bool)
    }


class MemoryAllocationClient:
    """HTTP client for the outside Memory Allocation service."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        attempts: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        base_url: Service base URL, e.g. ``http://memory-allocation:8080``.
        timeout_seconds: Per-request timeout.
        attempts: Bounded attempts for the status GET only.
        transport: Custom transport (tests inject an ``httpx.MockTransport``);
            ``None`` uses real sockets.
        """
        self._base_url = base_url.rstrip("/")
        self._attempts = attempts
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    @property
    def base_url(self) -> str:
        """The configured base URL (no trailing slash)."""
        return self._base_url

    async def get_status(self) -> OutsideStatus:
        """``GET /api/v2/lmcache``.

        Returns:
            The bare ``target_node -> device_path[]`` mapping.

        Raises:
            ClientError: On transport failure after every attempt or a
                non-2xx status.
            OutsideContractError: If the body is wrapped or not a mapping of
                string lists.
        """
        body = await get_json(
            self._client,
            f"{self._base_url}{OUTSIDE_STATUS_PATH}",
            attempts=self._attempts,
        )
        try:
            return parse_outside_status(body.value)
        except ValueError as exc:
            raise OutsideContractError(str(exc)) from exc

    async def deallocate(self, request: DeallocationRequest) -> DeallocationResponse:
        """``POST /api/v2/lmcache/deallocations``, once.

        Args:
            request: The exact request body.

        Returns:
            The validated response: ``status == "DONE"``, ``request_id``,
            ``target_node`` and ``device_path`` echoed exactly, and a
            positive ``released_size_gib``.

        Raises:
            OutsideExplicitFailure: On a non-2xx status.
            OutsideContractError: If a required field is missing, an echo
                differs, ``status`` is not ``DONE``, or the size is not
                positive.
            AmbiguousMutationError: If the request may have been sent but no
                response arrived.
            ClientConnectionError: If the request was never sent.
        """
        url = f"{self._base_url}{OUTSIDE_DEALLOCATIONS_PATH}"
        status_code, body = await post_json_once(
            self._client, url, request.model_dump(mode="json")
        )
        if not 200 <= status_code < 300:
            raise OutsideExplicitFailure(status_code, body.value, url)
        fields = _scalar_fields(body)
        response = _validate(DeallocationResponse, body, "deallocation", fields)
        _require_echo("request_id", request.request_id, response.request_id, fields)
        _require_echo("target_node", request.target_node, response.target_node, fields)
        _require_echo("device_path", request.device_path, response.device_path, fields)
        if response.status != OUTSIDE_STATUS_DONE:
            raise OutsideContractError(
                f"deallocation status is {response.status!r}, expected DONE", fields
            )
        if response.released_size_gib <= 0:
            raise OutsideContractError(
                f"released_size_gib must be positive, got {response.released_size_gib}",
                fields,
            )
        return response

    async def allocate(self, request: AllocationRequest) -> AllocationResponse:
        """``POST /api/v2/lmcache/allocations``, once.

        Args:
            request: The exact request body.

        Returns:
            The validated response: ``status == "DONE"``, ``request_id`` and
            ``target_node`` echoed exactly, a non-empty ``device_path``, and
            ``requested_size_gib == granted_size_gib == request_size_gib``.
            The returned path is *not* validated for ownership here; the
            caller proves it against the persisted before/after path sets.

        Raises:
            OutsideExplicitFailure: On a non-2xx status.
            OutsideContractError: If a required field is missing, an echo
                differs, ``status`` is not ``DONE``, the path is empty, or
                the three sizes disagree.
            AmbiguousMutationError: If the request may have been sent but no
                response arrived.
            ClientConnectionError: If the request was never sent.
        """
        url = f"{self._base_url}{OUTSIDE_ALLOCATIONS_PATH}"
        status_code, body = await post_json_once(
            self._client, url, request.model_dump(mode="json")
        )
        if not 200 <= status_code < 300:
            raise OutsideExplicitFailure(status_code, body.value, url)
        fields = _scalar_fields(body)
        response = _validate(AllocationResponse, body, "allocation", fields)
        _require_echo("request_id", request.request_id, response.request_id, fields)
        _require_echo("target_node", request.target_node, response.target_node, fields)
        if response.status != OUTSIDE_STATUS_DONE:
            raise OutsideContractError(
                f"allocation status is {response.status!r}, expected DONE", fields
            )
        if not response.device_path:
            raise OutsideContractError("allocation device_path is empty", fields)
        if not (
            response.requested_size_gib
            == response.granted_size_gib
            == request.request_size_gib
        ):
            raise OutsideContractError(
                "allocation sizes disagree: request_size_gib="
                f"{request.request_size_gib} requested_size_gib="
                f"{response.requested_size_gib} granted_size_gib="
                f"{response.granted_size_gib}",
                fields,
            )
        return response

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()


def _validate(
    model: type[ResponseT],
    body: JSONBody,
    label: str,
    fields: dict[str, str | int],
) -> ResponseT:
    """Validate a 2xx body against a response model.

    Raises:
        OutsideContractError: If a documented field is missing or mistyped.
    """
    try:
        return model.model_validate(body.as_dict())
    except (ValidationError, ClientResponseError) as exc:
        raise OutsideContractError(
            f"{label} response violates contract: {exc}", fields
        ) from exc


def _require_echo(
    field: str, sent: str, received: str, fields: dict[str, str | int]
) -> None:
    """Require an echoed field to match the sent value exactly.

    Raises:
        OutsideContractError: If they differ.
    """
    if sent != received:
        raise OutsideContractError(
            f"{field} echo mismatch: sent {sent!r}, received {received!r}", fields
        )
