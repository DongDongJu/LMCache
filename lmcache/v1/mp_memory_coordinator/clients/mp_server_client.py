# SPDX-License-Identifier: Apache-2.0
"""Client of one MP server's direct HTTP API (frozen routes).

Used routes: ``GET /healthcheck``, ``GET /status``,
``GET /reconfigure/dax/status``, ``POST /reconfigure/dax/remove``,
``POST /reconfigure/dax/add``. DAX POSTs are issued once per call; their
effect is always re-read from ``/reconfigure/dax/status`` by the caller.
"""

# Third Party
from pydantic import ValidationError
import httpx

# First Party
from lmcache.v1.mp_memory_coordinator.clients import (
    ClientHTTPError,
    ClientResponseError,
    get_json,
    post_json_once,
)
from lmcache.v1.mp_memory_coordinator.models import (
    MP_DAX_ADD_PATH,
    MP_DAX_REMOVE_PATH,
    MP_DAX_STATUS_PATH,
    MP_HEALTHCHECK_PATH,
    MP_STATUS_PATH,
    DaxAddResponse,
    DaxDeviceNotFound,
    DaxReconfigureStatus,
    DaxRemoveBlocked,
    DaxRemoveMode,
    DaxRemoveResponse,
    MPStatus,
)

DaxRemoveResult = DaxRemoveResponse | DaxRemoveBlocked | DaxDeviceNotFound
"""Typed outcome of a remove: success, busy (``409``), or unknown (``404``)."""


def format_gib(size_bytes: int) -> str:
    """Format a whole-GiB byte count as the ``"<n>GiB"`` add size string.

    Args:
        size_bytes: A positive multiple of GiB.

    Returns:
        ``"<n>GiB"``.

    Raises:
        ValueError: If ``size_bytes`` is not a positive whole number of GiB.
    """
    gib = 1024**3
    if size_bytes <= 0 or size_bytes % gib != 0:
        raise ValueError(f"size must be a positive whole GiB count, got {size_bytes}")
    return f"{size_bytes // gib}GiB"


class MPServerClient:
    """One shared connection pool for every MP server endpoint."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        attempts: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        timeout_seconds: Per-request timeout.
        attempts: Bounded attempts per GET.
        transport: Custom transport (tests inject an ``httpx.MockTransport``);
            ``None`` uses real sockets.
        """
        self._attempts = attempts
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    async def is_healthy(self, base_url: str) -> bool:
        """``GET /healthcheck``.

        Args:
            base_url: ``http://ip:port`` of the MP server.

        Returns:
            ``True`` on a 2xx response, ``False`` on any failure.
        """
        try:
            await get_json(self._client, f"{base_url}{MP_HEALTHCHECK_PATH}", attempts=1)
        except Exception:  # noqa: BLE001 -- health is a boolean probe
            return False
        return True

    async def get_status(self, base_url: str) -> MPStatus:
        """``GET /status``.

        Args:
            base_url: ``http://ip:port`` of the MP server.

        Returns:
            The validated status (unknown fields retained).

        Raises:
            ClientError: On transport failure, non-2xx, or schema mismatch.
        """
        body = await get_json(
            self._client, f"{base_url}{MP_STATUS_PATH}", attempts=self._attempts
        )
        try:
            return MPStatus.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(f"/status schema mismatch: {exc}") from exc

    async def get_dax_status(self, base_url: str) -> DaxReconfigureStatus:
        """``GET /reconfigure/dax/status``.

        Args:
            base_url: ``http://ip:port`` of the MP server.

        Returns:
            The validated DAX reconfiguration status.

        Raises:
            ClientError: On transport failure, non-2xx, or schema mismatch.
        """
        body = await get_json(
            self._client, f"{base_url}{MP_DAX_STATUS_PATH}", attempts=self._attempts
        )
        try:
            return DaxReconfigureStatus.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(
                f"/reconfigure/dax/status schema mismatch: {exc}"
            ) from exc

    async def remove_dax_device(
        self,
        base_url: str,
        *,
        adapter_index: int,
        device_path: str,
        mode: DaxRemoveMode,
    ) -> DaxRemoveResult:
        """``POST /reconfigure/dax/remove`` with ``force=false``, once.

        Args:
            base_url: ``http://ip:port`` of the MP server.
            adapter_index: Backend-local DAX adapter index.
            device_path: The device to drain or evict.
            mode: ``drain`` or ``evict``.

        Returns:
            :class:`DaxRemoveResponse` on 2xx, :class:`DaxRemoveBlocked` on
            ``409`` (busy: caller returns to status polling), or
            :class:`DaxDeviceNotFound` on ``404``.

        Raises:
            ClientHTTPError: On any other non-2xx status.
            AmbiguousMutationError: If the request may have been applied but
                no response arrived; reconcile with status.
            ClientConnectionError: If the request was never sent.
            ClientResponseError: On a malformed body.
        """
        url = f"{base_url}{MP_DAX_REMOVE_PATH}"
        status_code, body = await post_json_once(
            self._client,
            url,
            {
                "adapter_index": adapter_index,
                "device_path": device_path,
                "mode": mode.value,
                "force": False,
            },
        )
        try:
            if 200 <= status_code < 300:
                return DaxRemoveResponse.model_validate(body.as_dict())
            if status_code == 409:
                return DaxRemoveBlocked.model_validate(body.as_dict())
            if status_code == 404:
                return DaxDeviceNotFound.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(f"dax remove schema mismatch: {exc}") from exc
        raise ClientHTTPError(status_code, str(body.value), url)

    async def add_dax_device(
        self,
        base_url: str,
        *,
        adapter_index: int,
        device_path: str,
        size: str,
    ) -> DaxAddResponse:
        """``POST /reconfigure/dax/add``, once.

        Args:
            base_url: ``http://ip:port`` of the MP server.
            adapter_index: Backend-local DAX adapter index.
            device_path: The device to map.
            size: Size string such as ``"64GiB"`` (see :func:`format_gib`).

        Returns:
            The add response with the new device entry.

        Raises:
            ClientHTTPError: On a non-2xx status.
            AmbiguousMutationError: If the request may have been applied but
                no response arrived; reconcile with status.
            ClientConnectionError: If the request was never sent.
            ClientResponseError: On a malformed body.
        """
        url = f"{base_url}{MP_DAX_ADD_PATH}"
        status_code, body = await post_json_once(
            self._client,
            url,
            {"adapter_index": adapter_index, "device_path": device_path, "size": size},
        )
        if not 200 <= status_code < 300:
            raise ClientHTTPError(status_code, str(body.value), url)
        try:
            return DaxAddResponse.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(f"dax add schema mismatch: {exc}") from exc

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()
