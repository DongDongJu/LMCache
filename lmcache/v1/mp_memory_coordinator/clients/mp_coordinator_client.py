# SPDX-License-Identifier: Apache-2.0
"""Read-only client of the existing MP Coordinator.

The only allowed communication with the MP Coordinator is HTTP:
``GET /instances`` and ``GET /instances/usage``. Nothing here mutates.
"""

# Third Party
from pydantic import ValidationError
import httpx

# First Party
from lmcache.v1.mp_memory_coordinator.clients import (
    ClientResponseError,
    get_json,
)
from lmcache.v1.mp_memory_coordinator.models import (
    COORDINATOR_INSTANCES_PATH,
    COORDINATOR_USAGE_PATH,
    CoordinatorInstances,
    FleetUsage,
)


class MPCoordinatorClient:
    """Bounded-retry GET client for the MP Coordinator membership APIs."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float,
        attempts: int,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        base_url: Coordinator base URL, e.g. ``http://coordinator:9300``.
        timeout_seconds: Per-request timeout.
        attempts: Bounded attempts per GET.
        transport: Custom transport (tests inject an ``httpx.MockTransport``);
            ``None`` uses real sockets.
        """
        self._base_url = base_url.rstrip("/")
        self._attempts = attempts
        self._client = httpx.AsyncClient(timeout=timeout_seconds, transport=transport)

    @property
    def base_url(self) -> str:
        """The configured coordinator base URL (no trailing slash)."""
        return self._base_url

    async def get_instances(self) -> CoordinatorInstances:
        """Fetch ``GET /instances``.

        Returns:
            The membership list.

        Raises:
            ClientError: On transport failure after every attempt, a non-2xx
                status, or a body missing a required field.
        """
        body = await get_json(
            self._client,
            f"{self._base_url}{COORDINATOR_INSTANCES_PATH}",
            attempts=self._attempts,
        )
        try:
            return CoordinatorInstances.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(f"/instances schema mismatch: {exc}") from exc

    async def get_fleet_usage(self) -> FleetUsage:
        """Fetch ``GET /instances/usage``.

        Returns:
            The fleet memory view.

        Raises:
            ClientError: On transport failure after every attempt, a non-2xx
                status, or a body missing a required field.
        """
        body = await get_json(
            self._client,
            f"{self._base_url}{COORDINATOR_USAGE_PATH}",
            attempts=self._attempts,
        )
        try:
            return FleetUsage.model_validate(body.as_dict())
        except ValidationError as exc:
            raise ClientResponseError(
                f"/instances/usage schema mismatch: {exc}"
            ) from exc

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        await self._client.aclose()
