# SPDX-License-Identifier: Apache-2.0
"""Leadership: a Kubernetes Lease or a static single-process grant.

A Lease is *coordination*, not a downstream fencing token: production
safety comes from ``replicas: 1``, ``strategy: Recreate`` and a
``ReadWriteOncePod`` PVC. The elector therefore only has to be strict
about *itself*: any renewal conflict, timeout, or holder-identity change
is an immediate loss of permission to POST, and :meth:`ensure_leader`
re-renews immediately before every mutating call.

The Kubernetes implementation talks to the API server with ``httpx``
(``coordination.k8s.io/v1``, ``resourceVersion`` optimistic concurrency)
and expects the Lease to be pre-created by the manifests.
"""

# Standard
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol
import asyncio
import os
import socket

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.config import (
    LeaderElectionMode,
    MPMemoryCoordinatorConfig,
)

logger = init_logger(__name__)

_MICRO_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


class LeaderElector(Protocol):
    """What the controller needs from leadership."""

    @property
    def identity(self) -> str:
        """This process's holder identity."""
        ...

    def is_leader(self) -> bool:
        """Whether the last renewal granted leadership and is still valid."""
        ...

    async def ensure_leader(self) -> bool:
        """Renew now and return whether this process may mutate."""
        ...

    async def run(self, stop: asyncio.Event) -> None:
        """Renew on the configured interval until ``stop`` is set."""
        ...

    async def release(self) -> None:
        """Give up leadership on a clean stop (best effort)."""
        ...


class StaticLeader:
    """Always leader: single-process development and the local E2E harness."""

    def __init__(self, identity: str) -> None:
        """Args:
        identity: Reported holder identity.
        """
        self._identity = identity

    @property
    def identity(self) -> str:
        """See :class:`LeaderElector`."""
        return self._identity

    def is_leader(self) -> bool:
        """Always ``True``."""
        return True

    async def ensure_leader(self) -> bool:
        """Always ``True``."""
        return True

    async def run(self, stop: asyncio.Event) -> None:
        """Wait until ``stop`` is set."""
        await stop.wait()

    async def release(self) -> None:
        """Nothing to release."""
        return


def _now_micro(clock: Callable[[], float]) -> str:
    """Format ``clock()`` as Kubernetes ``MicroTime``."""
    return datetime.fromtimestamp(clock(), tz=timezone.utc).strftime(_MICRO_TIME_FORMAT)


def _parse_micro(value: object) -> float:
    """Parse a ``MicroTime``/``Time`` string into epoch seconds (``0`` if unset)."""
    if not isinstance(value, str) or not value:
        return 0.0
    text = value.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return 0.0


class KubernetesLeaseElector:
    """Holds a pre-created ``coordination.k8s.io/v1`` Lease.

    Acquisition: ``GET`` the Lease; if it has no holder, its holder is us, or
    its ``renewTime + leaseDurationSeconds`` is in the past, ``PUT`` it back
    with our identity, ``resourceVersion`` unchanged (the API server rejects
    the write with ``409`` if anyone else wrote in between). Any failure to
    renew leaves :meth:`is_leader` ``False``.
    """

    def __init__(
        self,
        config: MPMemoryCoordinatorConfig,
        *,
        clock: Callable[[], float],
        identity: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Args:
        config: Lease name/namespace/timings and API server settings.
        clock: Wall-clock source (Lease times are wall-clock).
        identity: Holder identity; empty means the hostname.
        transport: Custom transport for tests; ``None`` uses sockets.

        Raises:
            ValueError: If the namespace or API server URL cannot be
                resolved from the configuration or environment.
        """
        self._config = config
        self._clock = clock
        self._identity = identity or config.holder_identity or socket.gethostname()
        namespace = config.lease_namespace or os.environ.get("POD_NAMESPACE", "")
        if not namespace:
            raise ValueError("lease_namespace is empty and POD_NAMESPACE is unset")
        api_url = config.kubernetes_api_url
        if not api_url:
            host = os.environ.get("KUBERNETES_SERVICE_HOST", "")
            port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
            if not host:
                raise ValueError("kubernetes_api_url is empty and not in-cluster")
            api_url = f"https://{host}:{port}"
        self._url = (
            f"{api_url.rstrip('/')}/apis/coordination.k8s.io/v1/namespaces/"
            f"{namespace}/leases/{config.lease_name}"
        )
        headers: dict[str, str] = {}
        token_path = Path(config.kubernetes_token_path)
        if token_path.exists():
            headers["Authorization"] = f"Bearer {token_path.read_text().strip()}"
        verify: bool | str = config.kubernetes_ca_path or False
        if api_url.startswith("http://"):
            verify = False
        self._client = httpx.AsyncClient(
            timeout=config.request_timeout_seconds,
            headers=headers,
            verify=verify,
            transport=transport,
        )
        self._leader_until = 0.0
        self._lock = asyncio.Lock()

    @property
    def identity(self) -> str:
        """See :class:`LeaderElector`."""
        return self._identity

    def is_leader(self) -> bool:
        """Whether the last successful renewal is still within the lease."""
        return self._clock() < self._leader_until

    async def ensure_leader(self) -> bool:
        """Renew immediately; see :class:`LeaderElector`."""
        async with self._lock:
            return await self._renew()

    async def run(self, stop: asyncio.Event) -> None:
        """Renew every ``lease_renew_interval_seconds`` until ``stop``."""
        while not stop.is_set():
            await self.ensure_leader()
            try:
                await asyncio.wait_for(
                    stop.wait(), timeout=self._config.lease_renew_interval_seconds
                )
            except asyncio.TimeoutError:
                continue

    async def release(self) -> None:
        """Clear our holder identity if we hold the Lease (best effort)."""
        async with self._lock:
            self._leader_until = 0.0
            try:
                response = await self._client.get(self._url)
                if response.status_code != 200:
                    return
                lease = response.json()
                spec = lease.get("spec", {})
                if spec.get("holderIdentity") != self._identity:
                    return
                spec["holderIdentity"] = ""
                await self._client.put(self._url, json=lease)
            except (httpx.HTTPError, ValueError):
                return
            finally:
                await self._client.aclose()

    async def _renew(self) -> bool:
        """One acquire/renew round trip.

        Returns:
            ``True`` if the Lease is ours after the round trip.
        """
        now = self._clock()
        try:
            response = await self._client.get(self._url)
        except httpx.HTTPError as exc:
            logger.warning("lease GET failed: %s", exc)
            self._leader_until = 0.0
            return False
        if response.status_code != 200:
            logger.warning("lease GET returned %s", response.status_code)
            self._leader_until = 0.0
            return False
        try:
            lease = response.json()
        except ValueError:
            self._leader_until = 0.0
            return False
        spec = dict(lease.get("spec", {}))
        holder = str(spec.get("holderIdentity") or "")
        renew_time = _parse_micro(spec.get("renewTime"))
        duration = float(spec.get("leaseDurationSeconds") or 0)
        expired = renew_time + duration <= now
        if holder and holder != self._identity and not expired:
            self._leader_until = 0.0
            return False
        spec["holderIdentity"] = self._identity
        spec["leaseDurationSeconds"] = int(self._config.lease_duration_seconds)
        spec["renewTime"] = _now_micro(self._clock)
        if not spec.get("acquireTime") or holder != self._identity:
            spec["acquireTime"] = spec["renewTime"]
            spec["leaseTransitions"] = int(spec.get("leaseTransitions") or 0) + 1
        lease["spec"] = spec
        try:
            put = await self._client.put(self._url, json=lease)
        except httpx.HTTPError as exc:
            logger.warning("lease PUT failed: %s", exc)
            self._leader_until = 0.0
            return False
        if put.status_code != 200:
            logger.warning(
                "lease PUT returned %s (conflict or denied)", put.status_code
            )
            self._leader_until = 0.0
            return False
        self._leader_until = now + self._config.lease_duration_seconds
        return True


def build_leader(
    config: MPMemoryCoordinatorConfig, clock: Callable[[], float]
) -> LeaderElector:
    """Construct the elector selected by ``config.leader_election``.

    Args:
        config: The configuration.
        clock: Wall-clock source.

    Returns:
        A :class:`StaticLeader` or :class:`KubernetesLeaseElector`.
    """
    if config.leader_election is LeaderElectionMode.KUBERNETES:
        return KubernetesLeaseElector(config, clock=clock)
    return StaticLeader(config.holder_identity or socket.gethostname())
