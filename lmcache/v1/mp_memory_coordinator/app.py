# SPDX-License-Identifier: Apache-2.0
"""Process entry point: HTTP probes, metrics, and the control loop.

Endpoints:

* ``GET /healthz`` -- process alive and journal readable.
* ``GET /readyz`` -- current leader, MP Coordinator reachable, inventory
  reconciled, and no BLOCKED move.
* ``GET /status`` -- inventory, cooldowns, history, active move, counters,
  and the last cycle report.
* ``GET /journal`` -- the durable journal document (read-only).
* ``GET /metrics`` -- Prometheus counters for proposed / succeeded /
  rolled-back / blocked moves.

:func:`run_memory_coordinator` wires the clients, journal, leader elector,
and controller, serves the app with uvicorn, and stops gracefully: on
SIGTERM no new move is started and the current cycle persists its state.
"""

# Standard
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio
import signal
import time

# Third Party
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Gauge,
    generate_latest,
)
import uvicorn

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_memory_coordinator.adoption import (
    AdoptionResult,
    adopt,
    load_adoption_file,
)
from lmcache.v1.mp_memory_coordinator.clients import ClientError
from lmcache.v1.mp_memory_coordinator.clients.memory_allocation_client import (
    MemoryAllocationClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_coordinator_client import (
    MPCoordinatorClient,
)
from lmcache.v1.mp_memory_coordinator.clients.mp_server_client import MPServerClient
from lmcache.v1.mp_memory_coordinator.config import MPMemoryCoordinatorConfig
from lmcache.v1.mp_memory_coordinator.controller import (
    HttpRemote,
    RebalanceController,
)
from lmcache.v1.mp_memory_coordinator.leader import LeaderElector, build_leader
from lmcache.v1.mp_memory_coordinator.persistence.rebalance_journal import (
    JournalError,
    RebalanceJournal,
)

logger = init_logger(__name__)


class Metrics:
    """Minimal counters exported at ``/metrics``.

    Gauges mirror the persisted journal counters so a restart reports the
    durable totals rather than restarting from zero.
    """

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.proposed = Gauge(
            "lmcache_memcoord_moves_proposed_total",
            "Moves proposed by the dry-run policy",
            registry=self.registry,
        )
        self.succeeded = Gauge(
            "lmcache_memcoord_moves_succeeded_total",
            "Moves that completed with outcome SUCCEEDED",
            registry=self.registry,
        )
        self.rolled_back = Gauge(
            "lmcache_memcoord_moves_rolled_back_total",
            "Moves that completed with outcome ROLLED_BACK",
            registry=self.registry,
        )
        self.blocked = Gauge(
            "lmcache_memcoord_moves_blocked_total",
            "Moves that entered BLOCKED",
            registry=self.registry,
        )
        self.leader = Gauge(
            "lmcache_memcoord_leader",
            "1 when this process holds leadership",
            registry=self.registry,
        )

    def update(self, controller: RebalanceController) -> None:
        """Copy the controller's persisted counters into the gauges."""
        counters = controller.document.counters
        self.proposed.set(counters.proposed)
        self.succeeded.set(counters.succeeded)
        self.rolled_back.set(counters.rolled_back)
        self.blocked.set(counters.blocked)
        self.leader.set(1 if controller.readiness()[0] else 0)


def create_app(
    controller: RebalanceController,
    journal: RebalanceJournal,
    leader: LeaderElector,
    metrics: Metrics,
) -> FastAPI:
    """Build the probe/status app around a controller.

    Args:
        controller: The control loop owner.
        journal: The journal (for ``/healthz`` readability).
        leader: The leader elector (reported in ``/status``).
        metrics: The metrics registry.

    Returns:
        The FastAPI app. The control loop and leader renewal are started by
        :func:`run_memory_coordinator`, not by the app lifespan, so the app
        can be tested with a controller driven by hand.
    """
    app = FastAPI(
        title="LMCache MP Memory Coordinator",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """Process alive and journal readable."""
        if controller.journal_error:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": controller.journal_error},
            )
        return JSONResponse(content={"status": "healthy"})

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        """Leader, MP Coordinator reachable, reconciled, not BLOCKED."""
        ready, reason = controller.readiness()
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "unready", "reason": reason},
        )

    @app.get("/status")
    async def status() -> JSONResponse:
        """Inventory, cooldowns, history, active move, counters, last cycle."""
        return JSONResponse(content=controller.status())

    @app.get("/journal")
    async def journal_view() -> JSONResponse:
        """The durable journal document (read-only)."""
        return JSONResponse(content=controller.document.model_dump(mode="json"))

    @app.get("/metrics")
    async def metrics_view() -> Response:
        """Prometheus exposition of the move counters."""
        metrics.update(controller)
        return Response(
            content=generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST
        )

    return app


async def adoption_retry_loop(
    attempt: Callable[[], Awaitable[AdoptionResult]],
    is_initialized: Callable[[], bool],
    stop: asyncio.Event,
    interval_seconds: float,
) -> None:
    """Retry the startup adoption until the journal is initialized.

    A coordinator that starts while the MP Coordinator or the outside
    service is unreachable must still serve its probes (unready) rather
    than crash; adoption is attempted again every ``interval_seconds``
    until it succeeds, the journal becomes initialized, or ``stop`` is set.
    A failed pass never touches the journal, so nothing is half-adopted.

    Args:
        attempt: Runs one adoption pass (typically
            :meth:`RebalanceController.adopt_once`).
        is_initialized: Whether the journal now carries the marker.
        stop: Shutdown signal.
        interval_seconds: Delay between attempts.
    """
    while not stop.is_set() and not is_initialized():
        try:
            result = await attempt()
        except (ClientError, ValueError, JournalError) as exc:
            logger.warning("startup adoption deferred: %s", exc)
        else:
            logger.info(
                "adoption: %d adopted, %d rejected %s",
                len(result.adopted),
                len(result.rejected),
                result.rejected,
            )
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


def run_memory_coordinator(config: MPMemoryCoordinatorConfig) -> None:
    """Run the MP Memory Coordinator process until SIGTERM/SIGINT.

    Args:
        config: The validated configuration.
    """
    asyncio.run(_serve(config))


def run_adoption(config: MPMemoryCoordinatorConfig, allowlist: Path) -> None:
    """Explicitly adopt the allowlisted allocations into the journal and exit.

    Unlike ``adoption_file`` at startup, this command runs even when the
    journal already carries the initialization marker: it is the operator's
    explicit action. Entries already owned are rejected as such.

    Args:
        config: The validated configuration (state directory and remotes).
        allowlist: The allowlist YAML file.

    Raises:
        ValueError: If the allowlist is invalid.
        JournalError: If the journal cannot be trusted.
        ClientError: If the MP Coordinator or outside service is unreachable.
    """
    asyncio.run(_adopt(config, allowlist))


async def _adopt(config: MPMemoryCoordinatorConfig, allowlist: Path) -> None:
    """Async body of :func:`run_adoption`."""
    entries = load_adoption_file(allowlist)
    coordinator = MPCoordinatorClient(
        config.mp_coordinator_url,
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    mp_client = MPServerClient(
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    allocator = MemoryAllocationClient(
        config.memory_allocation_url,
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    journal = RebalanceJournal(Path(config.state_directory))
    try:
        document = journal.load()
        result = await adopt(
            entries,
            document,
            coordinator=coordinator,
            mp_client=mp_client,
            allocator=allocator,
            config=config,
            clock=time.time,
        )
        journal.save(document)
    finally:
        await coordinator.aclose()
        await mp_client.aclose()
        await allocator.aclose()
    for allocation in result.adopted:
        print(f"adopted {allocation.device_path} on {allocation.worker_ip}")
    for path, reason in result.rejected.items():
        print(f"rejected {path}: {reason}")


async def _serve(config: MPMemoryCoordinatorConfig) -> None:
    """Async body of :func:`run_memory_coordinator`."""
    coordinator = MPCoordinatorClient(
        config.mp_coordinator_url,
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    mp_client = MPServerClient(
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    allocator = MemoryAllocationClient(
        config.memory_allocation_url,
        timeout_seconds=config.request_timeout_seconds,
        attempts=config.get_retry_attempts,
    )
    remote = HttpRemote(
        coordinator,
        mp_client,
        allocator,
        adapter_index=config.adapter_index,
        clock=time.time,
    )
    journal = RebalanceJournal(Path(config.state_directory))
    leader = build_leader(config, time.time)
    controller = RebalanceController(config, journal, remote, leader)
    controller.load()
    metrics = Metrics()
    app = create_app(controller, journal, leader, metrics)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.http_host,
            port=config.http_port,
            log_level="info",
        )
    )

    async def _adopt_once() -> AdoptionResult:
        entries = load_adoption_file(Path(config.adoption_file))
        return await controller.adopt_once(
            entries,
            coordinator=coordinator,
            mp_client=mp_client,
            allocator=allocator,
        )

    @asynccontextmanager
    async def _tasks() -> AsyncIterator[None]:
        leader_task = asyncio.create_task(leader.run(stop))
        control_task = asyncio.create_task(controller.run_forever(stop))
        adoption_task = None
        if (
            config.adoption_file
            and not controller.journal_error
            and not controller.document.initialized
        ):
            adoption_task = asyncio.create_task(
                adoption_retry_loop(
                    _adopt_once,
                    lambda: controller.document.initialized,
                    stop,
                    config.poll_interval_seconds,
                )
            )
        try:
            yield
        finally:
            controller.request_stop()
            stop.set()
            await control_task
            await leader_task
            if adoption_task is not None:
                await adoption_task
            await leader.release()
            await coordinator.aclose()
            await mp_client.aclose()
            await allocator.aclose()

    logger.info(
        "MP Memory Coordinator listening on http://%s:%d (actuation_enabled=%s)",
        config.http_host,
        config.http_port,
        config.actuation_enabled,
    )
    async with _tasks():
        # uvicorn installs its own SIGTERM/SIGINT handlers while serving;
        # either signal path ends the server, and the tasks context above
        # then stops the control loop gracefully (no new move, state saved).
        serve_task = asyncio.create_task(server.serve())
        stop_task = asyncio.create_task(stop.wait())
        await asyncio.wait({serve_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        stop.set()
        server.should_exit = True
        await serve_task
        stop_task.cancel()
