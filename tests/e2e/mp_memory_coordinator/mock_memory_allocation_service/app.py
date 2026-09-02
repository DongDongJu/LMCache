# SPDX-License-Identifier: Apache-2.0
"""FastAPI applications of the mock Memory Allocation service.

Two separate applications are built so that they can be served on distinct
ports:

* the **public** app exposes exactly the three frozen outside routes of
  PLAN.md Section 2 and nothing else (no root, docs, OpenAPI or ``/__test``);
* the **admin** app exposes only the ``/__test`` routes used by E2E tests and
  never mounts the outside API.

A raw ASGI middleware on the public app records every observed request and
response in the audit log and implements the ``commit_then_drop`` fault by
aborting the response after its headers were sent.
"""

# Standard
from pathlib import Path
import asyncio
import json
import logging

# Third Party
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Local
from .faults import BarrierRegistry, BarrierSpec, FaultRegistry, FaultSpec
from .models import (
    AllocationRequest,
    DeallocationRequest,
    build_allocation_response,
    build_deallocation_response,
)
from .state import MockAllocatorState, MockServiceError, Operation

logger = logging.getLogger(__name__)

DROP_RESPONSE_STATE_KEY: str = "mock_drop_response"
"""``request.state`` flag a handler sets to have the middleware drop the
response after its headers (``commit_then_drop``)."""

STATUS_PATH: str = "/api/v2/apps/lmcache"
DEALLOCATIONS_PATH: str = "/api/v2/apps/lmcache/deallocations"
ALLOCATIONS_PATH: str = "/api/v2/apps/lmcache/allocations"

_ROUTE_OPERATIONS: dict[tuple[str, str], Operation] = {
    ("GET", STATUS_PATH): Operation.STATUS,
    ("POST", DEALLOCATIONS_PATH): Operation.DEALLOCATE,
    ("POST", ALLOCATIONS_PATH): Operation.ALLOCATE,
}

_SIZE_FIELDS: tuple[str, ...] = (
    "released_size_gib",
    "requested_size_gib",
    "granted_size_gib",
)


class PoolBudgetSpec(BaseModel):
    """Body of ``POST /__test/reset`` and ``POST /__test/pool_budget``.

    Attributes:
        pool_budget_gib: Maximum global assigned runtime GiB an allocation may
            reach; ``null`` (the default) means unlimited.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    pool_budget_gib: int | None = Field(default=None, ge=0)


class ResponseDropped(Exception):
    """Raised inside ``send`` to abort a response after its headers were sent.

    The exception propagates out of the ASGI application, so uvicorn closes
    the connection without completing the body and test clients observe an
    empty body instead of a valid ``DONE`` document.
    """


def create_state(fixture_path: Path, state_file: Path | None) -> MockAllocatorState:
    """Create the shared state for both applications.

    Args:
        fixture_path: YAML topology fixture.
        state_file: JSON file for persistence, or ``None`` for in-memory only.

    Returns:
        The loaded state (from ``state_file`` if it exists, else the fixture).

    Raises:
        ValueError: If the fixture or state file is invalid.
        OSError: If a file cannot be read or written.
    """
    return MockAllocatorState(fixture_path=fixture_path, state_file=state_file)


def build_apps(
    state: MockAllocatorState, faults: FaultRegistry, barriers: BarrierRegistry
) -> tuple[FastAPI, FastAPI]:
    """Build the public and admin applications over shared state.

    Args:
        state: Inventory, audit and persistence shared by both apps.
        faults: Registry driven by ``/__test/faults``.
        barriers: Registry driven by ``/__test/barriers``.

    Returns:
        ``(public_app, admin_app)``.
    """
    return build_public_app(state, faults, barriers), build_admin_app(
        state, faults, barriers
    )


def build_public_app(
    state: MockAllocatorState, faults: FaultRegistry, barriers: BarrierRegistry
) -> FastAPI:
    """Build the application serving only the three frozen outside routes.

    Args:
        state: Shared inventory and audit log.
        faults: Faults consulted by the two POST handlers.
        barriers: Barriers consulted by the two POST handlers.

    Returns:
        A FastAPI app with docs, OpenAPI and the root route disabled.
    """
    app = FastAPI(
        title="mock-memory-allocation-public",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(PublicAuditMiddleware, state=state)
    app.add_exception_handler(MockServiceError, _mock_service_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _internal_error_handler)

    @app.get(STATUS_PATH, response_model=None)
    async def status() -> JSONResponse:
        return JSONResponse(await state.status_view())

    @app.post(DEALLOCATIONS_PATH, response_model=None)
    async def deallocate(body: DeallocationRequest, request: Request) -> JSONResponse:
        fault = faults.take("deallocate")
        _raise_if_fault_rejects(fault)
        await barriers.wait("deallocate", "before")
        result = await state.deallocate(
            body.request_id, body.target_node, body.device_path
        )
        await barriers.wait("deallocate", "after")
        response_body = build_deallocation_response(
            result.request_id,
            result.target_node,
            result.device_path,
            result.released_size_gib,
        )
        return await _finish_response(request, fault, response_body)

    @app.post(ALLOCATIONS_PATH, response_model=None)
    async def allocate(body: AllocationRequest, request: Request) -> JSONResponse:
        # Pool admission precedes injected behaviour: a request the budget
        # refuses consumes no fault and hits no barrier, so faults always
        # target a request the pool would serve.
        admitted = await state.pool_admits(body.request_size_gib)
        fault = faults.take("allocate") if admitted else None
        _raise_if_fault_rejects(fault)
        if admitted:
            await barriers.wait("allocate", "before")
        result = await state.allocate(
            body.request_id, body.target_node, body.request_size_gib
        )
        await barriers.wait("allocate", "after")
        response_body = build_allocation_response(
            result.request_id,
            result.target_node,
            result.device_path,
            result.requested_size_gib,
            result.granted_size_gib,
        )
        return await _finish_response(request, fault, response_body)

    return app


def build_admin_app(
    state: MockAllocatorState, faults: FaultRegistry, barriers: BarrierRegistry
) -> FastAPI:
    """Build the test-only administration application.

    Args:
        state: Shared inventory and audit log.
        faults: Registry manipulated by ``/__test/faults``.
        barriers: Registry manipulated by ``/__test/barriers``.

    Returns:
        A FastAPI app exposing only the ``/__test`` routes.
    """
    app = FastAPI(
        title="mock-memory-allocation-admin",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    async def state_view() -> dict[str, object]:
        view = await state.snapshot()
        view["faults"] = {"active": faults.view()}
        view["barriers"] = barriers.view()
        return view

    @app.get("/__test/health", response_model=None)
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "fixture": str(state.fixture_path),
                "seq": await state.current_seq(),
            }
        )

    @app.post("/__test/reset", response_model=None)
    async def reset(request: Request) -> JSONResponse:
        # The body is optional: an empty body resets to an unlimited pool.
        raw = await request.body()
        try:
            spec = PoolBudgetSpec.model_validate_json(raw) if raw else PoolBudgetSpec()
        except ValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        barriers.release_all()
        faults.clear()
        await state.reset(pool_budget_gib=spec.pool_budget_gib)
        return JSONResponse(await state_view())

    @app.post("/__test/pool_budget", response_model=None)
    async def set_pool_budget(spec: PoolBudgetSpec) -> JSONResponse:
        await state.set_pool_budget(spec.pool_budget_gib)
        return JSONResponse(await state_view())

    @app.get("/__test/state", response_model=None)
    async def get_state() -> JSONResponse:
        return JSONResponse(await state_view())

    @app.get("/__test/audit", response_model=None)
    async def audit(after_seq: int = Query(default=0, ge=0)) -> JSONResponse:
        return JSONResponse({"records": await state.audit_after(after_seq)})

    @app.post("/__test/faults", response_model=None)
    async def install_fault(spec: FaultSpec) -> JSONResponse:
        active = faults.install(spec)
        return JSONResponse({"faults": [fault.model_dump() for fault in active]})

    @app.delete("/__test/faults", response_model=None)
    async def clear_faults() -> JSONResponse:
        faults.clear()
        return JSONResponse({"faults": []})

    @app.post("/__test/barriers", response_model=None)
    async def install_barrier(spec: BarrierSpec) -> JSONResponse:
        try:
            barriers.install(spec)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=409)
        return JSONResponse({"barriers": barriers.view()})

    @app.post("/__test/barriers/{name}/release", response_model=None)
    async def release_barrier(name: str) -> Response:
        try:
            barriers.release(name)
        except KeyError:
            return JSONResponse({"error": f"unknown barrier {name!r}"}, status_code=404)
        return Response(status_code=204)

    return app


def _raise_if_fault_rejects(fault: FaultSpec | None) -> None:
    """Raise the public error for faults that reject before any mutation."""
    if fault is None:
        return
    if fault.mode == "fail_before_mutation":
        raise MockServiceError(fault.status_code, "injected failure before mutation")
    if fault.mode == "insufficient_capacity":
        raise MockServiceError(409, "insufficient capacity: no matching free device")


async def _finish_response(
    request: Request, fault: FaultSpec | None, body: dict[str, str | int]
) -> JSONResponse:
    """Apply post-mutation fault behaviour and build the final response.

    Args:
        request: The current request; its ``state`` carries the drop flag.
        fault: The consumed fault, if any.
        body: The exact successful response body.

    Returns:
        The response to send (possibly corrupted by the fault).
    """
    if fault is None:
        return JSONResponse(body)
    if fault.mode in ("delay", "commit_then_drop") and fault.delay_seconds > 0:
        await asyncio.sleep(fault.delay_seconds)
    if fault.mode == "commit_then_drop":
        setattr(request.state, DROP_RESPONSE_STATE_KEY, True)
    elif fault.mode == "wrong_echo":
        body[fault.echo_field] = "wrong-" + str(body[fault.echo_field])
    elif fault.mode == "missing_field":
        body.pop(fault.missing_field_name, None)
    elif fault.mode == "wrong_size":
        for field in _SIZE_FIELDS:
            if field in body:
                body[field] = fault.size_gib_override
    elif fault.mode == "invalid_path":
        body["device_path"] = fault.path_override
    return JSONResponse(body)


async def _mock_service_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a :class:`MockServiceError` as ``{"error": message}``."""
    if not isinstance(exc, MockServiceError):
        raise exc
    return JSONResponse({"error": exc.message}, status_code=exc.status_code)


async def _http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render Starlette's own HTTP errors (404, 405, ...) as ``{"error": ...}``."""
    if not isinstance(exc, StarletteHTTPException):
        raise exc
    return JSONResponse(
        {"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers
    )


async def _internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render an unexpected exception (for example a failed self-check) as 500.

    Starlette also invokes this handler for :class:`ResponseDropped`, whose
    response headers were already sent; the returned document is then never
    transmitted, so it only needs to be logged at a quieter level.
    """
    if isinstance(exc, ResponseDropped):
        logger.info("%s: %s", request.url.path, exc)
        return JSONResponse({"error": "response dropped"}, status_code=500)
    logger.error("internal error while serving %s", request.url.path, exc_info=exc)
    return JSONResponse({"error": f"internal error: {exc}"}, status_code=500)


def _string_field(body: dict[str, object], key: str) -> str:
    """Return ``body[key]`` if it is a string, else ``""``."""
    value = body.get(key)
    return value if isinstance(value, str) else ""


def _parse_json_object(raw: bytes) -> dict[str, object]:
    """Decode a JSON object body; wrap anything else as ``{"raw": text}``."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError:
        return {"raw": raw.decode("utf-8", errors="replace")}
    if isinstance(parsed, dict):
        return {str(key): value for key, value in parsed.items()}
    return {"raw": parsed}


async def _read_body(receive: Receive) -> bytes:
    """Drain the request body from ``receive``."""
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


class PublicAuditMiddleware:
    """Raw ASGI middleware that audits public traffic and drops flagged responses.

    For each request to one of the three frozen routes it appends a ``request``
    audit record before the handler runs and a ``response`` record after the
    response body is complete.  When the handler set the
    :data:`DROP_RESPONSE_STATE_KEY` flag it forwards a 200 response start
    without a body and raises :class:`ResponseDropped`, so the client never
    receives a valid document although the mutation was committed.
    """

    def __init__(self, app: ASGIApp, state: MockAllocatorState) -> None:
        self._app = app
        self._state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve one connection; only ``http`` scopes on public routes are audited."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        operation = _ROUTE_OPERATIONS.get((scope["method"], scope["path"]))
        if operation is None:
            await self._app(scope, receive, send)
            return

        raw_body = await _read_body(receive)
        request_json = _parse_json_object(raw_body)
        request_id = _string_field(request_json, "request_id")
        target_node = _string_field(request_json, "target_node")
        device_path = _string_field(request_json, "device_path")
        await self._state.record_request(
            operation, request_id, target_node, device_path, request_json
        )
        request_state: dict[str, object] = scope.setdefault("state", {})

        replayed = False
        status_code = 0
        response_chunks: list[bytes] = []

        async def replay_receive() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": raw_body, "more_body": False}

        async def auditing_send(message: Message) -> None:
            nonlocal status_code
            drop = bool(request_state.get(DROP_RESPONSE_STATE_KEY, False))
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                if drop:
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 200,
                            "headers": [(b"connection", b"close")],
                        }
                    )
                    return
                await send(message)
                return
            if message["type"] == "http.response.body":
                if drop:
                    await self._state.record_response(
                        operation, request_id, target_node, device_path, 0, {}
                    )
                    raise ResponseDropped(
                        f"{operation.value} response dropped after commit"
                    )
                response_chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    await self._state.record_response(
                        operation,
                        request_id,
                        target_node,
                        device_path,
                        status_code,
                        _parse_json_object(b"".join(response_chunks)),
                    )
            await send(message)

        await self._app(scope, replay_receive, auditing_send)
