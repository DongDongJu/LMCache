# SPDX-License-Identifier: Apache-2.0
"""FastAPI applications for the four scenario-server listeners.

``build_apps`` returns one app per listener. Production-facing apps
(coordinator, donor MP, receiver MP) expose only the real routes with the
golden schemas, no docs, and audit every request and response. The admin app
exposes only ``/__test/*`` routes.
"""

# Standard
from dataclasses import dataclass
from decimal import Decimal
import json

# Third Party
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Local
from .faults import BarrierOperation, BarrierRequest, FaultSpec
from .state import (
    DONOR_ID,
    RECEIVER_ID,
    AuditLog,
    DaxAddRequest,
    DaxRemoveRequest,
    DeviceUpdate,
    HttpResult,
    IdentityBump,
    PresentDevice,
    ScenarioState,
    SizeRequest,
)

COORDINATOR_SERVICE = "coordinator"

_MAX_SIZE_STRING_LEN = 64
_SIZE_ERROR = "size must be a positive integer byte count or a string like '100GiB'"
_SIZE_UNITS: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
    "tib": 1024**4,
}


def _parse_size_string(size: str) -> int:
    """Parse ``"64GiB"``-style sizes exactly like the production API.

    Returns:
        The byte count.

    Raises:
        ValueError: If the text is not a positive number with a known unit.
    """
    text = size.strip()
    if not text or len(text) > _MAX_SIZE_STRING_LEN:
        raise ValueError(_SIZE_ERROR)
    unit_start = len(text)
    while unit_start > 0 and text[unit_start - 1].isalpha():
        unit_start -= 1
    value_text = text[:unit_start].strip()
    unit = text[unit_start:].lower()
    if unit not in _SIZE_UNITS:
        raise ValueError(_SIZE_ERROR)
    if "." in value_text:
        whole, fraction = value_text.split(".", 1)
        if not whole or not fraction or not whole.isdigit() or not fraction.isdigit():
            raise ValueError(_SIZE_ERROR)
    elif not value_text.isdigit():
        raise ValueError(_SIZE_ERROR)
    value = Decimal(value_text)
    if value <= 0:
        raise ValueError(_SIZE_ERROR)
    return int(value * _SIZE_UNITS[unit])


def resolve_size_bytes(size: SizeRequest) -> int:
    """Resolve an int or ``"64GiB"`` size request to bytes.

    Args:
        size: Positive integer byte count or a size string.

    Returns:
        The byte count.

    Raises:
        ValueError: If the size is a bool, non-positive, or unparsable.
    """
    if isinstance(size, bool):
        raise ValueError(_SIZE_ERROR)
    resolved = size if isinstance(size, int) else _parse_size_string(size)
    if resolved <= 0:
        raise ValueError(_SIZE_ERROR)
    return resolved


def _decode_json(raw: bytes) -> object:
    """Decode a JSON body, falling back to text; empty bodies become ``None``.

    Returns:
        The parsed JSON value, the raw text, or ``None``.
    """
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode("utf-8", errors="replace")


async def _read_body(receive: Receive) -> bytes:
    """Drain the ASGI request body.

    Returns:
        The complete body.
    """
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.request":
            chunks.append(bytes(message.get("body", b"")))
            if not message.get("more_body", False):
                break
        elif message["type"] == "http.disconnect":
            break
    return b"".join(chunks)


def _replay_receive(body: bytes, receive: Receive) -> Receive:
    """Return a receive callable that replays ``body`` once, then defers.

    Returns:
        The replaying callable; later calls forward to the original so a
        real disconnect is still observed.
    """
    replayed = False

    async def replay() -> Message:
        nonlocal replayed
        if not replayed:
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _to_response(result: HttpResult) -> JSONResponse:
    """Convert a state result into a JSON response.

    Returns:
        The response with the result's status code and body.
    """
    return JSONResponse(status_code=result.status_code, content=result.body)


def _error(status_code: int, message: str) -> JSONResponse:
    """Build ``{"error": message}`` with the given status.

    Returns:
        The error response.
    """
    return JSONResponse(status_code=status_code, content={"error": message})


def _validation_error(exc: ValidationError) -> JSONResponse:
    """Build the 422 body the production API returns for a bad payload.

    Returns:
        422 ``{"detail": [...]}``.
    """
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors(include_url=False))},
    )


def _production_app(service: str, audit: AuditLog) -> FastAPI:
    """Create a docs-less FastAPI app whose traffic is audited.

    Returns:
        The app with :class:`AuditMiddleware` installed.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(AuditMiddleware, service=service, audit=audit)
    return app


class AuditMiddleware:
    """Pure ASGI middleware auditing every request and response.

    The request body is read up front so the ``request`` record (with the
    exact JSON body) precedes any mutation record the handler appends, and
    the ``response`` record follows it. Non-HTTP scopes pass through.

    Args:
        app: Wrapped ASGI app.
        service: Audit ``service`` label.
        audit: Shared audit log.
    """

    def __init__(self, app: ASGIApp, service: str, audit: AuditLog) -> None:
        self._app = app
        self._service = service
        self._audit = audit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Audit one HTTP exchange around the wrapped app."""
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        method = str(scope["method"])
        path = str(scope["path"])
        query = bytes(scope.get("query_string", b"")).decode("latin-1")
        if query:
            path = f"{path}?{query}"
        body = await _read_body(receive)
        self._audit.record_request(self._service, method, path, _decode_json(body))
        status_code = 0
        chunks: list[bytes] = []

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            elif message["type"] == "http.response.body":
                chunks.append(bytes(message.get("body", b"")))
            await send(message)

        try:
            await self._app(scope, _replay_receive(body, receive), send_wrapper)
        finally:
            self._audit.record_response(
                self._service, method, path, status_code, _decode_json(b"".join(chunks))
            )


class UsageUpdate(BaseModel):
    """Body of ``POST /__test/usage``."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str
    used_bytes: int = Field(ge=0)


class ReregisterRequest(BaseModel):
    """Body of ``POST /__test/instances/{instance_id}/reregister``."""

    model_config = ConfigDict(extra="forbid")

    bump: IdentityBump


@dataclass
class ScenarioApps:
    """The four listener apps built from one :class:`ScenarioState`.

    Attributes:
        coordinator: Fake MP Coordinator (``/instances`` family).
        donor: Fake ``mp-donor`` MP server (also served on its alt port).
        receiver: Fake ``mp-receiver`` MP server (also served on its alt port).
        admin: ``/__test/*`` control plane.
    """

    coordinator: FastAPI
    donor: FastAPI
    receiver: FastAPI
    admin: FastAPI


def build_coordinator_app(state: ScenarioState) -> FastAPI:
    """Build the fake MP Coordinator app.

    Routes: ``GET /instances``, ``GET /instances/usage``,
    ``GET /instances/{instance_id}/usage``, ``GET /healthz``. Nothing else.

    Returns:
        The app.
    """
    app = _production_app(COORDINATOR_SERVICE, state.audit)

    @app.get("/instances")
    async def list_instances() -> JSONResponse:
        return _to_response(state.list_instances())

    @app.get("/instances/usage")
    async def fleet_usage() -> JSONResponse:
        return _to_response(state.fleet_usage())

    @app.get("/instances/{instance_id}/usage")
    async def instance_usage(instance_id: str) -> JSONResponse:
        return _to_response(state.instance_usage(instance_id))

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return _to_response(state.coordinator_health())

    return app


def build_mp_app(state: ScenarioState, instance_id: str) -> FastAPI:
    """Build one fake MP server app bound to ``instance_id``.

    Routes: ``GET /``, ``GET /healthcheck``, ``GET /status``,
    ``GET /reconfigure/dax/status``, ``POST /reconfigure/dax/remove``,
    ``POST /reconfigure/dax/add``. Mutations honour armed barriers before and
    after touching state.

    Args:
        state: Shared state.
        instance_id: ``mp-donor`` or ``mp-receiver``; also the audit service.

    Returns:
        The app.
    """
    app = _production_app(instance_id, state.audit)

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse({"status": "ok", "service": "LMCache HTTP API"})

    @app.get("/healthcheck")
    async def healthcheck() -> JSONResponse:
        return _to_response(state.healthcheck(instance_id))

    @app.get("/status")
    async def status() -> JSONResponse:
        return _to_response(state.mp_status(instance_id))

    @app.get("/reconfigure/dax/status")
    async def dax_status() -> JSONResponse:
        return _to_response(state.dax_status(instance_id))

    @app.post("/reconfigure/dax/remove")
    async def dax_remove(payload: dict[str, object]) -> JSONResponse:
        try:
            body = DaxRemoveRequest.model_validate(payload)
        except ValidationError as exc:
            return _validation_error(exc)
        operation: BarrierOperation = "drain" if body.mode == "drain" else "evict"
        await state.barriers.wait(instance_id, operation, "before")
        result = state.remove_device(instance_id, body)
        await state.barriers.wait(instance_id, operation, "after")
        return _to_response(result)

    @app.post("/reconfigure/dax/add")
    async def dax_add(payload: dict[str, object]) -> JSONResponse:
        try:
            body = DaxAddRequest.model_validate(payload)
        except ValidationError as exc:
            return _validation_error(exc)
        try:
            size_bytes = resolve_size_bytes(body.size)
        except ValueError:
            return _error(400, _SIZE_ERROR)
        await state.barriers.wait(instance_id, "add", "before")
        result = state.add_device(
            instance_id, body.adapter_index, body.device_path, size_bytes
        )
        await state.barriers.wait(instance_id, "add", "after")
        return _to_response(result)

    return app


def build_admin_app(state: ScenarioState) -> FastAPI:
    """Build the ``/__test/*`` admin app.

    Returns:
        The app (no docs, no production routes).
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/__test/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "seq": state.audit.seq})

    @app.post("/__test/reset")
    async def reset() -> JSONResponse:
        return JSONResponse(state.reset())

    @app.get("/__test/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(state.snapshot())

    @app.get("/__test/audit")
    async def get_audit(after_seq: int = 0) -> JSONResponse:
        return JSONResponse({"records": state.audit.after(after_seq)})

    @app.post("/__test/faults")
    async def set_faults(spec: FaultSpec) -> JSONResponse:
        try:
            return JSONResponse(state.apply_faults(spec))
        except KeyError as exc:
            return _error(404, f"unknown instance {exc.args[0]!r}")

    @app.delete("/__test/faults")
    async def clear_faults() -> JSONResponse:
        return JSONResponse(state.clear_faults())

    @app.post("/__test/usage")
    async def set_usage(body: UsageUpdate) -> JSONResponse:
        try:
            return JSONResponse(state.set_usage(body.instance_id, body.used_bytes))
        except KeyError:
            return _error(404, f"unknown instance {body.instance_id!r}")

    @app.post("/__test/devices")
    async def update_device(body: DeviceUpdate) -> JSONResponse:
        try:
            return JSONResponse(state.update_device(body))
        except KeyError:
            return _error(404, f"unknown instance {body.instance_id!r}")
        except LookupError:
            return _error(404, f"unknown device {body.device_path!r}")

    @app.post("/__test/present_devices")
    async def declare_present_device(body: PresentDevice) -> JSONResponse:
        try:
            return JSONResponse(state.declare_present_device(body))
        except KeyError:
            return _error(404, f"unknown instance {body.instance_id!r}")

    @app.post("/__test/instances/{instance_id}/reregister")
    async def reregister(instance_id: str, body: ReregisterRequest) -> JSONResponse:
        try:
            return JSONResponse(state.reregister(instance_id, body.bump))
        except KeyError:
            return _error(404, f"unknown instance {instance_id!r}")

    @app.post("/__test/barriers")
    async def arm_barrier(body: BarrierRequest) -> JSONResponse:
        if body.instance_id not in (DONOR_ID, RECEIVER_ID):
            return _error(404, f"unknown instance {body.instance_id!r}")
        try:
            barrier = state.barriers.arm(body)
        except ValueError as exc:
            return _error(409, str(exc))
        return JSONResponse(barrier.snapshot())

    @app.post("/__test/barriers/{name}/release")
    async def release_barrier(name: str) -> JSONResponse:
        try:
            barrier = state.barriers.release(name)
        except KeyError:
            return _error(404, f"unknown barrier {name!r}")
        return JSONResponse(barrier.snapshot())

    return app


def build_apps(state: ScenarioState) -> ScenarioApps:
    """Build all four listener apps over one shared state.

    Args:
        state: The scenario state.

    Returns:
        The apps; the donor/receiver apps are meant to be served on both
        their primary and alternate ports.
    """
    return ScenarioApps(
        coordinator=build_coordinator_app(state),
        donor=build_mp_app(state, DONOR_ID),
        receiver=build_mp_app(state, RECEIVER_ID),
        admin=build_admin_app(state),
    )
