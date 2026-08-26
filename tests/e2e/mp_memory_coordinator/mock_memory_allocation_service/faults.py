# SPDX-License-Identifier: Apache-2.0
"""Test-only fault injection and barriers for the mock Memory Allocation service.

Both registries are driven exclusively through the admin listener and consulted
by the public handlers.  They hold plain in-memory state and are only ever
touched from the single event loop that serves both listeners, so they need no
lock of their own.
"""

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import asyncio

# Third Party
from pydantic import BaseModel, ConfigDict, Field

FaultOperation = Literal["deallocate", "allocate"]
"""Which public POST a fault or barrier applies to."""

FaultMode = Literal[
    "fail_before_mutation",
    "commit_then_drop",
    "delay",
    "wrong_echo",
    "missing_field",
    "wrong_size",
    "invalid_path",
    "insufficient_capacity",
]
"""Supported fault behaviours; see :class:`FaultSpec` for their semantics."""

EchoField = Literal["request_id", "target_node", "device_path"]
"""Response fields that ``wrong_echo`` may corrupt."""

BarrierPoint = Literal["before", "after"]
"""Where a barrier blocks: before the mutation, or after it but before the
response is sent."""


class FaultSpec(BaseModel):
    """One injected fault, installed through ``POST /__test/faults``.

    Every field is optional and has a sane default so a test only needs to name
    the ``operation`` and ``mode`` it cares about.  Modes:

    * ``fail_before_mutation``: respond ``status_code`` with an error body and
      change nothing.
    * ``commit_then_drop``: apply and audit the mutation, wait
      ``delay_seconds``, then abort the response after the headers so the
      client never receives a valid ``DONE`` body.
    * ``delay``: sleep ``delay_seconds`` and then respond normally.
    * ``wrong_echo``: mutate and respond 200, but replace ``echo_field`` with
      ``"wrong-" + original``.
    * ``missing_field``: mutate and respond 200 without ``missing_field_name``.
    * ``wrong_size``: mutate and respond 200 with every size field
      (``released_size_gib``, ``requested_size_gib``, ``granted_size_gib``)
      replaced by ``size_gib_override``.
    * ``invalid_path``: mutate and respond 200 with ``device_path`` replaced by
      ``path_override``.
    * ``insufficient_capacity``: respond 409 and change nothing, even when a
      matching free device exists.

    Attributes:
        operation: Public POST the fault applies to.
        mode: Behaviour, as listed above.
        count: Number of matching requests the fault applies to before it is
            removed automatically.
        status_code: HTTP status used by ``fail_before_mutation``.
        delay_seconds: Sleep used by ``delay`` and ``commit_then_drop``.
        echo_field: Field corrupted by ``wrong_echo``.
        missing_field_name: Field omitted by ``missing_field``.
        size_gib_override: Value used by ``wrong_size``.
        path_override: Value used by ``invalid_path``.
    """

    model_config = ConfigDict(extra="forbid")

    operation: FaultOperation = "allocate"
    mode: FaultMode = "fail_before_mutation"
    count: int = Field(default=1, ge=1)
    status_code: int = Field(default=500, ge=400, le=599)
    delay_seconds: float = Field(default=0.0, ge=0.0)
    echo_field: EchoField = "request_id"
    missing_field_name: str = "released_size_gib"
    size_gib_override: int = 0
    path_override: str = ""


class FaultRegistry:
    """Ordered list of active faults.

    Faults are matched by ``operation`` in installation order.  Each match
    consumes one unit of the fault's ``count``; a fault whose count reaches
    zero is removed.
    """

    def __init__(self) -> None:
        self._faults: list[FaultSpec] = []

    def install(self, spec: FaultSpec) -> list[FaultSpec]:
        """Install a fault, replacing any active fault with the same operation and mode.

        Args:
            spec: The fault to install.

        Returns:
            A copy of the active fault list after installation.
        """
        for index, existing in enumerate(self._faults):
            if existing.operation == spec.operation and existing.mode == spec.mode:
                self._faults[index] = spec
                break
        else:
            self._faults.append(spec)
        return list(self._faults)

    def clear(self) -> None:
        """Remove every active fault."""
        self._faults.clear()

    def take(self, operation: str) -> FaultSpec | None:
        """Consume one unit of the first active fault for ``operation``.

        Args:
            operation: ``"deallocate"`` or ``"allocate"``.

        Returns:
            The matching fault (with its pre-decrement ``count``), or ``None``
            when no fault is active for that operation.
        """
        for index, fault in enumerate(self._faults):
            if fault.operation != operation:
                continue
            if fault.count > 1:
                self._faults[index] = fault.model_copy(
                    update={"count": fault.count - 1}
                )
            else:
                del self._faults[index]
            return fault
        return None

    def view(self) -> list[dict[str, object]]:
        """Return the active faults as JSON objects, in matching order."""
        return [fault.model_dump() for fault in self._faults]


class BarrierSpec(BaseModel):
    """One named barrier, installed through ``POST /__test/barriers``.

    Attributes:
        operation: Public POST the barrier applies to.
        when: ``"before"`` blocks before the mutation; ``"after"`` blocks after
            the mutation is committed but before the response is sent.
        name: Unique name used to release the barrier.
    """

    model_config = ConfigDict(extra="forbid")

    operation: FaultOperation
    when: BarrierPoint
    name: str = Field(min_length=1)


class BarrierStatus(str, Enum):
    """Lifecycle of a barrier as reported by the admin state view."""

    ARMED = "armed"
    WAITING = "waiting"
    RELEASED = "released"


@dataclass
class _Barrier:
    """Installed barrier plus the event a blocked request waits on."""

    spec: BarrierSpec
    event: asyncio.Event
    status: BarrierStatus


class BarrierRegistry:
    """Named one-shot barriers that block a public request at a chosen point.

    A barrier is consumed by the first matching request that reaches it: that
    request blocks until ``release`` is called, and the barrier is then
    discarded.  Releasing a barrier that no request has reached yet discards
    it without blocking anyone.  Waiters and releasers must share one event
    loop (both listeners of the mock always do).
    """

    def __init__(self) -> None:
        self._barriers: dict[str, _Barrier] = {}

    def install(self, spec: BarrierSpec) -> None:
        """Install an armed barrier.

        Args:
            spec: The barrier to install.

        Raises:
            ValueError: If a barrier with the same name already exists.
        """
        if spec.name in self._barriers:
            raise ValueError(f"barrier {spec.name!r} already exists")
        self._barriers[spec.name] = _Barrier(
            spec=spec, event=asyncio.Event(), status=BarrierStatus.ARMED
        )

    async def wait(self, operation: str, when: str) -> None:
        """Block on the first armed barrier matching ``operation`` and ``when``.

        Returns immediately when no armed barrier matches.  The matched barrier
        is consumed: it is marked waiting while blocked and removed once
        released.

        Args:
            operation: ``"deallocate"`` or ``"allocate"``.
            when: ``"before"`` or ``"after"``.
        """
        for name, barrier in self._barriers.items():
            if barrier.status is not BarrierStatus.ARMED:
                continue
            if barrier.spec.operation != operation or barrier.spec.when != when:
                continue
            barrier.status = BarrierStatus.WAITING
            await barrier.event.wait()
            self._barriers.pop(name, None)
            return

    def release(self, name: str) -> None:
        """Release the named barrier.

        A waiting request resumes and the barrier is discarded; an armed barrier
        nobody reached yet is discarded immediately.

        Args:
            name: Barrier name given at installation.

        Raises:
            KeyError: If no barrier with that name exists.
        """
        barrier = self._barriers[name]
        was_waiting = barrier.status is BarrierStatus.WAITING
        barrier.status = BarrierStatus.RELEASED
        barrier.event.set()
        if not was_waiting:
            # Nobody can be blocked on an armed barrier, so drop it right away.
            # A waiting request removes its own entry when it resumes.
            self._barriers.pop(name, None)

    def release_all(self) -> None:
        """Release every barrier, waking any blocked request."""
        for name in list(self._barriers):
            self.release(name)

    def view(self) -> dict[str, dict[str, str]]:
        """Return ``{name: {"operation", "when", "status"}}`` for every barrier."""
        return {
            name: {
                "operation": barrier.spec.operation,
                "when": barrier.spec.when,
                "status": barrier.status.value,
            }
            for name, barrier in self._barriers.items()
        }
