# SPDX-License-Identifier: Apache-2.0
"""Read-only test EventBus sink loaded through the existing server-module API."""

# Standard
from dataclasses import asdict, is_dataclass
from enum import Enum
import base64
import json
import os
import time

# First Party
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventCallback, EventSubscriber
from lmcache.v1.multiprocess.ext_server_module import ExtServerModuleBuildContext


def encode(value: object) -> object:
    """Preserve primitive descriptors and opaque byte identities in JSON."""
    if isinstance(value, bytes):
        return {"bytes_base64": base64.b64encode(value).decode()}
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    raise TypeError(f"Unserializable MP evidence value: {type(value).__name__}")


class Trace(EventSubscriber):
    """Write exact operation keys, bindings and stream-completion timestamps."""

    def __init__(self, path: str) -> None:
        self.stream = open(path, "a", buffering=1)

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Observe existing events without changing their ordering or contents."""
        selected = (
            EventType.MP_LOOKUP,
            EventType.MP_LOOKUP_PREFETCH_START,
            EventType.MP_LOOKUP_PREFETCH_END,
            EventType.MP_TOKENS,
            EventType.MP_STORE_START,
            EventType.MP_STORE_END,
            EventType.MP_RETRIEVE_START,
            EventType.MP_RETRIEVE_END,
            EventType.MP_REQUEST_START,
            EventType.MP_REQUEST_END,
            EventType.L1_WRITE_FINISHED,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
            EventType.L1_KEYS_EVICTED,
            EventType.L1_READ_FINISHED,
        )
        return {kind: self.record for kind in selected}

    def record(self, event: Event) -> None:
        """Append one actual EventBus record; failed serialization is visible."""
        self.stream.write(
            json.dumps(
                {
                    "event": event.event_type.value,
                    "timestamp": event.timestamp,
                    "received_ns": time.time_ns(),
                    "session_id": event.session_id,
                    "metadata": event.metadata,
                },
                default=encode,
            )
            + "\n"
        )

    def shutdown(self) -> None:
        """Flush the owned trace stream on normal EventBus shutdown."""
        self.stream.close()


def build_server_modules(context: ExtServerModuleBuildContext) -> None:
    """Register a read-only subscriber only in explicitly enabled test pods."""
    if os.environ.get("LMCACHE_KV_EVENTS_E2E") != "1":
        raise RuntimeError("MP trace requires LMCACHE_KV_EVENTS_E2E=1")
    context.server_context.event_bus.register_subscriber(
        Trace("/artifacts/mp-events.jsonl")
    )
