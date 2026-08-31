# SPDX-License-Identifier: Apache-2.0
"""Configuration for the standalone MP Memory Coordinator.

A frozen dataclass with safe defaults (``actuation_enabled=False``) and a
strict YAML loader: unknown keys, wrong types, and out-of-range values are
rejected at startup rather than surfacing as a bad decision later.
"""

# Standard
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
import math

# Third Party
import yaml

_HTTP_SCHEMES = ("http://", "https://")


class LeaderElectionMode(str, Enum):
    """How the process decides it may mutate.

    ``NONE`` always grants leadership (single-process development and the
    local E2E harness). ``KUBERNETES`` holds a ``coordination.k8s.io/v1``
    Lease and loses permission to POST on any renewal conflict or timeout.
    """

    NONE = "none"
    KUBERNETES = "kubernetes"


@dataclass(frozen=True)
class MPMemoryCoordinatorConfig:
    """Settings of the MP Memory Coordinator process.

    Attributes:
        mp_coordinator_url: Base URL of the existing MP Coordinator. Only
            ``GET /instances`` and ``GET /instances/usage`` are called.
        memory_allocation_url: Base URL of the outside Memory Allocation
            service (frozen ``/api/v2/lmcache`` API).
        poll_interval_seconds: Seconds between observation cycles.
        stable_samples: Consecutive eligible samples with the same
            LOW/HIGH classification required before a move is proposed.
        high_ratio: ``used/capacity`` at or above which an instance is HIGH.
        low_ratio: ``used/capacity`` at or below which an instance is LOW.
        minimum_ratio_gap: Required ``receiver_ratio - donor_ratio``.
        projected_donor_max_ratio: Upper bound of the donor's ratio after
            its device is removed (``used / (capacity - slot_capacity)``).
        cooldown_seconds: Seconds after a completed move during which
            neither participant may be selected again.
        adapter_index: Backend-local DAX adapter index. Only ``0`` is
            supported (one adapter per MP instance).
        min_devices_per_instance: Active DAX devices a donor must keep
            after its device is removed.
        allowed_device_path_prefix: Required prefix of every device path
            the coordinator manages or accepts from the outside service.
        drain_timeout_seconds: Seconds a donor drain may take before the
            move enters BLOCKED (there is no undrain API).
        state_directory: Absolute directory holding the journal.
        actuation_enabled: When ``False`` no new move is started; recovery of
            an already durable move still runs.
        http_host: Bind address of the health/readiness HTTP server.
        http_port: Bind port of the health/readiness HTTP server.
        request_timeout_seconds: Per-request HTTP timeout for every remote.
        get_retry_attempts: Bounded attempts for GET requests and status
            polls. POST requests are never retried.
        dax_poll_interval_seconds: Seconds between DAX status polls while a
            drain or capacity convergence is awaited.
        dax_add_max_attempts: Attempts for a receiver ``add`` before the
            attach is considered persistently failed.
        capacity_convergence_timeout_seconds: Seconds to wait for the MP
            Coordinator usage view to reflect the moved capacity before the
            move is logged as delayed (it keeps waiting; nothing is retried).
        leader_election: See :class:`LeaderElectionMode`.
        lease_name: Name of the pre-created Lease.
        lease_namespace: Namespace of the Lease. Empty means the value of
            the ``POD_NAMESPACE`` environment variable at startup.
        lease_duration_seconds: Lease validity window.
        lease_renew_interval_seconds: Seconds between renewals.
        holder_identity: Lease holder identity. Empty means the hostname.
        kubernetes_api_url: API server base URL. Empty means the in-cluster
            ``KUBERNETES_SERVICE_HOST``/``KUBERNETES_SERVICE_PORT`` pair.
        kubernetes_token_path: Service-account bearer token file.
        kubernetes_ca_path: CA bundle for the API server; empty disables
            verification (only sensible with a plain-``http`` test server).
        adoption_file: Optional operator-approved allowlist applied once,
            when the journal carries no initialization marker.
    """

    mp_coordinator_url: str = "http://lmcache-mp-coordinator:8000"
    memory_allocation_url: str = "http://memory-allocation-service:8080"
    poll_interval_seconds: float = 10.0
    stable_samples: int = 3
    high_ratio: float = 0.75
    low_ratio: float = 0.40
    minimum_ratio_gap: float = 0.25
    projected_donor_max_ratio: float = 0.70
    cooldown_seconds: float = 300.0
    adapter_index: int = 0
    min_devices_per_instance: int = 1
    allowed_device_path_prefix: str = "/dev/dax-cxl/"
    drain_timeout_seconds: float = 300.0
    state_directory: str = "/var/lib/lmcache-memory-coordinator"
    actuation_enabled: bool = False
    http_host: str = "0.0.0.0"
    http_port: int = 9400
    request_timeout_seconds: float = 5.0
    get_retry_attempts: int = 3
    dax_poll_interval_seconds: float = 2.0
    dax_add_max_attempts: int = 3
    capacity_convergence_timeout_seconds: float = 120.0
    leader_election: LeaderElectionMode = LeaderElectionMode.NONE
    lease_name: str = "lmcache-mp-memory-coordinator"
    lease_namespace: str = ""
    lease_duration_seconds: float = 15.0
    lease_renew_interval_seconds: float = 5.0
    holder_identity: str = ""
    kubernetes_api_url: str = ""
    kubernetes_token_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    kubernetes_ca_path: str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
    adoption_file: str = ""

    def __post_init__(self) -> None:
        """Validate every field.

        Raises:
            ValueError: If a URL is not explicit ``http(s)://``, a ratio is
                outside ``0 <= low < high <= 1``, an interval or count is
                not positive, the adapter index is not ``0``, the state
                directory is not absolute, or the path prefix is not an
                absolute directory prefix.
        """
        for name in ("mp_coordinator_url", "memory_allocation_url"):
            value = getattr(self, name)
            if not value.startswith(_HTTP_SCHEMES):
                raise ValueError(
                    f"{name} must be an explicit http(s) URL, got {value!r}"
                )
        for name in (
            "poll_interval_seconds",
            "cooldown_seconds",
            "drain_timeout_seconds",
            "request_timeout_seconds",
            "dax_poll_interval_seconds",
            "capacity_convergence_timeout_seconds",
            "lease_duration_seconds",
            "lease_renew_interval_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        for name in (
            "stable_samples",
            "min_devices_per_instance",
            "get_retry_attempts",
            "dax_add_max_attempts",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not (0.0 <= self.low_ratio < self.high_ratio <= 1.0):
            raise ValueError("ratios must satisfy 0 <= low_ratio < high_ratio <= 1")
        if not (0.0 <= self.minimum_ratio_gap <= 1.0):
            raise ValueError("minimum_ratio_gap must be within [0, 1]")
        if not (0.0 < self.projected_donor_max_ratio <= 1.0):
            raise ValueError("projected_donor_max_ratio must be within (0, 1]")
        if self.adapter_index != 0:
            raise ValueError("adapter_index must be 0: one DAX adapter per instance")
        if not Path(self.state_directory).is_absolute():
            raise ValueError("state_directory must be an absolute path")
        if not self.allowed_device_path_prefix.startswith("/"):
            raise ValueError("allowed_device_path_prefix must be an absolute prefix")
        if self.lease_renew_interval_seconds >= self.lease_duration_seconds:
            raise ValueError("lease_renew_interval_seconds must be < lease_duration")
        if not (1 <= self.http_port <= 65535):
            raise ValueError("http_port must be within [1, 65535]")


def load_config(path: Path) -> MPMemoryCoordinatorConfig:
    """Load and validate a YAML configuration file.

    The document must be a mapping whose keys are exactly the fields of
    :class:`MPMemoryCoordinatorConfig`; a missing key keeps the default.
    Values are type-checked strictly (``true`` is not accepted for an
    integer, ``"10"`` is not accepted for a number).

    Args:
        path: The YAML file to read.

    Returns:
        The validated configuration.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document is not a mapping, contains an unknown
            key, a value has the wrong type, or validation fails.
    """
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: configuration must be a mapping")
    return config_from_mapping(raw, source=str(path))


def config_from_mapping(
    raw: dict[object, object], source: str = "<mapping>"
) -> MPMemoryCoordinatorConfig:
    """Build a configuration from an already-parsed mapping.

    Args:
        raw: Key/value pairs as produced by a YAML or JSON parser.
        source: Label used in error messages.

    Returns:
        The validated configuration.

    Raises:
        ValueError: On an unknown key, a wrongly typed value, or a failed
            validation.
    """
    known = {field.name: field for field in fields(MPMemoryCoordinatorConfig)}
    unknown = sorted(str(key) for key in raw if key not in known)
    if unknown:
        raise ValueError(f"{source}: unknown configuration keys: {unknown}")
    values: dict[str, object] = {}
    for key, value in raw.items():
        name = str(key)
        values[name] = _coerce(name, known[name].type, value, source)
    return MPMemoryCoordinatorConfig(**values)  # type: ignore[arg-type]


def _coerce(name: str, declared: object, value: object, source: str) -> object:
    """Check ``value`` against the declared dataclass field type strictly.

    Args:
        name: Field name, for error messages.
        declared: The field's declared type (``str``, ``int``, ``float``,
            ``bool``, or :class:`LeaderElectionMode`).
        value: The parsed YAML value.
        source: Label used in error messages.

    Returns:
        The value converted to the declared type.

    Raises:
        ValueError: If the value has the wrong type.
    """
    if declared is bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"{source}: {name} must be a boolean")
    if declared is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        raise ValueError(f"{source}: {name} must be an integer")
    if declared is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ValueError(f"{source}: {name} must be a number")
    if declared is str:
        if isinstance(value, str):
            return value
        raise ValueError(f"{source}: {name} must be a string")
    if declared is LeaderElectionMode:
        if isinstance(value, str):
            try:
                return LeaderElectionMode(value)
            except ValueError as exc:
                raise ValueError(
                    f"{source}: {name} must be one of "
                    f"{[mode.value for mode in LeaderElectionMode]}"
                ) from exc
        raise ValueError(f"{source}: {name} must be a string")
    raise ValueError(f"{source}: {name} has unsupported type {declared!r}")
