# SPDX-License-Identifier: Apache-2.0
"""E2E harness: real MP Memory Coordinator against detached test services.

Two modes:

* **local** (default): the scenario server, the strict mock Memory
  Allocation service, and the real ``lmcache mp-memory-coordinator`` run as
  separate subprocesses on loopback with free ports and a temporary state
  directory. Every test resets both services and starts a fresh coordinator.
* **cluster** (``--kube-context CTX``): the services and the coordinator run
  as Pods deployed from ``manifests/overlays/kind``; the harness reaches them
  through ``kubectl port-forward``. A requested cluster that is unreachable
  is a test *failure*, not a skip. Tests that need to kill or restart the
  coordinator process are marked ``local_only`` and are skipped in this mode.

The coordinator receives ordinary HTTP URLs only; it cannot tell that its
dependencies are fakes. Test controls never touch production code.
"""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time

# Third Party
import pytest
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "two_server_local_dax.yaml"
PYTHON = sys.executable

DONOR_IP = "192.0.2.40"
RECEIVER_IP = "192.0.2.41"
DONOR_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.0"
DONOR_RUNTIME = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.1"
DONOR_SPARE = "/dev/dax-cxl/lmcache-e2e--mp-196/dax0.2"
RECEIVER_BOOT = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.0"
RECEIVER_RUNTIME = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.1"
RECEIVER_SPARE = "/dev/dax-cxl/lmcache-e2e--mp-197/dax0.2"
GIB = 1 << 30

E2E_NAMESPACE = "lmcache-memcoord-e2e"
CLUSTER_ADOPTION_FILE = "/etc/lmcache/adoption.yaml"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kube-context",
        action="store",
        default="",
        help="Run against the Pods deployed from manifests/overlays/kind in this "
        "kube context instead of local subprocesses.",
    )
    parser.addoption(
        "--e2e-namespace",
        action="store",
        default=E2E_NAMESPACE,
        help="Namespace of the deployed E2E topology (cluster mode).",
    )
    # Real two-worker Device-DAX gate (PLAN.md section 9); see test_hardware.py.
    for name, default, help_text in (
        ("--donor-node", "", "Kubernetes node name pinned to the donor MP Pod"),
        ("--receiver-node", "", "Kubernetes node name pinned to the receiver MP Pod"),
        (
            "--dax-inventory-file",
            "",
            "Explicit allowlist (two_server_local_dax.yaml schema) with real worker "
            "IPs, paths, sizes, roles, and initial states",
        ),
        ("--move-size-gib", "0", "Size of one complete donor/receiver candidate"),
        ("--outside-api-url", "", "Public URL of the strict mock allocator"),
        ("--dax-state-timeout", "120s", "Deadline for DAX state transitions"),
    ):
        parser.addoption(name, action="store", default=default, help=help_text)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    cluster = bool(config.getoption("--kube-context"))
    markexpr = str(getattr(config.option, "markexpr", "") or "")
    hardware_requested = "hardware" in markexpr
    conformance_requested = "outside_api" in markexpr
    for item in items:
        if "hardware" in item.keywords and not hardware_requested:
            # The real-DAX gate runs only when explicitly selected with
            # ``-m hardware``; then a missing prerequisite is a failure.
            item.add_marker(pytest.mark.skip(reason="hardware gate not requested"))
            continue
        if "outside_api" in item.keywords and not conformance_requested:
            # Likewise the outside-API conformance suite targets a URL the
            # caller names with ``-m outside_api --outside-api-url``.
            item.add_marker(pytest.mark.skip(reason="conformance not requested"))
            continue
        if cluster and "local_only" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="needs process control (local mode)")
            )
        if (
            not cluster
            and "kind" in item.keywords
            and "local_only" not in item.keywords
        ):
            # ``kind``-marked tests also run locally; the marker only selects
            # them for a cluster run via ``-m kind``.
            continue


def free_port() -> int:
    """Return a free loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_until(
    predicate: Callable[[], bool], timeout: float, interval: float = 0.1, what: str = ""
) -> None:
    """Deadline-poll ``predicate``; never a fixed sleep.

    Raises:
        AssertionError: If the deadline passes.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what or predicate}")


def wait_http(url: str, timeout: float, *, any_status: bool = False) -> None:
    """Wait until ``url`` answers 2xx (or any status when ``any_status``)."""

    def _up() -> bool:
        try:
            status = requests.get(url, timeout=1.0).status_code
        except requests.RequestException:
            return False
        return any_status or status < 300

    wait_until(_up, timeout, what=url)


class AdminClient:
    """Thin client of a test service's ``/__test`` admin port."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def reset(self) -> dict:
        response = requests.post(self._url("/__test/reset"), timeout=5)
        response.raise_for_status()
        return response.json()

    def health(self) -> dict:
        response = requests.get(self._url("/__test/health"), timeout=5)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict:
        response = requests.get(self._url("/__test/state"), timeout=5)
        response.raise_for_status()
        return response.json()

    def audit(self, after_seq: int = 0) -> list[dict]:
        response = requests.get(
            self._url("/__test/audit"), params={"after_seq": after_seq}, timeout=5
        )
        response.raise_for_status()
        return list(response.json()["records"])

    def faults(self, spec: dict) -> dict:
        response = requests.post(self._url("/__test/faults"), json=spec, timeout=5)
        assert response.status_code < 300, response.text
        return response.json()

    def clear_faults(self) -> None:
        requests.delete(self._url("/__test/faults"), timeout=5).raise_for_status()

    def barrier(self, spec: dict) -> dict:
        response = requests.post(self._url("/__test/barriers"), json=spec, timeout=5)
        assert response.status_code < 300, response.text
        return response.json()

    def release(self, name: str) -> None:
        response = requests.post(
            self._url(f"/__test/barriers/{name}/release"), timeout=5
        )
        assert response.status_code < 300, response.text

    def post(self, path: str, body: dict) -> dict:
        response = requests.post(self._url(path), json=body, timeout=5)
        assert response.status_code < 300, response.text
        return response.json() if response.text else {}


class MemcoordClient:
    """Client of the real coordinator's probe/status endpoints."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> requests.Response:
        return requests.get(f"{self.base_url}{path}", timeout=5)

    def healthz(self) -> requests.Response:
        return self._get("/healthz")

    def readyz(self) -> requests.Response:
        return self._get("/readyz")

    def status(self) -> dict:
        response = self._get("/status")
        response.raise_for_status()
        return response.json()

    def journal(self) -> dict:
        response = self._get("/journal")
        response.raise_for_status()
        return response.json()

    def active_move(self) -> dict | None:
        return self.journal().get("active_move")

    def last_move(self) -> dict | None:
        history = self.journal().get("history", [])
        return history[-1] if history else None

    def cycles(self) -> float:
        """Wall-clock stamp of the last cycle (changes every cycle)."""
        return float(self.status()["last_cycle"]["at"])

    def wait_cycles(self, count: int, timeout: float = 60.0) -> None:
        """Wait for ``count`` further cycles to complete."""
        seen: set[float] = set()

        def _tick() -> bool:
            try:
                seen.add(self.cycles())
            except requests.RequestException:
                return False
            return len(seen) > count

        wait_until(_tick, timeout, what=f"{count} cycles")

    def wait_terminal(self, timeout: float = 120.0) -> dict:
        """Wait until the active move is COMPLETE or BLOCKED; return it."""

        def _done() -> bool:
            try:
                move = self.active_move()
            except requests.RequestException:
                return False
            if move is None:
                return self.last_move() is not None
            return move["state"] == "BLOCKED"

        wait_until(_done, timeout, what="terminal move")
        move = self.active_move()
        if move is not None:
            return move
        last = self.last_move()
        assert last is not None
        return last

    def wait_state(self, states: set[str], timeout: float = 60.0) -> dict:
        """Wait until the active move is in one of ``states``."""

        def _in() -> bool:
            try:
                move = self.active_move()
            except requests.RequestException:
                return False
            return move is not None and move["state"] in states

        wait_until(_in, timeout, what=f"move in {states}")
        move = self.active_move()
        assert move is not None
        return move


@dataclass
class Endpoints:
    """URLs the harness and the coordinator use."""

    coordinator_url: str
    donor_url: str
    receiver_url: str
    scenario_admin_url: str
    allocator_public_url: str
    allocator_admin_url: str


@dataclass
class ServiceProcess:
    """A subprocess with its log file."""

    process: subprocess.Popen
    log_path: Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def _spawn(
    argv: list[str], log_path: Path, env: dict[str, str] | None = None
) -> ServiceProcess:
    log = open(log_path, "ab")  # noqa: SIM115 -- kept open for the process lifetime
    merged = dict(os.environ)
    merged.update(env or {})
    merged.setdefault("LMCACHE_TRACK_USAGE", "false")
    process = subprocess.Popen(
        argv, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT, env=merged
    )
    return ServiceProcess(process=process, log_path=log_path)


class Memcoord:
    """Process-control facade for the coordinator under test (local mode)."""

    def __init__(self, endpoints: Endpoints, work_dir: Path) -> None:
        self._endpoints = endpoints
        self._work_dir = work_dir
        self.state_dir = work_dir / "state"
        self.port = free_port()
        self.client = MemcoordClient(f"http://127.0.0.1:{self.port}")
        self._process: ServiceProcess | None = None
        self.config_overrides: dict[str, object] = {}
        self._starts = 0

    def config(self, overrides: dict[str, object]) -> dict[str, object]:
        base: dict[str, object] = {
            "mp_coordinator_url": self._endpoints.coordinator_url,
            "memory_allocation_url": self._endpoints.allocator_public_url,
            "poll_interval_seconds": 0.5,
            "stable_samples": 3,
            "cooldown_seconds": 10.0,
            "drain_timeout_seconds": 300.0,
            "dax_poll_interval_seconds": 0.3,
            "request_timeout_seconds": 3.0,
            "get_retry_attempts": 2,
            "dax_add_max_attempts": 3,
            "state_directory": str(self.state_dir),
            "actuation_enabled": True,
            "http_host": "127.0.0.1",
            "http_port": self.port,
            "leader_election": "none",
        }
        base.update(self.config_overrides)
        base.update(overrides)
        return base

    def start(self, **overrides: object) -> "Memcoord":
        """Start (or restart) the real coordinator with the given overrides."""
        self.stop()
        self._starts += 1
        self._work_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._work_dir / f"memcoord-{self._starts}.yaml"
        config_path.write_text(yaml.safe_dump(self.config(overrides)))
        self._process = _spawn(
            [
                PYTHON,
                "-m",
                "lmcache.cli.main",
                "mp-memory-coordinator",
                "--config",
                str(config_path),
            ],
            self._work_dir / f"memcoord-{self._starts}.log",
        )
        # A corrupt journal answers 503 on purpose; the process is still up.
        wait_http(f"{self.client.base_url}/healthz", 60.0, any_status=True)
        return self

    def adopt(
        self, allowlist: dict, **overrides: object
    ) -> subprocess.CompletedProcess:
        """Run the explicit one-time adoption command."""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        config_path = self._work_dir / "memcoord-adopt.yaml"
        config_path.write_text(yaml.safe_dump(self.config(overrides)))
        allow_path = self._work_dir / "adopt.yaml"
        allow_path.write_text(yaml.safe_dump(allowlist))
        env = dict(os.environ)
        env.setdefault("LMCACHE_TRACK_USAGE", "false")
        return subprocess.run(
            [
                PYTHON,
                "-m",
                "lmcache.cli.main",
                "mp-memory-coordinator",
                "--config",
                str(config_path),
                "--adopt",
                str(allow_path),
            ],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def seed_inventory(self) -> None:
        """Adopt the donor's runtime path through the real ``--adopt`` command.

        Discovery would find the donor's device on its own; scenarios that
        want a deterministic, explicitly adopted inventory (or that exercise
        the ``--adopt`` command itself) still seed the exact
        ``(worker_ip, path, size)`` tuple the fixture declares.
        """
        result = self.adopt(
            {
                "allocations": [
                    {
                        "worker_ip": DONOR_IP,
                        "device_path": DONOR_RUNTIME,
                        "allocation_size_gib": 64,
                        "device_map_size_bytes": 64 * GIB,
                    }
                ]
            }
        )
        assert result.returncode == 0, result.stderr
        assert f"adopted {DONOR_RUNTIME}" in result.stdout, result.stdout

    def kill(self) -> None:
        """SIGKILL the coordinator (simulated crash)."""
        if self._process is not None and self._process.process.poll() is None:
            self._process.process.kill()
            self._process.process.wait(timeout=10)

    def stop(self) -> None:
        if self._process is not None:
            self._process.stop()
            self._process = None

    def journal_path(self) -> Path:
        return self.state_dir / "journal.json"

    def logs(self) -> str:
        if self._process is None:
            return ""
        return self._process.log_path.read_text(errors="replace")


_WIPE_POD = """\
apiVersion: v1
kind: Pod
metadata:
  name: memcoord-journal-wipe
spec:
  restartPolicy: Never
  containers:
    - name: wipe
      image: lmcache-memory-allocation-mock:e2e
      imagePullPolicy: IfNotPresent
      command:
        - sh
        - -c
        - rm -f /var/lib/lmcache-memory-coordinator/journal.json
          /var/lib/lmcache-memory-coordinator/journal.json.tmp
      volumeMounts:
        - name: journal
          mountPath: /var/lib/lmcache-memory-coordinator
  volumes:
    - name: journal
      persistentVolumeClaim:
        claimName: lmcache-mp-memory-coordinator-journal
"""


class ClusterMemcoord:
    """Cluster-mode facade over the deployed coordinator.

    ``start`` rewrites the ConfigMap, scales the Deployment to zero, wipes the
    journal on the PVC with a one-off Pod (so every test starts from an
    uninitialized journal, exactly like local mode), scales back up, and
    re-establishes a port-forward. Adoption happens through the ConfigMap's
    ``adoption_file`` when ``seed_inventory`` was called.
    """

    DEPLOYMENT = "lmcache-mp-memory-coordinator"
    SELECTOR = "app.kubernetes.io/name=lmcache-mp-memory-coordinator"

    def __init__(self, context: str, namespace: str, work_dir: Path) -> None:
        self._context = context
        self._namespace = namespace
        self._work_dir = work_dir
        self.state_dir = work_dir / "unused"
        self.config_overrides: dict[str, object] = {}
        self._seeded = False
        self._forward: PortForward | None = None
        self._pod = ""
        self.client = MemcoordClient("http://127.0.0.1:1")
        # The pristine overlay configuration, never the live ConfigMap (an
        # earlier test rewrites it): each start applies exactly the overlay
        # plus this test's overrides.
        overlay = (
            Path(__file__).resolve().parent
            / "manifests"
            / "overlays"
            / "kind"
            / "configmap-patch.yaml"
        )
        data = yaml.safe_load(overlay.read_text())["data"]
        self._base_config = dict(yaml.safe_load(data["mp-memory-coordinator.yaml"]))

    def _kubectl(self, *args: str, timeout: float = 180) -> str:
        result = subprocess.run(
            ["kubectl", "--context", self._context, "-n", self._namespace, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        assert result.returncode == 0, f"kubectl {' '.join(args)}: {result.stderr}"
        return result.stdout

    def seed_inventory(self) -> None:
        """Adopt the donor runtime path at the next start (ConfigMap allowlist)."""
        self._seeded = True

    def config(self, overrides: dict[str, object]) -> dict[str, object]:
        config = dict(self._base_config)
        # The live ConfigMap may have been rewritten by an earlier test;
        # adoption is decided per test, never inherited.
        if self._seeded:
            config["adoption_file"] = CLUSTER_ADOPTION_FILE
        else:
            config.pop("adoption_file", None)
        config.update(self.config_overrides)
        config.update(overrides)
        return config

    def start(self, **overrides: object) -> "ClusterMemcoord":
        self.stop()
        patch = {
            "data": {
                "mp-memory-coordinator.yaml": yaml.safe_dump(self.config(overrides))
            }
        }
        self._kubectl(
            "patch",
            "configmap",
            self.DEPLOYMENT,
            "--type",
            "merge",
            "-p",
            json.dumps(patch),
        )
        self._kubectl("scale", f"deploy/{self.DEPLOYMENT}", "--replicas=0")
        self._kubectl(
            "wait", "--for=delete", "pod", "-l", self.SELECTOR, "--timeout=180s"
        )
        self._wipe_journal()
        self._kubectl("scale", f"deploy/{self.DEPLOYMENT}", "--replicas=1")
        self._reforward()
        return self

    def _pod_name(self) -> str:
        """Wait for the single coordinator Pod to be Running; return its name.

        Readiness is deliberately not awaited: a coordinator whose MP
        Coordinator is unreachable (or whose journal is corrupt) is *meant*
        to stay unready, and tests assert exactly that.
        """

        def _running() -> bool:
            out = subprocess.run(
                [
                    "kubectl",
                    "--context",
                    self._context,
                    "-n",
                    self._namespace,
                    "get",
                    "pod",
                    "-l",
                    self.SELECTOR,
                    "-o",
                    "jsonpath={range .items[*]}{.metadata.name}={.status.phase}"
                    '={.metadata.deletionTimestamp}{"\\n"}{end}',
                ],
                capture_output=True,
                text=True,
                timeout=60,
            ).stdout
            running = [
                line.split("=")[0]
                for line in out.splitlines()
                if line.endswith("=Running=")
            ]
            self._pod = running[0] if len(running) == 1 else ""
            return bool(self._pod)

        self._pod = ""
        wait_until(_running, 300.0, interval=1.0, what="coordinator pod Running")
        return self._pod

    def _wipe_journal(self) -> None:
        manifest = self._work_dir / "wipe-pod.yaml"
        self._work_dir.mkdir(parents=True, exist_ok=True)
        manifest.write_text(_WIPE_POD)
        subprocess.run(
            [
                "kubectl",
                "--context",
                self._context,
                "-n",
                self._namespace,
                "delete",
                "pod",
                "memcoord-journal-wipe",
                "--ignore-not-found",
                "--wait=true",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self._kubectl("apply", "-f", str(manifest))
        self._kubectl(
            "wait",
            "--for=jsonpath={.status.phase}=Succeeded",
            "pod/memcoord-journal-wipe",
            "--timeout=180s",
        )
        self._kubectl("delete", "pod", "memcoord-journal-wipe", "--wait=true")

    def _reforward(self) -> None:
        """Port-forward to the Running Pod, retrying until it answers.

        A forward opened before the container listens is dropped by kubectl
        on the first refused connection, so the forward is re-created until
        ``/healthz`` answers with any status (a corrupt journal is 503).
        """
        if self._forward is not None:
            self._forward.stop()
            self._forward = None
        pod = self._pod_name()
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            forward = PortForward(self._context, self._namespace, f"pod/{pod}", 9400)
            client = MemcoordClient(forward.url)
            probe_deadline = time.monotonic() + 10.0
            while time.monotonic() < probe_deadline and forward.process.poll() is None:
                try:
                    client.healthz()
                    self._forward = forward
                    self.client = client
                    return
                except requests.RequestException:
                    time.sleep(0.5)
            forward.stop()
            time.sleep(1.0)
        raise AssertionError(f"coordinator pod {pod} never answered /healthz")

    def adopt(
        self, allowlist: dict, **overrides: object
    ) -> subprocess.CompletedProcess:
        raise NotImplementedError("explicit adoption command is local-only")

    def kill(self) -> None:
        self._kubectl(
            "delete", "pod", "-l", self.SELECTOR, "--grace-period=0", "--force"
        )
        self._reforward()

    def stop(self) -> None:
        if self._forward is not None:
            self._forward.stop()
            self._forward = None

    def journal_path(self) -> Path:
        raise NotImplementedError("journal file access is local-only")

    def logs(self) -> str:
        try:
            return self._kubectl("logs", f"deploy/{self.DEPLOYMENT}", "--tail=2000")
        except AssertionError as exc:
            return str(exc)


@dataclass
class Harness:
    """Everything a test needs."""

    endpoints: Endpoints
    scenario: AdminClient
    allocator: AdminClient
    memcoord: "Memcoord | ClusterMemcoord"
    mode: str
    artifacts: Path
    cleanup: list[Callable[[], None]] = field(default_factory=list)

    def reset(self) -> None:
        """Reset both test services and verify their health."""
        self.scenario.reset()
        self.allocator.reset()
        assert self.scenario.health()["status"] == "ok"
        assert self.allocator.health()["status"] == "ok"

    def outside_status(self) -> dict:
        response = requests.get(
            f"{self.endpoints.allocator_public_url}/api/v2/apps/lmcache", timeout=5
        )
        response.raise_for_status()
        return response.json()

    def scenario_posts(self) -> list[dict]:
        """MP mutation requests observed by the scenario server, in order."""
        return [
            r
            for r in self.scenario.audit()
            if r["kind"] == "request" and r["method"] == "POST"
        ]

    def allocator_posts(self) -> list[dict]:
        """Outside POST requests observed by the allocator, in order."""
        return [
            r
            for r in self.allocator.audit()
            if r["kind"] == "request" and r["operation"] in ("deallocate", "allocate")
        ]

    def mutation_sequence(self) -> list[str]:
        """Combined MP + outside mutation kinds, each service in its own order.

        The scenario server and the allocator keep independent audits; the
        cross-service order is established from the journal's effect ledger
        by the tests, not from wall clocks. This helper returns the MP
        sequence followed by the outside sequence for counting.
        """
        kinds = []
        for record in self.scenario_posts():
            body = record.get("body") or {}
            kinds.append(f"mp:{record['service']}:{body.get('mode', 'add')}")
        for record in self.allocator_posts():
            kinds.append(f"outside:{record['operation']}")
        return kinds

    def collect_artifacts(self, name: str) -> None:
        """Dump audits, state, journal, and logs for a failed test."""
        target = self.artifacts / name
        target.mkdir(parents=True, exist_ok=True)
        try:
            (target / "scenario-audit.json").write_text(
                json.dumps(self.scenario.audit(), indent=2)
            )
            (target / "scenario-state.json").write_text(
                json.dumps(self.scenario.state(), indent=2)
            )
            (target / "allocator-audit.json").write_text(
                json.dumps(self.allocator.audit(), indent=2)
            )
            (target / "allocator-state.json").write_text(
                json.dumps(self.allocator.state(), indent=2)
            )
            (target / "outside-status.json").write_text(
                json.dumps(self.outside_status(), indent=2)
            )
        except Exception as exc:  # noqa: BLE001 -- best effort
            (target / "collect-error.txt").write_text(str(exc))
        if (
            isinstance(self.memcoord, Memcoord)
            and self.memcoord.journal_path().exists()
        ):
            shutil.copy(self.memcoord.journal_path(), target / "journal.json")
        try:
            (target / "journal.json").write_text(
                json.dumps(self.memcoord.client.journal())
            )
        except Exception:  # noqa: BLE001 -- best effort
            pass
        (target / "memcoord.log").write_text(self.memcoord.logs())


@pytest.fixture(scope="session")
def e2e_mode(request: pytest.FixtureRequest) -> str:
    return "cluster" if request.config.getoption("--kube-context") else "local"


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    configured = os.environ.get("MEMCOORD_E2E_ARTIFACTS", "")
    if configured:
        path = Path(configured)
        path.mkdir(parents=True, exist_ok=True)
        return path
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="session")
def local_services(
    e2e_mode: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Endpoints]:
    """Spawn the scenario server and the mock allocator (local mode)."""
    if e2e_mode != "local":
        yield Endpoints("", "", "", "", "", "")
        return
    work = tmp_path_factory.mktemp("services")
    coordinator_port, donor_port, receiver_port, admin_port = (
        free_port(),
        free_port(),
        free_port(),
        free_port(),
    )
    donor_alt, receiver_alt = free_port(), free_port()
    public_port, alloc_admin_port = free_port(), free_port()
    scenario = _spawn(
        [
            PYTHON,
            "-m",
            "tests.e2e.mp_memory_coordinator.scenario_server",
            "--fixture",
            str(FIXTURE),
            "--host",
            "127.0.0.1",
            "--advertise-ip",
            "127.0.0.1",
            "--coordinator-port",
            str(coordinator_port),
            "--donor-port",
            str(donor_port),
            "--receiver-port",
            str(receiver_port),
            "--donor-alt-port",
            str(donor_alt),
            "--receiver-alt-port",
            str(receiver_alt),
            "--admin-port",
            str(admin_port),
        ],
        work / "scenario.log",
    )
    allocator = _spawn(
        [
            PYTHON,
            "-m",
            "tests.e2e.mp_memory_coordinator.mock_memory_allocation_service",
            "--fixture",
            str(FIXTURE),
            "--public-host",
            "127.0.0.1",
            "--public-port",
            str(public_port),
            "--admin-host",
            "127.0.0.1",
            "--admin-port",
            str(alloc_admin_port),
        ],
        work / "allocator.log",
    )
    try:
        wait_http(f"http://127.0.0.1:{admin_port}/__test/health", 60.0)
        wait_http(f"http://127.0.0.1:{alloc_admin_port}/__test/health", 60.0)
        yield Endpoints(
            coordinator_url=f"http://127.0.0.1:{coordinator_port}",
            donor_url=f"http://127.0.0.1:{donor_port}",
            receiver_url=f"http://127.0.0.1:{receiver_port}",
            scenario_admin_url=f"http://127.0.0.1:{admin_port}",
            allocator_public_url=f"http://127.0.0.1:{public_port}",
            allocator_admin_url=f"http://127.0.0.1:{alloc_admin_port}",
        )
    finally:
        scenario.stop()
        allocator.stop()


class PortForward:
    """``kubectl port-forward`` to a Service; parses the local port."""

    def __init__(self, context: str, namespace: str, service: str, port: int) -> None:
        """Args:
        context: kube context.
        namespace: namespace of the resource.
        service: a Service name, or an explicit ``pod/<name>`` resource.
        port: remote port.
        """
        resource = service if "/" in service else f"svc/{service}"
        self.process = subprocess.Popen(
            [
                "kubectl",
                "--context",
                context,
                "-n",
                namespace,
                "port-forward",
                resource,
                f":{port}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert self.process.stdout is not None
        line = ""
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if "Forwarding from 127.0.0.1:" in line:
                break
        if "Forwarding from 127.0.0.1:" not in line:
            self.process.kill()
            raise AssertionError(f"port-forward to {service}:{port} failed: {line}")
        self.local_port = int(line.split("127.0.0.1:")[1].split(" ")[0])
        self.url = f"http://127.0.0.1:{self.local_port}"

    def stop(self) -> None:
        self.process.kill()


@pytest.fixture(scope="session")
def cluster_services(
    request: pytest.FixtureRequest, e2e_mode: str
) -> Iterator[Endpoints]:
    """Port-forward to the deployed topology (cluster mode)."""
    if e2e_mode != "cluster":
        yield Endpoints("", "", "", "", "", "")
        return
    context = str(request.config.getoption("--kube-context"))
    namespace = str(request.config.getoption("--e2e-namespace"))
    probe = subprocess.run(
        [
            "kubectl",
            "--context",
            context,
            "-n",
            namespace,
            "get",
            "deploy",
            "-o",
            "name",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, f"cluster prerequisite failed: {probe.stderr}"
    for name in (
        "deployment.apps/scenario-server",
        "deployment.apps/mock-memory-allocation-service",
        "deployment.apps/lmcache-mp-memory-coordinator",
    ):
        assert name in probe.stdout, f"missing {name} in {namespace}: {probe.stdout}"
    forwards = [
        PortForward(context, namespace, "scenario-server", 9300),
        PortForward(context, namespace, "scenario-server", 8081),
        PortForward(context, namespace, "scenario-server", 8082),
        PortForward(context, namespace, "scenario-server-admin", 9091),
        PortForward(context, namespace, "mock-memory-allocation-service", 8080),
        PortForward(context, namespace, "mock-memory-allocation-service-admin", 9090),
    ]
    try:
        yield Endpoints(
            coordinator_url=forwards[0].url,
            donor_url=forwards[1].url,
            receiver_url=forwards[2].url,
            scenario_admin_url=forwards[3].url,
            allocator_public_url=forwards[4].url,
            allocator_admin_url=forwards[5].url,
        )
    finally:
        for forward in forwards:
            forward.stop()


@pytest.fixture
def harness(
    request: pytest.FixtureRequest,
    e2e_mode: str,
    local_services: Endpoints,
    cluster_services: Endpoints,
    tmp_path: Path,
    artifacts_dir: Path,
) -> Iterator[Harness]:
    """Reset both services and hand the test a fresh coordinator handle."""
    endpoints = local_services if e2e_mode == "local" else cluster_services
    memcoord: Memcoord | ClusterMemcoord
    if e2e_mode == "cluster":
        memcoord = ClusterMemcoord(
            str(request.config.getoption("--kube-context")),
            str(request.config.getoption("--e2e-namespace")),
            tmp_path,
        )
    else:
        memcoord = Memcoord(endpoints, tmp_path)
    harness = Harness(
        endpoints=endpoints,
        scenario=AdminClient(endpoints.scenario_admin_url),
        allocator=AdminClient(endpoints.allocator_admin_url),
        memcoord=memcoord,
        mode=e2e_mode,
        artifacts=artifacts_dir,
    )
    harness.reset()
    try:
        yield harness
    finally:
        failed = getattr(request.node, "rep_call", None)
        if failed is not None and failed.failed:
            harness.collect_artifacts(request.node.name)
        memcoord.stop()
        for hook in harness.cleanup:
            hook()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    setattr(item, f"rep_{report.when}", report)
