# SPDX-License-Identifier: Apache-2.0
"""Tests for the process layer: probe/status app, the ``lmcache
mp-memory-coordinator`` CLI, startup adoption, and the Lease elector."""

# Standard
from pathlib import Path
import argparse
import asyncio
import json

# Third Party
from fastapi.testclient import TestClient
import httpx
import pytest
import yaml

# First Party
from lmcache.cli.commands import ALL_COMMANDS
from lmcache.cli.commands.mp_memory_coordinator import MPMemoryCoordinatorCommand
from lmcache.v1.mp_memory_coordinator.adoption import AdoptionResult
from lmcache.v1.mp_memory_coordinator.app import (
    Metrics,
    adoption_retry_loop,
    create_app,
)
from lmcache.v1.mp_memory_coordinator.clients import ClientConnectionError
from lmcache.v1.mp_memory_coordinator.config import (
    MPMemoryCoordinatorConfig,
    config_from_mapping,
)
from lmcache.v1.mp_memory_coordinator.controller import RebalanceController
from lmcache.v1.mp_memory_coordinator.leader import (
    KubernetesLeaseElector,
    StaticLeader,
    build_leader,
)
from lmcache.v1.mp_memory_coordinator.models import JournalDocument
from lmcache.v1.mp_memory_coordinator.persistence.rebalance_journal import (
    RebalanceJournal,
)
from tests.v1.mp_memory_coordinator.test_controller import FakeWorld, _inventory


def _controller(
    tmp_path: Path, world: FakeWorld
) -> tuple[RebalanceController, RebalanceJournal]:
    journal = RebalanceJournal(tmp_path / "state")
    journal.save(JournalDocument(initialized=True, inventory=_inventory()))
    config = MPMemoryCoordinatorConfig(
        state_directory=str(tmp_path / "state"), actuation_enabled=False
    )
    controller = RebalanceController(config, journal, world, StaticLeader("t"))
    controller.load()
    return controller, journal


def test_probes_status_journal_and_metrics(tmp_path: Path) -> None:
    world = FakeWorld()
    controller, journal = _controller(tmp_path, world)
    app = create_app(controller, journal, StaticLeader("t"), Metrics())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "healthy"}
        # Not ready before the first cycle reached the coordinator.
        unready = client.get("/readyz")
        assert unready.status_code == 503
        assert unready.json()["status"] == "unready"

        asyncio.run(controller.run_once())
        ready = client.get("/readyz")
        assert ready.status_code == 200 and ready.json() == {
            "status": "ready",
            "reason": "ok",
        }

        status = client.get("/status").json()
        assert status["actuation_enabled"] is False
        assert status["inventory"][0]["device_path"] == _inventory()[0].device_path
        assert status["last_cycle"]["coordinator_reachable"] is True

        document = client.get("/journal").json()
        assert document["initialized"] is True
        assert document["active_move"] is None

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "lmcache_memcoord_moves_proposed_total" in metrics.text
        assert "lmcache_memcoord_leader 1.0" in metrics.text

        assert client.get("/docs").status_code == 404
        assert client.get("/").status_code == 404

        world.coordinator_up = False
        asyncio.run(controller.run_once())
        assert client.get("/readyz").status_code == 503
        assert world.audit == []


def test_healthz_reports_unusable_journal(tmp_path: Path) -> None:
    world = FakeWorld()
    directory = tmp_path / "state"
    journal = RebalanceJournal(directory)
    journal.save(JournalDocument())
    journal.path.write_text("{corrupt")
    config = MPMemoryCoordinatorConfig(state_directory=str(directory))
    controller = RebalanceController(config, journal, world, StaticLeader("t"))
    controller.load()
    app = create_app(controller, journal, StaticLeader("t"), Metrics())
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 503
        assert "journal" in response.json()["reason"]
        assert client.get("/readyz").status_code == 503


# -- CLI --------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmcache")
    subparsers = parser.add_subparsers(dest="command")
    MPMemoryCoordinatorCommand().register(subparsers)
    return parser


def test_command_is_discovered_by_the_cli() -> None:
    assert "mp-memory-coordinator" in {cmd.name() for cmd in ALL_COMMANDS}


def test_cli_help_lists_config_adopt_and_check(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parser().parse_args(["mp-memory-coordinator", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--config" in out and "--adopt" in out and "--check" in out


def test_cli_config_is_required(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parser().parse_args(["mp-memory-coordinator"])
    assert excinfo.value.code == 2


def test_cli_check_accepts_valid_config(tmp_path: Path, capsys) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump({"state_directory": str(tmp_path / "s")}))
    args = _parser().parse_args(
        ["mp-memory-coordinator", "--config", str(config), "--check"]
    )
    args.func(args)
    assert "configuration OK" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content",
    [
        "poll_interval: 5\n",  # unknown key
        "actuation_enabled: sure\n",  # wrong type
        "low_ratio: 0.9\nhigh_ratio: 0.5\n",  # failed validation
        "- not\n- a mapping\n",
    ],
)
def test_cli_config_errors_exit_2(tmp_path: Path, content: str, capsys) -> None:
    config = tmp_path / "cfg.yaml"
    config.write_text(content)
    args = _parser().parse_args(
        ["mp-memory-coordinator", "--config", str(config), "--check"]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 2
    assert "configuration error" in capsys.readouterr().err


def test_cli_missing_config_file_exits_2(tmp_path: Path, capsys) -> None:
    args = _parser().parse_args(
        ["mp-memory-coordinator", "--config", str(tmp_path / "missing.yaml"), "--check"]
    )
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 2


# -- startup adoption --------------------------------------------------------------


def test_adoption_retry_loop_retries_until_initialized_and_never_crashes() -> None:
    """An unreachable MP Coordinator at startup defers adoption; the loop
    keeps trying (the process serves meanwhile) and stops once initialized."""
    attempts = {"n": 0}
    initialized = {"value": False}

    async def attempt() -> AdoptionResult:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ClientConnectionError("MP Coordinator unreachable")
        initialized["value"] = True
        return AdoptionResult()

    async def run() -> None:
        stop = asyncio.Event()
        await asyncio.wait_for(
            adoption_retry_loop(attempt, lambda: initialized["value"], stop, 0.01),
            timeout=5.0,
        )

    asyncio.run(run())
    assert attempts["n"] == 3
    assert initialized["value"] is True


def test_adoption_retry_loop_stops_on_shutdown() -> None:
    async def attempt() -> AdoptionResult:
        raise ClientConnectionError("down")

    async def run() -> int:
        stop = asyncio.Event()
        task = asyncio.create_task(
            adoption_retry_loop(attempt, lambda: False, stop, 0.05)
        )
        await asyncio.sleep(0.12)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
        return 0

    assert asyncio.run(run()) == 0


# -- Kubernetes Lease elector (fake API server) ----------------------------


LEASE_PATH = (
    "/apis/coordination.k8s.io/v1/namespaces/ns/leases/lmcache-mp-memory-coordinator"
)


class FakeLeaseServer:
    """An in-memory Lease with ``resourceVersion`` optimistic concurrency."""

    def __init__(self) -> None:
        self.lease: dict = {
            "metadata": {
                "name": "lmcache-mp-memory-coordinator",
                "resourceVersion": "1",
            },
            "spec": {},
        }
        self.down = False
        self.puts: list[dict] = []
        self.tokens: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.tokens.append(request.headers.get("authorization", ""))
        if self.down:
            return httpx.Response(503, json={"message": "down"})
        if request.url.path != LEASE_PATH:
            return httpx.Response(404, json={"message": "not found"})
        if request.method == "GET":
            return httpx.Response(200, json=self.lease)
        body = json.loads(request.content)
        self.puts.append(body)
        if (
            body["metadata"]["resourceVersion"]
            != self.lease["metadata"]["resourceVersion"]
        ):
            return httpx.Response(409, json={"message": "conflict"})
        version = str(int(self.lease["metadata"]["resourceVersion"]) + 1)
        body["metadata"]["resourceVersion"] = version
        self.lease = body
        return httpx.Response(200, json=self.lease)


class Clock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


def _config(tmp_path: Path, **overrides: object) -> MPMemoryCoordinatorConfig:
    token = tmp_path / "token"
    token.write_text("secret-token\n")
    fields: dict[object, object] = dict(
        state_directory="/tmp/unused",
        leader_election="kubernetes",
        lease_namespace="ns",
        kubernetes_api_url="http://apiserver",
        kubernetes_token_path=str(token),
        kubernetes_ca_path="",
        lease_duration_seconds=15.0,
        lease_renew_interval_seconds=5.0,
    )
    fields.update(overrides)
    return config_from_mapping(fields)


def _elector(server: FakeLeaseServer, config, clock, identity: str):
    return KubernetesLeaseElector(
        config,
        clock=clock,
        identity=identity,
        transport=httpx.MockTransport(server.handler),
    )


def test_acquires_free_lease_with_bearer_token_and_renews(tmp_path: Path) -> None:
    server = FakeLeaseServer()
    clock = Clock()
    elector = _elector(server, _config(tmp_path), clock, "pod-a")
    assert not elector.is_leader()
    assert asyncio.run(elector.ensure_leader()) is True
    assert elector.is_leader()
    assert server.tokens[0] == "Bearer secret-token"
    spec = server.lease["spec"]
    assert spec["holderIdentity"] == "pod-a"
    assert spec["leaseDurationSeconds"] == 15
    assert spec["leaseTransitions"] == 1
    assert spec["renewTime"].endswith("Z")
    # Leadership is bounded by the lease duration when not renewed.
    clock.now += 14.0
    assert elector.is_leader()
    clock.now += 2.0
    assert not elector.is_leader()
    # Renewal keeps the same holder and does not bump transitions.
    assert asyncio.run(elector.ensure_leader()) is True
    assert server.lease["spec"]["leaseTransitions"] == 1


def test_does_not_steal_a_live_lease_but_takes_an_expired_one(tmp_path: Path) -> None:
    server = FakeLeaseServer()
    clock = Clock()
    holder = _elector(server, _config(tmp_path), clock, "pod-a")
    assert asyncio.run(holder.ensure_leader())

    other = _elector(server, _config(tmp_path), clock, "pod-b")
    assert asyncio.run(other.ensure_leader()) is False
    assert not other.is_leader()
    assert server.lease["spec"]["holderIdentity"] == "pod-a"

    clock.now += 16.0  # pod-a's lease expired without renewal
    assert asyncio.run(other.ensure_leader()) is True
    assert server.lease["spec"]["holderIdentity"] == "pod-b"
    assert server.lease["spec"]["leaseTransitions"] == 2
    # The old holder loses permission on its next renewal.
    assert asyncio.run(holder.ensure_leader()) is False
    assert not holder.is_leader()


def test_conflict_or_outage_is_immediate_loss(tmp_path: Path) -> None:
    server = FakeLeaseServer()
    clock = Clock()
    elector = _elector(server, _config(tmp_path), clock, "pod-a")
    assert asyncio.run(elector.ensure_leader())

    server.down = True
    assert asyncio.run(elector.ensure_leader()) is False
    assert not elector.is_leader()
    server.down = False

    assert asyncio.run(elector.ensure_leader())
    # Someone else wrote in between: the resourceVersion moves on.
    server.lease["metadata"]["resourceVersion"] = "999"
    server.lease["spec"]["holderIdentity"] = "pod-a"

    def conflicting(request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            return httpx.Response(409, json={"message": "conflict"})
        return server.handler(request)

    conflicted = KubernetesLeaseElector(
        _config(tmp_path),
        clock=clock,
        identity="pod-a",
        transport=httpx.MockTransport(conflicting),
    )
    assert asyncio.run(conflicted.ensure_leader()) is False
    assert not conflicted.is_leader()


def test_release_clears_holder_only_when_held(tmp_path: Path) -> None:
    server = FakeLeaseServer()
    clock = Clock()
    elector = _elector(server, _config(tmp_path), clock, "pod-a")
    assert asyncio.run(elector.ensure_leader())
    asyncio.run(elector.release())
    assert server.lease["spec"]["holderIdentity"] == ""
    assert not elector.is_leader()

    server.lease["spec"]["holderIdentity"] = "pod-b"
    other = _elector(server, _config(tmp_path), clock, "pod-a")
    asyncio.run(other.release())
    assert server.lease["spec"]["holderIdentity"] == "pod-b"


def test_run_loop_renews_until_stopped(tmp_path: Path) -> None:
    server = FakeLeaseServer()
    clock = Clock()
    elector = _elector(
        server, _config(tmp_path, lease_renew_interval_seconds=0.05), clock, "pod-a"
    )

    async def run():
        stop = asyncio.Event()
        task = asyncio.create_task(elector.run(stop))
        await asyncio.sleep(0.2)
        stop.set()
        await task

    asyncio.run(run())
    assert len(server.puts) >= 2
    assert elector.is_leader()


def test_build_leader_selects_mode_and_validates_namespace(tmp_path: Path) -> None:
    assert isinstance(
        build_leader(MPMemoryCoordinatorConfig(holder_identity="x"), Clock()),
        StaticLeader,
    )
    with pytest.raises(ValueError, match="lease_namespace"):
        build_leader(_config(tmp_path, lease_namespace=""), Clock())
    with pytest.raises(ValueError, match="kubernetes_api_url"):
        build_leader(_config(tmp_path, kubernetes_api_url=""), Clock())


def test_static_leader_is_always_leader() -> None:
    leader = StaticLeader("dev")
    assert leader.identity == "dev"
    assert leader.is_leader()
    assert asyncio.run(leader.ensure_leader())
