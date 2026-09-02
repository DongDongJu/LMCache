# SPDX-License-Identifier: Apache-2.0
"""Ordinary Lease handoff between two coordinator processes (local mode).

A tiny in-process fake of the Kubernetes Lease API serves one
``coordination.k8s.io/v1`` Lease with ``resourceVersion`` optimistic
concurrency. Two real coordinator processes share the same state directory
(the test-only "shared storage" of the two-replica overlay). Only the
leader mutates; when it is killed, the follower takes the Lease after it
expires and continues the same move from the durable journal. This proves
ordinary handoff only -- it makes no claim about partitioned stale-leader
fencing, which production avoids with ``replicas: 1`` and a
``ReadWriteOncePod`` PVC.
"""

# Standard
from collections.abc import Iterator
from pathlib import Path
import threading

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pytest
import uvicorn

# Local
from .conftest import Harness, Memcoord, free_port, wait_http, wait_until

pytestmark = pytest.mark.local_only

LEASE_PATH = "/apis/coordination.k8s.io/v1/namespaces/{namespace}/leases/{name}"


class FakeLeaseAPI:
    """One Lease object with optimistic concurrency."""

    def __init__(self) -> None:
        self.lease: dict = {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {
                "name": "lmcache-mp-memory-coordinator",
                "resourceVersion": "1",
            },
            "spec": {},
        }
        self.lock = threading.Lock()
        self.holders: list[str] = []

    def app(self) -> FastAPI:
        app = FastAPI()

        @app.get(LEASE_PATH)
        async def get_lease(namespace: str, name: str) -> JSONResponse:
            with self.lock:
                return JSONResponse(content=self.lease)

        @app.put(LEASE_PATH)
        async def put_lease(
            namespace: str, name: str, request: Request
        ) -> JSONResponse:
            body = await request.json()
            with self.lock:
                if (
                    body["metadata"]["resourceVersion"]
                    != self.lease["metadata"]["resourceVersion"]
                ):
                    return JSONResponse(
                        status_code=409, content={"message": "conflict"}
                    )
                body["metadata"]["resourceVersion"] = str(
                    int(self.lease["metadata"]["resourceVersion"]) + 1
                )
                self.lease = body
                holder = body["spec"].get("holderIdentity", "")
                if holder and (not self.holders or self.holders[-1] != holder):
                    self.holders.append(holder)
                return JSONResponse(content=self.lease)

        return app


@pytest.fixture
def lease_api() -> Iterator[tuple[FakeLeaseAPI, str]]:
    api = FakeLeaseAPI()
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(api.app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_http(
        f"http://127.0.0.1:{port}"
        + LEASE_PATH.format(namespace="e2e", name="lmcache-mp-memory-coordinator"),
        30,
    )
    try:
        yield api, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _barrier_hit(harness: Harness, name: str) -> bool:
    barriers = harness.scenario.state().get("barriers", [])
    if isinstance(barriers, dict):
        entry = barriers.get(name, {})
        return bool(entry.get("hit")) or entry.get("status") == "waiting"
    return any(
        b.get("name") == name and (b.get("hit") or b.get("status") == "waiting")
        for b in barriers
    )


def _lease_config(api_url: str, token: Path, identity: str) -> dict[str, object]:
    return {
        "leader_election": "kubernetes",
        "lease_namespace": "e2e",
        "kubernetes_api_url": api_url,
        "kubernetes_token_path": str(token),
        "kubernetes_ca_path": "",
        "lease_duration_seconds": 3.0,
        "lease_renew_interval_seconds": 1.0,
        "holder_identity": identity,
        "cooldown_seconds": 600.0,
    }


def test_lease_handoff_continues_the_move_with_single_writer(
    harness: Harness, lease_api: tuple[FakeLeaseAPI, str], tmp_path: Path
) -> None:
    api, api_url = lease_api
    token = tmp_path / "token"
    token.write_text("t")
    harness.memcoord.seed_inventory()
    harness.scenario.barrier(
        {"instance_id": "mp-donor", "operation": "evict", "when": "after", "name": "h"}
    )

    # Replica A holds the Lease and drives the move; replica B stands by on
    # the same state directory.
    replica_a = harness.memcoord
    replica_a.start(**_lease_config(api_url, token, "replica-a"))
    replica_b = Memcoord(harness.endpoints, tmp_path / "b")
    replica_b.state_dir = replica_a.state_dir
    replica_b.start(**_lease_config(api_url, token, "replica-b"))
    harness.cleanup.append(replica_b.stop)

    wait_until(
        lambda: api.holders[:1] == ["replica-a"] or api.holders[:1] == ["replica-b"], 30
    )
    leader, follower = (
        (replica_a, replica_b)
        if api.holders[0] == "replica-a"
        else (replica_b, replica_a)
    )
    wait_until(
        lambda: leader.client.readyz().status_code == 200, 60, what="leader ready"
    )
    assert follower.client.readyz().status_code == 503
    assert follower.client.readyz().json()["reason"] == "not leader"

    # Park the move at the donor evict (barrier hit = request in flight),
    # then kill the leader mid-flight and let the evict complete.
    wait_until(lambda: _barrier_hit(harness, "h"), 90, what="evict barrier hit")
    leader.kill()
    harness.scenario.release("h")

    # The follower acquires the Lease after expiry and completes the move.
    wait_until(lambda: api.holders[-1] != api.holders[0], 60, what="handoff")
    assert api.holders == [api.holders[0], api.holders[-1]]
    wait_until(
        lambda: follower.client.readyz().status_code == 200, 60, what="follower ready"
    )
    move = follower.client.wait_terminal(timeout=120)
    assert move["state"] == "COMPLETE" and move["outcome"] in (
        "SUCCEEDED",
        "ROLLED_BACK",
    )
    outside = [
        r["operation"] for r in harness.move_allocator_posts(follower.client.journal())
    ]
    assert outside.count("deallocate") <= 1 and outside.count("allocate") <= 1
    request_ids = [r["body"]["request_id"] for r in harness.allocator_posts()]
    assert len(request_ids) == len(set(request_ids))
    assert move["outcome"] == "SUCCEEDED", move


def test_follower_never_mutates_or_writes_the_journal(
    harness: Harness, lease_api: tuple[FakeLeaseAPI, str], tmp_path: Path
) -> None:
    api, api_url = lease_api
    token = tmp_path / "token"
    token.write_text("t")
    # Pre-hold the Lease for an outsider so both replicas are followers.
    api.lease["spec"] = {
        "holderIdentity": "someone-else",
        "leaseDurationSeconds": 3600,
        "renewTime": "2999-01-01T00:00:00.000000Z",
    }
    harness.memcoord.seed_inventory()
    journal_before = harness.memcoord.journal_path().read_bytes()
    harness.memcoord.start(**_lease_config(api_url, token, "replica-a"))
    harness.memcoord.client.wait_cycles(4, timeout=60)
    assert harness.memcoord.client.readyz().status_code == 503
    assert harness.scenario_posts() == [] and harness.allocator_posts() == []
    assert harness.memcoord.journal_path().read_bytes() == journal_before
    assert api.holders == []
