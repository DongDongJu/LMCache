# SPDX-License-Identifier: Apache-2.0
"""Tests for the memory orchestrator process and its client.

Every orchestrator here is a real ``python -m lmcache.v1.memory_orchestrator``
process whose endpoint is read from its startup marker.
"""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

# Third Party
import grpc
import pytest

# First Party
from lmcache.cli.commands.memory import MemoryCommand
from lmcache.v1 import memory_orchestrator
from lmcache.v1.memory_orchestrator._proto_gen import memory_orchestrator_pb2 as pb2
from lmcache.v1.memory_orchestrator._proto_gen.memory_orchestrator_pb2_grpc import (
    MemoryOrchestratorStub,
)
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    CloseResult,
    OrchestratorUnavailableError,
    ReadRequest,
    ReadStatus,
    RegionFencedError,
    RegisterResult,
    RequestRejectedError,
    TokenStatus,
    WireLayout,
    WireObjectKey,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.memory_orchestrator.server import OrchestratorConfig, parse_args

REGION = "pool-test"
ALIGN = 4096
CAPACITY = 1024 * ALIGN
MAX_BATCH = 4
MODE = "software_fenced"
LAYOUT = WireLayout(shapes=((2, 8),), dtypes=("bfloat16",))
STARTUP_TIMEOUT_S = 60.0
EXIT_TIMEOUT_S = 30.0
# Far above any id a client allocates, so raw requests never collide with them.
RAW_REQUEST_ID = 1 << 40
# Root of the ``lmcache`` tree under test; spawned orchestrators import it too.
SOURCE_ROOT = Path(memory_orchestrator.__file__).resolve().parents[3]


@dataclass
class Orchestrator:
    """A spawned orchestrator process."""

    proc: subprocess.Popen[bytes]
    state_dir: Path
    log_path: Path

    @property
    def marker_path(self) -> Path:
        return self.state_dir / f"{REGION}.marker"

    def marker(self) -> dict[str, object]:
        return json.loads(self.marker_path.read_text())

    @property
    def endpoint(self) -> str:
        return str(self.marker()["endpoint"])

    def log(self) -> str:
        return self.log_path.read_text()

    def stop(self) -> int:
        self.proc.send_signal(signal.SIGTERM)
        return self.proc.wait(timeout=EXIT_TIMEOUT_S)

    def wait_for_marker(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(
                    f"orchestrator exited with {self.proc.returncode}:\n{self.log()}"
                )
            try:
                if self.marker()["pid"] == self.proc.pid:
                    return
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # not created yet, or being written
            time.sleep(0.05)
        pytest.fail(f"no startup marker within {STARTUP_TIMEOUT_S}s:\n{self.log()}")


def start_orchestrator(state_dir: Path, log_path: Path) -> Orchestrator:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(SOURCE_ROOT), env.get("PYTHONPATH")) if path
    )
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "lmcache.v1.memory_orchestrator",
                "--region-id",
                REGION,
                "--capacity-bytes",
                str(CAPACITY),
                "--alignment",
                "4K",
                "--max-batch-entries",
                str(MAX_BATCH),
                "--listen",
                "127.0.0.1:0",
                "--state-dir",
                str(state_dir),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return Orchestrator(proc, state_dir, log_path)


def kill(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def connect(orchestrator: Orchestrator, client_id: str) -> MemoryOrchestratorClient:
    client = MemoryOrchestratorClient(orchestrator.endpoint, REGION, client_id)
    client.register(
        layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
        mapped_bytes=CAPACITY,
        visibility_mode=MODE,
    )
    return client


def key(name: str) -> WireObjectKey:
    return WireObjectKey(
        chunk_hash=name.encode(),
        model_name="model",
        kv_rank=0,
        object_group_id=0,
        cache_salt="",
    )


def write(name: str, payload_bytes: int = ALIGN) -> WriteRequest:
    return WriteRequest(key=key(name), payload_bytes=payload_bytes, layout=LAYOUT)


def raw_reserve_write(
    client: MemoryOrchestratorClient, request_id: int, names: list[str]
) -> pb2.ReserveWriteRequest:
    """A ReserveWrite request in the client's name with a chosen request id."""
    return pb2.ReserveWriteRequest(
        env=pb2.Envelope(
            region_id=REGION,
            expected_region_epoch=client.region_epoch,
            client_id=client.client_id,
            client_incarnation=client.client_incarnation,
            request_id=request_id,
        ),
        entries=[
            pb2.WriteEntry(
                key=pb2.ObjectKey(chunk_hash=name.encode(), model_name="model"),
                payload_bytes=ALIGN,
                layout=pb2.ObjectLayout(
                    shapes=[pb2.TensorShape(dims=[2, 8])], dtypes=["bfloat16"]
                ),
            )
            for name in names
        ],
    )


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Orchestrator]:
    """One orchestrator for the tests that only need a running region."""
    root = tmp_path_factory.mktemp("shared-orchestrator")
    orchestrator = start_orchestrator(root / "state", root / "orchestrator.log")
    try:
        orchestrator.wait_for_marker()
        yield orchestrator
    finally:
        kill(orchestrator.proc)


@pytest.fixture
def clients(
    shared: Orchestrator,
) -> Iterator[Callable[[str], MemoryOrchestratorClient]]:
    """Registers clients with the shared orchestrator; closes them after."""
    opened: list[MemoryOrchestratorClient] = []

    def open_client(client_id: str) -> MemoryOrchestratorClient:
        client = connect(shared, client_id)
        opened.append(client)
        return client

    yield open_client
    for client in opened:
        client.close()


@pytest.fixture
def raw_stub(shared: Orchestrator) -> Iterator[MemoryOrchestratorStub]:
    with grpc.insecure_channel(shared.endpoint) as channel:
        yield MemoryOrchestratorStub(channel)


@pytest.fixture
def spawn(tmp_path: Path) -> Iterator[Callable[[Path], Orchestrator]]:
    """Starts orchestrators on a given state dir; kills them after the test."""
    spawned: list[Orchestrator] = []

    def start(state_dir: Path) -> Orchestrator:
        log_path = tmp_path / f"orchestrator-{len(spawned)}.log"
        orchestrator = start_orchestrator(state_dir, log_path)
        spawned.append(orchestrator)
        return orchestrator

    yield start
    for orchestrator in spawned:
        kill(orchestrator.proc)


def test_write_finish_read_flow(clients):
    writer = clients("mp-writer")
    reader = clients("mp-reader")
    names = ["flow-0", "flow-1", "flow-2"]

    grants = writer.reserve_write(
        [write(name, 100 + i) for i, name in enumerate(names)]
    )
    assert [g.status for g in grants] == [WriteStatus.WRITE_GRANTED] * 3
    assert all(
        g.handle.offset % ALIGN == 0 and g.handle.length == ALIGN for g in grants
    )
    assert len({g.handle.offset for g in grants}) == 3
    busy = reader.reserve_read([ReadRequest(key(name), 1) for name in names])
    assert [r.status for r in busy] == [ReadStatus.BUSY_WRITING] * 3

    assert writer.finish_write([g.token for g in grants]) == [TokenStatus.OK] * 3
    reads = reader.reserve_read(
        [ReadRequest(key(name), 2) for name in names]
        + [ReadRequest(key("flow-absent"), 1)]
    )
    assert [r.status for r in reads] == [ReadStatus.READ_GRANTED] * 3 + [
        ReadStatus.MISS
    ]
    for i, (grant, granted) in enumerate(zip(grants, reads[:3], strict=True)):
        assert granted.handle == grant.handle
        assert granted.layout == LAYOUT
        assert granted.payload_bytes == 100 + i
        assert len(granted.leases) == 2

    leases = [lease for granted in reads for lease in granted.leases]
    leased = reader.usage().read_leases
    assert reader.finish_read(leases) == [TokenStatus.OK] * 6
    assert reader.finish_read(leases[:1]) == [TokenStatus.OK]
    assert reader.usage().read_leases == leased - 6
    again = reader.reserve_write([write(name) for name in names])
    assert [g.status for g in again] == [WriteStatus.EXISTS_VALID] * 3


def test_replayed_reserve_write_returns_the_recorded_grants(clients, raw_stub):
    client = clients("mp-replay")
    request = raw_reserve_write(client, RAW_REQUEST_ID, ["replay-0", "replay-1"])
    first = raw_stub.ReserveWrite(request, timeout=5)
    allocated = client.usage().allocated_bytes

    second = raw_stub.ReserveWrite(request, timeout=5)
    assert second == first
    assert [g.status for g in first.grants] == [pb2.WriteGrant.WRITE_GRANTED] * 2
    assert client.usage().allocated_bytes == allocated
    # The replayed tokens are the live ones.
    tokens = [grant.token for grant in second.grants]
    assert client.finish_write(tokens) == [TokenStatus.OK] * 2


def test_reused_request_id_with_another_payload_is_rejected(clients, raw_stub):
    client = clients("mp-reuse")
    raw_stub.ReserveWrite(
        raw_reserve_write(client, RAW_REQUEST_ID, ["reuse-0"]), timeout=5
    )
    allocated = client.usage().allocated_bytes

    with pytest.raises(grpc.RpcError) as excinfo:
        raw_stub.ReserveWrite(
            raw_reserve_write(client, RAW_REQUEST_ID, ["reuse-1"]), timeout=5
        )
    assert excinfo.value.code() is grpc.StatusCode.ALREADY_EXISTS
    assert client.usage().allocated_bytes == allocated
    assert client.reserve_write([write("reuse-1")])[0].status is (
        WriteStatus.WRITE_GRANTED
    )


def test_client_splits_batches_above_max_batch_entries(clients):
    client = clients("mp-batch")
    assert client.describe_region().max_batch_entries == MAX_BATCH
    names = [f"batch-{i}" for i in range(2 * MAX_BATCH + 2)]

    grants = client.reserve_write([write(name) for name in names])
    assert [g.status for g in grants] == [WriteStatus.WRITE_GRANTED] * len(names)
    offsets = [g.handle.offset for g in grants]
    assert offsets == sorted(set(offsets))
    assert client.finish_write([g.token for g in grants]) == [TokenStatus.OK] * len(
        names
    )
    reads = client.reserve_read([ReadRequest(key(name), 2) for name in names])
    assert [r.handle for r in reads] == [g.handle for g in grants]
    leases = [lease for granted in reads for lease in granted.leases]
    assert client.finish_read(leases) == [TokenStatus.OK] * len(leases)


def test_failed_split_batch_aborts_the_earlier_grants(clients):
    client = clients("mp-partial")
    names = [f"partial-{i}" for i in range(MAX_BATCH)]
    consumed = client.usage().consumed

    with pytest.raises(RequestRejectedError) as excinfo:
        client.reserve_write([write(name) for name in names] + [write("partial-x", 0)])
    assert excinfo.value.code == "INVALID_ARGUMENT"
    usage = client.usage()
    assert usage.consumed == consumed + MAX_BATCH
    regrants = client.reserve_write([write(name) for name in names])
    assert [g.status for g in regrants] == [WriteStatus.WRITE_GRANTED] * MAX_BATCH


def test_close_client_retires_writes(clients):
    closer = clients("mp-closer")
    observer = clients("mp-observer")
    committed = closer.reserve_write([write("close-valid")])[0]
    closer.finish_write([committed.token])
    closer.reserve_read([ReadRequest(key("close-valid"), 1)])
    busy = closer.reserve_write([write("close-busy")])[0]
    assert observer.reserve_write([write("close-busy")])[0].status is (
        WriteStatus.BUSY_WRITING
    )
    before = observer.usage()

    assert closer.close() == CloseResult(aborted_writes=1, released_leases=1)
    assert closer.close() is None
    after = observer.usage()
    assert after.consumed == before.consumed + 1
    assert after.read_leases == before.read_leases - 1
    assert after.clients == before.clients - 1
    regrant = observer.reserve_write([write("close-busy")])[0]
    assert regrant.status is WriteStatus.WRITE_GRANTED
    assert regrant.handle.offset > busy.handle.offset


def test_new_incarnation_fences_the_previous_client(shared):
    # Not closed at the end: closing a retired client only logs its refusal.
    old = connect(shared, "mp-restarted")
    grant = old.reserve_write([write("restart-busy")])[0]

    new = MemoryOrchestratorClient(shared.endpoint, REGION, "mp-restarted")
    result = new.register(
        layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
        mapped_bytes=CAPACITY,
        visibility_mode=MODE,
    )
    assert result == RegisterResult(
        region_epoch=old.region_epoch, retired_writes=1, released_leases=0
    )
    with pytest.raises(RegionFencedError) as excinfo:
        old.finish_write([grant.token])
    assert excinfo.value.reset_required is False
    assert "client not registered" in str(excinfo.value)
    assert new.finish_write([grant.token]) == [TokenStatus.STALE_TOKEN]
    assert new.close() == CloseResult(aborted_writes=0, released_leases=0)


def test_wrong_region_is_fenced(shared):
    client = MemoryOrchestratorClient(shared.endpoint, "pool-other", "mp-lost")
    with pytest.raises(RegionFencedError) as excinfo:
        client.describe_region()
    assert excinfo.value.reset_required is False
    assert client.close() is None


def test_unreachable_orchestrator_raises_unavailable():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    client = MemoryOrchestratorClient(
        f"127.0.0.1:{port}", REGION, "mp-alone", rpc_timeout_s=1.0, max_attempts=2
    )
    # Empty batches send nothing, so they succeed even without a server.
    assert client.reserve_write([]) == []
    assert client.finish_read([]) == []
    with pytest.raises(OrchestratorUnavailableError):
        client.describe_region()
    assert client.close() is None


def test_second_orchestrator_on_the_same_state_dir_exits_2(shared, spawn):
    second = spawn(shared.state_dir)
    assert second.proc.wait(timeout=STARTUP_TIMEOUT_S) == 2
    assert shared.marker()["pid"] == shared.proc.pid
    assert str(shared.proc.pid) in second.log()


def test_restart_after_kill_requires_reset(spawn, tmp_path):
    state_dir = tmp_path / "state"
    first = spawn(state_dir)
    first.wait_for_marker()
    client = connect(first, "mp-crash")
    assert client.reserve_write([write("crash-busy")])[0].status is (
        WriteStatus.WRITE_GRANTED
    )
    first.proc.kill()
    # Reap it: a zombie still counts as a live pid.
    first.proc.wait()

    second = spawn(state_dir)
    second.wait_for_marker()
    assert second.marker()["reset_required"] is True
    fresh = MemoryOrchestratorClient(second.endpoint, REGION, "mp-crash")
    assert fresh.describe_region().reset_required is True
    with pytest.raises(RegionFencedError) as excinfo:
        fresh.register(
            layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
            mapped_bytes=CAPACITY,
            visibility_mode=MODE,
        )
    assert excinfo.value.reset_required is True
    with pytest.raises(RegionFencedError) as excinfo:
        fresh.reserve_write([write("crash-busy")])
    assert excinfo.value.reset_required is True
    assert fresh.close() is None

    # A clean stop keeps the marker: the region still needs an offline reset.
    assert second.stop() == 0
    assert second.marker_path.exists()


def test_graceful_stop_without_clients_removes_the_marker(spawn, tmp_path):
    orchestrator = spawn(tmp_path / "state")
    orchestrator.wait_for_marker()
    client = connect(orchestrator, "mp-graceful")
    assert client.close() == CloseResult(aborted_writes=0, released_leases=0)

    assert orchestrator.stop() == 0
    assert not orchestrator.marker_path.exists()


def test_graceful_stop_with_a_registered_client_keeps_the_marker(spawn, tmp_path):
    orchestrator = spawn(tmp_path / "state")
    orchestrator.wait_for_marker()
    connect(orchestrator, "mp-lingering")

    assert orchestrator.stop() == 0
    assert orchestrator.marker_path.exists()
    assert "clients are still registered" in orchestrator.log()


def test_parse_args_defaults_and_capacity_rounding(tmp_path):
    config = parse_args(
        ["--region-id", "r", "--capacity-gb", "1.001", "--state-dir", str(tmp_path)]
    )
    assert config == OrchestratorConfig(
        region_id="r",
        capacity_bytes=1 << 30,
        alignment_bytes=2 << 20,
        layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
        visibility_mode="software_fenced",
        max_batch_entries=4096,
        listen="0.0.0.0:7700",
        state_dir=tmp_path,
        max_workers=16,
    )
    config = parse_args(
        [
            "--region-id",
            "r",
            "--capacity-bytes",
            str(10 * ALIGN + 5),
            "--alignment",
            "4k",
            "--layout-fingerprint",
            "custom",
            "--visibility-mode",
            "coherent",
            "--listen",
            "[::]:0",
            "--state-dir",
            str(tmp_path),
        ]
    )
    assert config.capacity_bytes == 10 * ALIGN
    assert config.alignment_bytes == ALIGN
    assert config.layout_fingerprint == b"custom"
    assert config.visibility_mode == "coherent"
    assert config.listen == "[::]:0"


@pytest.mark.parametrize(
    "extra",
    [
        ["--capacity-gb", "1"],
        ["--alignment", "3000"],
        ["--alignment", "2X"],
        ["--capacity-bytes", "4095", "--alignment", "4K"],
        ["--listen", "7700"],
        ["--region-id", "../escape"],
        ["--max-batch-entries", "0"],
    ],
)
def test_parse_args_rejects_invalid_flags(tmp_path, extra):
    argv = ["--region-id", "r", "--capacity-bytes", str(CAPACITY)]
    with pytest.raises(SystemExit) as excinfo:
        parse_args([*argv, "--state-dir", str(tmp_path), *extra])
    assert excinfo.value.code == 2


def test_memory_command_serves_the_parsed_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    argv = ["--region-id", "r", "--capacity-bytes", str(CAPACITY)]
    argv += ["--alignment", "4K", "--state-dir", str(tmp_path)]
    parser = argparse.ArgumentParser()
    command = MemoryCommand()
    command.register(parser.add_subparsers())
    args = parser.parse_args([command.name(), *argv])
    served: list[OrchestratorConfig] = []

    def fake_serve(config: OrchestratorConfig) -> int:
        served.append(config)
        return 2

    monkeypatch.setattr("lmcache.v1.memory_orchestrator.server.serve", fake_serve)
    with pytest.raises(SystemExit) as excinfo:
        args.func(args)
    assert excinfo.value.code == 2
    assert served == [parse_args(argv)]
