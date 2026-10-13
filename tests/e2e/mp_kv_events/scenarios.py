# SPDX-License-Identifier: Apache-2.0
"""Real router/engine/MP scenarios; native placements are observed, never injected."""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.client import HTTPException
from typing import Any
import argparse
import base64
import hashlib
import json
import re
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET

# Third Party
from fixtures import fixture
from residency import assert_target_resident, identity, target_l1_keys
from response_oracle import validate_inference_response
from run import HERE, NAMESPACE, command, deployment, kubectl, write_json


class NativeLedger:
    """Project independently captured engine events into CPU/GPU placement sets."""

    def __init__(self, records: list[dict]) -> None:
        self.blocks: dict[str, dict[str, dict]] = {"CPU": {}, "GPU": {}}
        self.records = records
        previous_sequence = None
        for record in records:
            if "frames_base64" in record:
                sequence = int.from_bytes(
                    base64.b64decode(record["frames_base64"][1]), "big"
                )
                if previous_sequence is not None:
                    assert sequence == previous_sequence + 1, (
                        "Native observer publication gap or epoch change"
                    )
                previous_sequence = sequence
            for event in record["payload"][1]:
                kind = event["type"]
                if kind == "AllBlocksCleared":
                    for blocks in self.blocks.values():
                        blocks.clear()
                    continue
                medium = event.get("medium")
                if medium in ("CPU", "CPU_PINNED"):
                    medium = "CPU"
                if medium not in self.blocks:
                    continue
                if kind == "BlockRemoved":
                    for block_hash in event["block_hashes"]:
                        self.blocks[medium].pop(str(block_hash), None)
                elif kind == "BlockStored":
                    size = event["block_size"]
                    parent = event["parent_block_hash"]
                    assert len(event["token_ids"]) == size * len(
                        event["block_hashes"]
                    ), event
                    for index, block_hash in enumerate(event["block_hashes"]):
                        descriptor = {
                            "hash": block_hash,
                            "parent": parent,
                            "tokens": event["token_ids"][
                                index * size : (index + 1) * size
                            ],
                            "block_size": size,
                        }
                        self.blocks[medium][str(block_hash)] = descriptor
                        parent = block_hash

    def prefix(self, tokens: list[int], medium: str) -> list[dict]:
        """Find a native parent chain matching the exact request token prefix."""
        result = []
        parent = None
        offset = 0
        while offset < len(tokens):
            matches = [
                block
                for block in self.blocks[medium].values()
                if block["parent"] == parent
                and block["tokens"] == tokens[offset : offset + block["block_size"]]
            ]
            if not matches:
                break
            assert len(matches) == 1, "Ambiguous native identity for one prefix"
            block = matches[0]
            result.append(block)
            offset += block["block_size"]
            parent = block["hash"]
        return result


def counter(text: str, name: str) -> float:
    """Sum a metric's label series in a replica-isolated no-other-traffic window."""
    return sum(
        float(line.rsplit(" ", 1)[-1])
        for line in text.splitlines()
        if line.startswith(name + "{") or line.startswith(name + " ")
    )


class Driver:
    """Execute one persisted deployment using its actual discovered worker IDs."""

    def __init__(
        self, args: argparse.Namespace, *, prepare_kernels: bool = True
    ) -> None:
        self.args = args
        self.state = deployment(args)
        if self.state.get("pending_event_mode_change"):
            raise ValueError(
                "An event-mode change is incomplete; finish it before tests"
            )
        self.root = args.artifacts
        self.root.mkdir(parents=True, exist_ok=True)
        source_hashes = {}
        for path in sorted(HERE.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(HERE)
            content = path.read_bytes()
            archived = self.root / "host-source" / relative
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(content)
            source_hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        write_json(
            self.root / "host-harness.json",
            {
                "files_sha256": source_hashes,
                "source_directory": "host-source",
                "kernel_setup_requested": prepare_kernels,
            },
        )
        self.workers: dict[str, int] = {}
        self.native_epoch_ns = {"a": 0, "b": 0}
        self.last_publication_opportunity_ns = {"a": 0, "b": 0}
        self.requests: list[dict] = []
        self.serial = 0
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.forward_log = (self.root / "port-forward.log").open("w")
        self.forward = subprocess.Popen(
            kubectl(
                "port-forward", "-n", NAMESPACE, "pod/frontend", f"{self.port}:8000"
            ),
            stdout=self.forward_log,
            stderr=subprocess.STDOUT,
        )
        try:
            self.wait(lambda: self.http_ready(), "frontend readiness", timeout=600)
            for replica in ("a", "b"):
                self.wait(
                    partial(self.worker_ready, replica),
                    f"worker {replica} startup",
                    timeout=600,
                )
                self.workers[replica] = self.worker_identity(replica)
            assert self.workers["a"] != self.workers["b"], self.workers
            write_json(self.root / "worker-identities.json", self.workers)
            write_json(self.root / "deployment.json", self.state)
            if prepare_kernels:
                self.prepare_kernels()
        except Exception:
            self.close()
            raise

    def prepare_kernels(self) -> None:
        """Compile decode/cache-hit kernels, then reset both caches before cases."""
        previous_workers = dict(self.workers)
        warmups = []
        started = time.time_ns()
        for role, replica in enumerate(("a", "b")):
            tokens = fixture(48000 + role, 1, self.state["chunk_size"])
            warmups.append(self.request(tokens, replica, max_tokens=2, timeout=600))
            warmups.append(self.request(tokens, replica, max_tokens=1, timeout=600))

        def reset(replica: str) -> None:
            content = self.exec(
                replica,
                "from pathlib import Path; "
                "print(Path('/config/launcher.json').read_text(),end='')",
            )
            self.local_http(
                replica,
                9900,
                "/reload",
                "POST",
                {"launcher_sha256": hashlib.sha256(content.encode()).hexdigest()},
                timeout=720,
            )
            self.workers[replica] = self.worker_identity(replica)
            self.last_publication_opportunity_ns[replica] = 0
            ledger = self.ledger(replica)
            assert not ledger.blocks["CPU"] and not ledger.blocks["GPU"], (
                "Kernel setup placements survived the fresh-process reset"
            )
            status = json.loads(self.local_http(replica, 8080, "/status"))
            assert status["kv_events"]["live_objects"] == 0, status["kv_events"]

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(reset, ("a", "b")))
        script = """
import asyncio, json, sys
from dynamo.runtime import DistributedRuntime
async def inspect():
    runtime = DistributedRuntime(asyncio.get_running_loop(), 'etcd', 'nats')
    try:
        client = await runtime.endpoint(sys.argv[1]).client()
        expected = set(json.loads(sys.argv[2]))
        deadline = asyncio.get_running_loop().time() + 60
        while asyncio.get_running_loop().time() < deadline:
            if set(client.instance_ids()) == expected:
                print(json.dumps(sorted(expected)))
                return
            await asyncio.sleep(0.2)
        raise RuntimeError('Fresh serving identities did not converge in discovery')
    finally:
        runtime.shutdown()
asyncio.run(inspect())
"""
        metric = next(
            line
            for line in self.local_http("a", 8081, "/metrics").splitlines()
            if line.startswith("dynamo_component_inflight_requests{")
            and 'dynamo_endpoint="generate"' in line
        )
        endpoint = []
        for label in ("dynamo_namespace", "dynamo_component", "dynamo_endpoint"):
            values = set(re.findall(label + r'="([^"]+)"', metric))
            assert len(values) == 1, (label, values)
            endpoint.append(values.pop())
        discovery = self.exec(
            "a", script, ".".join(endpoint), json.dumps(list(self.workers.values()))
        )
        write_json(
            self.root / "kernel-setup.json",
            {
                "started_ns": started,
                "ended_ns": time.time_ns(),
                "requests": warmups,
                "previous_workers": previous_workers,
                "fresh_workers": self.workers,
                "discovery": discovery,
                "scope": "untimed JIT setup followed by empty engine and MP caches",
            },
        )
        write_json(self.root / "worker-identities.json", self.workers)

    def worker_identity(self, replica: str) -> int:
        """Read Dynamo's hexadecimal connection ID, excluding engine rank labels."""
        metrics = self.local_http(replica, 8081, "/metrics")
        identifiers = {
            value
            for line in metrics.splitlines()
            if line.startswith("dynamo_component_")
            for value in re.findall(r'worker_id="([0-9a-f]+)"', line)
        }
        assert len(identifiers) == 1, (
            f"Expected one Dynamo connection ID in {replica}: {identifiers}"
        )
        return int(identifiers.pop(), 16)

    def model_name(self, replica: str) -> str:
        """Use the actual registered ObjectKey namespace, not the served alias."""
        status = json.loads(self.local_http(replica, 8080, "/status"))
        names = {item["model_name"] for item in status["cache_context_meta"].values()}
        assert len(names) == 1, f"Expected one registered model namespace: {names}"
        return names.pop()

    def close(self) -> None:
        """Stop only the driver's own port-forward process."""
        self.forward.terminate()
        self.forward.wait(timeout=10)
        self.forward_log.close()
        write_json(self.root / "requests.json", self.requests)

    def exec(self, replica: str, script: str, *argv: str, timeout: float = 120) -> str:
        """Execute a bounded read/control script in the selected private pod."""
        return command(
            kubectl(
                "exec",
                "-n",
                NAMESPACE,
                f"worker-{replica}",
                "--",
                "python3",
                "-c",
                script,
                *argv,
            ),
            timeout=timeout,
        ).stdout

    def local_http(
        self,
        replica: str,
        port: int,
        path: str,
        method: str = "GET",
        body: dict | None = None,
        timeout: float = 90,
    ) -> str:
        """Use the selected worker's localhost API, preserving private MP scope."""
        script = (
            "import urllib.request,sys; "
            "r=urllib.request.Request(sys.argv[1],method=sys.argv[2],"
            "data=None if sys.argv[3]=='null' else sys.argv[3].encode(),"
            "headers={'Content-Type':'application/json'}); "
            "print(urllib.request.urlopen(r,timeout=float(sys.argv[4])).read().decode())"
        )
        return self.exec(
            replica,
            script,
            f"http://127.0.0.1:{port}{path}",
            method,
            json.dumps(body),
            str(timeout),
            timeout=timeout + 30,
        )

    def worker_ready(self, replica: str) -> bool:
        """Wait for actual engine health without triggering inference."""
        try:
            self.local_http(replica, 8081, "/health")
            return True
        except subprocess.CalledProcessError:
            return False

    def http_ready(self) -> bool:
        """Check frontend model readiness without submitting inference."""
        try:
            if self.forward.poll() is not None:
                self.forward = subprocess.Popen(
                    kubectl(
                        "port-forward",
                        "-n",
                        NAMESPACE,
                        "pod/frontend",
                        f"{self.port}:8000",
                    ),
                    stdout=self.forward_log,
                    stderr=subprocess.STDOUT,
                )
            with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}/v1/models", timeout=2
            ) as response:
                return bool(json.load(response).get("data"))
        except (urllib.error.URLError, TimeoutError, HTTPException, ConnectionError):
            return False

    def wait(
        self, predicate: Callable[[], bool], label: str, timeout: float = 5
    ) -> None:
        """Poll an observed state within its explicit convergence budget."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        raise AssertionError(f"Deadline exceeded: {label} ({timeout}s)")

    def records(self, replica: str, name: str) -> list[dict]:
        """Read complete append-only evidence lines from one running worker."""
        data = self.exec(
            replica,
            "import pathlib,sys; "
            "print(pathlib.Path('/artifacts',sys.argv[1]).read_text())",
            name,
        )
        records = [json.loads(line) for line in data.splitlines() if line]
        if name == "native-events.jsonl":
            write_json(self.root / f"native-{replica}-all-epochs.json", records)
            return [
                record
                for record in records
                if record["received_ns"] >= self.native_epoch_ns[replica]
                and record["payload"][0] * 1e9 >= self.native_epoch_ns[replica]
            ]
        return records

    def ledger(self, replica: str) -> NativeLedger:
        """Rebuild observed state in native publication order."""
        return NativeLedger(self.records(replica, "native-events.jsonl"))

    def snapshot(self, label: str) -> dict:
        """Capture both private MPs, native publishers and worker counters."""
        result = {}
        for replica in ("a", "b"):
            result[replica] = {
                "native": self.records(replica, "native-events.jsonl"),
                "mp": self.records(replica, "mp-events.jsonl"),
                "reads": self.records(replica, "mp-reads.jsonl"),
                "native_epoch_ns": self.native_epoch_ns[replica],
                "metrics": self.local_http(replica, 8080, "/metrics"),
                "engine_metrics": self.local_http(replica, 8081, "/metrics"),
                "status": json.loads(self.local_http(replica, 8080, "/status")),
                "engine_log": self.exec(
                    replica,
                    "from pathlib import Path; "
                    "print(Path('/artifacts/engine.log').read_text())",
                ),
            }
        write_json(self.root / f"{label}.json", result)
        return result

    def request(
        self,
        tokens: list[int],
        owner: str | None = None,
        query: bool = False,
        max_tokens: int = 1,
        timeout: float = 90,
    ) -> dict:
        """Send targeted setup or unbiased measured traffic through real Dynamo."""
        request_id = f"kv-events-{uuid.uuid4().hex}"
        headers = {"Content-Type": "application/json", "x-request-id": request_id}
        if owner:
            headers.update(
                {
                    "x-dynamo-worker-instance-id": str(self.workers[owner]),
                    "x-dynamo-dp-rank": "0",
                }
            )
        payload = {
            "model": self.state["lock"]["model"]["id"],
            "prompt": tokens,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "nvext": {
                "extra_fields": ["worker_id", "timing"],
                "annotations": ["query_instance_id:"] if query else [],
            },
        }
        started = time.time_ns()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
        )
        chunks = []
        saw_done = False
        first_token_ns = None
        record_path = self.root / f"request-{request_id}.json"
        submitted = {
            "frontend_request_id": request_id,
            "tokens": tokens,
            "setup_target": owner,
            "query_only": query,
            "started_ns": started,
            "payload": payload,
        }
        write_json(record_path, submitted)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                for line in response:
                    text = line.decode().strip()
                    if text == "data: [DONE]":
                        saw_done = True
                        continue
                    if not text.startswith("data: "):
                        continue
                    if saw_done:
                        raise ValueError("SSE payload arrived after the terminal DONE")
                    chunk = json.loads(text[6:])
                    chunks.append(chunk)
                    if first_token_ns is None and any(
                        choice.get("text") for choice in chunk.get("choices", [])
                    ):
                        first_token_ns = time.time_ns()
        except Exception as error:
            submitted.update(
                error=repr(error), ended_ns=time.time_ns(), response=chunks
            )
            write_json(record_path, submitted)
            raise
        raw = json.dumps(chunks)
        choices = [chunk.get("nvext", {}).get("worker_id") for chunk in chunks]
        selected = next((value for value in choices if value), None)
        result: dict[str, Any] = {
            "frontend_request_id": request_id,
            "tokens": tokens,
            "setup_target": owner,
            "query_only": query,
            "selected": selected,
            "started_ns": started,
            "first_token_ns": first_token_ns,
            "ended_ns": time.time_ns(),
            "response": chunks,
            "saw_done": saw_done,
            "request_payload": payload,
            "output": "".join(
                choice.get("text", "")
                for chunk in chunks
                for choice in chunk.get("choices", [])
            ),
        }
        write_json(record_path, result)
        try:
            if not query:
                result["completion_tokens"] = validate_inference_response(
                    chunks,
                    saw_done=saw_done,
                    max_tokens=max_tokens,
                )
            assert selected is not None, raw
        except Exception as error:
            result["error"] = repr(error)
            write_json(record_path, result)
            raise
        if owner:
            assert selected["decode_worker_id"] == self.workers[owner], result
        if not query:
            selected_owner = next(
                replica
                for replica, identity in self.workers.items()
                if identity == selected["decode_worker_id"]
            )
            self.last_publication_opportunity_ns[selected_owner] = result["ended_ns"]
        write_json(record_path, result)
        self.requests.append(result)
        write_json(self.root / "requests.json", self.requests)
        return result

    def probe_owner(
        self,
        tokens: list[int],
        owner: str,
        trials: int = 20,
        *,
        opportunity_ns: int | None = None,
    ) -> None:
        """Verify routing without mutating KV caches, then retain every decision."""
        opportunity = (
            self.last_publication_opportunity_ns[owner]
            if opportunity_ns is None
            else opportunity_ns
        )
        remaining = 5 - (time.time_ns() - opportunity) / 1e9 if opportunity else 5
        self.wait(
            lambda: self.request(tokens, query=True)["selected"]["decode_worker_id"]
            == self.workers[owner],
            f"router placement converges to {owner} after recorded active step",
            timeout=max(0, remaining),
        )
        write_json(
            self.root / f"router-convergence-{owner}-{len(self.requests)}.json",
            {
                "publication_opportunity_ns": opportunity,
                "observed_router_ns": time.time_ns(),
                "scope": (
                    "upper bound from completed ordinary active inference; "
                    "no request-free idle claim"
                ),
            },
        )
        for _ in range(trials):
            assert (
                self.request(tokens, query=True)["selected"]["decode_worker_id"]
                == self.workers[owner]
            )

    def active_barrier(self, owner: str) -> dict:
        """Give a step-dependent engine one recorded unrelated publication step."""
        self.serial += 1
        return self.request(
            fixture(6000 + self.serial, 1, self.state["chunk_size"]), owner
        )

    def evict_gpu(self, tokens: list[int], owner: str) -> None:
        """Use genuine capacity pressure until every target GPU hash is absent."""
        expected = len(tokens) // self.state["chunk_size"] * self.state["chunk_size"]
        initial = self.ledger(owner).prefix(tokens, "CPU")
        assert sum(block["block_size"] for block in initial) == expected
        target_hashes = {str(block["hash"]) for block in initial}
        pressure_chunks = min(
            1024 // self.state["chunk_size"],
            max(1, self.state["event_log_size"] // 2),
        )
        for _ in range(256):
            ledger = self.ledger(owner)
            if not target_hashes.intersection(ledger.blocks["GPU"]):
                assert (
                    sum(block["block_size"] for block in ledger.prefix(tokens, "CPU"))
                    == expected
                ), "CPU copy evicted during GPU fixture setup"
                records = self.records(owner, "mp-events.jsonl")
                keys = target_l1_keys(
                    records,
                    tokens,
                    self.state["chunk_size"],
                    self.model_name(owner),
                    self.state["tp"],
                )
                assert_target_resident(records, keys)
                return
            self.serial += 1
            self.request(
                fixture(5000 + self.serial, pressure_chunks, self.state["chunk_size"]),
                owner,
            )
        raise AssertionError("Capacity pressure did not remove every target GPU block")

    def wait_received(self, owner: str) -> None:
        """Await actual subscriber receipt through the current MP committed cursor."""
        status = json.loads(self.local_http(owner, 8080, "/status"))["kv_events"]

        def received() -> bool:
            log = self.exec(
                owner,
                "from pathlib import Path; "
                "print(Path('/artifacts/engine.log').read_text())",
            )
            return any(
                int(epoch) == status["incarnation"]
                and int(cursor) >= status["next_seq"] - 1
                for epoch, cursor in re.findall(
                    r"MP_KV_SUBSCRIBER received_ns=\d+ incarnation=(\d+) cursor=(\d+)",
                    log,
                )
            )

        self.wait(received, "subscriber receives the committed fixture")

    def warm(self, tokens: list[int], owner: str) -> dict:
        """Create a cold direct baseline and await real complete CPU placement."""
        before = self.snapshot(f"warm-{owner}-{self.serial}-before")
        ledger = NativeLedger(before[owner]["native"])
        assert not ledger.prefix(tokens, "GPU") and not ledger.prefix(tokens, "CPU"), (
            "Warmup fixture is not cold"
        )
        baseline = self.request(tokens, owner)
        expected = len(tokens) // self.state["chunk_size"] * self.state["chunk_size"]

        def physically_ready() -> bool:
            records = self.records(owner, "mp-events.jsonl")
            try:
                keys = target_l1_keys(
                    records,
                    tokens,
                    self.state["chunk_size"],
                    self.model_name(owner),
                    self.state["tp"],
                )
                assert_target_resident(records, keys)
            except AssertionError:
                return False
            return True

        self.wait(physically_ready, "all exact fixture L1 objects committed")
        self.wait_received(owner)
        self.active_barrier(owner)
        self.wait(
            lambda: sum(
                block["block_size"]
                for block in self.ledger(owner).prefix(tokens, "CPU")
            )
            == expected,
            "complete committed CPU prefix",
        )
        after = self.snapshot(f"warm-{owner}-{self.serial}-after")
        assert not after[owner]["reads"][len(before[owner]["reads"]) :], (
            "Cold baseline or its fresh publication barrier read MP objects"
        )
        assert not any(
            event["event"] == "mp.retrieve.end"
            for event in after[owner]["mp"][len(before[owner]["mp"]) :]
        ), "Cold baseline unexpectedly completed an MP load"
        assert counter(
            after[owner]["metrics"], "lmcache_mp_l1_read_chunks_total"
        ) == counter(before[owner]["metrics"], "lmcache_mp_l1_read_chunks_total"), (
            "Baseline unexpectedly loaded from MP"
        )
        return baseline

    def measured(
        self, tokens: list[int], owner: str, cpu_load: bool, baseline: dict, label: str
    ) -> dict:
        """Require actual selected-engine response and CUDA-stream MP completion."""
        before = self.snapshot(label + "-before")
        result = self.request(tokens)
        after = self.snapshot(label + "-after")
        assert result["selected"]["decode_worker_id"] == self.workers[owner], result
        assert result["output"] == baseline["output"], "Cold/cache-hit output mismatch"
        router_hit_rates = [
            chunk["nvext"]["timing"]["kv_hit_rate"]
            for chunk in result["response"]
            if "kv_hit_rate" in chunk.get("nvext", {}).get("timing", {})
        ]
        assert router_hit_rates and all(0 < rate <= 1 for rate in router_hit_rates), (
            "Selected cache owner received no positive router overlap credit",
            router_hit_rates,
        )
        completions = {}
        for replica in ("a", "b"):
            new = after[replica]["mp"][len(before[replica]["mp"]) :]
            completions[replica] = [
                event for event in new if event["event"] == "mp.retrieve.end"
            ]
            assert all(
                event["metadata"]["total_bytes"] > 0 for event in completions[replica]
            )
        other = "b" if owner == "a" else "a"
        assert not completions[other], "Nonselected private MP unexpectedly loaded KV"
        if cpu_load:
            assert completions[owner], "Selected MP never completed H2D"
            assert all(
                event["metadata"]["num_tokens"] > 0 for event in completions[owner]
            )
        else:
            assert not completions[owner], (
                "GPU-only/cold request unexpectedly loaded CPU KV"
            )
            assert not after[owner]["reads"][len(before[owner]["reads"]) :], (
                "GPU-only request read CPU objects"
            )
        engine_log = after[owner]["engine_log"][len(before[owner]["engine_log"]) :]
        traces = []
        for line in engine_log.splitlines():
            match = re.search(r"MP_KV_TRACE (\w+) (.*)", line)
            if match:
                fields = dict(re.findall(r"(\w+)=([^\s]+)", match[2]))
                traces.append({"event": match[1], **fields})
        lookups = [trace for trace in traces if trace["event"] == "lookup"]
        if not cpu_load and not lookups:
            lookups = [trace for trace in traces if trace["event"] == "gpu_match"]
        assert lookups, "Missing actual engine GPU-match/lookup trace"
        engine_ids = {trace["request_id"] for trace in lookups}
        assert len(engine_ids) == 1, "Evidence window contains multiple requests"
        engine_request = engine_ids.pop()
        native_gpu_tokens = sum(
            block["block_size"]
            for block in NativeLedger(before[owner]["native"]).prefix(tokens, "GPU")
        )
        observed_gpu_tokens = {int(trace["gpu_hit_tokens"]) for trace in lookups}
        assert observed_gpu_tokens == {native_gpu_tokens}, (
            observed_gpu_tokens,
            native_gpu_tokens,
        )
        if cpu_load:
            assert {event["session_id"] for event in completions[owner]} == {
                engine_request
            }
            submitted = [
                trace for trace in traces if trace["event"] == "retrieve_submitted"
            ]
            consumed = [
                trace
                for trace in traces
                if trace["event"] in ("retrieve_completed", "retrieve_consumer_wait")
            ]
            required_ranks = set(range(self.state["tp"]))
            assert len(submitted) == self.state["tp"]
            assert {int(trace["rank"]) for trace in submitted} == required_ranks
            consumed = [
                trace
                for trace in consumed
                if trace["request_id"] == engine_request
                and (
                    trace["event"] == "retrieve_consumer_wait"
                    or trace.get("success") == "True"
                )
            ]
            assert {int(trace["rank"]) for trace in consumed} == required_ranks, traces
            assert not [
                trace
                for trace in traces
                if trace["event"] == "retrieve_completed"
                and trace["request_id"] == engine_request
                and trace.get("success") != "True"
            ]
            complete_tokens = (
                len(tokens) // self.state["chunk_size"] * self.state["chunk_size"]
            )
            eligible = complete_tokens - native_gpu_tokens
            expected_start = (
                native_gpu_tokens // self.state["chunk_size"] * self.state["chunk_size"]
            )
            expected_skip = native_gpu_tokens - expected_start
            assert eligible > 0
            assert all(
                int(trace["external_hit_tokens"]) == complete_tokens
                for trace in lookups
            )
            mp_lookups = [
                event
                for event in after[owner]["mp"][len(before[owner]["mp"]) :]
                if event["event"] == "mp.lookup_prefetch.end"
                and event["session_id"] == engine_request
            ]
            assert mp_lookups, "Missing actual MP L1 lookup result"
            assert all(
                event["metadata"]["l1_hit_tokens"] == complete_tokens
                and event["metadata"]["l2_hit_tokens"] == 0
                and event["metadata"]["l1_hit_keys"]
                == complete_tokens // self.state["chunk_size"] * self.state["tp"]
                and event["metadata"]["l2_hit_keys"] == 0
                for event in mp_lookups
            )
            for trace in submitted:
                assert trace["request_id"] == engine_request
                assert (int(trace["start"]), int(trace["end"]), int(trace["skip"])) == (
                    expected_start,
                    complete_tokens,
                    expected_skip,
                ), trace
                assert (
                    int(trace["end"]) - int(trace["start"]) - int(trace["skip"])
                    == eligible
                ), trace
            assert len(completions[owner]) == self.state["tp"]
            expected_objects = (
                complete_tokens
                - (
                    native_gpu_tokens
                    // self.state["chunk_size"]
                    * self.state["chunk_size"]
                )
            ) // self.state["chunk_size"]
            assert all(
                event["metadata"]["retrieved_count"] == expected_objects
                and event["metadata"]["num_tokens"] == complete_tokens - expected_start
                for event in completions[owner]
            )
            reads = after[owner]["reads"][len(before[owner]["reads"]) :]
            assert reads and not after[other]["reads"][len(before[other]["reads"]) :]
            assert {row["request_id"] for row in reads} == {engine_request}
            assert {row["worker_id"] for row in reads} == required_ranks
            expected_ipc_tokens = tokens
            if self.state["engine"] == "sglang":
                # SGLang matches N-1 input tokens, then page-aligns the key
                # before submit_lookup; vLLM retains the full request key.
                aligned = (
                    (len(tokens) - 1)
                    // self.state["native_block_size"]
                    * self.state["native_block_size"]
                )
                expected_ipc_tokens = tokens[:aligned]
            assert all(
                row["token_ids"] == expected_ipc_tokens
                and row["requested_keys"] == row["found_keys"]
                and (row["start"], row["end"], row["skip_first_n_tokens"])
                == (expected_start, complete_tokens, expected_skip)
                for row in reads
            )
            assert {row["instance_id"] for row in reads} == {
                event["metadata"]["engine_id"] for event in completions[owner]
            }
            lookup_records = [
                event
                for event in after[owner]["mp"][len(before[owner]["mp"]) :]
                if event["event"] == "mp.lookup"
                and event["session_id"] == engine_request
            ]
            assert lookup_records, "Missing exact MP lookup chunk identities"
            lookup_hashes = lookup_records[0]["metadata"]["chunk_hashes"]
            load_hashes = {
                base64.b64decode(value["bytes_base64"]).hex()
                for value in lookup_hashes[
                    native_gpu_tokens // self.state["chunk_size"] :
                ]
            }
            expected_keys = [
                key
                for key in target_l1_keys(
                    before[owner]["mp"],
                    tokens,
                    self.state["chunk_size"],
                    self.model_name(owner),
                    self.state["tp"],
                )
                if key["chunk_hash_hex"] in load_hashes
            ]
            assert {identity(key) for row in reads for key in row["found_keys"]} == {
                identity(key) for key in expected_keys
            }
            assert sum(row["bytes"] for row in reads) == sum(
                event["metadata"]["total_bytes"] for event in completions[owner]
            )
        evidence = {
            "request": result,
            "router_kv_hit_rates": router_hit_rates,
            "id_mapping": {
                "frontend": result["frontend_request_id"],
                "engine": engine_request,
            },
            "engine_trace": traces,
            "mp_h2d_completions": completions,
            "mp_reads": {
                replica: after[replica]["reads"][len(before[replica]["reads"]) :]
                for replica in ("a", "b")
            },
            "isolation": (
                "concurrency=1, no background inference; "
                "snapshots bracket only measured request"
            ),
        }
        write_json(self.root / (label + "-trace.json"), evidence)
        return evidence

    def cpu_only(self, owner: str, index: int) -> None:
        """T02: native GPU eviction, twenty router probes, then actual CPU H2D."""
        tokens = fixture(index)
        baseline = self.warm(tokens, owner)
        self.evict_gpu(tokens, owner)
        other = "b" if owner == "a" else "a"
        assert not self.ledger(other).prefix(tokens, "CPU")
        assert not self.ledger(other).prefix(tokens, "GPU")
        self.probe_owner(tokens, owner)
        self.measured(tokens, owner, True, baseline, f"T02-{owner}")

    def mixed(self, owner: str, index: int) -> None:
        """T03: native GPU prefix followed by its exact CPU parent-chain suffix."""
        tokens = fixture(index, 4)
        baseline = self.warm(tokens, owner)
        other = "b" if owner == "a" else "a"
        # The competing GPU prefix is longer than A's GPU prefix, but shorter
        # than A's combined GPU/CPU prefix. A can win only with CPU credit.
        self.warm(tokens[:512] + [198], other)
        self.evict_gpu(tokens, owner)
        self.request(tokens[:256] + [198], owner)
        ledger = self.ledger(owner)
        gpu = ledger.prefix(tokens, "GPU")
        cpu = ledger.prefix(tokens, "CPU")
        assert 0 < len(gpu) < len(cpu), (gpu, cpu)
        assert gpu == cpu[: len(gpu)], "GPU/CPU native boundary differs"
        competitor = self.ledger(other)
        other_gpu = competitor.prefix(tokens, "GPU")
        other_cpu = competitor.prefix(tokens, "CPU")
        assert len(gpu) < len(other_gpu) < len(cpu), (gpu, other_gpu, cpu)
        assert len(other_cpu) <= len(other_gpu), (other_cpu, other_gpu)
        write_json(
            self.root / f"T03-{owner}-competition.json",
            {
                "owner_gpu": gpu,
                "owner_cpu": cpu,
                "competitor_gpu": other_gpu,
                "competitor_cpu": other_cpu,
            },
        )
        self.probe_owner(tokens, owner)
        self.measured(tokens, owner, True, baseline, f"T03-{owner}")

    def cold(self, owner: str, index: int) -> None:
        """T01: submit an untargeted cold fixture and prove neither MP loads it."""
        tokens = fixture(8000 + index)
        before = self.snapshot(f"T01-{owner}-before")
        for replica in ("a", "b"):
            ledger = NativeLedger(before[replica]["native"])
            assert not ledger.prefix(tokens, "GPU") and not ledger.prefix(tokens, "CPU")
        response = self.request(tokens)
        after = self.snapshot(f"T01-{owner}-after")
        for replica in ("a", "b"):
            assert not [
                event
                for event in after[replica]["mp"][len(before[replica]["mp"]) :]
                if event["event"] == "mp.retrieve.end"
            ]
            assert not after[replica]["reads"][len(before[replica]["reads"]) :], (
                "Cold request unexpectedly read L1 objects"
            )
        write_json(
            self.root / f"T01-{owner}-trace.json", {"request": response, "mp_loads": 0}
        )

    def gpu_survival(self, owner: str, index: int) -> None:
        """T01/T05: physical CPU clear preserves an independently observed GPU hit."""
        tokens = fixture(index)
        baseline = self.warm(tokens, owner)
        gpu = self.ledger(owner).prefix(tokens, "GPU")
        assert gpu
        self.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        self.wait_received(owner)
        self.active_barrier(owner)
        self.wait(
            lambda: not self.ledger(owner).prefix(tokens, "CPU"), "CPU scoped removal"
        )
        assert self.ledger(owner).prefix(tokens, "GPU") == gpu
        self.probe_owner(tokens, owner)
        self.measured(tokens, owner, False, baseline, f"T05-{owner}")

    def removal(self, owner: str, index: int) -> None:
        """T04: remove one actual private CPU copy and select/load the other."""
        tokens = fixture(index)
        other = "b" if owner == "a" else "a"
        baseline = self.warm(tokens, owner)
        self.warm(tokens, other)
        self.evict_gpu(tokens, owner)
        self.evict_gpu(tokens, other)
        self.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        self.wait_received(owner)
        barrier = self.active_barrier(owner)
        self.wait(
            lambda: not self.ledger(owner).prefix(tokens, "CPU"),
            "CPU removal publication",
        )
        assert self.ledger(other).prefix(tokens, "CPU")
        self.probe_owner(tokens, other, opportunity_ns=barrier["ended_ns"])
        self.measured(tokens, other, True, baseline, f"T04-remove-{owner}")

    def geometry(self, owner: str, index: int) -> None:
        """T06: compare all complete native subblocks and exact scoped removals."""
        chunk_size = self.state["chunk_size"]
        tokens = fixture(index, 2, chunk_size)[:-1] + [11111] * 7
        self.warm(tokens, owner)
        ledger = self.ledger(owner)
        cpu = ledger.prefix(tokens, "CPU")
        gpu = ledger.prefix(tokens, "GPU")
        expected_count = 2 * chunk_size // self.state["native_block_size"]
        assert len(cpu) == expected_count, (len(cpu), expected_count)
        assert cpu == gpu[:expected_count], "Independent native GPU/CPU tuples differ"
        previous = len(ledger.records)
        self.local_http(owner, 8080, "/cache/clear", "POST", {"tier": "l1"})
        self.wait_received(owner)
        self.active_barrier(owner)
        self.wait(
            lambda: not self.ledger(owner).prefix(tokens, "CPU"),
            "exact native removals",
        )
        expected = {str(block["hash"]) for block in cpu}
        removed = [
            str(block_hash)
            for record in self.ledger(owner).records[previous:]
            for event in record["payload"][1]
            if event["type"] == "BlockRemoved"
            and event.get("medium") in ("CPU", "CPU_PINNED")
            for block_hash in event["block_hashes"]
            if str(block_hash) in expected
        ]
        assert len(removed) == len(expected) and set(removed) == expected, removed
        write_json(
            self.root / f"T06-{owner}-native-identity.json",
            {"CPU": cpu, "GPU": gpu, "removed": removed},
        )


def test(args: argparse.Namespace) -> int:
    """Execute real scenarios and preserve PASS/FAIL/BLOCKED/LIMITATION statuses."""
    # Third Party
    from capacity_scenarios import run_capacity
    from overhead import run_overhead
    from recovery import (
        idle,
        incarnation,
        overflow,
        reconnect,
        truncated_restart,
        warm_restart,
    )
    from tp_scenarios import run_tp

    args.artifacts.mkdir(parents=True, exist_ok=True)
    try:
        driver = Driver(args)
    except Exception as error:
        trace = traceback.format_exc()
        (args.artifacts / "startup-exception.txt").write_text(trace)
        result = {"scenario": "startup", "status": "FAIL", "reason": repr(error)}
        write_json(args.artifacts / "results.json", [result])
        suite = ET.Element("testsuite", name="mp_kv_events_startup", tests="1")
        case = ET.SubElement(suite, "testcase", name="startup")
        ET.SubElement(case, "failure", message=repr(error))
        ET.ElementTree(suite).write(args.artifacts / "junit.xml", encoding="unicode")
        (args.artifacts / "REPORT.md").write_text("# FAIL\n\n" + trace + "\n")
        return 1
    results: list[dict[str, Any]] = []
    try:
        callbacks = []
        if args.suite == "core":
            callbacks = [
                ("T01", driver.cold),
                ("T05", driver.gpu_survival),
                ("T02", driver.cpu_only),
                ("T03", driver.mixed),
                ("T04", driver.removal),
                ("T06", driver.geometry),
            ]
        elif args.suite == "recovery":
            callbacks = [
                ("T08", lambda owner, index: reconnect(driver, owner, index)),
                ("T10-engine", lambda owner, index: warm_restart(driver, owner, index)),
                ("T10-mp", lambda owner, index: incarnation(driver, owner, index)),
                ("T11-idle", lambda owner, index: idle(driver, owner, index)),
                (
                    "T11-paused",
                    lambda owner, index: idle(driver, owner, index, paused=True),
                ),
            ]
        elif args.suite == "overflow":
            callbacks = [
                ("T09", lambda owner, index: overflow(driver, owner, index)),
                (
                    "T10-truncated",
                    lambda owner, index: truncated_restart(driver, owner, index),
                ),
            ]
        elif args.suite == "capacity":
            results.extend(run_capacity(driver))
        elif args.suite == "tp":
            results.extend(run_tp(driver))
        elif args.suite == "overhead":
            results.append(
                {"scenario": "overhead", "owner": "a", **run_overhead(driver)}
            )
        requested = set(args.cases or [])
        if requested - {name for name, _ in callbacks}:
            raise ValueError(f"Unknown case selection: {requested}")
        for number, (name, scenario) in enumerate(callbacks):
            if requested and name not in requested:
                continue
            for role, owner in enumerate(("a", "b")):
                started = time.monotonic()
                row: dict[str, Any] = {"scenario": name, "owner": owner}
                try:
                    base = (
                        10000
                        if args.suite == "recovery"
                        else 20000
                        if args.suite == "overflow"
                        else 1000
                    )
                    stride = 1000 if args.suite in ("recovery", "overflow") else 200
                    role_stride = 200 if args.suite in ("recovery", "overflow") else 100
                    detail = scenario(
                        owner, base + stride * number + role_stride * role + 1
                    )
                    row.update(
                        detail if isinstance(detail, dict) else {"status": "PASS"}
                    )
                except Exception as error:
                    # A scenario boundary records unexpected oracle/control bugs
                    # while preserving the rest of the requested result matrix.
                    trace = traceback.format_exc()
                    (args.artifacts / f"{name}-{owner}-exception.txt").write_text(trace)
                    row.update(status="FAIL", reason=str(error))
                row["duration_seconds"] = time.monotonic() - started
                results.append(row)
                write_json(args.artifacts / "results.json", results)
    except Exception as error:
        (args.artifacts / "suite-exception.txt").write_text(traceback.format_exc())
        results.append(
            {"scenario": args.suite, "status": "FAIL", "reason": repr(error)}
        )
    finally:
        driver.close()
    write_json(args.artifacts / "results.json", results)
    suite = ET.Element(
        "testsuite", name="mp_kv_events_" + args.suite, tests=str(len(results))
    )
    for result in results:
        case = ET.SubElement(
            suite,
            "testcase",
            name=result.get("scenario", args.suite) + "-" + result.get("owner", "both"),
            time=str(result.get("duration_seconds", 0)),
        )
        if result["status"] in ("BLOCKED", "LIMITATION"):
            ET.SubElement(
                case, "skipped", message=result.get("reason", result["status"])
            )
        elif result["status"] != "PASS":
            ET.SubElement(
                case, "failure", message=result.get("reason", "See JSON evidence")
            )
    ET.ElementTree(suite).write(args.artifacts / "junit.xml", encoding="unicode")
    lines = [
        "# MP KV events evidence",
        "",
        "| Scenario | Owner | Result | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row.get('scenario', args.suite)} | {row.get('owner', 'both')} | "
        f"{row['status']} | "
        f"{row.get('reason', 'See JSON traces')} |"
        for row in results
    )
    (args.artifacts / "REPORT.md").write_text("\n".join(lines) + "\n")
    return (
        0
        if results and all(row["status"] in ("PASS", "LIMITATION") for row in results)
        else 1
    )
