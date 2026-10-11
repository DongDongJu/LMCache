# SPDX-License-Identifier: Apache-2.0
"""Run isolated MP trials, or a matched original/disabled/worker comparison."""

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

# First Party
from benchmarks.gds_optimization.process_sampler import process_identity, sample_cpu
from benchmarks.gds_optimization.slab import SlabLease, save_json
from benchmarks.gds_optimization.support import server_owns_slab, wait_ready

ROOT = Path(__file__).resolve().parent
SLAB_BYTES = 16 << 30


def capture(argv: list[str], cwd: Path | None = None) -> dict[str, object]:
    """Retain a bounded metadata command's output and exit code without shell use."""
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=30)
    return {
        "argv": argv,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def source_identity(source: Path) -> dict[str, object]:
    """Hash the selected checkout's Python/native files and record Git state."""
    return {
        "source": str(source),
        "commit": capture(["git", "rev-parse", "HEAD"], source),
        "changes": capture(["git", "status", "--porcelain=v1"], source),
        "source_sha256": {
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((source / "lmcache").rglob("*"))
            if path.is_file() and path.suffix in {".py", ".so"}
        },
    }


def build_plan(args: argparse.Namespace) -> list[dict[str, object]]:
    """Create sequential repeated arms with identical geometry and slab identity.

    Original source omits new CLI flags entirely. Candidate zero explicitly
    disables the optional pool. Repetitions rotate arm order to expose drift.
    """
    arms: list[tuple[str, Path, int | None]] = []
    if args.baseline_source is not None:
        arms.append(("original", args.baseline_source, None))
    workers = (
        args.workers
        if args.workers is not None
        else ([0, 1, 4] if args.baseline_source is not None else [4])
    )
    arms.extend((f"workers{count}", args.source, count) for count in workers)
    plan: list[dict[str, object]] = []
    for repetition in range(args.repetitions):
        ordered = arms[repetition % len(arms) :] + arms[: repetition % len(arms)]
        for name, source, count in ordered:
            plan.append(
                {
                    "name": f"r{repetition}-{name}",
                    "source": str(source),
                    "workers": count,
                    "repetition": repetition,
                }
            )
    return plan


def stop_owned(process: subprocess.Popen[bytes] | None) -> None:
    """Stop only a child process group created by this controller, then reap it."""
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def run_trial(
    args: argparse.Namespace, arm: dict[str, object], lease: SlabLease
) -> None:
    """Run one owned server/client pair; record failures and always drain children.

    The lease remains held by the campaign until all servers exit. Instrumented
    trials are labelled diagnostics and must not be pooled with steady rates.
    """
    source = Path(str(arm["source"]))
    output = args.output / str(arm["name"])
    output.mkdir(exist_ok=False)
    lease.validate()
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    if any(row.split(",", 1)[0].strip() == args.gpu_uuid for row in busy.splitlines()):
        raise RuntimeError("Selected GPU has an active compute process")
    for port in (args.rpc_port, args.http_port):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    env = dict(os.environ)
    config = json.loads(args.cufile_config.read_text())
    config.setdefault("logging", {})["dir"] = str(output)
    save_json(output / "cufile.json", config)
    overrides = {
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "PYTHONPATH": os.pathsep.join((str(source), str(ROOT.parents[1]))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LMCACHE_DISABLE_BANNER": "1",
        "LMCACHE_LOG_LEVEL": "WARNING",
        "CUFILE_ENV_PATH_JSON": str(output / "cufile.json"),
        "GDS_BENCH_SOURCE": str(source),
        "GDS_BENCH_SLAB_IDENTITY": json.dumps(lease.identity()),
    }
    if args.cuda_lib_dir is not None:
        overrides["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(None, (str(args.cuda_lib_dir), env.get("LD_LIBRARY_PATH", "")))
        )
    if args.instrument:
        overrides["GDS_DIAGNOSTIC_EVENTS"] = str(output / "server-events.jsonl")
    else:
        env.pop("GDS_DIAGNOSTIC_EVENTS", None)
    env.update(overrides)
    affinity = (
        []
        if args.numa_node is None
        else [
            "numactl",
            f"--cpunodebind={args.numa_node}",
            f"--membind={args.numa_node}",
        ]
    )
    prefix = affinity + [sys.executable]
    server_argv = prefix + [
        "-m",
        "benchmarks.gds_optimization.server_entry",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.rpc_port),
        "--http-host",
        "127.0.0.1",
        "--http-port",
        str(args.http_port),
        "--chunk-size",
        "256",
        "--supported-transfer-mode",
        "lmcache_driven",
        "--gds-l1-path",
        str(lease.path.parent),
        "--gds-l1-backend",
        "cufile",
        "--gds-l1-use-direct-io",
        "--l1-size-gb",
        "16",
        "--no-l1-use-lazy",
        "--disable-observability",
        "--eviction-policy",
        "LRU",
    ]
    if arm["workers"] is not None:
        server_argv += [
            "--gds-l1-io-workers",
            str(arm["workers"]),
            "--gds-l1-max-inflight-bytes",
            str(args.max_inflight_bytes),
        ]
    client_argv = prefix + [
        "-m",
        "benchmarks.gds_optimization.replay",
        "--rpc-url",
        f"tcp://127.0.0.1:{args.rpc_port}",
        "--http-url",
        f"http://127.0.0.1:{args.http_port}",
        "--gpu",
        "0",
        "--num-tokens",
        "4096",
        "--num-layers",
        str(args.layers),
        "--num-heads",
        str(args.heads),
        "--num-blocks",
        "1024",
        "--requests",
        str(args.requests),
        "--warmup",
        str(args.warmup),
        "--duration",
        str(args.duration),
        "--output",
        str(output / "result.json"),
        "--samples",
        str(output / "samples.jsonl"),
    ]
    record: dict[str, object] = {
        **arm,
        **source_identity(source),
        "status": "running",
        "controller": process_identity(os.getpid()),
        "slab": lease.identity(),
        "environment_overrides": overrides,
        "inherited_runtime_environment": {
            name: env.get(name)
            for name in ("CUFILE_FORCE_COMPAT_MODE", "LD_LIBRARY_PATH")
        },
        "diagnostic": args.instrument,
        "numa_node": args.numa_node,
        "geometry": {
            "tokens": 4096,
            "chunks_per_request": 16,
            "chunk_bytes": 2 * args.layers * 256 * args.heads * 128 * 2,
            "layers": args.layers,
            "heads": args.heads,
            "dtype": "float16",
            "layout": "NHD",
            "tp_size": 1,
            "block_size": 16,
            "num_blocks": 1024,
            "slab_bytes": SLAB_BYTES,
            "gpu_cache_bytes": 2 * args.layers * 1024 * 16 * args.heads * 128 * 2,
        },
        "started_wall_ns": time.time_ns(),
        "server_argv": server_argv,
        "client_argv": client_argv,
        "path_verdict": "not established by O_DIRECT",
    }
    server = client = None
    save_json(output / "manifest.json", record)
    try:
        with (output / "server.log").open("wb") as log:
            server = subprocess.Popen(
                server_argv,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lease.fd,),
            )
        record["server"] = process_identity(server.pid)
        save_json(output / "manifest.json", record)
        wait_ready(server, f"http://127.0.0.1:{args.http_port}", args.timeout)
        record["slab_ownership"] = server_owns_slab(server, lease.path, SLAB_BYTES)
        with (output / "client.log").open("wb") as log:
            client = subprocess.Popen(
                client_argv,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        record["client"] = process_identity(client.pid)
        save_json(output / "manifest.json", record)
        save_json(
            output / "commands.json",
            {"rank": 0, "server_pid": server.pid, "client_pid": client.pid},
        )
        identities: dict[str, dict[str, object]] = {}
        deadline = time.monotonic() + args.timeout
        with (output / "process-counters.jsonl").open("x") as counters:
            while client.poll() is None:
                counters.write(
                    json.dumps(sample_cpu(output, os.getpid(), identities)) + "\n"
                )
                counters.flush()
                if server.poll() is not None:
                    raise RuntimeError(
                        f"Server exited during replay: {server.returncode}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError("Owned client exceeded the trial timeout")
                time.sleep(1)
        record["client_returncode"] = client.returncode
        if client.returncode:
            raise RuntimeError(
                f"Client returned {client.returncode}; inspect retained logs"
            )
        result = json.loads((output / "result.json").read_text())
        if result["correctness"] != "passed" or result["errors"]:
            raise RuntimeError("Replay did not report verified successful completion")
        lease.validate()
        if server_owns_slab(server, lease.path, SLAB_BYTES) != record["slab_ownership"]:
            raise RuntimeError("Server slab ownership changed during replay")
        record["status"] = "passed"
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        raise
    finally:
        stop_owned(client)
        stop_owned(server)
        record.update(
            server_returncode=None if server is None else server.returncode,
            ended_wall_ns=time.time_ns(),
        )
        save_json(output / "manifest.json", record)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable paths/device selection; reject geometry exceeding the slab."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Candidate checkout")
    parser.add_argument(
        "--baseline-source", type=Path, help="Original checkout; omits new flags"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh artifact directory"
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="Dedicated existing filesystem directory",
    )
    parser.add_argument(
        "--reuse-slab", type=Path, help="Previously prepared package-owned slab"
    )
    parser.add_argument(
        "--gpu-uuid", required=True, help="Stable physical GPU-... UUID"
    )
    parser.add_argument("--cufile-config", type=Path, required=True)
    parser.add_argument("--cuda-lib-dir", type=Path)
    parser.add_argument(
        "--numa-node",
        type=int,
        help="Explicit CPU/memory numactl node; omitted means unbound",
    )
    parser.add_argument("--workers", type=int, nargs="+", choices=(0, 1, 4))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-inflight-bytes", type=int, default=128 << 20)
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rpc-port", type=int, default=25555)
    parser.add_argument("--http-port", type=int, default=25880)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="Diagnostic spans; exclude from steady rates",
    )
    args = parser.parse_args(argv)
    for name in (
        "source",
        "baseline_source",
        "output",
        "scratch_root",
        "cufile_config",
        "cuda_lib_dir",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (
        min(
            args.repetitions,
            args.requests,
            args.layers,
            args.heads,
            args.max_inflight_bytes,
            args.duration,
            args.timeout,
        )
        <= 0
        or args.warmup < 0
    ):
        parser.error("Positive counts/durations and nonnegative warmup are required")
    payload = 2 * args.layers * 4096 * args.heads * 128 * 2
    if max(16, args.warmup) * payload > SLAB_BYTES:
        parser.error("Payloads between owned cache clears exceed the 16 GiB slab")
    if not args.gpu_uuid.startswith("GPU-"):
        parser.error("Select a physical GPU by its stable GPU-... UUID")
    if args.rpc_port == args.http_port or not all(
        0 < p < 65536 for p in (args.rpc_port, args.http_port)
    ):
        parser.error("RPC and HTTP ports must be distinct valid ports")
    if args.workers is not None and len(set(args.workers)) != len(args.workers):
        parser.error("Worker arms must be unique")
    return args


def main() -> None:
    """Create one owned allocation, run the requested plan, retain all artifacts."""
    args = parse_args()
    if not args.scratch_root.is_dir():
        raise ValueError("Create the dedicated scratch root before running")
    for source in filter(None, (args.source, args.baseline_source)):
        if not (source / "lmcache/cli/main.py").is_file():
            raise ValueError(f"Not an LMCache source checkout: {source}")
    args.output.mkdir(parents=True, exist_ok=False)
    available = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if args.gpu_uuid not in [value.strip() for value in available]:
        raise ValueError("Selected physical GPU UUID is absent")
    plan = build_plan(args)
    completed: list[object] = []
    campaign: dict[str, object] = {
        "status": "running",
        "plan": plan,
        "started_wall_ns": time.time_ns(),
        "helper_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ROOT.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".json"}
        },
        "config_sha256": hashlib.sha256(args.cufile_config.read_bytes()).hexdigest(),
        "gpu_uuid": args.gpu_uuid,
        "completed": completed,
    }
    save_json(
        args.output / "inventory.json",
        {
            "gpu": capture(["nvidia-smi", "-q", "-x"]),
            "topology": capture(["nvidia-smi", "topo", "-m"]),
            "mount": capture(["findmnt", "--json", "--target", str(args.scratch_root)]),
            "storage": capture(
                ["lsblk", "--json", "-o", "NAME,TYPE,SIZE,MODEL,SERIAL,MOUNTPOINTS"]
            ),
            "kernel": capture(["uname", "-a"]),
        },
    )
    lease = None
    try:
        lease = (
            SlabLease.create(
                args.scratch_root / args.output.name,
                SLAB_BYTES,
                args.output / "preparation.json",
            )
            if args.reuse_slab is None
            else SlabLease.open_existing(args.reuse_slab, args.scratch_root, SLAB_BYTES)
        )
        campaign["slab"] = lease.identity()
        save_json(args.output / "campaign.json", campaign)
        for arm in plan:
            run_trial(args, arm, lease)
            completed.append(arm["name"])
            save_json(args.output / "campaign.json", campaign)
        campaign["status"] = "passed"
    except BaseException as error:
        campaign.update(status="failed", error=repr(error))
        raise
    finally:
        if lease is not None:
            lease.close()
        campaign["ended_wall_ns"] = time.time_ns()
        save_json(args.output / "campaign.json", campaign)


if __name__ == "__main__":
    main()
