#!/usr/bin/env python3
"""Run LMCache + vLLM on a single GPU and benchmark multiple cache policies."""

from __future__ import annotations

import argparse
import os
import signal
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Iterable

import requests

DEFAULT_POLICIES = ["LRU", "MRU", "SIEVE", "SIEVE_SLRU", "SIEVE_PDG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        default="NousResearch/Llama-2-7b-chat-hf",
        help="Model name or local path passed to vLLM",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=DEFAULT_POLICIES,
        help="Cache policies to benchmark. Values are passed to LMCACHE_CACHE_POLICY.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port for the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host used when polling server readiness",
    )
    parser.add_argument(
        "--dtype",
        default="half",
        help="Value for vLLM --dtype",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Value for vLLM --max-model-len",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Value for vLLM --max-num-seqs",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.9,
        help="Value for vLLM --gpu-memory-utilization",
    )
    parser.add_argument(
        "--swap-space",
        type=int,
        default=4,
        help="Value for vLLM --swap-space (GB)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional quantization argument passed to vLLM",
    )
    parser.add_argument(
        "--bench-random-input-len",
        type=int,
        default=1024,
        help="Random input length used by vLLM benchmark",
    )
    parser.add_argument(
        "--benchmark-type",
        choices=["serve", "long_doc_qa"],
        default="serve",
        help="Which benchmark harness to run after the server is ready",
    )
    parser.add_argument(
        "--bench-random-output-len",
        type=int,
        default=128,
        help="Random output length used by vLLM bench serve (ignored for long_doc_qa)",
    )
    parser.add_argument(
        "--bench-dataset",
        default=None,
        help="Optional dataset name (e.g. 'sharegpt') passed to vLLM bench instead of random prompts",
    )
    parser.add_argument(
        "--bench-dataset-path",
        default=None,
        help="Optional dataset path used when --bench-dataset is provided",
    )
    parser.add_argument(
        "--long-doc-num-documents",
        type=int,
        default=51,
        help="Number of documents for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-document-length",
        type=int,
        default=10000,
        help="Document length (tokens) for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-output-len",
        type=int,
        default=100,
        help="Output length for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-repeat-count",
        type=int,
        default=1,
        help="Repeat count for prompts in long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-repeat-mode",
        choices=["random", "tile", "interleave"],
        default="tile",
        help="Repeat mode for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-max-inflight",
        type=int,
        default=4,
        help="Max inflight requests for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-hit-miss-ratio",
        default=None,
        help="Optional hit:miss ratio (e.g. 3:1) for long_doc_qa benchmark",
    )
    parser.add_argument(
        "--long-doc-extra-arg",
        dest="long_doc_extra_args",
        action="append",
        default=[],
        help="Additional raw arguments to pass through to long_doc_qa.py",
    )
    parser.add_argument(
        "--long-doc-api-mode",
        choices=["auto", "chat", "completions"],
        default="auto",
        help=(
            "API mode used by long_doc_qa benchmark. "
            "'auto' selects completions for models without a chat template."
        ),
    )
    parser.add_argument(
        "--bench-num-prompts",
        type=int,
        default=30,
        help="Number of prompts used in the benchmark",
    )
    parser.add_argument(
        "--bench-request-rate",
        type=float,
        default=1.0,
        help="Poisson arrival rate used in throughput test",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=5.0,
        help="Time to sleep after the server reports ready before benchmarking",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for the vLLM server to become ready",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./benchmark_logs"),
        help="Directory for per-policy stdout/stderr logs",
    )
    parser.add_argument(
        "--extra-vllm-arg",
        dest="extra_vllm_args",
        action="append",
        default=[],
        help="Additional raw arguments to append to the vLLM server command",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    return parser.parse_args()


def ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_vllm_server(
    policy: str, args: argparse.Namespace, env: dict[str, str], log_dir: Path
) -> tuple[subprocess.Popen[str], Path, Path]:
    stdout_path = log_dir / f"server_{policy}.stdout.log"
    stderr_path = log_dir / f"server_{policy}.stderr.log"
    server_cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--tensor-parallel-size",
        "1",
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_mem_util),
        "--swap-space",
        str(args.swap_space),
        "--port",
        str(args.port),
    ]
    if args.quantization:
        server_cmd.extend(["--quantization", args.quantization])
    if args.extra_vllm_args:
        server_cmd.extend(args.extra_vllm_args)

    env = env.copy()
    env["LMCACHE_CACHE_POLICY"] = policy

    if args.dry_run:
        print("Server command:", " ".join(server_cmd))
        return None, stdout_path, stderr_path  # type: ignore

    stdout_file = stdout_path.open("w", encoding="utf-8")
    stderr_file = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        server_cmd,
        stdout=stdout_file,
        stderr=stderr_file,
        text=True,
        env=env,
    )
    print(
        f"  > vLLM server spawned for policy {policy}. Logs: {stdout_path} / {stderr_path}"
    )
    return proc, stdout_path, stderr_path


def wait_for_server(host: str, port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    url = f"http://{host}:{port}/health"
    attempts = 0
    while time.time() < deadline:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                print(f"  > Server healthcheck succeeded (attempt {attempts + 1}).")
                return True
        except requests.RequestException:
            pass
        attempts += 1
        print(
            f"  > Waiting for server on {host}:{port} (attempt {attempts},"
            f" elapsed {attempts} s)..."
        )
        time.sleep(1)
    return False


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def resolve_vllm_cli(env: dict[str, str]) -> str:
    candidate = shutil.which("vllm", path=env.get("PATH"))
    if candidate:
        return candidate
    user_cli = Path.home() / ".local" / "bin" / "vllm"
    if user_cli.exists():
        return str(user_cli)
    raise RuntimeError(
        "Could not find the 'vllm' CLI. Ensure it is installed and on PATH."
    )


def run_long_doc_benchmark(
    args: argparse.Namespace, env: dict[str, str], policy: str, log_dir: Path
) -> tuple[int, str]:
    script_path = Path(__file__).resolve().parent.parent / "benchmarks" / "long_doc_qa" / "long_doc_qa.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--num-documents",
        str(args.long_doc_num_documents),
        "--document-length",
        str(args.long_doc_document_length),
        "--output-len",
        str(args.long_doc_output_len),
        "--repeat-count",
        str(args.long_doc_repeat_count),
        "--repeat-mode",
        args.long_doc_repeat_mode,
        "--max-inflight-requests",
        str(args.long_doc_max_inflight),
    ]
    if args.long_doc_hit_miss_ratio:
        cmd.extend(["--hit-miss-ratio", args.long_doc_hit_miss_ratio])
    if args.long_doc_extra_args:
        cmd.extend(args.long_doc_extra_args)

    use_completions = False
    if args.long_doc_api_mode == "completions":
        use_completions = True
    elif args.long_doc_api_mode == "auto":
        model_lower = args.model.lower()
        if model_lower.startswith("facebook/opt") or "opt-" in model_lower:
            use_completions = True

    already_requested = any(arg == "--completions" for arg in cmd)
    if use_completions and not already_requested:
        cmd.append("--completions")

    if args.dry_run:
        print("Benchmark command (long_doc_qa):", " ".join(cmd))
        return 0, ""

    bench_log = log_dir / f"benchmark_{policy}.log"
    print(
        "  > Running long_doc_qa benchmark:",
        " ".join(cmd),
        f"(logging to {bench_log})",
    )
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output, _ = proc.communicate()
    bench_log.write_text(output, encoding="utf-8")
    print(f"  > Benchmark completed with return code {proc.returncode}")
    return proc.returncode, output


def run_benchmark(
    args: argparse.Namespace, env: dict[str, str], policy: str, log_dir: Path
) -> tuple[int, str]:
    if args.benchmark_type == "long_doc_qa":
        return run_long_doc_benchmark(args, env, policy, log_dir)

    vllm_cli = resolve_vllm_cli(env)
    bench_cmd = [
        vllm_cli,
        "bench",
        "serve",
        "--port",
        str(args.port),
        "--model",
        args.model,
    ]
    if args.bench_dataset:
        bench_cmd.extend(["--dataset-name", args.bench_dataset])
        if args.bench_dataset_path:
            bench_cmd.extend(["--dataset-path", args.bench_dataset_path])
    else:
        bench_cmd.extend(
            [
                "--dataset-name",
                "random",
                "--random-input-len",
                str(args.bench_random_input_len),
                "--random-output-len",
                str(args.bench_random_output_len),
            ]
        )
    bench_cmd.extend(
        [
            "--num-prompts",
            str(args.bench_num_prompts),
            "--request-rate",
            str(args.bench_request_rate),
            "--ignore-eos",
        ]
    )
    if args.dry_run:
        print("Benchmark command:", " ".join(bench_cmd))
        return 0, ""

    bench_log = log_dir / f"benchmark_{policy}.log"
    print(
        "  > Running benchmark:",
        " ".join(bench_cmd),
        f"(logging to {bench_log})",
    )
    proc = subprocess.Popen(
        bench_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    output, _ = proc.communicate()
    bench_log.write_text(output, encoding="utf-8")
    print(f"  > Benchmark completed with return code {proc.returncode}")
    return proc.returncode, output


def extract_summary(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "(no output)"
    last_block = lines[-40:]
    return "\n".join(last_block)


def main() -> None:
    args = parse_args()
    ensure_log_dir(args.log_dir)

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

    summaries: list[str] = []

    for policy in args.policies:
        print(f"=== Policy: {policy} ===")
        if args.dry_run:
            run_vllm_server(policy, args, env, args.log_dir)
            run_benchmark(args, env, policy, args.log_dir)
            continue

        server_proc, stdout_path, stderr_path = run_vllm_server(policy, args, env, args.log_dir)
        try:
            print(
                f"Waiting for vLLM server to become ready (policy {policy}) "
                f"with timeout {args.timeout}s"
            )
            if not wait_for_server(args.host, args.port, args.timeout):
                stop_process(server_proc)
                raise RuntimeError(
                    f"Server did not become ready for policy {policy}."
                    f" Check {stdout_path} / {stderr_path}."
                )
            print(
                f"Server ready on {args.host}:{args.port};"
                f" sleeping {args.warmup_seconds:.1f}s before benchmarking"
            )
            time.sleep(args.warmup_seconds)
            code, bench_output = run_benchmark(args, env, policy, args.log_dir)
            summary = extract_summary(bench_output)
            summaries.append(
                textwrap.dedent(
                    f"""
                    Policy: {policy}
                    Return code: {code}
                    Log file: {args.log_dir / f'benchmark_{policy}.log'}
                    --- Tail ---
                    {summary}
                    """
                ).strip()
            )
            print(summary)
        finally:
            stop_process(server_proc)

    if summaries and not args.dry_run:
        summary_path = args.log_dir / "summary.txt"
        summary_path.write_text("\n\n".join(summaries), encoding="utf-8")
        print(f"\n=== Summary saved to {summary_path} ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)
