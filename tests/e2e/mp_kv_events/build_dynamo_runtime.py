# SPDX-License-Identifier: Apache-2.0
"""Build the pinned Dynamo companion wheel in an isolated large-disk workspace."""

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FEATURES = (
    "kv-indexer,slot-tracker,select-service,mm-routing,"
    "aic-forward-pass,request-trace-s3"
)


def build(source: Path, work: Path, rustup: Path, jobs: int) -> dict:
    """Require the committed patch, build a release wheel, and record provenance."""
    source, work, rustup = source.resolve(), work.resolve(), rustup.resolve()
    if sys.version_info[:2] != (3, 12) or jobs < 1:
        raise ValueError("Use Python 3.12 and a positive build job count")
    lock = json.loads((HERE / "versions.lock.yaml").read_text())["dynamo"]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args], text=True
        ).strip()

    tree, revision = git("rev-parse", "HEAD^{tree}"), git("rev-parse", "HEAD")
    if tree != lock["bridge_tree"] or git("diff", "HEAD", "--name-only"):
        raise ValueError("Dynamo source must match the committed companion tree")
    patch = HERE / lock["bridge_patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != lock["bridge_patch_sha256"]:
        raise ValueError("Companion patch differs from the lock")
    for tool in ("gcc", "g++", "clang-18", "pkg-config", "protoc", "cmake", "ninja"):
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required build tool is unavailable: {tool}")
    if not rustup.is_file():
        raise FileNotFoundError("Pass the installed rustup executable with --rustup")
    work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    paths = {
        "CARGO_HOME": "cargo",
        "RUSTUP_HOME": "rustup",
        "CARGO_TARGET_DIR": "target",
        "TMPDIR": "tmp",
        "PIP_CACHE_DIR": "pip-cache",
    }
    for name, directory in paths.items():
        path = work / directory
        path.mkdir(exist_ok=True)
        env[name] = str(path)
    venv = work / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, env=env)
    python = venv / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "maturin==1.15.0", "patchelf==0.19.1.0"],
        check=True,
        env=env,
    )
    subprocess.run(
        [str(rustup), "set", "auto-self-update", "disable"], check=True, env=env
    )
    subprocess.run(
        [str(rustup), "toolchain", "install", "1.96.1", "--profile", "minimal"],
        check=True,
        env=env,
    )
    toolchain = work / "rustup/toolchains/1.96.1-x86_64-unknown-linux-gnu/bin"
    env.update(
        {
            "PATH": f"{toolchain}:{venv / 'bin'}:/usr/local/cuda/bin:{env['PATH']}",
            "LIBCLANG_PATH": "/usr/lib/llvm-18/lib",
            "CUDA_HOME": "/usr/local/cuda",
            "RUSTUP_TOOLCHAIN": "1.96.1",
            "PYO3_PYTHON": str(python),
            "CARGO_BUILD_JOBS": str(jobs),
        }
    )
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    if list(wheels.glob("*.whl")):
        raise FileExistsError(
            "Use a fresh wheel output directory to avoid stale artifacts"
        )
    argv = [
        str(venv / "bin/maturin"),
        "build",
        "--release",
        "--locked",
        "--features",
        FEATURES,
        "--out",
        str(wheels),
    ]
    logfile = work / "build.log"
    print(f"Building Dynamo; full output: {logfile}", flush=True)
    with logfile.open("w") as output:
        subprocess.run(
            argv,
            cwd=source / "lib/bindings/python",
            env=env,
            check=True,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    if git("rev-parse", "HEAD^{tree}") != tree or git("diff", "HEAD", "--name-only"):
        raise ValueError("Dynamo source changed during the build")
    matches = list(wheels.glob("ai_dynamo_runtime-*.whl"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one built runtime wheel")
    wheel = matches[0]
    with wheel.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt = {
        "source_revision": revision,
        "source_tree": tree,
        "wheel_path": str(wheel),
        "wheel_sha256": digest,
        "build_command": argv,
        "build_cwd": str(source / "lib/bindings/python"),
        "build_environment": {
            key: env[key]
            for key in (
                *paths,
                "PATH",
                "CUDA_HOME",
                "LIBCLANG_PATH",
                "PYO3_PYTHON",
                "CARGO_BUILD_JOBS",
                "RUSTUP_TOOLCHAIN",
            )
        },
        "toolchain": {
            tool: subprocess.check_output(
                [tool, "--version"], env=env, text=True
            ).strip()
            for tool in ("rustc", "cargo", "maturin", "gcc", "protoc")
        },
    }
    (work / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--rustup",
        type=Path,
        default=Path(shutil.which("rustup") or "/nonexistent/rustup"),
    )
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    build(args.source, args.work, args.rustup, args.jobs)
