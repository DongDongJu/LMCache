# SPDX-License-Identifier: Apache-2.0
"""Build a tracked OCI update over an already unpacked test image."""

# Standard
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read_output(argv: list[str]) -> str:
    """Run an inspection command without invoking a shell."""
    return subprocess.check_output(argv, text=True).strip()


def encode(value: object) -> bytes:
    """Encode deterministic OCI JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    """Compute an OCI content digest."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def add_file(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    """Append a deterministic regular file to a task-owned tar."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 1
    archive.addfile(info, io.BytesIO(data))


def read_json(archive: tarfile.TarFile, name: str) -> dict:
    """Read required OCI metadata from an archive."""
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"Missing OCI metadata: {name}")
    return json.load(stream)


def verify_dynamo_build(wheel: Path, receipt: Path, checkout: Path, lock: dict) -> dict:
    """Match a built wheel to the committed companion tree and pinned patch."""
    result = json.loads(receipt.read_text())
    tree = read_output(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"])
    if tree != lock["dynamo"]["bridge_tree"]:
        raise ValueError("Dynamo checkout differs from its pinned patch")
    if result["source_tree"] != tree:
        raise ValueError("Dynamo wheel build and final source trees differ")
    if read_output(["git", "-C", str(checkout), "diff", "HEAD", "--name-only"]):
        raise ValueError("Dynamo has uncommitted tracked changes")
    with wheel.open("rb") as stream:
        wheel_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if wheel_sha != result["wheel_sha256"]:
        raise ValueError("Dynamo wheel differs from its build receipt")
    patch = HERE / lock["dynamo"]["bridge_patch"]
    if (
        hashlib.sha256(patch.read_bytes()).hexdigest()
        != lock["dynamo"]["bridge_patch_sha256"]
    ):
        raise ValueError("Dynamo companion patch differs from lock")
    return result


def build(
    base_image: str,
    base_archive: Path,
    engine: str,
    output: Path,
    base_state: Path | None = None,
    dynamo_wheel: Path | None = None,
    dynamo_build_receipt: Path | None = None,
    dynamo_source: Path | None = None,
) -> dict:
    """Keep engine binaries and optionally replace the verified Dynamo runtime.

    The sparse OCI archive deliberately omits the immutable base layer. Import it
    only on a node where that exact base snapshot is already unpacked. It is not
    a standalone image archive; the report records its required base archive.
    """
    if not base_image.startswith(("mp-kv-events-", "docker.io/library/mp-kv-events-")):
        raise ValueError("Only task-owned base images are accepted")
    lock = json.loads((HERE / "versions.lock.yaml").read_text())
    if bool(dynamo_wheel) != bool(dynamo_build_receipt):
        raise ValueError("Dynamo wheel and build receipt must be supplied together")
    runtime_build = None
    if dynamo_wheel is not None:
        if engine != "sglang" or dynamo_build_receipt is None:
            raise ValueError("The Dynamo replay overlay applies only to SGLang")
        dynamo_source = dynamo_source or REPO.parent / "dynamo-pinned"
        runtime_build = verify_dynamo_build(
            dynamo_wheel, dynamo_build_receipt, dynamo_source, lock
        )
    pin = lock[engine]
    engine_root = REPO.parent / engine
    expected_tree = pin.get("bridge_tree")
    reference = "HEAD^{tree}" if expected_tree else "HEAD"
    if read_output(["git", "-C", str(engine_root), "rev-parse", reference]) != (
        expected_tree or pin["bridge_revision"]
    ):
        raise ValueError("Engine bridge checkout differs from lock")
    if read_output(
        ["git", "-C", str(engine_root), "status", "--porcelain", "--untracked-files=no"]
    ):
        raise ValueError("Engine bridge has uncommitted tracked changes")
    patch = HERE / pin["bridge_patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != pin["bridge_patch_sha256"]:
        raise ValueError("Bridge patch differs from lock")
    inspection = json.loads(read_output(["docker", "image", "inspect", base_image]))[0]
    parent_state = json.loads(base_state.read_text()) if base_state else {}
    if parent_state and parent_state["image_id"] != inspection["Id"]:
        raise ValueError("Base state and Docker runtime have different configs")
    with tarfile.open(base_archive) as archive:
        index = read_json(archive, "index.json")
        descriptor = index["manifests"][0]
        manifest = read_json(
            archive, "blobs/sha256/" + descriptor["digest"].split(":")[1]
        )
        if manifest["config"]["digest"] != inspection["Id"]:
            raise ValueError("Base archive and Docker runtime have different configs")
        config = read_json(
            archive, "blobs/sha256/" + manifest["config"]["digest"].split(":")[1]
        )
    if config["rootfs"]["diff_ids"] != inspection["RootFS"]["Layers"]:
        raise ValueError("Base archive and Docker runtime have different filesystems")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="python-overlay-", dir=output) as temp:
        context = Path(temp)
        inputs = {}
        checkouts = [("lmcache", REPO), ("engine", engine_root)]
        if dynamo_wheel is not None and dynamo_source is not None:
            checkouts.append(("dynamo", dynamo_source))
        for name, checkout in checkouts:
            paths = read_output(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "ls-files",
                    "-z",
                    "--cached",
                    "--others",
                    "--exclude-standard",
                ]
            ).split("\0")
            files = {}
            for relative in sorted(set(paths) - {""}):
                source_path = checkout / relative
                if not source_path.is_file():
                    continue
                target = context / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                data = source_path.read_bytes()
                target.write_bytes(data)
                files[relative] = hashlib.sha256(data).hexdigest()
            inputs[name] = {
                "head": read_output(["git", "-C", str(checkout), "rev-parse", "HEAD"]),
                "files": files,
            }
        source = {
            "inputs": inputs,
            "lock": lock,
            "base_image_id": inspection["Id"],
        }
        if runtime_build is not None and dynamo_wheel is not None:
            source["dynamo_runtime_build"] = runtime_build
            shutil.copyfile(dynamo_wheel, context / "dynamo-runtime.whl")
            (context / "dynamo-build.json").write_bytes(encode(runtime_build))
        source["sha256"] = hashlib.sha256(encode(source)).hexdigest()
        (context / "source.json").write_bytes(encode(source))
        staged = context / "staged"
        staged.mkdir(mode=0o777)
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--mount",
                f"type=bind,source={context},target=/overlay,readonly",
                "--mount",
                f"type=bind,source={staged},target=/staged",
                "--entrypoint",
                "python3",
                base_image,
                "/overlay/lmcache/tests/e2e/mp_kv_events/stage_python_overlay.py",
                "--context",
                "/overlay",
                "--output",
                "/staged",
                "--engine",
                engine,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if dynamo_wheel is not None:
            subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--user",
                    f"{os.getuid()}:{os.getgid()}",
                    "--mount",
                    f"type=bind,source={context},target=/overlay,readonly",
                    "--mount",
                    f"type=bind,source={staged},target=/staged",
                    "--entrypoint",
                    "python3",
                    base_image,
                    "/overlay/lmcache/tests/e2e/mp_kv_events/stage_dynamo_overlay.py",
                    "--context",
                    "/overlay",
                    "--output",
                    "/staged",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        receipt = json.loads(
            (staged / "opt/mp-kv-events/python-overlay.json").read_text()
        )
        staged_native = {
            path.relative_to(staged).as_posix() for path in staged.rglob("*.so")
        }
        allowed_native = set()
        if dynamo_wheel is not None:
            runtime_receipt = json.loads(
                (staged / "opt/mp-kv-events/dynamo-overlay.json").read_text()
            )
            allowed_native.add(runtime_receipt["native_change"]["path"].lstrip("/"))
        if staged_native != allowed_native:
            raise ValueError(f"Unexpected native files in overlay: {staged_native}")
        layer_buffer = io.BytesIO()
        with tarfile.open(fileobj=layer_buffer, mode="w") as archive:
            for path in sorted(staged.rglob("*")):
                if path.is_file():
                    add_file(
                        archive, path.relative_to(staged).as_posix(), path.read_bytes()
                    )
        layer = layer_buffer.getvalue()
        compressed = gzip.compress(layer, mtime=0)
        base_diff_ids = list(config["rootfs"]["diff_ids"])
        config["rootfs"]["diff_ids"].append(digest(layer))
        kind = "runtime" if dynamo_wheel is not None else "python"
        config.setdefault("history", []).append(
            {"created_by": f"mp-kv-events tracked {kind} overlay " + source["sha256"]}
        )
        config_bytes = encode(config)
        manifest = copy.deepcopy(manifest)
        manifest["config"].update(digest=digest(config_bytes), size=len(config_bytes))
        manifest["layers"].append(
            {
                "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                "digest": digest(compressed),
                "size": len(compressed),
            }
        )
        manifest_bytes = encode(manifest)
        image = (
            f"docker.io/library/mp-kv-events-{engine}:{source['sha256'][:20]}-{kind}"
        )
        descriptor.update(digest=digest(manifest_bytes), size=len(manifest_bytes))
        descriptor["annotations"] = {
            "io.containerd.image.name": image,
            "org.opencontainers.image.ref.name": image.rsplit(":", 1)[1],
        }
        archive_path = output / (source["sha256"][:20] + ".incremental.oci.tar")
        with tarfile.open(archive_path, "w") as archive:
            add_file(archive, "oci-layout", encode({"imageLayoutVersion": "1.0.0"}))
            add_file(archive, "index.json", encode(index))
            for data in (config_bytes, manifest_bytes, compressed):
                add_file(archive, "blobs/sha256/" + digest(data).split(":")[1], data)
        report = {
            "image": image,
            "image_id": digest(config_bytes),
            "source": source,
            "archive": str(archive_path.resolve()),
            "base_archive": str(base_archive.resolve()),
            "base_image": base_image,
            "base_image_id": inspection["Id"],
            "required_base_diff_ids": base_diff_ids,
            "overlay": receipt,
            "layer_bytes": len(layer),
            "compressed_bytes": len(compressed),
        }
        if "external_runtime" in parent_state:
            report["external_runtime"] = parent_state["external_runtime"]
        if "base_compact" in parent_state:
            report["base_compact"] = parent_state["base_compact"]
        elif "compression" in parent_state:
            report["base_compact"] = {
                key: parent_state[key]
                for key in ("image", "image_id", "compression", "compaction")
                if key in parent_state
            }
        if dynamo_wheel is not None:
            report["dynamo_overlay"] = json.loads(
                (staged / "opt/mp-kv-events/dynamo-overlay.json").read_text()
            )
        report_path = output / (source["sha256"][:20] + ".json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ("image", "image_id", "archive", "compressed_bytes")
                }
            )
        )
        print(str(report_path))
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-state", type=Path)
    parser.add_argument("--dynamo-wheel", type=Path)
    parser.add_argument("--dynamo-build-receipt", type=Path)
    parser.add_argument("--dynamo-source", type=Path)
    args = parser.parse_args()
    build(
        args.base_image,
        args.base_archive,
        args.engine,
        args.output,
        args.base_state,
        args.dynamo_wheel,
        args.dynamo_build_receipt,
        args.dynamo_source,
    )
