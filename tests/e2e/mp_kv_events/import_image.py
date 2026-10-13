# SPDX-License-Identifier: Apache-2.0
"""Import one compact test image with its unique compressed blob staged on NVMe."""

# Standard
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile

CONTENT_ROOT = Path(
    "/var/lib/rancher/k3s/agent/containerd/io.containerd.content.v1.content"
)


def ctr(*argv: str) -> str:
    """Call the local k3s containerd API in Kubernetes' namespace."""
    return subprocess.check_output(["k3s", "ctr", "-n", "k8s.io", *argv], text=True)


def digest_file(path: Path) -> str:
    """Hash the actual file bytes without buffering a model-sized layer."""
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(archive: tarfile.TarFile, name: str) -> dict:
    """Read a required regular OCI metadata member."""
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f"Missing OCI metadata: {name}")
    return json.load(stream)


def import_image(
    archive: Path,
    scratch: Path,
    report: Path,
    unpacked_bytes: int,
    runtime_state: Path | None = None,
) -> None:
    """Stage only this image's unique compressed layer, then unpack and clean up.

    Requires root on the selected local k3s node. No shared content-store setting,
    snapshot, metadata database, or image belonging to another run is changed.
    A hardlink retains the original blob until the read-only bind is verified.
    """
    if os.geteuid() != 0:
        raise PermissionError("Use sudo for local k3s content staging")
    with tarfile.open(archive) as source:
        index = read_json(source, "index.json")
        assert len(index["manifests"]) == 1
        descriptor = index["manifests"][0]
        image = descriptor["annotations"]["io.containerd.image.name"]
        if not image.startswith("docker.io/library/mp-kv-events-"):
            raise ValueError("Only task-owned images may use external staging")
        manifest = read_json(
            source, "blobs/sha256/" + descriptor["digest"].split(":")[1]
        )
        assert len(manifest["layers"]) == 1
        layer = manifest["layers"][0]
        assert layer["mediaType"].endswith(("+zstd", "+gzip"))
        config = read_json(
            source, "blobs/sha256/" + manifest["config"]["digest"].split(":")[1]
        )
        assert len(config["rootfs"]["diff_ids"]) == 1
    runtime = None
    mount_args: list[str] = []
    if runtime_state is not None:
        saved = json.loads(runtime_state.read_text())
        accepted_images = {
            saved["image_id"],
            saved.get("base_compact", {}).get("image_id"),
        }
        if manifest["config"]["digest"] not in accepted_images:
            raise ValueError("Runtime state does not describe this image")
        runtime = saved["external_runtime"]
        runtime_manifest = json.loads(Path(runtime["manifest"]).read_text())
        if runtime_manifest["content_sha256"] != runtime["content_sha256"]:
            raise ValueError("Runtime state and immutable manifest differ")
        expected_paths = {d["package_path"] for d in runtime_manifest["directories"]}
        if expected_paths != {d["package_path"] for d in runtime["directories"]}:
            raise ValueError("Runtime mount paths differ from the immutable manifest")
        for directory in runtime["directories"]:
            host = Path(directory["host_path"])
            package = Path(directory["package_path"])
            if not host.is_absolute() or not host.is_dir() or not package.is_absolute():
                raise ValueError("Runtime mounts need existing absolute directories")
            if any("," in str(path) for path in (host, package)):
                raise ValueError("Runtime mount paths cannot contain ctr separators")
            mount_args.extend(
                ["--mount", f"type=bind,src={host},dst={package},options=rbind:ro"]
            )
    disk = shutil.disk_usage("/")
    reserve = disk.total * 0.05 + 1024**3
    if disk.free < reserve + max(layer["size"], int(unpacked_bytes * 1.05)):
        raise RuntimeError(
            "Insufficient root headroom for either staged or unpacked image"
        )
    scratch.mkdir(parents=True, exist_ok=True)
    encoded = layer["digest"].removeprefix("sha256:")
    if len(encoded) != 64 or any(c not in "0123456789abcdef" for c in encoded):
        raise ValueError("Expected a SHA256 layer digest")
    blob = CONTENT_ROOT / "blobs/sha256" / encoded
    backup = CONTENT_ROOT.parent / ("mp-kv-events-blob-backup-" + encoded)
    external = scratch / (encoded + ".layer")
    if blob.exists() or backup.exists() or external.exists():
        raise FileExistsError(
            "Layer already exists; do not replace shared or active content"
        )
    print(
        ctr(
            "images",
            "import",
            "--local",
            "--no-unpack",
            "--label",
            "io.cri-containerd.pinned=pinned",
            str(archive),
        )
    )
    listing = ctr("content", "list")
    references = {
        line.split()[0]
        for line in listing.splitlines()
        if "=" + layer["digest"] in line
    }
    assert references == {descriptor["digest"]}, references
    assert blob.stat().st_size == layer["size"]
    assert digest_file(blob) == layer["digest"]
    shutil.copyfile(blob, external)
    assert digest_file(external) == layer["digest"]
    external.chmod(0o444)
    state = {
        "image": image,
        "layer": layer,
        "blob": str(blob),
        "external": str(external),
        "backup": str(backup),
        "phase": "staged",
    }
    if runtime is not None:
        state["external_runtime"] = runtime
        state["runtime_state"] = str(runtime_state)
    report.write_text(json.dumps(state, indent=2) + "\n")
    mounted = False
    unpacked = False
    try:
        os.link(blob, backup)
        blob.unlink()
        blob.touch(mode=0o444)
        subprocess.run(["mount", "--bind", str(external), str(blob)], check=True)
        mounted = True
        subprocess.run(["mount", "-o", "remount,bind,ro", str(blob)], check=True)
        assert (
            blob.stat().st_size == layer["size"]
            and digest_file(blob) == layer["digest"]
        )
        # Verify the actual containerd ReaderAt path, not only filesystem access.
        process = subprocess.Popen(
            ["k3s", "ctr", "-n", "k8s.io", "content", "get", layer["digest"]],
            stdout=subprocess.PIPE,
        )
        assert isinstance(process.stdout, io.BufferedReader)
        read_digest = (
            "sha256:" + hashlib.file_digest(process.stdout, "sha256").hexdigest()
        )
        assert process.wait() == 0 and read_digest == layer["digest"]
        backup.unlink()
        state["phase"] = "mounted_verified"
        report.write_text(json.dumps(state, indent=2) + "\n")
        print(
            ctr(
                "images",
                "import",
                "--local",
                "--label",
                "io.cri-containerd.pinned=pinned",
                str(archive),
            )
        )
        snapshot = json.loads(ctr("snapshots", "info", config["rootfs"]["diff_ids"][0]))
        assert snapshot["Kind"] == "Committed", snapshot
        smoke_id = "mp-kv-image-smoke-" + encoded[:12]
        smoke_command = [
            "python3",
            "-c",
            "import torch, lmcache.cuda_ops, lmcache.lmcache_native; "
            "print('native_import_ok', torch.__version__)",
        ]
        if runtime is not None:
            smoke_command = [
                "python3",
                "/opt/mp-kv-events/verify_runtime.py",
                "--exec",
                *smoke_command,
            ]
        state["native_import_smoke_command"] = [*mount_args, *smoke_command]
        smoke = ctr(
            "run",
            "--rm",
            "--runc-binary",
            "/usr/bin/nvidia-container-runtime",
            "--env",
            "NVIDIA_VISIBLE_DEVICES=none",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            *mount_args,
            image,
            smoke_id,
            *smoke_command,
        )
        state["native_import_smoke"] = smoke
        unpacked = True
    finally:
        if mounted:
            subprocess.run(["umount", str(blob)], check=True)
        if backup.exists():
            blob.unlink(missing_ok=True)
            backup.rename(blob)
        else:
            # The installed image's committed snapshot is authoritative after
            # unpack. Raw cache layers may be discarded, as with ctr's regular
            # --discard-unpacked-layers option.
            if not unpacked:
                ctr("images", "remove", image)
            ctr("content", "delete", layer["digest"])
        external.unlink(missing_ok=True)
        state["phase"] = "complete" if unpacked else "failed_cleaned"
        state["root_free_bytes"] = shutil.disk_usage("/").free
        report.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--unpacked-bytes", type=int, required=True)
    parser.add_argument("--runtime-state", type=Path)
    args = parser.parse_args()
    import_image(
        args.archive, args.scratch, args.report, args.unpacked_bytes, args.runtime_state
    )
