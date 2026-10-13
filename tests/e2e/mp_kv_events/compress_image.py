# SPDX-License-Identifier: Apache-2.0
"""Compress a compact OCI layer before import, bounding root-disk staging space."""

# Standard
from pathlib import Path
from typing import IO
import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tempfile


def regular_member(archive: tarfile.TarFile, name: str) -> IO[bytes]:
    """Open a required regular OCI archive entry."""
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected a regular OCI archive entry: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read required OCI archive entry: {name}")
    return stream


def compress(source: Path, output: Path) -> dict:
    """Repack one compact OCI image with a standard zstd-compressed layer.

    The image configuration and uncompressed rootfs identity remain unchanged.
    Compression occurs on the archive filesystem, before touching containerd.
    """
    with tempfile.TemporaryDirectory(prefix="oci-compress-", dir=output.parent) as tmp:
        layer_path = Path(tmp) / "layer.zst"
        with tarfile.open(source) as archive:
            with regular_member(archive, "index.json") as stream:
                index = json.load(stream)
            assert len(index["manifests"]) == 1, "Expected one local image manifest"
            descriptor = index["manifests"][0]
            with regular_member(
                archive, "blobs/sha256/" + descriptor["digest"].split(":")[1]
            ) as stream:
                manifest = json.load(stream)
            assert len(manifest["layers"]) == 1, "Compact the image before compression"
            layer = manifest["layers"][0]
            assert layer["mediaType"] == "application/vnd.oci.image.layer.v1.tar"
            uncompressed_bytes = layer["size"]
            with (
                regular_member(
                    archive, "blobs/sha256/" + layer["digest"].split(":")[1]
                ) as original,
                layer_path.open("wb") as stream,
            ):
                process = subprocess.Popen(
                    ["zstd", "-q", "-T16", "-3", "-c"],
                    stdin=subprocess.PIPE,
                    stdout=stream,
                )
                assert process.stdin is not None
                shutil.copyfileobj(original, process.stdin, 1024 * 1024)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("OCI layer compression failed")
            with layer_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            layer.update(
                mediaType="application/vnd.oci.image.layer.v1.tar+zstd",
                digest="sha256:" + digest,
                size=layer_path.stat().st_size,
            )
            manifest_data = json.dumps(manifest, separators=(",", ":")).encode()
            manifest_hash = hashlib.sha256(manifest_data).hexdigest()
            descriptor.update(digest="sha256:" + manifest_hash, size=len(manifest_data))
            config_path = "blobs/sha256/" + manifest["config"]["digest"].split(":")[1]
            with regular_member(archive, config_path) as stream:
                config = stream.read()
        with tarfile.open(output, "w") as target:
            payloads = {
                "index.json": json.dumps(index).encode(),
                "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
                "blobs/sha256/" + manifest_hash: manifest_data,
                config_path: config,
            }
            for name, data in payloads.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                target.addfile(entry, io.BytesIO(data))
            target.add(layer_path, arcname="blobs/sha256/" + digest, recursive=False)
        return {
            "archive": str(output),
            "compressed_bytes": layer["size"],
            "unpacked_bytes": uncompressed_bytes,
            "config_digest": manifest["config"]["digest"],
            "manifest_digest": descriptor["digest"],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    args.report.write_text(
        json.dumps(compress(args.source, args.output), indent=2) + "\n"
    )
