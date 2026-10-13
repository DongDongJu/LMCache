# SPDX-License-Identifier: Apache-2.0
"""Apply the tracked Python bridge to the pinned CI wheel, preserving native code."""

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile


def digest(data: bytes) -> str:
    """Return a deterministic content identity for wheel provenance."""
    return hashlib.sha256(data).hexdigest()


def repack(wheel: Path, source: Path, manifest: Path, output: Path) -> Path:
    """Repack the declared bridge over verified baseline Python files.

    Args:
        wheel: Exact pinned SGLang CI wheel with precompiled native extensions.
        source: SGLang checkout copied into the build context.
        manifest: run.py source manifest containing the lock and per-file hashes.
        output: Directory receiving the patched wheel and its native hash manifest.

    Returns:
        The patched wheel path. Native extension bytes remain unchanged.

    Raises:
        ValueError: If the wheel or bridge inputs differ from their pinned hashes.
    """
    provenance = json.loads(manifest.read_text())
    pin = provenance["lock"]["sglang"]
    if digest(wheel.read_bytes()) != pin["wheel_sha256"]:
        raise ValueError("SGLang wheel does not match the pinned SHA256")
    with zipfile.ZipFile(wheel) as archive:
        native = {
            path: digest(archive.read(path))
            for path in archive.namelist()
            if path.endswith(".so")
        }
        if not native:
            raise ValueError("Pinned SGLang wheel contains no native extensions")
        for relative, expected in pin["bridge_baseline_sha256"].items():
            if digest(archive.read(relative)) != expected:
                raise ValueError(f"SGLang bridge baseline mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", temp],
            check=True,
        )
        unpacked = next(directory.iterdir())
        for relative in pin["bridge_baseline_sha256"]:
            relative_source = "python/" + relative
            path = source / relative_source
            expected = provenance["inputs"]["engine"]["files"][relative_source]
            if digest(path.read_bytes()) != expected:
                raise ValueError(f"SGLang bridge source mismatch: {relative}")
            shutil.copy2(path, unpacked / relative)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked),
                "--build-number",
                "1",
                "-d",
                str(output),
            ],
            check=True,
        )
    patched = next(output.glob("*.whl"))
    with zipfile.ZipFile(patched) as archive:
        for relative, expected in native.items():
            if digest(archive.read(relative)) != expected:
                raise ValueError(f"Native SGLang extension changed: {relative}")
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "base_wheel_sha256": pin["wheel_sha256"],
                "patched_wheel_sha256": digest(patched.read_bytes()),
                "bridge_revision": pin["bridge_revision"],
                "native_extension_sha256": native,
            },
            indent=2,
        )
        + "\n"
    )
    return patched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(repack(args.wheel, args.source, args.manifest, args.output))
