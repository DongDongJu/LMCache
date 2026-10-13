# SPDX-License-Identifier: Apache-2.0
"""Verify immutable NVMe runtime directories before starting any worker process."""

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import os
import stat


def inventory(root: Path) -> dict:
    """Describe every file, symlink, and permission without following symlinks."""
    if not root.is_dir():
        raise ValueError(f"Required runtime directory is missing: {root}")
    entries = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        entry: dict = {"mode": stat.S_IMODE(info.st_mode)}
        if path.is_symlink():
            entry.update(kind="symlink", target=os.readlink(path))
        elif path.is_file():
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            entry.update(kind="file", sha256=checksum, size=info.st_size)
        elif path.is_dir():
            entry.update(kind="directory")
        else:
            raise ValueError(f"Unexpected runtime file type: {path}")
        entries[relative] = entry
    return entries


def verify(manifest: Path) -> dict:
    """Fail before engine startup if a required read-only mount is absent or changed."""
    descriptor = json.loads(manifest.read_text())
    for directory in descriptor["directories"]:
        path = Path(directory["package_path"])
        actual = inventory(path)
        if actual != directory["entries"]:
            raise ValueError(
                f"Immutable runtime directory does not match manifest: {path}"
            )
    return descriptor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("/opt/mp-kv-events/external-runtime.json")
    )
    parser.add_argument("--exec", dest="command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    result = verify(args.manifest)
    print("Verified immutable runtime " + result["content_sha256"], flush=True)
    if args.command:
        os.execvpe(args.command[0], args.command, os.environ)
