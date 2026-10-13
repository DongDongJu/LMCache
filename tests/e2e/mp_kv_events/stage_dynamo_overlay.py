# SPDX-License-Identifier: Apache-2.0
"""Install a verified Dynamo runtime wheel into a sparse image layer."""

# Standard
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import zipfile

# Third Party
from stage_python_overlay import sha256


def wheel_files(wheel: Path) -> tuple[dict[str, bytes], dict]:
    """Check wheel paths and every RECORD hash before exposing its contents."""
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != len(set(names)):
            raise ValueError("Wheel contains duplicate paths")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ".data" in path.parts:
                raise ValueError(f"Unsupported wheel path: {name}")
            if not (
                name.startswith("dynamo/")
                or path.parts[0].startswith("ai_dynamo_runtime-")
                and path.parts[0].endswith(".dist-info")
            ):
                raise ValueError(f"Wheel writes outside the Dynamo runtime: {name}")
        contents = {name: archive.read(name) for name in names}
    records = [name for name in names if name.endswith(".dist-info/RECORD")]
    metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
    if len(records) != 1 or len(metadata) != 1:
        raise ValueError("Expected one runtime wheel distribution")
    rows = list(csv.reader(io.StringIO(contents[records[0]].decode())))
    if len(rows) != len(contents) or {row[0] for row in rows} != set(contents):
        raise ValueError("Wheel RECORD does not cover its exact file set")
    for name, encoded, size in rows:
        if name == records[0] and not encoded and not size:
            continue
        data = contents[name]
        expected = "sha256=" + base64.urlsafe_b64encode(
            hashlib.sha256(data).digest()
        ).decode().rstrip("=")
        if encoded != expected or size != str(len(data)):
            raise ValueError(f"Wheel RECORD mismatch: {name}")
    parsed = BytesParser().parsebytes(contents[metadata[0]])
    return contents, {
        "name": parsed["Name"],
        "version": parsed["Version"],
        "requires_dist": sorted(parsed.get_all("Requires-Dist", [])),
    }


def stage(context: Path, output: Path) -> dict:
    """Replace only wheel-owned runtime files and the tracked SG wrapper."""
    source = json.loads((context / "source.json").read_text())
    pin = source["lock"]["dynamo"]
    wheel = context / "dynamo-runtime.whl"
    build = json.loads((context / "dynamo-build.json").read_text())
    if sha256(wheel) != build["wheel_sha256"]:
        raise ValueError("Runtime wheel differs from its build receipt")
    files, metadata = wheel_files(wheel)
    runtime = importlib.metadata.distribution("ai-dynamo-runtime")
    if metadata["name"].replace("_", "-") != "ai-dynamo-runtime":
        raise ValueError("Expected ai-dynamo-runtime wheel")
    if metadata["version"] != runtime.version or runtime.version != pin["version"]:
        raise ValueError("Runtime version changed; this overlay is a pinned patch")
    if metadata["requires_dist"] != sorted(runtime.requires or []):
        raise ValueError("Runtime dependency requirements changed")
    root = Path(str(runtime.locate_file("")))
    previous = {
        str(path): root / str(path)
        for path in runtime.files or []
        if "__pycache__" not in str(path)
    }
    if any(
        PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
        for name in previous
    ):
        raise ValueError("Installed runtime RECORD contains external paths")
    native_names = {name for name in files if name.endswith(".so")}
    if native_names != {"dynamo/_core.abi3.so"}:
        raise ValueError(f"Unexpected changed native library set: {native_names}")
    native_before = sha256(root / "dynamo/_core.abi3.so")
    changed = {}

    def remove(installed: Path) -> None:
        if output == Path("/"):
            installed.unlink(missing_ok=True)
            return
        whiteout = (
            output / installed.parent.relative_to("/") / (".wh." + installed.name)
        )
        whiteout.parent.mkdir(parents=True, exist_ok=True)
        whiteout.touch()

    for name in previous.keys() - files.keys():
        installed = previous[name]
        if not installed.is_file():
            continue
        remove(installed)
        changed[str(installed)] = "deleted"
    for name, data in files.items():
        installed = root / name
        target = output / installed.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        changed[str(installed)] = hashlib.sha256(data).hexdigest()
        if name.endswith(".py"):
            for cached in installed.parent.glob(f"__pycache__/{installed.stem}.*.pyc"):
                remove(cached)
    frontend = importlib.metadata.distribution("ai-dynamo")
    records = [
        path for path in frontend.files or [] if str(path).endswith(".dist-info/RECORD")
    ]
    if len(records) != 1:
        raise ValueError("Expected exactly one installed ai-dynamo RECORD")
    record_path = Path(str(frontend.locate_file(records[0])))
    rows = {row[0]: row for row in csv.reader(io.StringIO(record_path.read_text()))}
    for relative, baseline in pin["python_baseline_sha256"].items():
        if relative != "dynamo/sglang/publisher.py":
            raise ValueError(f"Unexpected Dynamo Python replacement: {relative}")
        installed = Path(str(frontend.locate_file(relative)))
        if sha256(installed) != baseline:
            raise ValueError(f"Dynamo wrapper baseline mismatch: {relative}")
        data = (context / "dynamo" / "components/src" / relative).read_bytes()
        if (
            hashlib.sha256(data).hexdigest()
            != (source["inputs"]["dynamo"]["files"]["components/src/" + relative])
        ):
            raise ValueError(f"Dynamo wrapper source mismatch: {relative}")
        target = output / installed.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode()
        rows[relative] = [relative, "sha256=" + encoded.rstrip("="), str(len(data))]
        changed[str(installed)] = hashlib.sha256(data).hexdigest()
        for cached in installed.parent.glob(f"__pycache__/{installed.stem}.*.pyc"):
            remove(cached)
    target = output / record_path.relative_to("/")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows[key] for key in sorted(rows))
    receipt = {
        "build": build,
        "wheel": metadata,
        "wheel_sha256": sha256(wheel),
        "source_revision": pin["bridge_revision"],
        "changed": changed,
        "native_change": {
            "path": str(root / "dynamo/_core.abi3.so"),
            "before_sha256": native_before,
            "after_sha256": hashlib.sha256(files["dynamo/_core.abi3.so"]).hexdigest(),
        },
        "scope": "Dynamo runtime wheel and existing SGLang publisher wrapper only",
    }
    target = output / "opt/mp-kv-events/dynamo-overlay.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage(args.context, args.output)))
