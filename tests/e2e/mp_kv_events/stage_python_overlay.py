# SPDX-License-Identifier: Apache-2.0
"""Stage verified Python changes over an installed, immutable native runtime."""

# Standard
from pathlib import Path
import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json


def sha256(path: Path) -> str:
    """Hash an input without buffering native libraries in memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(context: Path, output: Path, engine: str, bridge_only: bool) -> dict:
    """Validate baseline files, stage replacements, and regenerate RECORD entries."""
    current = json.loads((context / "source.json").read_text())
    pin = current["lock"][engine]
    previous_path = Path("/opt/mp-kv-events/source.json")
    previous = json.loads(previous_path.read_text()) if not bridge_only else None
    if previous is not None and engine == "sglang":
        old_dynamo = previous["lock"]["dynamo"].get("bridge_patch_sha256")
        new_dynamo = current["lock"]["dynamo"].get("bridge_patch_sha256")
        if old_dynamo != new_dynamo and not (context / "dynamo-runtime.whl").is_file():
            raise ValueError(
                "Changed Dynamo source pin requires its verified runtime wheel"
            )
    changed: dict[str, str] = {}
    native: dict[str, dict[str, str]] = {}

    def package(name: str, replacements: dict[str, Path | None], baseline: dict):
        distribution = importlib.metadata.distribution(name)
        root = Path(str(distribution.locate_file("")))
        native[name] = {
            str(path.relative_to(root)): sha256(path)
            for path in sorted((root / name).rglob("*.so"))
        }
        if not native[name]:
            raise ValueError(f"No native extensions found for {name}")
        records = [
            f for f in distribution.files or [] if str(f).endswith(".dist-info/RECORD")
        ]
        if len(records) != 1:
            raise ValueError(f"Expected exactly one installed RECORD for {name}")
        record_path = Path(str(distribution.locate_file(records[0])))
        rows = {row[0]: row for row in csv.reader(io.StringIO(record_path.read_text()))}
        for relative, source in replacements.items():
            installed = root / relative
            expected = baseline.get(relative)
            if expected is not None:
                if not installed.is_file() or sha256(installed) != expected:
                    raise ValueError(f"Installed baseline mismatch: {installed}")
            elif installed.exists():
                raise ValueError(
                    f"Undeclared installed file already exists: {installed}"
                )
            target = output / installed.relative_to("/")
            target.parent.mkdir(parents=True, exist_ok=True)
            if source is None:
                if output == Path("/"):
                    target.unlink()
                else:
                    (target.parent / (".wh." + target.name)).touch()
                rows.pop(relative, None)
                changed[str(installed)] = "deleted"
            else:
                data = source.read_bytes()
                target.write_bytes(data)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                rows[relative] = [
                    relative,
                    "sha256=" + encoded.decode().rstrip("="),
                    str(len(data)),
                ]
                changed[str(installed)] = hashlib.sha256(data).hexdigest()
            # Invalidate any compiled cache even when source size is unchanged.
            for cached in installed.parent.glob(f"__pycache__/{installed.stem}.*.pyc"):
                if output == Path("/"):
                    cached.unlink()
                    continue
                whiteout = (
                    output / cached.parent.relative_to("/") / (".wh." + cached.name)
                )
                whiteout.parent.mkdir(parents=True, exist_ok=True)
                whiteout.touch()
        record_output = output / record_path.relative_to("/")
        record_output.parent.mkdir(parents=True, exist_ok=True)
        with record_output.open("w", newline="") as stream:
            csv.writer(stream).writerows(rows[key] for key in sorted(rows))

    if importlib.metadata.version(engine) != pin["version"]:
        raise ValueError("Installed engine version differs from the pinned wheel")
    prefix = "python/" if engine == "sglang" else ""
    engine_files = current["inputs"]["engine"]["files"]
    replacements = {}
    for relative in pin["bridge_baseline_sha256"]:
        source = context / "engine" / (prefix + relative)
        if sha256(source) != engine_files[prefix + relative]:
            raise ValueError(f"Bridge source mismatch: {relative}")
        replacements[relative] = source
    engine_baseline = dict(pin["bridge_baseline_sha256"])
    if previous is not None:
        old_engine = previous["inputs"]["engine"]["files"]
        for relative in replacements:
            engine_baseline[relative] = old_engine[prefix + relative]
    package(engine, replacements, engine_baseline)
    if previous is not None:
        old = previous["inputs"]["lmcache"]["files"]
        new = current["inputs"]["lmcache"]["files"]
        replacements = {}
        for relative in old.keys() | new.keys():
            if old.get(relative) == new.get(relative):
                continue
            if relative.startswith(
                ("csrc/", "rust/", "include/", "cmake/", "requirements/")
            ) or relative in ("setup.py", "pyproject.toml", "CMakeLists.txt"):
                raise ValueError(
                    f"Native/dependency input changed; rebuild required: {relative}"
                )
            if relative.startswith("lmcache/"):
                if not relative.endswith(".py"):
                    raise ValueError(f"Non-Python runtime input changed: {relative}")
                source = context / "lmcache" / relative if relative in new else None
                if source is not None and sha256(source) != new[relative]:
                    raise ValueError(f"LMCache source mismatch: {relative}")
                replacements[relative] = source
        package("lmcache", replacements, old)
        harness = context / "lmcache/tests/e2e/mp_kv_events"
        for source in harness.rglob("*"):
            if source.is_file():
                target = output / "opt/mp-kv-events" / source.relative_to(harness)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        target = output / "opt/mp-kv-events/source.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((context / "source.json").read_bytes())
    receipt = {
        "engine": engine,
        "bridge_revision": pin["bridge_revision"],
        "source_sha256": current["sha256"],
        "changed": changed,
        "native_unchanged_sha256": native,
    }
    target = output / "opt/mp-kv-events/python-overlay.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--bridge-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(stage(args.context, args.output, args.engine, args.bridge_only)))
