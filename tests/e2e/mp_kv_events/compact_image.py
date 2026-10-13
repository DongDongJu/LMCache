# SPDX-License-Identifier: Apache-2.0
"""Flatten only the test image, keeping its runtime configuration and provenance."""

# Standard
from pathlib import Path
import argparse
import json
import subprocess
import tempfile


def runtime_fingerprint(image: str) -> dict:
    """Read package versions, source provenance and native hashes from the image."""
    script = """
import hashlib, importlib.metadata, importlib.util, json, pathlib
source = pathlib.Path('/opt/mp-kv-events/source.json')
spec = importlib.util.find_spec('lmcache')
root = pathlib.Path(spec.origin).parent
native = {}
for path in sorted(root.glob('*.so')):
    with path.open('rb') as stream:
        native[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
print(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                  'native':native,
                  'packages':{d.metadata['Name']:d.version
                              for d in importlib.metadata.distributions()}}))
"""
    return json.loads(
        subprocess.check_output(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python3",
                image,
                "-c",
                script,
            ]
        )
    )


def compact(image: str, archive_dir: Path) -> dict:
    """Remove redundant source/build caches and discard superseded image layers.

    Args:
        image: The task-owned image built by run.py.
        archive_dir: Large local disk receiving temporary rootfs data.

    Returns:
        Original/compact image identities and the preserved OCI configuration.
    """
    if not image.startswith(("docker.io/library/mp-kv-events-", "mp-kv-events-")):
        raise ValueError("Compaction accepts only this harness's images")
    original = json.loads(
        subprocess.check_output(["docker", "image", "inspect", image])
    )[0]
    config = original["Config"]
    fingerprint = runtime_fingerprint(image)
    script = """
import importlib.util, pathlib, shutil
for name in ('lmcache','vllm','sglang'):
    spec = importlib.util.find_spec(name)
    if spec and spec.origin and pathlib.Path(spec.origin).is_relative_to('/src'):
        raise RuntimeError('Refusing to remove an editable package source: ' + name)
for directory in ('/src/lmcache','/src/engine','/opt/uv/cache','/root/.cache/uv'):
    shutil.rmtree(directory, ignore_errors=True)
"""
    container = subprocess.check_output(
        [
            "docker",
            "create",
            "--network",
            "none",
            "--entrypoint",
            "python3",
            image,
            "-c",
            script,
        ],
        text=True,
    ).strip()
    with tempfile.NamedTemporaryFile(
        prefix="compact-rootfs-", suffix=".tar", dir=archive_dir, delete=False
    ) as temporary:
        archive = Path(temporary.name)
    target = image + "-compact"
    try:
        subprocess.run(["docker", "start", "--attach", container], check=True)
        status = int(
            subprocess.check_output(
                ["docker", "inspect", "--format", "{{.State.ExitCode}}", container]
            )
        )
        if status:
            raise RuntimeError(f"Image cleanup exited with {status}")
        subprocess.run(["docker", "export", "-o", str(archive), container], check=True)
        argv = ["docker", "import"]
        for value in config.get("Env", []):
            key, value = value.split("=", 1)
            argv.extend(["--change", f"ENV {key}={json.dumps(value)}"])
        for key, value in config.get("Labels", {}).items():
            argv.extend(["--change", f"LABEL {key}={json.dumps(value)}"])
        for key, directive in (("User", "USER"), ("WorkingDir", "WORKDIR")):
            if config.get(key):
                argv.extend(["--change", directive + " " + config[key]])
        for key in ("Entrypoint", "Cmd", "OnBuild"):
            if config.get(key):
                if key == "OnBuild":
                    for value in config[key]:
                        argv.extend(["--change", "ONBUILD " + value])
                else:
                    argv.extend(
                        ["--change", key.upper() + " " + json.dumps(config[key])]
                    )
        for port in config.get("ExposedPorts", {}):
            argv.extend(["--change", "EXPOSE " + port])
        if config.get("Volumes"):
            argv.extend(["--change", "VOLUME " + json.dumps(list(config["Volumes"]))])
        # Import cannot preserve these instructions. Fail visibly if a pinned
        # base starts relying on them instead of changing shutdown semantics.
        if config.get("Healthcheck") or config.get("StopSignal") or config.get("Shell"):
            raise ValueError("Pinned image has unsupported compaction configuration")
        subprocess.run([*argv, str(archive), target], check=True)
        result = json.loads(
            subprocess.check_output(["docker", "image", "inspect", target])
        )[0]
        metadata_differences = {}
        for key, value in config.items():
            # OCI specifies ArgsEscaped only for legacy Windows CommandLine.
            # containerd's Linux path always uses the preserved argv arrays.
            if key == "ArgsEscaped" and original["Os"] == "linux":
                if result["Config"].get(key) != value:
                    metadata_differences[key] = {
                        "original": value,
                        "compact": result["Config"].get(key),
                        "reason": "Legacy Windows-only metadata on a Linux image",
                    }
                continue
            assert result["Config"].get(key) == value, (
                key,
                value,
                result["Config"].get(key),
            )
        assert runtime_fingerprint(target) == fingerprint, (
            "Compaction changed runtime files"
        )
        return {
            "image": target,
            "image_id": result["Id"],
            "original_image_id": original["Id"],
            "original_size_bytes": original["Size"],
            "compact_size_bytes": result["Size"],
            "config": config,
            "metadata_differences": metadata_differences,
            "runtime_fingerprint": fingerprint,
        }
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container], check=True, stdout=subprocess.DEVNULL
        )
        archive.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compact(args.image, args.archive_dir), indent=2) + "\n"
    )
