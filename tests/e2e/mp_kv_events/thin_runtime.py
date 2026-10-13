# SPDX-License-Identifier: Apache-2.0
"""Move immutable runtime bytes to explicit read-only volumes without deleting deps."""

# Standard
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile

# Third Party
from compact_image import compact
from compress_image import compress
from verify_runtime import inventory

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = "/usr/local/lib/python3.12/dist-packages"
PACKAGES = ("flashinfer_cubin", "nvidia")


def create(base_image: str, output: Path) -> dict:
    """Export exact runtime trees, flatten their image, and record required mounts."""
    if not base_image.startswith(
        ("mp-kv-events-sglang:", "docker.io/library/mp-kv-events-sglang:")
    ):
        raise ValueError("Only task-owned SGLang images are supported")
    output.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(
        subprocess.check_output(["docker", "image", "inspect", base_image])
    )[0]
    with tempfile.TemporaryDirectory(prefix="immutable-runtime-", dir=output) as temp:
        context = Path(temp)
        payload = context / "payload"
        payload.mkdir()
        container = subprocess.check_output(
            [
                "docker",
                "create",
                "--network",
                "none",
                "--entrypoint",
                "/bin/true",
                base_image,
            ],
            text=True,
        ).strip()
        try:
            subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container}:/opt/mp-kv-events/source.json",
                    str(context / "source.json"),
                ],
                check=True,
            )
            for name in PACKAGES:
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{container}:{PACKAGE_ROOT}/{name}",
                        str(payload / name),
                    ],
                    check=True,
                )
        finally:
            subprocess.run(
                ["docker", "rm", container], check=True, stdout=subprocess.DEVNULL
            )
        directories = [
            {
                "package_path": PACKAGE_ROOT + "/" + name,
                "entries": inventory(payload / name),
            }
            for name in PACKAGES
        ]
        content_sha256 = hashlib.sha256(
            json.dumps(directories, sort_keys=True).encode()
        ).hexdigest()
        descriptor = {"content_sha256": content_sha256, "directories": directories}
        artifact = output / "runtime" / content_sha256
        artifact.parent.mkdir(exist_ok=True)
        if artifact.exists():
            for name, directory in zip(PACKAGES, directories, strict=True):
                if inventory(artifact / name) != directory["entries"]:
                    raise ValueError("Existing immutable runtime artifact changed")
        else:
            payload.rename(artifact)
        manifest = context / "external-runtime.json"
        manifest.write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
        (artifact / "manifest.json").write_bytes(manifest.read_bytes())
        shutil.copy2(HERE / "verify_runtime.py", context / "verify_runtime.py")
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nUSER root\n"
            "COPY external-runtime.json verify_runtime.py /opt/mp-kv-events/\n"
            "RUN python3 -c 'import pathlib,shutil; "
            '[shutil.rmtree(pathlib.Path("' + PACKAGE_ROOT + '") / name) '
            'for name in ("flashinfer_cubin", "nvidia")]\'\n'
            'ENTRYPOINT ["python3", "/opt/mp-kv-events/verify_runtime.py", "--exec"]\n'
        )
        identity = hashlib.sha256(
            (
                baseline["Id"]
                + content_sha256
                + hashlib.sha256(
                    (context / "verify_runtime.py").read_bytes()
                ).hexdigest()
            ).encode()
        ).hexdigest()[:20]
        image = "docker.io/library/mp-kv-events-sglang:" + identity + "-thin"
        subprocess.run(
            [
                "docker",
                "build",
                "--network=none",
                "--build-arg",
                "BASE_IMAGE=" + base_image,
                "-t",
                image,
                str(context),
            ],
            check=True,
        )
        compacted = compact(image, output)
        archive = output / (identity + ".tar")
        subprocess.run(
            ["docker", "save", "-o", str(archive), compacted["image"]], check=True
        )
        compression = compress(archive, output / (identity + ".oci.tar"))
        mounts = [
            {
                "package_path": PACKAGE_ROOT + "/" + name,
                "host_path": str((artifact / name).resolve()),
            }
            for name in PACKAGES
        ]
        state = {
            "source": json.loads((context / "source.json").read_text()),
            "image": compacted["image"],
            "image_id": compacted["image_id"],
            "base_image": base_image,
            "base_image_id": baseline["Id"],
            "compaction": compacted,
            "compression": compression,
            "external_runtime": {
                "content_sha256": content_sha256,
                "manifest": str((artifact / "manifest.json").resolve()),
                "directories": mounts,
            },
        }
        report = output / (identity + ".json")
        report.write_text(json.dumps(state, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "state": str(report),
                    "image": state["image"],
                    "size_bytes": compacted["compact_size_bytes"],
                    "external_runtime": state["external_runtime"],
                }
            )
        )
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    result = create(args.base_image, args.output)
    if args.state:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(result, indent=2) + "\n")
