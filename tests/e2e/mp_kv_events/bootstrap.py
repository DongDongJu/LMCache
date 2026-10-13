# SPDX-License-Identifier: Apache-2.0
"""Install or remove a run-scoped device plugin on one local k3s GPU node."""

# Standard
from pathlib import Path
import argparse
import json
import subprocess
import time
import uuid

# Third Party
from run import NAMESPACE, command, kubectl, metadata, read_lock, write_json


def manifest(node: str, gpu_uuids: list[str], run: str, lock: dict) -> dict:
    """Render a nonprivileged plugin with an explicit visible-device allowlist."""
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": metadata("device-plugin-" + run, run),
        "spec": {
            "runtimeClassName": "nvidia",
            "nodeSelector": {"kubernetes.io/hostname": node},
            "containers": [
                {
                    "name": "device-plugin",
                    "image": lock["cluster"]["device_plugin_image"],
                    "env": [
                        {
                            "name": "NVIDIA_VISIBLE_DEVICES",
                            "value": ",".join(gpu_uuids),
                        },
                        {"name": "FAIL_ON_INIT_ERROR", "value": "true"},
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [
                        {
                            "name": "plugins",
                            "mountPath": "/var/lib/kubelet/device-plugins",
                        }
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "plugins",
                    "hostPath": {
                        "path": "/var/lib/kubelet/device-plugins",
                        "type": "Directory",
                    },
                }
            ],
        },
    }


def main() -> None:
    """Persist bootstrap identity independently of every serving deployment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("render", "install", "cleanup"))
    parser.add_argument("--node", default="")
    parser.add_argument("--gpu-uuids", default="")
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument(
        "--lock", type=Path, default=Path(__file__).with_name("versions.lock.yaml")
    )
    args = parser.parse_args()
    context = command(kubectl("config", "current-context")).stdout.strip()
    if args.action == "cleanup":
        saved = json.loads(args.state.read_text())
        if context != saved["context"]:
            raise ValueError("Kubernetes context changed since plugin installation")
        pod = json.loads(
            command(
                kubectl("get", "pod", "-n", NAMESPACE, saved["name"], "-o", "json")
            ).stdout
        )
        if pod["metadata"]["uid"] != saved["uid"]:
            raise ValueError("Plugin pod identity changed; refusing deletion")
        command(kubectl("delete", "pod", "-n", NAMESPACE, saved["name"], "--wait=true"))
        return
    gpu_uuids = args.gpu_uuids.split(",")
    available = command(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    ).stdout.splitlines()
    if not args.node or not gpu_uuids or len(set(gpu_uuids)) != len(gpu_uuids):
        raise ValueError("Choose one node and a nonempty, distinct GPU UUID allowlist")
    if not set(gpu_uuids).issubset(available):
        raise ValueError("Every selected GPU must exist on this local host")
    run = "bootstrap-" + uuid.uuid4().hex[:12]
    pod = manifest(args.node, gpu_uuids, run, read_lock(args.lock))
    if args.action == "render":
        print(json.dumps(pod, indent=2))
        return
    node = json.loads(command(kubectl("get", "node", args.node, "-o", "json")).stdout)
    local_addresses = {
        entry["local"]
        for interface in json.loads(command(["ip", "-j", "address"]).stdout)
        for entry in interface.get("addr_info", [])
    }
    node_addresses = {item["address"] for item in node["status"]["addresses"]}
    if not local_addresses.intersection(node_addresses):
        raise ValueError("Selected node is not this local GPU host")
    if int(node["status"].get("allocatable", {}).get("nvidia.com/gpu", 0)):
        raise ValueError("Node already advertises GPUs; reuse its existing plugin")
    command(kubectl("create", "namespace", NAMESPACE), check=False)
    subprocess.run(
        kubectl("apply", "-f", "-"), input=json.dumps(pod), text=True, check=True
    )
    name = pod["metadata"]["name"]
    created = json.loads(
        command(kubectl("get", "pod", "-n", NAMESPACE, name, "-o", "json")).stdout
    )
    saved = {
        "context": context,
        "name": name,
        "uid": created["metadata"]["uid"],
        "manifest": pod,
    }
    write_json(args.state, saved)
    command(
        kubectl(
            "wait",
            "-n",
            NAMESPACE,
            "pod/" + name,
            "--for=condition=Ready",
            "--timeout=90s",
        )
    )
    visible = command(
        kubectl(
            "exec",
            "-n",
            NAMESPACE,
            name,
            "--",
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        )
    ).stdout.splitlines()
    if set(visible) != set(gpu_uuids):
        raise RuntimeError(f"Plugin sees unexpected devices: {visible}")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        node = json.loads(
            command(kubectl("get", "node", args.node, "-o", "json")).stdout
        )
        if int(node["status"]["allocatable"].get("nvidia.com/gpu", 0)) == len(
            gpu_uuids
        ):
            saved.update(visible_gpu_uuids=visible, verified_allocatable=len(visible))
            write_json(args.state, saved)
            return
        time.sleep(1)
    raise RuntimeError("Node allocation did not match the plugin's verified device set")


if __name__ == "__main__":
    main()
