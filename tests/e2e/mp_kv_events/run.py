# SPDX-License-Identifier: Apache-2.0
"""Run-scoped Kubernetes controls for the MP native KV-event acceptance suite."""

# Standard
from pathlib import Path
from typing import Any
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid

# Third Party
from compact_image import compact
from compress_image import compress

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORK_ROOT = Path(
    os.environ.get("LMCACHE_E2E_WORK_ROOT", REPO / "artifacts/mp-kv-events/work")
)
NAMESPACE = "kv-events-e2e"
LABEL = "mp-kv-events/run"


def command(
    argv: list[str], *, check: bool = True, timeout: float = 120
) -> subprocess.CompletedProcess:
    """Run a bounded external command without a shell and return its output."""
    return subprocess.run(
        argv, check=check, capture_output=True, text=True, timeout=timeout
    )


def kubectl(*args: str) -> list[str]:
    """Build the kubectl argv; deployment commands supply the persisted context."""
    return ["kubectl", *args]


def write_json(path: Path, value: object) -> None:
    """Persist an indented JSON artifact, creating only its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_lock(path: Path) -> dict:
    """Load the checked-in JSON-subset YAML lock without a YAML dependency."""
    return json.loads(path.read_text())


def preflight(args: argparse.Namespace) -> int:
    """Inspect the existing cluster without installing or changing workloads."""
    checks: list[dict] = []
    raw: dict[str, Any] = {}
    for name, argv in (
        (
            "gpu",
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv",
            ],
        ),
        (
            "gpu_processes",
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv",
            ],
        ),
        ("context", kubectl("config", "current-context")),
        ("nodes", kubectl("get", "nodes", "-o", "json")),
        ("pods", kubectl("get", "pods", "-A", "-o", "json")),
        ("runtimeclasses", kubectl("get", "runtimeclass", "-o", "json")),
        ("toolkit", ["nvidia-ctk", "--version"]),
    ):
        if not shutil.which(argv[0]):
            checks.append(
                {"name": name, "status": "BLOCKED", "reason": f"Missing {argv[0]}"}
            )
            continue
        result = command(argv, check=False)
        # Pod specs can contain inline secrets. Persist only inventory fields.
        if name == "pods" and result.returncode == 0:
            pods = json.loads(result.stdout)["items"]
            raw[name] = [
                {
                    "namespace": pod["metadata"]["namespace"],
                    "name": pod["metadata"]["name"],
                    "node": pod["spec"].get("nodeName"),
                    "phase": pod["status"]["phase"],
                    "containers": [
                        {
                            "name": c["name"],
                            "image": c["image"],
                            "resources": c.get("resources", {}),
                        }
                        for c in pod["spec"]["containers"]
                    ],
                }
                for pod in pods
            ]
        else:
            raw[name] = result.stdout if result.returncode == 0 else result.stderr
        checks.append(
            {
                "name": name,
                "status": "PASS" if result.returncode == 0 else "BLOCKED",
                "reason": "" if result.returncode == 0 else result.stderr.strip(),
            }
        )
    nodes = json.loads(raw.get("nodes", '{"items":[]}'))["items"]
    candidates = []
    for node in nodes:
        spec = node["spec"]
        ready = any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in node["status"]["conditions"]
        )
        tainted = any(
            t["effect"] in ("NoSchedule", "NoExecute") for t in spec.get("taints", [])
        )
        available = int(node["status"]["allocatable"].get("nvidia.com/gpu", 0))
        for pod in raw.get("pods", []):
            if pod["node"] == node["metadata"]["name"] and pod["phase"] not in (
                "Succeeded",
                "Failed",
            ):
                available -= sum(
                    int(c["resources"].get("requests", {}).get("nvidia.com/gpu", 0))
                    for c in pod["containers"]
                )
        if (
            ready
            and not spec.get("unschedulable", False)
            and not tainted
            and available >= 2 * args.tp
        ):
            candidates.append(node["metadata"]["name"])
    selected = args.node or (candidates[0] if len(candidates) == 1 else "")
    args.node = selected
    checks.append(
        {
            "name": "single_node_allocation",
            "status": "PASS" if selected in candidates else "BLOCKED",
            "reason": (
                f"Need one selected Ready node with {2 * args.tp} free GPUs; "
                f"candidates={candidates}"
            ),
            "selected_node": selected,
        }
    )
    runtime = json.loads(raw.get("runtimeclasses", '{"items":[]}'))
    checks.append(
        {
            "name": "nvidia_runtime",
            "status": "PASS"
            if any(x["metadata"]["name"] == "nvidia" for x in runtime["items"])
            else "BLOCKED",
        }
    )
    disk = shutil.disk_usage(args.artifacts.parent.resolve())
    raw["artifact_disk_free_bytes"] = disk.free
    root_disk = shutil.disk_usage("/")
    checks.append(
        {
            "name": "node_disk_headroom",
            "status": "PASS"
            if root_disk.free > root_disk.total * 0.05 + 1024**3
            else "BLOCKED",
            "reason": "Require kubelet 5% eviction reserve plus 1 GiB process headroom",
            "free_bytes": root_disk.free,
        }
    )
    lock = read_lock(args.lock)
    if selected in candidates:
        selected_node = next(
            node for node in nodes if node["metadata"]["name"] == selected
        )
        actual = selected_node["status"]["nodeInfo"]["kubeletVersion"]
        checks.append(
            {
                "name": "pinned_kubernetes",
                "status": "PASS" if actual == lock["cluster"]["k3s"] else "BLOCKED",
                "actual": actual,
                "required": lock["cluster"]["k3s"],
            }
        )
    raw["memory"] = Path("/proc/meminfo").read_text()
    report = {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "BLOCKED",
        "scope": "read-only preflight; GPU pod and real CUDA IPC remain separate gates",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cluster": args.cluster,
        "tp": args.tp,
        "checks": checks,
        "raw": raw,
    }
    write_json(args.artifacts / "preflight.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "node": selected,
                "artifact": str(args.artifacts / "preflight.json"),
            }
        )
    )
    return 0 if report["status"] == "PASS" else 2


def metadata(name: str, run: str) -> dict:
    """Name and label every namespaced resource for precise cleanup."""
    return {
        "name": name,
        "namespace": NAMESPACE,
        "labels": {LABEL: run, "app.kubernetes.io/part-of": "mp-kv-events"},
    }


def service(name: str, run: str, ports: dict[str, int]) -> dict:
    """Create a Service selecting one component, never both serving replicas."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": metadata(name, run),
        "spec": {
            "selector": {LABEL: run, "component": name},
            "ports": [
                {"name": key, "port": value, "targetPort": value}
                for key, value in ports.items()
            ],
        },
    }


def launcher(
    engine: str,
    tp: int,
    transport: str,
    chunk_size: int,
    lock: dict,
    event_log_size: int = 32768,
    events_disabled: bool = False,
) -> dict:
    """Construct explicit engine and MP argv from the pinned integration APIs."""
    model = lock["model"]
    events = json.dumps(
        {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": "tcp://*:20080",
            "enable_kv_cache_events": True,
        }
    )
    if engine == "sglang":
        event_config = json.loads(events)
        event_config["format"] = "dynamo"
        event_config["replay_endpoint"] = "tcp://*:20081"
        event_config["replay_handshake"] = True
        event_config.pop("enable_kv_cache_events")
        events = json.dumps(event_config)
    common = ["--revision", model["revision"]]
    if not events_disabled:
        common.extend(["--kv-events-config", events])
    elif engine == "vllm":
        common.extend(
            ["--kv-events-config", json.dumps({"enable_kv_cache_events": False})]
        )
    url = f"{'tcp' if transport == 'zmq' else 'grpc'}://127.0.0.1:5555"
    if engine == "vllm":
        argv = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            model["id"],
            "--tokenizer-revision",
            model["tokenizer_revision"],
            "--tensor-parallel-size",
            str(tp),
            "--block-size",
            "16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
            "--enforce-eager",
            "--disable-hybrid-kv-cache-manager",
            "--num-gpu-blocks-override",
            "257",
            "--prefix-caching-hash-algo",
            "sha256_cbor",
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "LMCacheMPConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "lmcache.mp.server_urls": [url],
                        "lmcache.mp.mp_transfer_mode": "lmcache_driven",
                    },
                }
            ),
            *common,
        ]
    else:
        argv = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            model["id"],
            "--tp-size",
            str(tp),
            "--page-size",
            "16",
            "--max-running-requests",
            "1",
            "--max-total-tokens",
            "4096",
            "--context-length",
            "4096",
            "--enable-lmcache",
            "--lmcache-config-file",
            "/config/lmcache.json",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--log-level",
            "debug",
            *common,
        ]
    argv.extend(
        [
            "--request-plane",
            "nats",
            "--event-plane",
            "nats",
            "--discovery-backend",
            "etcd",
        ]
    )
    return {
        "mp": {
            "argv": [
                "lmcache",
                "server",
                "--port",
                "5555",
                "--http-port",
                "8080",
                "--l1-size-gb",
                "0.25",
                "--max-gpu-workers",
                str(tp),
                "--eviction-policy",
                "LRU",
                "--chunk-size",
                str(chunk_size),
                "--transport",
                transport,
                "--kv-event-log-size",
                str(0 if events_disabled else event_log_size),
                "--server-module",
                json.dumps({"module_path": "mp_trace"}),
                "--server-module",
                json.dumps({"module_path": "mp_test_hooks"}),
            ],
            "health": "http://127.0.0.1:8080/healthcheck",
        },
        "observer": {
            "argv": [
                "python3",
                "/opt/mp-kv-events/observe.py",
                "--endpoint",
                "tcp://127.0.0.1:20080",
                "--output",
                "/artifacts/native-events.jsonl",
            ]
            + (
                ["--replay-endpoint", "tcp://127.0.0.1:20081", "--replay-handshake"]
                if engine == "sglang" and not events_disabled
                else []
            )
        },
        "engine": {"argv": argv, "health": "http://127.0.0.1:8081/health"},
    }


def render(
    args: argparse.Namespace,
    lock: dict,
    image: str,
    run: str,
    external_runtime: dict | None = None,
) -> list[dict]:
    """Render one node, two private MP/engine pods and Dynamo dependencies."""
    items: list[dict] = []
    config = {
        "launcher.json": json.dumps(
            launcher(
                args.engine,
                args.tp,
                args.transport,
                args.chunk_size,
                lock,
                args.event_log_size,
                args.events_disabled,
            )
        ),
        "lmcache.json": json.dumps(
            {
                "mp_host": "tcp://127.0.0.1"
                if args.transport == "zmq"
                else "grpc://127.0.0.1",
                "mp_port": 5555,
                "extra_config": {"lmcache.mp.mp_transfer_mode": "lmcache_driven"},
            }
        ),
    }
    runtime_files = ("worker.py", "observe.py", "mp_trace.py", "mp_test_hooks.py")
    for name in runtime_files:
        config[name] = (HERE / name).read_text()
    resolved = json.loads(config["launcher.json"])
    if external_runtime:
        resolved["external_runtime"] = external_runtime
    resolved["runtime_files_sha256"] = {
        name: hashlib.sha256(config[name].encode()).hexdigest()
        for name in runtime_files
    }
    config["launcher.json"] = json.dumps(resolved)
    items.append(
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": metadata("launcher", run),
            "data": config,
        }
    )
    for name, service_image, argv, ports in (
        (
            "nats",
            lock["services"]["nats"],
            ["nats-server", "-js", "-m", "8222"],
            {"client": 4222, "http": 8222},
        ),
        (
            "etcd",
            lock["services"]["etcd"],
            [
                "etcd",
                "--data-dir",
                "/tmp/mp-kv-events-etcd",
                "--listen-client-urls",
                "http://0.0.0.0:2379",
                "--advertise-client-urls",
                "http://etcd:2379",
            ],
            {"client": 2379},
        ),
        (
            "frontend",
            image,
            [
                "python3",
                "-m",
                "dynamo.frontend",
                "--router-mode",
                "kv",
                "--router-kv-events",
                "--router-host-cache-hit-weight",
                "0.75",
                "--http-port",
                "8000",
                "--request-plane",
                "nats",
                "--event-plane",
                "nats",
                "--discovery-backend",
                "etcd",
            ],
            {"http": 8000},
        ),
    ):
        meta = metadata(name, run)
        meta["labels"]["component"] = name
        items.append(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": meta,
                "spec": {
                    "nodeSelector": {"kubernetes.io/hostname": args.node},
                    "containers": [
                        {
                            "name": name,
                            "image": service_image,
                            "command": argv,
                            "env": [
                                {"name": "NATS_SERVER", "value": "nats://nats:4222"},
                                {"name": "ETCD_ENDPOINTS", "value": "http://etcd:2379"},
                            ],
                            "resources": {
                                "requests": {"cpu": "1", "memory": "1Gi"},
                                "limits": {"cpu": "4", "memory": "8Gi"},
                            },
                        }
                    ],
                },
            }
        )
        items.append(service(name, run, ports))
    for replica in ("a", "b"):
        name = f"worker-{replica}"
        meta = metadata(name, run)
        meta["labels"]["component"] = name
        items.append(
            {
                "apiVersion": "v1",
                "kind": "Pod",
                "metadata": meta,
                "spec": {
                    "runtimeClassName": "nvidia",
                    "nodeSelector": {"kubernetes.io/hostname": args.node},
                    "terminationGracePeriodSeconds": 90,
                    "containers": [
                        {
                            "name": "worker",
                            "image": image,
                            "command": [
                                "python3",
                                "/opt/mp-kv-events/worker.py",
                                "--config",
                                "/config/launcher.json",
                            ],
                            "env": [
                                {"name": "LMCACHE_KV_EVENTS_E2E", "value": "1"},
                                {"name": "NATS_SERVER", "value": "nats://nats:4222"},
                                {"name": "ETCD_ENDPOINTS", "value": "http://etcd:2379"},
                                {"name": "DYN_SYSTEM_PORT", "value": "8081"},
                                {"name": "PYTHONHASHSEED", "value": "0"},
                                {"name": "HF_HOME", "value": "/models"},
                                {
                                    "name": "CUDA_CACHE_PATH",
                                    "value": "/home/dynamo/.nv/ComputeCache",
                                },
                                {"name": "CUDA_CACHE_MAXSIZE", "value": "4294967296"},
                                {"name": "LMCACHE_LOG_LEVEL", "value": "DEBUG"},
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": "8",
                                    "memory": "16Gi",
                                    "nvidia.com/gpu": str(args.tp),
                                },
                                "limits": {
                                    "cpu": "16",
                                    "memory": "32Gi",
                                    "nvidia.com/gpu": str(args.tp),
                                },
                            },
                            "volumeMounts": [
                                {"name": "shm", "mountPath": "/dev/shm"},
                                {
                                    "name": "config",
                                    "mountPath": "/config",
                                    "readOnly": True,
                                },
                                {"name": "models", "mountPath": "/models"},
                                {"name": "artifacts", "mountPath": "/artifacts"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "4Gi"},
                        },
                        {"name": "config", "configMap": {"name": "launcher"}},
                        {
                            "name": "models",
                            "hostPath": {
                                "path": str(args.model_cache),
                                "type": "DirectoryOrCreate",
                            },
                        },
                        {"name": "artifacts", "emptyDir": {}},
                    ],
                },
            }
        )
        items.append(service(name, run, {"engine": 8081, "mp": 8080, "events": 20080}))
    for item in items:
        if item["kind"] != "Pod" or not item["metadata"]["name"].startswith("worker-"):
            continue
        spec = item["spec"]
        worker = spec["containers"][0]
        worker["env"].append({"name": "PYTHONDONTWRITEBYTECODE", "value": "1"})
        worker["env"].extend(
            [{"name": "CPATH", "value": ""}, {"name": "LIBRARY_PATH", "value": ""}]
        )
        if args.engine == "vllm":
            worker["env"].append({"name": "VLLM_PLUGINS", "value": ""})
        else:
            worker["env"].append({"name": "FLASHINFER_USE_CUDA_NORM", "value": "1"})
        worker["volumeMounts"].extend(
            {
                "name": "config",
                "mountPath": "/opt/mp-kv-events/" + name,
                "subPath": name,
                "readOnly": True,
            }
            for name in runtime_files
        )
        worker["resources"]["requests"]["ephemeral-storage"] = "128Mi"
        worker["resources"]["limits"]["ephemeral-storage"] = "256Mi"
        for volume in spec["volumes"]:
            if volume["name"] == "artifacts":
                volume.pop("emptyDir")
                volume["hostPath"] = {
                    "path": str(
                        args.model_cache.parent
                        / "runs"
                        / run
                        / item["metadata"]["name"]
                    ),
                    "type": "DirectoryOrCreate",
                }
        spec["volumes"].append(
            {"name": "tmp", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}}
        )
        worker["volumeMounts"].append({"name": "tmp", "mountPath": "/tmp"})
        for name, path in (
            ("jit-cache", "/root/.cache"),
            ("cuda-cache", "/root/.nv"),
            ("triton-cache", "/root/.triton"),
        ):
            spec["volumes"].append(
                {
                    "name": name,
                    "hostPath": {
                        "path": str(
                            args.model_cache.parent
                            / "compiler-cache"
                            / args.engine
                            / item["metadata"]["name"]
                            / name
                        ),
                        "type": "DirectoryOrCreate",
                    },
                }
            )
            worker["volumeMounts"].append({"name": name, "mountPath": path})
            worker["volumeMounts"].append(
                {"name": name, "mountPath": path.replace("/root/", "/home/dynamo/")}
            )
    if external_runtime:
        for item in items:
            if item["kind"] != "Pod":
                continue
            spec = item["spec"]
            container = spec["containers"][0]
            if container["image"] != image:
                continue
            container["command"] = [
                "python3",
                "/opt/mp-kv-events/verify_runtime.py",
                "--exec",
                *container["command"],
            ]
            for index, directory in enumerate(external_runtime["directories"]):
                name = f"runtime-{index}"
                spec.setdefault("volumes", []).append(
                    {
                        "name": name,
                        "hostPath": {
                            "path": directory["host_path"],
                            "type": "Directory",
                        },
                    }
                )
                container.setdefault("volumeMounts", []).append(
                    {
                        "name": name,
                        "mountPath": directory["package_path"],
                        "readOnly": True,
                    }
                )
    return items


def deployment(args: argparse.Namespace) -> dict:
    """Read the saved deployment and reject context or namespace identity drift."""
    state = json.loads((args.state / "deployment.json").read_text())
    context = command(kubectl("config", "current-context")).stdout.strip()
    if context != state["context"]:
        raise ValueError(
            f"Current context {context!r} differs from deployment {state['context']!r}"
        )
    namespace = json.loads(
        command(kubectl("get", "namespace", NAMESPACE, "-o", "json")).stdout
    )
    if namespace["metadata"]["uid"] != state["namespace_uid"]:
        raise ValueError("Namespace UID changed since deployment")
    return state


def deploy(args: argparse.Namespace) -> int:
    """Apply the recorded, labeled manifest only after a successful preflight."""
    if preflight(args):
        return 2
    lock = read_lock(args.lock)
    build = json.loads((args.state / f"build-{args.engine}.json").read_text())
    image_lock = build["source"]["lock"]
    for component in (args.engine, "dynamo"):
        keys = ["version", "revision"]
        if component == args.engine or args.engine == "sglang":
            keys.append("bridge_patch_sha256")
        if any(
            image_lock[component].get(key) != lock[component].get(key) for key in keys
        ):
            raise ValueError(
                f"Built image has stale {component} source pins; rebuild it"
            )
    if image_lock["model"] != lock["model"]:
        raise ValueError("Built image and deployment model pins differ")
    run = uuid.uuid4().hex[:12]
    manifest = {
        "apiVersion": "v1",
        "kind": "List",
        "items": render(args, lock, build["image"], run, build.get("external_runtime")),
    }
    # Refuse to replace any existing serving run; cleanup requires its identity.
    existing = json.loads(
        command(
            kubectl("get", "pods", "-n", NAMESPACE, "-l", "component", "-o", "json"),
            check=False,
        ).stdout
        or '{"items":[]}'
    )
    if existing["items"]:
        raise ValueError(
            "A serving run already exists in kv-events-e2e; "
            "collect and clean up that run first"
        )
    command(kubectl("create", "namespace", NAMESPACE), check=False)
    write_json(args.state / "manifest.json", manifest)
    namespace = json.loads(
        command(kubectl("get", "namespace", NAMESPACE, "-o", "json")).stdout
    )
    state = {
        "run": run,
        "context": command(kubectl("config", "current-context")).stdout.strip(),
        "namespace_uid": namespace["metadata"]["uid"],
        "engine": args.engine,
        "tp": args.tp,
        "gpu_affinity_workers": args.tp,
        "transport": args.transport,
        "chunk_size": args.chunk_size,
        "native_block_size": 16,
        "event_log_size": args.event_log_size,
        "events_disabled": args.events_disabled,
        "node": args.node,
        "image": build["image"],
        "external_runtime": build.get("external_runtime"),
        "lock": lock,
    }
    write_json(args.state / "deployment.json", state)
    print(command(kubectl("apply", "-f", str(args.state / "manifest.json"))).stdout)
    return 0


def reconfigure_events(args: argparse.Namespace) -> int:
    """Reload only event publication mode while retaining both GPU allocations."""
    state = deployment(args)
    if state["events_disabled"] == args.events_disabled:
        raise ValueError("Requested event mode already active")
    current = json.loads(
        command(
            kubectl("get", "configmap", "-n", NAMESPACE, "launcher", "-o", "json")
        ).stdout
    )
    if current["metadata"]["labels"][LABEL] != state["run"]:
        raise ValueError("Launcher ConfigMap does not belong to this serving run")
    previous = json.loads(current["data"]["launcher.json"])
    updated = launcher(
        state["engine"],
        state["tp"],
        state["transport"],
        state["chunk_size"],
        state["lock"],
        state["event_log_size"],
        args.events_disabled,
    )
    updated["runtime_files_sha256"] = previous["runtime_files_sha256"]
    if "external_runtime" in previous:
        updated["external_runtime"] = previous["external_runtime"]
    content = json.dumps(updated)
    checksum = hashlib.sha256(content.encode()).hexdigest()
    current["data"]["launcher.json"] = content
    args.artifacts.mkdir(parents=True, exist_ok=True)
    write_json(
        args.artifacts / "event-mode-change.json",
        {
            "run": state["run"],
            "previous": previous,
            "updated": updated,
            "launcher_sha256": checksum,
            "events_disabled": args.events_disabled,
        },
    )
    state["pending_event_mode_change"] = {
        "events_disabled": args.events_disabled,
        "launcher_sha256": checksum,
    }
    write_json(args.state / "deployment.json", state)
    subprocess.run(
        kubectl("replace", "-f", "-"), input=json.dumps(current), text=True, check=True
    )
    script = """
from pathlib import Path
import hashlib, json, sys, time, urllib.request
expected = sys.argv[1]
deadline = time.monotonic() + 180
while (hashlib.sha256(Path('/config/launcher.json').read_bytes()).hexdigest()
       != expected):
    if time.monotonic() > deadline:
        raise TimeoutError('ConfigMap projection checksum did not converge')
    time.sleep(1)
request = urllib.request.Request('http://127.0.0.1:9900/reload',
    data=json.dumps({'launcher_sha256': expected}).encode(),
    headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(request, timeout=900) as response:
    assert response.status == 200
print(json.dumps({'launcher_sha256': expected, 'ready_ns': time.time_ns()}))
"""
    for replica in ("a", "b"):
        result = subprocess.run(
            kubectl(
                "exec",
                "-n",
                NAMESPACE,
                "worker-" + replica,
                "--",
                "python3",
                "-c",
                script,
                checksum,
            ),
            capture_output=True,
            text=True,
            timeout=1100,
            check=True,
        )
        (args.artifacts / f"reload-{replica}.log").write_text(
            result.stdout + result.stderr
        )
    write_json(args.artifacts / "previous-deployment.json", state)
    state["events_disabled"] = args.events_disabled
    state.pop("pending_event_mode_change", None)
    state["event_mode_launcher_sha256"] = checksum
    write_json(args.state / "deployment.json", state)
    write_json(args.artifacts / "deployment.json", state)
    return 0


def collect(args: argparse.Namespace) -> int:
    """Collect only this run's logs, native events, metrics and resolved image IDs."""
    state = deployment(args)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts / "deployment.json", state)
    pods = json.loads(
        command(
            kubectl(
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                f"{LABEL}={state['run']}",
                "-o",
                "json",
            )
        ).stdout
    )
    for pod in pods["items"]:
        name = pod["metadata"]["name"]
        write_json(args.artifacts / name / "status.json", pod["status"])
        result = command(
            kubectl("logs", "-n", NAMESPACE, name, "--timestamps"), check=False
        )
        (args.artifacts / name / "container.log").write_text(
            result.stdout + result.stderr
        )
        if name.startswith("worker-"):
            result = command(
                kubectl(
                    "cp", f"{NAMESPACE}/{name}:/artifacts/.", str(args.artifacts / name)
                ),
                check=False,
            )
            (args.artifacts / name / "collection.log").write_text(
                result.stdout + result.stderr
            )
    return 0


def cleanup(args: argparse.Namespace) -> int:
    """Delete only the saved run's resources, preserving namespace/model data."""
    state = deployment(args)
    print(
        command(
            kubectl(
                "delete",
                "pod,service,configmap",
                "-n",
                NAMESPACE,
                "-l",
                f"{LABEL}={state['run']}",
                "--wait=true",
            )
        ).stdout
    )
    if args.remove_image:
        command(["sudo", "-n", "k3s", "ctr", "images", "remove", state["image"]])
    return 0


def build(args: argparse.Namespace) -> int:
    """Build and import a content-addressed local image from the current sources.

    A source manifest hashes every input file, including uncommitted changes.
    The resulting image ID and immutable local tag are persisted for deploy.
    """
    lock = read_lock(args.lock)
    pin = lock[args.engine]
    runtime_build = None
    dynamo_source = args.dynamo_source or REPO.parent / "dynamo-pinned"
    needs_runtime = args.engine == "sglang" and "bridge_revision" in lock["dynamo"]
    if needs_runtime or args.dynamo_wheel or args.dynamo_build_receipt:
        if args.engine != "sglang" or not (
            args.dynamo_wheel and args.dynamo_build_receipt
        ):
            raise ValueError(
                "SGLang replay requires --dynamo-wheel and --dynamo-build-receipt"
            )
        # Third Party
        from incremental_image import verify_dynamo_build

        runtime_build = verify_dynamo_build(
            args.dynamo_wheel, args.dynamo_build_receipt, dynamo_source, lock
        )
    engine_source = REPO.parent / args.engine
    source_tree = command(
        ["git", "-C", str(engine_source), "rev-parse", "HEAD^{tree}"]
    ).stdout.strip()
    if source_tree != pin["bridge_tree"]:
        raise ValueError("Engine source tree differs from the pinned companion patch")
    if command(["git", "-C", str(engine_source), "diff", "HEAD", "--name-only"]).stdout:
        raise ValueError("Engine has uncommitted changes outside the pinned tree")
    command(
        [
            "git",
            "-C",
            str(engine_source),
            "merge-base",
            "--is-ancestor",
            pin["revision"],
            "HEAD",
        ]
    )
    with tempfile.TemporaryDirectory(
        prefix="mp-kv-events-", dir=args.build_dir
    ) as temp:
        context = Path(temp)
        (context / "dynamo").mkdir()
        (context / "dynamo-runtime").mkdir()
        manifests = {}
        checkouts = [("lmcache", REPO), ("engine", engine_source)]
        if runtime_build is not None:
            checkouts.append(("dynamo", dynamo_source))
            shutil.copyfile(
                args.dynamo_wheel, context / "dynamo-runtime/dynamo-runtime.whl"
            )
            write_json(context / "dynamo-runtime/dynamo-build.json", runtime_build)
        for name, checkout in checkouts:
            paths = command(
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
            ).stdout.split("\0")
            manifest = {}
            for relative in sorted(set(paths) - {""}):
                source = checkout / relative
                if not source.is_file():
                    continue
                target = context / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifests[name] = {
                "head": command(
                    ["git", "-C", str(checkout), "rev-parse", "HEAD"]
                ).stdout.strip(),
                "files": manifest,
            }
        source = {"inputs": manifests, "lock": lock}
        if runtime_build is not None:
            source["dynamo_runtime_build"] = runtime_build
        source_digest = hashlib.sha256(
            json.dumps(source, sort_keys=True).encode()
        ).hexdigest()
        source["sha256"] = source_digest
        write_json(context / "source.json", source)
        image = f"docker.io/library/mp-kv-events-{args.engine}:{source_digest[:20]}"
        argv = [
            "docker",
            "build",
            "--network=host",
            "-f",
            str(HERE / "Dockerfile"),
            "-t",
            image,
            "--build-arg",
            f"BASE_IMAGE={pin['base_image']}",
            "--build-arg",
            f"ENGINE={args.engine}",
            "--build-arg",
            f"ENGINE_VERSION={pin['version']}",
            "--build-arg",
            f"ENGINE_INDEX={pin['wheel_index']}",
            "--build-arg",
            "LMCACHE_VERSION=0.5.5.dev0",
            str(context),
        ]
        # Native CUDA compilation can exceed the bounded read-command timeout.
        subprocess.run(argv, check=True)
        compacted = compact(image, args.build_dir)
        image = compacted["image"]
        image_id = compacted["image_id"]
        archive = args.build_dir / f"{source_digest[:20]}.tar"
        subprocess.run(["docker", "save", "-o", str(archive), image], check=True)
        compressed_archive = archive.with_suffix(".oci.tar")
        compression = compress(archive, compressed_archive)
        write_json(compressed_archive.with_suffix(".json"), compression)
        if not args.no_import:
            subprocess.run(
                [
                    "sudo",
                    "-n",
                    "python3",
                    str(HERE / "import_image.py"),
                    str(compressed_archive),
                    "--scratch",
                    str(args.build_dir / "import-staging"),
                    "--report",
                    str(args.build_dir / f"{source_digest[:20]}-import.json"),
                    "--unpacked-bytes",
                    str(compression["unpacked_bytes"]),
                ],
                check=True,
            )
        write_json(
            args.state / f"build-{args.engine}.json",
            {
                "image": image,
                "image_id": image_id,
                "source": source,
                "compaction": compacted,
                "compression": compression,
            },
        )
    return 0


def main() -> int:
    """Parse the command contract and execute a single run-scoped operation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "preflight",
            "reconfigure-events",
            "deploy",
            "collect",
            "cleanup",
            "render",
            "build",
            "test",
        ),
    )
    parser.add_argument("--cluster", choices=("k3s",), default="k3s")
    parser.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--router", choices=("dynamo",), default="dynamo")
    parser.add_argument("--tp", choices=(1, 2), type=int, default=1)
    parser.add_argument("--transport", choices=("zmq", "grpc"), default="zmq")
    parser.add_argument("--chunk-size", choices=(16, 256), type=int, default=256)
    parser.add_argument("--node", default="")
    parser.add_argument("--event-log-size", type=int, default=32768)
    parser.add_argument("--events-disabled", action="store_true")
    parser.add_argument("--remove-image", action="store_true")
    parser.add_argument("--no-import", action="store_true")
    parser.add_argument("--dynamo-wheel", type=Path)
    parser.add_argument("--dynamo-build-receipt", type=Path)
    parser.add_argument("--dynamo-source", type=Path)
    parser.add_argument("--namespace", choices=(NAMESPACE,), default=NAMESPACE)
    parser.add_argument("--lock", type=Path, default=HERE / "versions.lock.yaml")
    parser.add_argument(
        "--state", type=Path, default=REPO / "artifacts/mp-kv-events/state"
    )
    parser.add_argument(
        "--artifacts", type=Path, default=REPO / "artifacts/mp-kv-events/preflight"
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=WORK_ROOT / "models",
    )
    parser.add_argument("--build-dir", type=Path, default=WORK_ROOT / "build")
    parser.add_argument(
        "--cases", nargs="+", help="Named cases within the selected suite"
    )
    parser.add_argument(
        "--suite",
        choices=("core", "recovery", "overflow", "tp", "overhead", "capacity"),
        default="core",
    )
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts.parent.mkdir(parents=True, exist_ok=True)
    if args.action == "render":
        built_image = json.loads((args.state / f"build-{args.engine}.json").read_text())
        print(
            json.dumps(
                {
                    "apiVersion": "v1",
                    "kind": "List",
                    "items": render(
                        args,
                        read_lock(args.lock),
                        built_image["image"],
                        "render",
                        built_image.get("external_runtime"),
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.action == "test":
        # Third Party
        from scenarios import test

        return test(args)
    return {
        "preflight": preflight,
        "reconfigure-events": reconfigure_events,
        "deploy": deploy,
        "collect": collect,
        "cleanup": cleanup,
        "build": build,
    }[args.action](args)


if __name__ == "__main__":
    sys.exit(main())
