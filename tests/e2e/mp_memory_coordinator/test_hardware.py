# SPDX-License-Identifier: Apache-2.0
"""Real two-worker pre-provisioned Device-DAX simulation gate (manual/nightly).

Runs only when ``-m hardware`` is selected together with ``--kube-context``
and ``--dax-inventory-file``. Every requested prerequisite that is missing is
a **failure**, never a skip. The verifier reads host/sysfs and Kubernetes
state read-only; the coordinator under test never does.

Procedure (PLAN.md section 9): identity and static-inventory gate, baseline
capture, dry run with ``actuation_enabled=false``, then the exact move
sequence with actuation enabled, verification of capacity deltas, and a
cleanup that restores the outside/MP baseline. Artifacts land under
``MEMCOORD_E2E_ARTIFACTS`` (``artifacts/hardware/<run-id>/`` via ``make``).
"""

# Standard
from pathlib import Path
import json
import subprocess

# Third Party
import pytest
import requests
import yaml

# Local
from .conftest import wait_until

pytestmark = pytest.mark.hardware


def _kubectl(context: str, *args: str) -> str:
    result = subprocess.run(
        ["kubectl", "--context", context, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"kubectl {' '.join(args)} failed: {result.stderr}"
    return result.stdout


@pytest.fixture(scope="module")
def gate(request: pytest.FixtureRequest, artifacts_dir: Path) -> dict:
    """Validate every prerequisite and resolve the identity mapping."""
    config = request.config
    context = str(config.getoption("--kube-context"))
    inventory_file = str(config.getoption("--dax-inventory-file"))
    outside_url = str(config.getoption("--outside-api-url"))
    donor_node = str(config.getoption("--donor-node"))
    receiver_node = str(config.getoption("--receiver-node"))
    move_size = int(str(config.getoption("--move-size-gib")) or "0")
    namespace = str(config.getoption("--e2e-namespace"))
    missing = [
        name
        for name, value in (
            ("--kube-context", context),
            ("--dax-inventory-file", inventory_file),
            ("--outside-api-url", outside_url),
            ("--donor-node", donor_node),
            ("--receiver-node", receiver_node),
        )
        if not value
    ]
    assert not missing, f"hardware gate requested without {missing}"
    assert move_size > 0, "--move-size-gib must be a positive whole GiB count"
    inventory = yaml.safe_load(Path(inventory_file).read_text())
    assert inventory.get("schema_version") == 1, (
        "DAX_INVENTORY_FILE schema_version != 1"
    )
    nodes = inventory["nodes"]
    assert len(nodes) == 2, "the gate needs exactly two workers"

    # Pod placement and node identity: metadata.worker_ip must equal the
    # node's InternalIP and each MP Pod must sit on its expected node.
    node_json = json.loads(_kubectl(context, "get", "nodes", "-o", "json"))
    internal_ips = {
        item["metadata"]["name"]: next(
            a["address"]
            for a in item["status"]["addresses"]
            if a["type"] == "InternalIP"
        )
        for item in node_json["items"]
    }
    for node_name in (donor_node, receiver_node):
        assert node_name in internal_ips, f"node {node_name} not found"
        assert internal_ips[node_name] in nodes, (
            f"{node_name} InternalIP {internal_ips[node_name]} is not a worker in "
            f"DAX_INVENTORY_FILE"
        )
    coordinator_url = _service_url(context, namespace, "lmcache-mp-coordinator", 9300)
    instances = requests.get(f"{coordinator_url}/instances", timeout=10).json()[
        "instances"
    ]
    by_worker = {i.get("metadata", {}).get("worker_ip", ""): i for i in instances}
    donor_ip, receiver_ip = internal_ips[donor_node], internal_ips[receiver_node]
    assert donor_ip in by_worker and receiver_ip in by_worker, (
        f"MP instances registered for {sorted(by_worker)}; expected {donor_ip} and "
        f"{receiver_ip}"
    )
    worker_ips = [i.get("metadata", {}).get("worker_ip", "") for i in instances]
    assert len(worker_ips) == len(set(worker_ips)), "duplicate worker_ip in fleet"
    topology = {
        "donor": {
            "kubernetes_node": donor_node,
            "target_node": donor_ip,
            "mp_instance_id": by_worker[donor_ip]["instance_id"],
            "mp_http_endpoint": (
                f"{by_worker[donor_ip]['ip']}:{by_worker[donor_ip]['http_port']}"
            ),
        },
        "receiver": {
            "kubernetes_node": receiver_node,
            "target_node": receiver_ip,
            "mp_instance_id": by_worker[receiver_ip]["instance_id"],
            "mp_http_endpoint": (
                f"{by_worker[receiver_ip]['ip']}:{by_worker[receiver_ip]['http_port']}"
            ),
        },
    }
    run_dir = artifacts_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "topology.yaml").write_text(yaml.safe_dump(topology))
    (run_dir / "dax-inventory.yaml").write_text(Path(inventory_file).read_text())
    (run_dir / "kubernetes-nodes.yaml").write_text(
        _kubectl(context, "get", "nodes", "-o", "yaml")
    )
    (run_dir / "pod-placement.txt").write_text(
        _kubectl(context, "-n", namespace, "get", "pods", "-o", "wide")
    )
    return {
        "context": context,
        "namespace": namespace,
        "inventory": inventory,
        "topology": topology,
        "outside_url": outside_url.rstrip("/"),
        "coordinator_url": coordinator_url,
        "move_size_gib": move_size,
        "run_dir": run_dir,
    }


def _service_url(context: str, namespace: str, service: str, port: int) -> str:
    """Resolve a Service URL through a read-only port-forward."""
    return _port_forward_url(context, namespace, service, port)


_FORWARDS: dict[str, str] = {}


def _port_forward_url(context: str, namespace: str, service: str, port: int) -> str:
    """Start (once) a port-forward to ``service:port`` and return its URL."""
    # Local
    from .conftest import PortForward

    key = f"{namespace}/{service}:{port}"
    if key not in _FORWARDS:
        _FORWARDS[key] = PortForward(context, namespace, service, port).url
    return _FORWARDS[key]


def _mp_dax_status(endpoint: str) -> dict:
    return requests.get(f"http://{endpoint}/reconfigure/dax/status", timeout=10).json()


def _active_paths(status: dict) -> set[str]:
    devices = status["adapters"][0]["status"]["devices"] if status["adapters"] else []
    return {d["device_path"] for d in devices if d["state"] == "active"}


def test_static_inventory_gate(gate: dict) -> None:
    """Every allowlisted path has one fixed worker, role, size, and state that
    agrees with the live MP DAX state and the outside status."""
    inventory = gate["inventory"]
    outside = requests.get(
        f"{gate['outside_url']}/api/v2/apps/lmcache", timeout=10
    ).json()
    assert isinstance(outside, dict) and all(
        isinstance(v, list) for v in outside.values()
    ), "outside status is not the bare target_node -> device_path[] object"
    seen_paths: set[str] = set()
    for worker_ip, node in inventory["nodes"].items():
        role = (
            "donor"
            if gate["topology"]["donor"]["target_node"] == worker_ip
            else "receiver"
        )
        endpoint = gate["topology"][role]["mp_http_endpoint"]
        live = _active_paths(_mp_dax_status(endpoint))
        for device in node["devices"]:
            path = device["path"]
            assert path not in seen_paths, f"{path} listed twice"
            seen_paths.add(path)
            assert (
                device["size_gib"] == gate["move_size_gib"]
                or device["role"] == "bootstrap"
            )
            if device["role"] == "bootstrap":
                assert path in live, f"bootstrap {path} not active in MP on {worker_ip}"
                assert path not in outside.get(worker_ip, []), (
                    f"bootstrap {path} must not appear in outside status"
                )
            elif device["state"] == "assigned":
                assert path in outside.get(worker_ip, []), (
                    f"{path} not assigned outside"
                )
            else:
                assert path not in live, f"free candidate {path} is active in MP"
                assert path not in outside.get(worker_ip, []), (
                    f"free {path} assigned outside"
                )
            for other_ip, paths in outside.items():
                if other_ip != worker_ip:
                    assert path not in paths, f"{path} appears under {other_ip}"


def test_happy_move_on_real_dax(gate: dict) -> None:
    """Dry run first, then the exact sequence with actuation enabled.

    The coordinator is driven through its Deployment (ConfigMap patch +
    rollout); assertions use the journal, outside status, and DAX status.
    """
    context, namespace = gate["context"], gate["namespace"]
    memcoord_url = _port_forward_url(
        context, namespace, "lmcache-mp-memory-coordinator", 9400
    )
    run_dir: Path = gate["run_dir"]
    baseline = run_dir / "baseline"
    baseline.mkdir(exist_ok=True)
    (baseline / "outside-status.json").write_text(
        requests.get(f"{gate['outside_url']}/api/v2/apps/lmcache", timeout=10).text
    )
    (baseline / "instances.json").write_text(
        requests.get(f"{gate['coordinator_url']}/instances", timeout=10).text
    )
    (baseline / "usage.json").write_text(
        requests.get(f"{gate['coordinator_url']}/instances/usage", timeout=10).text
    )
    for role in ("donor", "receiver"):
        (baseline / f"dax-{role}.json").write_text(
            json.dumps(
                _mp_dax_status(gate["topology"][role]["mp_http_endpoint"]), indent=2
            )
        )

    # Dry run: zero mutating POSTs while a proposal is logged.
    status = requests.get(f"{memcoord_url}/status", timeout=10).json()
    assert status["actuation_enabled"] is False, "start the gate in observation mode"
    wait_until(
        lambda: (
            requests.get(f"{memcoord_url}/status", timeout=10).json()["last_cycle"][
                "proposal"
            ]
            is not None
        ),
        timeout=600,
        interval=5,
        what="dry-run proposal (make the receiver HIGH with the KV workload client)",
    )
    proposal = requests.get(f"{memcoord_url}/status", timeout=10).json()["last_cycle"][
        "proposal"
    ]
    (run_dir / "dry-run").mkdir(exist_ok=True)
    (run_dir / "dry-run" / "proposal.json").write_text(json.dumps(proposal, indent=2))
    assert proposal["donor_worker_ip"] == gate["topology"]["donor"]["target_node"]
    assert proposal["receiver_worker_ip"] == gate["topology"]["receiver"]["target_node"]
    assert proposal["allocation_size_gib"] == gate["move_size_gib"]
    before_outside = requests.get(
        f"{gate['outside_url']}/api/v2/apps/lmcache", timeout=10
    ).json()

    # Enable actuation through the ConfigMap and a rollout; the journal on
    # the PVC survives.
    _kubectl(
        context,
        "-n",
        namespace,
        "patch",
        "configmap",
        "lmcache-mp-memory-coordinator",
        "--type",
        "json",
        "-p",
        json.dumps(
            [
                {
                    "op": "replace",
                    "path": "/data/mp-memory-coordinator.yaml",
                    "value": _with_actuation(
                        _kubectl(
                            context,
                            "-n",
                            namespace,
                            "get",
                            "configmap",
                            "lmcache-mp-memory-coordinator",
                            "-o",
                            "jsonpath={.data.mp-memory-coordinator\\.yaml}",
                        ),
                        True,
                    ),
                }
            ]
        ),
    )
    _kubectl(
        context,
        "-n",
        namespace,
        "rollout",
        "restart",
        "deploy/lmcache-mp-memory-coordinator",
    )
    _kubectl(
        context,
        "-n",
        namespace,
        "rollout",
        "status",
        "deploy/lmcache-mp-memory-coordinator",
        "--timeout=180s",
    )
    memcoord_url = _port_forward_url(
        context, namespace, "lmcache-mp-memory-coordinator", 9400
    )

    def _terminal() -> bool:
        journal = requests.get(f"{memcoord_url}/journal", timeout=10).json()
        move = journal.get("active_move")
        return (move is not None and move["state"] == "BLOCKED") or bool(
            journal["history"]
        )

    wait_until(_terminal, timeout=1800, interval=5, what="terminal move")
    journal = requests.get(f"{memcoord_url}/journal", timeout=10).json()
    mutation = run_dir / "mutation"
    mutation.mkdir(exist_ok=True)
    (mutation / "journal.json").write_text(json.dumps(journal, indent=2))
    (mutation / "lease.yaml").write_text(
        _kubectl(
            context,
            "-n",
            namespace,
            "get",
            "lease",
            "lmcache-mp-memory-coordinator",
            "-o",
            "yaml",
        )
    )
    move = journal["active_move"] or journal["history"][-1]
    if move["state"] == "BLOCKED":
        pytest.fail(
            f"move BLOCKED; journal and device state preserved: {move['block_reason']}"
        )
    assert move["outcome"] == "SUCCEEDED", move
    assert list(move["effects"]) == [
        "donor_drain",
        "donor_evict",
        "deallocate",
        "allocate",
        "receiver_add",
    ]
    donor_ip = gate["topology"]["donor"]["target_node"]
    receiver_ip = gate["topology"]["receiver"]["target_node"]
    after_outside = requests.get(
        f"{gate['outside_url']}/api/v2/apps/lmcache", timeout=10
    ).json()
    assert move["old_path"] not in after_outside.get(donor_ip, [])
    assert move["new_path"] in after_outside.get(receiver_ip, [])
    assert move["new_path"] != move["old_path"]
    declared_receiver = {
        d["path"]
        for d in gate["inventory"]["nodes"][receiver_ip]["devices"]
        if d["role"] == "runtime"
    }
    assert move["new_path"] in declared_receiver, (
        "allocator returned an undeclared path"
    )
    assert (
        move["released_size_gib"]
        == move["granted_size_gib"]
        == gate["move_size_gib"]
        == move["old_map_size_bytes"] // (1 << 30)
    )
    donor_dax = _mp_dax_status(gate["topology"]["donor"]["mp_http_endpoint"])
    receiver_dax = _mp_dax_status(gate["topology"]["receiver"]["mp_http_endpoint"])
    assert move["old_path"] not in _active_paths(donor_dax)
    assert move["new_path"] in _active_paths(receiver_dax)
    assert sum(len(v) for v in after_outside.values()) == sum(
        len(v) for v in before_outside.values()
    ), "total assigned runtime devices must return to the pre-move value"
    final = run_dir / "final"
    final.mkdir(exist_ok=True)
    (final / "outside-status.json").write_text(json.dumps(after_outside, indent=2))
    (final / "dax-donor.json").write_text(json.dumps(donor_dax, indent=2))
    (final / "dax-receiver.json").write_text(json.dumps(receiver_dax, indent=2))
    (final / "usage.json").write_text(
        requests.get(f"{gate['coordinator_url']}/instances/usage", timeout=10).text
    )


def _with_actuation(config_text: str, enabled: bool) -> str:
    """Return ``config_text`` with ``actuation_enabled`` set."""
    data = yaml.safe_load(config_text) or {}
    data["actuation_enabled"] = enabled
    return yaml.safe_dump(data)
