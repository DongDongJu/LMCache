# SPDX-License-Identifier: Apache-2.0
"""Static contract tests for the MP Memory Coordinator Kubernetes example."""

# Standard
from pathlib import Path

# Third Party
import yaml

# First Party
from lmcache.v1.mp_memory_coordinator.adoption import load_adoption_file
from lmcache.v1.mp_memory_coordinator.config import LeaderElectionMode, load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "mp_memory_coordinator" / "kubernetes"
CONFIG_PATH = EXAMPLE_DIR / "config" / "mp-memory-coordinator.yaml"
ADOPTION_PATH = EXAMPLE_DIR / "config" / "adoption.yaml"
WORKLOAD_NAME = "lmcache-mp-memory-coordinator"
RESOURCE_FILES = {
    "namespace.yaml",
    "serviceaccount.yaml",
    "rbac.yaml",
    "lease.yaml",
    "pvc.yaml",
    "deployment.yaml",
    "service.yaml",
}


def _documents(path: Path) -> list[dict]:
    """Parse all non-empty YAML documents from ``path``."""
    assert path.is_file(), f"missing Kubernetes example file: {path}"
    documents = [
        document for document in yaml.safe_load_all(path.read_text()) if document
    ]
    assert documents, f"{path} contains no YAML document"
    assert all(isinstance(document, dict) for document in documents), path
    return documents


def _only(resources: list[dict], kind: str) -> dict:
    """Return the example's sole resource of ``kind``."""
    matches = [resource for resource in resources if resource.get("kind") == kind]
    assert len(matches) == 1, f"expected one {kind}, found {len(matches)}"
    return matches[0]


def _named(items: list[dict], name: str) -> dict:
    """Return the sole list item whose ``name`` equals ``name``."""
    matches = [item for item in items if item.get("name") == name]
    assert len(matches) == 1, f"expected one item named {name}, found {len(matches)}"
    return matches[0]


def _generated_files(generator: dict) -> dict[str, str]:
    """Return ``ConfigMap`` output-key to source-path mappings."""
    files: dict[str, str] = {}
    for value in generator.get("files", []):
        assert isinstance(value, str), value
        key, separator, source = value.partition("=")
        if not separator:
            source = key
            key = Path(source).name
        files[key] = source
    return files


def test_kubernetes_example_preserves_the_safe_single_writer_contract() -> None:
    """Validate the example's config and production-safety manifest invariants."""
    expected_files = {
        "kustomization.yaml",
        "config/mp-memory-coordinator.yaml",
        "config/adoption.yaml",
        *RESOURCE_FILES,
    }
    missing = sorted(
        relative
        for relative in expected_files
        if not (EXAMPLE_DIR / relative).is_file()
    )
    assert missing == [], f"missing Kubernetes example files: {missing}"

    kustomization = _documents(EXAMPLE_DIR / "kustomization.yaml")[0]
    assert kustomization["kind"] == "Kustomization"
    assert set(kustomization.get("resources", [])) == RESOURCE_FILES

    resources = [
        document
        for relative in sorted(RESOURCE_FILES)
        for document in _documents(EXAMPLE_DIR / relative)
    ]
    expected_kinds = {
        "Namespace",
        "ServiceAccount",
        "Role",
        "RoleBinding",
        "Lease",
        "PersistentVolumeClaim",
        "Deployment",
        "Service",
    }
    assert {resource.get("kind") for resource in resources} == expected_kinds
    for resource in resources:
        assert resource.get("apiVersion"), resource
        assert resource.get("metadata", {}).get("name"), resource

    namespace = _only(resources, "Namespace")
    assert kustomization.get("namespace") == namespace["metadata"]["name"]
    service_account = _only(resources, "ServiceAccount")
    role = _only(resources, "Role")
    role_binding = _only(resources, "RoleBinding")
    lease = _only(resources, "Lease")
    pvc = _only(resources, "PersistentVolumeClaim")
    deployment = _only(resources, "Deployment")
    service = _only(resources, "Service")
    assert deployment["metadata"]["name"] == WORKLOAD_NAME

    config = load_config(CONFIG_PATH)
    assert config.actuation_enabled is False
    assert config.leader_election is LeaderElectionMode.KUBERNETES
    assert config.adoption_file == ""
    load_adoption_file(ADOPTION_PATH)

    deployment_spec = deployment["spec"]
    assert deployment_spec["replicas"] == 1
    assert deployment_spec["strategy"]["type"] == "Recreate"
    pod_spec = deployment_spec["template"]["spec"]
    assert pod_spec["terminationGracePeriodSeconds"] >= 300
    assert pod_spec["serviceAccountName"] == service_account["metadata"]["name"]
    container = _named(pod_spec["containers"], "coordinator")
    assert container["image"] == "lmcache/standalone:REPLACE_WITH_RELEASE_TAG"
    assert kustomization["images"] == [
        {
            "name": "lmcache/standalone",
            "newTag": "REPLACE_WITH_RELEASE_TAG",
        }
    ]
    assert container["livenessProbe"]["tcpSocket"] == {"port": "http"}
    assert container["readinessProbe"]["httpGet"] == {
        "path": "/readyz",
        "port": "http",
    }

    config_mount = next(
        mount
        for mount in container["volumeMounts"]
        if mount["mountPath"] == "/etc/lmcache"
    )
    assert config_mount.get("readOnly") is True
    journal_mount = next(
        mount
        for mount in container["volumeMounts"]
        if mount["mountPath"] == config.state_directory
    )
    config_volume = _named(pod_spec["volumes"], config_mount["name"])
    journal_volume = _named(pod_spec["volumes"], journal_mount["name"])
    assert (
        journal_volume["persistentVolumeClaim"]["claimName"] == pvc["metadata"]["name"]
    )

    config_map_name = config_volume["configMap"]["name"]
    generators = [
        generator
        for generator in kustomization.get("configMapGenerator", [])
        if generator.get("name") == config_map_name
    ]
    assert len(generators) == 1, f"missing ConfigMap generator {config_map_name}"
    assert _generated_files(generators[0]) == {
        "mp-memory-coordinator.yaml": "config/mp-memory-coordinator.yaml",
        "adoption.yaml": "config/adoption.yaml",
    }

    assert pvc["spec"]["accessModes"] == ["ReadWriteOncePod"]
    assert config.lease_name == lease["metadata"]["name"]
    assert lease["spec"]["leaseDurationSeconds"] == config.lease_duration_seconds

    assert role["rules"] == [
        {
            "apiGroups": ["coordination.k8s.io"],
            "resources": ["leases"],
            "resourceNames": [lease["metadata"]["name"]],
            "verbs": ["get", "update"],
        }
    ]
    assert role_binding["roleRef"] == {
        "apiGroup": "rbac.authorization.k8s.io",
        "kind": "Role",
        "name": role["metadata"]["name"],
    }
    assert role_binding["subjects"] == [
        {
            "kind": "ServiceAccount",
            "name": service_account["metadata"]["name"],
        }
    ]

    service_port = _named(service["spec"]["ports"], "http")
    container_port = _named(container["ports"], "http")
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert service_port["port"] == config.http_port
    assert service_port["targetPort"] == "http"
    assert container_port["containerPort"] == config.http_port
