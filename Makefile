# Convenience wrappers for the MP Memory Coordinator development and E2E
# flows (PLAN: Standalone LMCache MP Memory Coordinator). Python commands use
# `uv run` so they resolve the project environment.

E2E_DIR       := tests/e2e/mp_memory_coordinator
KIND_CLUSTER  := lmcache-memcoord-e2e
KIND_CONTEXT  := kind-$(KIND_CLUSTER)
E2E_NAMESPACE := lmcache-memcoord-e2e
ARTIFACTS     ?= artifacts/e2e/$(shell date +%Y%m%d-%H%M%S)

.PHONY: help test-mp-memory-coordinator run-mp-memory-allocation-mock \
        test-mp-memory-allocation-mock e2e-mp-memory-coordinator-local \
        e2e-mp-memory-coordinator-images e2e-mp-memory-coordinator-kind \
        e2e-mp-memory-coordinator-kind-collect e2e-mp-memory-coordinator-hardware \
        test-outside-api-conformance

help:
	@echo "test-mp-memory-coordinator          unit + contract tests of the coordinator"
	@echo "run-mp-memory-allocation-mock       docker compose up the strict mock allocator"
	@echo "test-mp-memory-allocation-mock      tests of the mock allocator"
	@echo "e2e-mp-memory-coordinator-local     E2E with local subprocesses (no cluster)"
	@echo "e2e-mp-memory-coordinator-kind      build images, create kind cluster, run E2E, delete"
	@echo "e2e-mp-memory-coordinator-hardware  real two-worker Device-DAX gate (manual)"
	@echo "test-outside-api-conformance        frozen outside API conformance: OUTSIDE_API_URL=http://host:port"

test-mp-memory-coordinator:
	uv run pytest -q tests/v1/mp_memory_coordinator

run-mp-memory-allocation-mock:
	docker compose -f $(E2E_DIR)/dev/docker-compose.yaml up --build

test-mp-memory-allocation-mock:
	uv run pytest -q $(E2E_DIR)/test_mock_memory_allocation_service.py

# Conformance of ANY implementation of the frozen outside API (real service or
# mock). Mutating round trip: deallocates one assigned runtime path and
# allocates the same size back to the same node.
test-outside-api-conformance:
	@test -n "$(OUTSIDE_API_URL)" || { echo "OUTSIDE_API_URL is required"; exit 2; }
	uv run pytest -q $(E2E_DIR)/test_outside_api_conformance.py -m outside_api --outside-api-url $(OUTSIDE_API_URL)

e2e-mp-memory-coordinator-local:
	MEMCOORD_E2E_ARTIFACTS=$(ARTIFACTS) uv run pytest -q $(E2E_DIR)

e2e-mp-memory-coordinator-images:
	docker build -t lmcache-memcoord:e2e -f $(E2E_DIR)/Dockerfile .
	docker build -t lmcache-memcoord-scenario:e2e -f $(E2E_DIR)/scenario_server/Dockerfile .
	docker build -t lmcache-memory-allocation-mock:e2e -f $(E2E_DIR)/mock_memory_allocation_service/Dockerfile .

e2e-mp-memory-coordinator-kind: e2e-mp-memory-coordinator-images
	kind create cluster --name $(KIND_CLUSTER) --config $(E2E_DIR)/manifests/kind.yaml
	kind load docker-image --name $(KIND_CLUSTER) \
	  lmcache-memcoord:e2e lmcache-memcoord-scenario:e2e lmcache-memory-allocation-mock:e2e
	kubectl --context $(KIND_CONTEXT) apply -k $(E2E_DIR)/manifests/overlays/kind
	kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) rollout status deploy/scenario-server --timeout=120s
	kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) rollout status deploy/mock-memory-allocation-service --timeout=120s
	kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) rollout status deploy/lmcache-mp-memory-coordinator --timeout=120s
	MEMCOORD_E2E_ARTIFACTS=$(ARTIFACTS) uv run pytest -q $(E2E_DIR) -m kind --kube-context $(KIND_CONTEXT) \
	  || { $(MAKE) e2e-mp-memory-coordinator-kind-collect; kind delete cluster --name $(KIND_CLUSTER); exit 1; }
	kind delete cluster --name $(KIND_CLUSTER)

# Always collected on failure: both test-service logs and audits, the Lease,
# the journal, outside status, DAX status, and usage snapshots.
e2e-mp-memory-coordinator-kind-collect:
	mkdir -p $(ARTIFACTS)/kind
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) logs deploy/scenario-server > $(ARTIFACTS)/kind/scenario-server.log
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) logs deploy/mock-memory-allocation-service > $(ARTIFACTS)/kind/mock-allocator.log
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) logs deploy/lmcache-mp-memory-coordinator > $(ARTIFACTS)/kind/memcoord.log
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) get lease lmcache-mp-memory-coordinator -o yaml > $(ARTIFACTS)/kind/lease.yaml
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) exec deploy/lmcache-mp-memory-coordinator -- cat /var/lib/lmcache-memory-coordinator/journal.json > $(ARTIFACTS)/kind/journal.json
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) exec deploy/scenario-server -- python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:9091/__test/audit').read().decode())" > $(ARTIFACTS)/kind/scenario-audit.json
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) exec deploy/mock-memory-allocation-service -- python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:9090/__test/audit').read().decode())" > $(ARTIFACTS)/kind/allocator-audit.json
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) exec deploy/mock-memory-allocation-service -- python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/api/v2/apps/lmcache').read().decode())" > $(ARTIFACTS)/kind/outside-status.json
	-kubectl --context $(KIND_CONTEXT) -n $(E2E_NAMESPACE) exec deploy/scenario-server -- python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8081/reconfigure/dax/status').read().decode());print(urllib.request.urlopen('http://127.0.0.1:8082/reconfigure/dax/status').read().decode());print(urllib.request.urlopen('http://127.0.0.1:9300/instances/usage').read().decode())" > $(ARTIFACTS)/kind/dax-and-usage.json

# Manual / nightly gate on real Device-DAX hardware (PLAN.md section 9).
# Required: KUBE_CONTEXT DONOR_NODE RECEIVER_NODE DAX_INVENTORY_FILE MOVE_SIZE_GIB OUTSIDE_API_URL
e2e-mp-memory-coordinator-hardware:
	@test -n "$(KUBE_CONTEXT)" || { echo "KUBE_CONTEXT is required"; exit 2; }
	@test -n "$(DAX_INVENTORY_FILE)" || { echo "DAX_INVENTORY_FILE is required"; exit 2; }
	@test -n "$(OUTSIDE_API_URL)" || { echo "OUTSIDE_API_URL is required"; exit 2; }
	MEMCOORD_E2E_ARTIFACTS=$(ARTIFACTS)/hardware uv run pytest -q $(E2E_DIR) -m hardware \
	  --kube-context $(KUBE_CONTEXT) --e2e-namespace $(or $(TEST_NAMESPACE),$(E2E_NAMESPACE)) \
	  --donor-node $(DONOR_NODE) --receiver-node $(RECEIVER_NODE) \
	  --dax-inventory-file $(DAX_INVENTORY_FILE) --move-size-gib $(MOVE_SIZE_GIB) \
	  --outside-api-url $(OUTSIDE_API_URL) --dax-state-timeout $(or $(DAX_STATE_TIMEOUT),120s)
