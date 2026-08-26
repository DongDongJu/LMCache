# SPDX-License-Identifier: Apache-2.0
# Deterministic scenario server (fake MP Coordinator + fake donor/receiver MP).
# Build context is the repository root:
#   docker build -t lmcache-memcoord-scenario:e2e \
#     -f tests/e2e/mp_memory_coordinator/scenario_server/Dockerfile .
FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi uvicorn pydantic pyyaml

WORKDIR /app
COPY tests/__init__.py /app/tests/__init__.py
COPY tests/e2e/__init__.py /app/tests/e2e/__init__.py
COPY tests/e2e/mp_memory_coordinator/ /app/tests/e2e/mp_memory_coordinator/

ENV PYTHONUNBUFFERED=1
# coordinator, donor MP (+alt), receiver MP (+alt), admin
EXPOSE 9300 8081 8181 8082 8182 9091

# Advertised identity comes from the environment in Kubernetes:
#   SCENARIO_ADVERTISE_IP, SCENARIO_DONOR_HTTP_PORT, SCENARIO_RECEIVER_HTTP_PORT
ENTRYPOINT ["python", "-m", "tests.e2e.mp_memory_coordinator.scenario_server", \
    "--fixture", "/app/tests/e2e/mp_memory_coordinator/fixtures/two_server_local_dax.yaml", \
    "--host", "0.0.0.0", \
    "--coordinator-port", "9300", \
    "--donor-port", "8081", \
    "--receiver-port", "8082", \
    "--admin-port", "9091"]
