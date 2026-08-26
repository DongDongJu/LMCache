# Strict mock of the outside Memory Allocation service (PLAN.md Phase 1A).
#
# Build from the repository root so the package path is preserved:
#   docker build -t lmcache-memory-allocation-mock:e2e \
#     -f tests/e2e/mp_memory_coordinator/mock_memory_allocation_service/Dockerfile .
#
# The image deliberately does not install LMCache: the mock must stay independent
# from the production client so the two sides cannot drift together.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn pydantic pyyaml httpx

# Keep the `tests.e2e.mp_memory_coordinator...` package path importable.
RUN mkdir -p /app/tests/e2e \
    && touch /app/tests/__init__.py /app/tests/e2e/__init__.py

COPY tests/e2e/mp_memory_coordinator/ /app/tests/e2e/mp_memory_coordinator/

EXPOSE 8080 9090

ENTRYPOINT ["python", "-m", "tests.e2e.mp_memory_coordinator.mock_memory_allocation_service", \
    "--fixture", "/app/tests/e2e/mp_memory_coordinator/fixtures/two_server_local_dax.yaml", \
    "--public-host", "0.0.0.0", "--public-port", "8080", \
    "--admin-host", "0.0.0.0", "--admin-port", "9090"]
