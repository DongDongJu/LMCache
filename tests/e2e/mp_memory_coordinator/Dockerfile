# CPU-only image of the MP Memory Coordinator for the CI E2E (kind).
# The coordinator is pure Python (httpx/fastapi/pydantic) but lives inside the
# lmcache package, whose import needs torch; a CPU torch wheel keeps the image
# small enough for CI. Build context is the repository root:
#   docker build -t lmcache-memcoord:e2e -f tests/e2e/mp_memory_coordinator/Dockerfile .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    LMCACHE_TRACK_USAGE=false \
    NO_NATIVE_EXT=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# torch (CPU) in its own layer so source changes do not re-download it.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace/LMCache
COPY . /workspace/LMCache

# requirements/cli.txt is needed because ``lmcache`` discovers every CLI
# command module eagerly (the bench command imports ``openai``).
RUN pip install -r requirements/build.txt -r requirements/cli.txt \
    && NO_NATIVE_EXT=1 pip install --no-build-isolation .

RUN mkdir -p /var/lib/lmcache-memory-coordinator /etc/lmcache

ENTRYPOINT ["lmcache", "mp-memory-coordinator"]
CMD ["--config", "/etc/lmcache/mp-memory-coordinator.yaml"]
