# SPDX-License-Identifier: Apache-2.0
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG ENGINE
ARG ENGINE_VERSION
ARG ENGINE_INDEX
ARG LMCACHE_VERSION
ENV TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=8 UV_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONPATH=/opt/mp-kv-events
USER root
COPY lmcache/tests/e2e/mp_kv_events/requirements-sglang.txt /tmp/requirements-sglang.txt
RUN if [ "${ENGINE}" = sglang ]; then \
      python3 -m pip install --break-system-packages 'uv==0.11.14' 'wheel==0.45.1' \
      && uv pip install --system --no-deps -r /tmp/requirements-sglang.txt \
      && uv pip install --system --no-deps 'flashinfer-cubin==0.7.1' \
           --index-url https://flashinfer.ai/whl \
      && uv pip install --system --no-deps 'flashinfer-jit-cache==0.7.1+cu130' \
           --index-url https://flashinfer.ai/whl/cu130; \
    else \
      uv pip install --system --no-deps --index-url ${ENGINE_INDEX} \
        "${ENGINE}==${ENGINE_VERSION}"; \
    fi
COPY lmcache /src/lmcache
COPY engine /src/engine
COPY dynamo /src/dynamo
COPY dynamo-runtime/ /src/
COPY source.json /opt/mp-kv-events/source.json
COPY source.json /src/source.json
# Tracked Python bridge files differ from the pinned CI wheel. Repack
# them with a fresh RECORD while preserving every native Rust extension byte.
RUN if [ "${ENGINE}" = sglang ]; then \
      python3 -m pip download --no-deps --index-url ${ENGINE_INDEX} \
        --dest /tmp/sglang-ci-wheel "sglang==${ENGINE_VERSION}" \
      && python3 /src/lmcache/tests/e2e/mp_kv_events/repack_sglang.py \
        --wheel /tmp/sglang-ci-wheel/*.whl --source /src/engine \
        --manifest /opt/mp-kv-events/source.json --output /tmp/sglang-patched-wheel \
      && uv pip install --system --no-deps /tmp/sglang-patched-wheel/*.whl \
      && cp /tmp/sglang-patched-wheel/provenance.json /opt/mp-kv-events/sglang-wheel.json \
      && rm -rf /tmp/sglang-ci-wheel /tmp/sglang-patched-wheel; \
    else \
      cp /opt/mp-kv-events/source.json /src/source.json \
      && python3 /src/lmcache/tests/e2e/mp_kv_events/stage_python_overlay.py \
        --context /src --output / --engine vllm --bridge-only; \
    fi
RUN uv pip install --system 'setuptools==80.9.0' 'setuptools-scm==9.2.2' \
    'grpcio==1.78.0' 'grpcio-tools==1.78.0' \
    && CPATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include \
       LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
       SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=${LMCACHE_VERSION} \
       uv pip install --system --no-deps --no-build-isolation /src/lmcache
COPY lmcache/tests/e2e/mp_kv_events /opt/mp-kv-events
RUN if [ "${ENGINE}" = sglang ] && [ -f /src/dynamo-runtime.whl ]; then \
      python3 /opt/mp-kv-events/stage_dynamo_overlay.py --context /src --output /; \
    fi
ENTRYPOINT []
