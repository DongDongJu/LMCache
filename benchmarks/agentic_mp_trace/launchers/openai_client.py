# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible request driver for recording jobs."""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any
import json
import time

# Third Party
import httpx

# First Party
from benchmarks.agentic_mp_trace.datasets import AgenticRequest


@dataclass
class RequestResult:
    request_id: str
    status_code: int | None
    latency_s: float
    error: str | None
    response: dict[str, Any] | None


def request_payload(request: AgenticRequest, *, model_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_id,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.messages:
        payload["messages"] = request.messages
    else:
        payload["prompt"] = request.prompt_text or ""
    return payload


def send_one(
    request: AgenticRequest,
    *,
    base_url: str,
    model_id: str,
    timeout_s: float = 300.0,
) -> RequestResult:
    endpoint = "/v1/chat/completions" if request.messages else "/v1/completions"
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            response = client.post(
                f"{base_url.rstrip('/')}{endpoint}",
                json=request_payload(request, model_id=model_id),
            )
        payload = response.json() if response.content else None
        return RequestResult(
            request_id=request.request_id,
            status_code=response.status_code,
            latency_s=time.monotonic() - started,
            error=None if response.is_success else response.text,
            response=payload,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request.request_id,
            status_code=None,
            latency_s=time.monotonic() - started,
            error=str(exc),
            response=None,
        )


def drive_requests(
    requests: list[AgenticRequest],
    *,
    base_url: str,
    model_id: str,
    concurrency: int,
    request_rate_qps: float,
    log_path: str,
) -> list[RequestResult]:
    delay_s = 0.0 if request_rate_qps <= 0 else 1.0 / request_rate_qps
    results: list[RequestResult] = []
    with open(log_path, "w") as log_file:
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = []
            for request in requests:
                futures.append(
                    executor.submit(
                        send_one,
                        request,
                        base_url=base_url,
                        model_id=model_id,
                    )
                )
                if delay_s:
                    time.sleep(delay_s)
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log_file.write(json.dumps(asdict(result), sort_keys=True) + "\n")
    return results

