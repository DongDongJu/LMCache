# SPDX-License-Identifier: Apache-2.0
"""The MP P/D proxy waits for all workers and for the published shared prefix."""

# Standard
import asyncio
import json

# Third Party
import httpx
import pytest

# First Party
from examples.disagg_prefill_mp import disagg_proxy_server as proxy


@pytest.mark.asyncio
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
async def test_store_barrier_counts_distinct_ranks(
    monkeypatch: pytest.MonkeyPatch, tp: int
) -> None:
    monkeypatch.setattr(proxy, "main_event_loop", asyncio.get_running_loop())
    monkeypatch.setattr(proxy, "pending_requests", {})
    request_id = "0123456789abcdef"
    pending = proxy.create_pending_request(request_id)
    for rank in range(tp - 1):
        assert not proxy.notify_request(f"cmpl-{request_id}", "model", tp, rank)
        assert not proxy.notify_request(f"cmpl-{request_id}", "model", tp, rank)
    await asyncio.sleep(0)
    assert not pending.event.is_set()
    assert proxy.notify_request(f"cmpl-{request_id}", "model", tp, tp - 1)
    await asyncio.wait_for(pending.event.wait(), 1)
    assert pending.ranks == set(range(tp))
    assert (pending.model_name, pending.world_size) == ("model", tp)
    proxy.remove_pending_request(request_id)
    assert not proxy.notify_request(f"cmpl-{request_id}", "model", tp, 0)


@pytest.mark.asyncio
async def test_inconsistent_rank_reports_cannot_open_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(proxy, "main_event_loop", asyncio.get_running_loop())
    monkeypatch.setattr(proxy, "pending_requests", {})
    pending = proxy.create_pending_request("0123456789abcdef")
    assert not proxy.notify_request("cmpl-0123456789abcdef", "model", 8, 0)
    for model, size, rank in (("other", 8, 1), ("model", 2, 1), ("model", 8, 8)):
        with pytest.raises(ValueError):
            proxy.notify_request("cmpl-0123456789abcdef", model, size, rank)
    await asyncio.sleep(0)
    assert not pending.event.is_set()
    assert pending.ranks == {0}


@pytest.mark.asyncio
async def test_shared_readiness_retries_partial_prefix() -> None:
    posts = 0
    polls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal posts, polls
        if request.method == "POST":
            assert json.loads(request.content) == {
                "model_name": "model",
                "world_size": 8,
                "token_ids": [1, 2, 3, 4],
                "cache_salt": "tenant",
            }
            posts += 1
            return httpx.Response(
                202,
                json={
                    "status": "submitted",
                    "request_id": str(posts),
                    "chunks": 2,
                },
            )
        polls += 1
        if polls == 1:
            return httpx.Response(200, json={"status": "pending"})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "ready_chunks": posts,
                "found_keys": 32,
                "total_keys": 32,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        await asyncio.wait_for(
            proxy.wait_for_shared_prefix(
                client,
                "http://cache",
                proxy.PendingRequest(model_name="model", world_size=8),
                [1, 2, 3, 4],
                "tenant",
            ),
            1,
        )
    assert (posts, polls) == (2, 3)


@pytest.mark.asyncio
async def test_shared_readiness_propagates_cache_failure() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503))
    ) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await proxy.wait_for_shared_prefix(
                client,
                "http://cache",
                proxy.PendingRequest(model_name="model", world_size=8),
                [1, 2],
                "",
            )
