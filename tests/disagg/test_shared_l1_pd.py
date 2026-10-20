# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-host vLLM P/D check through the MP proxy and shared L1.

Set LMCACHE_SHARED_PD_TEST_CONFIG to JSON with ``proxy``, ``prefill``, ``decode``,
``prefill_cache``, ``decode_cache`` URLs, ``model``, ``prompt_token_ids``,
``expected_token_ids`` from an uncached reference at the same TP size,
``cache_salt``, ``chunk_size``, and ``evidence_dir``. Both engines must enable
the prefix-cache reset endpoint; the proxy must use --decoder-cache-url.
"""

# Standard
from pathlib import Path
import json
import os
import time

# Third Party
import pytest

# First Party
from tests.disagg.test_multi_l1_pd import _metric, _request, _snapshot

pytestmark = pytest.mark.no_shared_allocator


def test_shared_l1_pd_reuses_every_required_shard() -> None:
    """Repeat P/D with cleared GPU caches, equal output and actual decoder loads."""
    config_path = os.environ.get("LMCACHE_SHARED_PD_TEST_CONFIG", "")
    if not config_path:
        pytest.skip("Set LMCACHE_SHARED_PD_TEST_CONFIG for the running deployment")
    config = json.loads(Path(config_path).read_text())
    directory = Path(config["evidence_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    tokens = config["prompt_token_ids"]
    expected = config["expected_token_ids"]
    chunk_size = config["chunk_size"]
    assert len(tokens) > chunk_size
    assert len(tokens) % chunk_size, "Include a trailing prompt token for decode"
    body = {
        "model": config["model"],
        "prompt": tokens,
        "max_tokens": len(expected),
        "temperature": 0,
        "seed": 42,
        "return_token_ids": True,
        "cache_salt": config["cache_salt"],
        "kv_transfer_params": {"cached_token_stats": True},
    }
    for iteration in range(2):
        for role in ("prefill", "decode"):
            reset = json.loads(
                _request(config[role] + "/reset_prefix_cache?reset_external=false", {})
            )
            assert reset["success"]
        before = _request(config["decode_cache"] + "/metrics")
        response = json.loads(_request(config["proxy"] + "/v1/completions", body))
        (directory / f"response-{iteration}.json").write_text(
            json.dumps(response, indent=2)
        )
        assert response["choices"][0]["token_ids"] == expected
        cached = response["kv_transfer_params"]["cached_token_stats"]
        assert cached["num_lmcache_extra_cached_tokens"] >= (
            len(tokens) // chunk_size * chunk_size
        )
        # Shared-region usage is cached for one second; collect after it settles.
        time.sleep(1.1)
        for role in ("prefill", "decode"):
            after = _snapshot(config[role + "_cache"], directory, f"{iteration}-{role}")
            if role == "decode":
                name = "lmcache_mp_num_chunks_loaded_total"
                assert _metric(after, name) > _metric(before, name)
