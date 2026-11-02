# SPDX-License-Identifier: Apache-2.0
"""Native block contributions preserve shared prefixes and validate atomically."""

# Standard
from dataclasses import replace

# Third Party
import pytest

# First Party
from lmcache.utils import CacheRemoveEvent, CacheStoreEvent
from lmcache.v1.multiprocess.custom_types import KVEventRecord
from lmcache.v1.multiprocess.kv_event_subscriber import KVEventLedger


def record(hashes: tuple[int, ...], tokens: tuple[int, ...]) -> KVEventRecord:
    """Build one complete native chunk in the supported dense group-zero profile."""
    return KVEventRecord(
        1, "stored", "CPU", "model", hashes, None, tokens, 2, "dense-native", 1
    )


def test_shared_blocks_are_removed_after_the_final_chunk() -> None:
    ledger = KVEventLedger()
    first = record((1, 2), (10, 11, 20, 21))
    sibling = record((1, 3), (10, 11, 30, 31))
    [stored] = ledger.apply(first)
    assert isinstance(stored, CacheStoreEvent)
    assert stored.block_hashes == [1, 2]
    [branch] = ledger.apply(sibling)
    assert isinstance(branch, CacheStoreEvent)
    assert branch.block_hashes == [3]
    assert branch.parent_block_hash == 1
    assert branch.token_ids == [30, 31]
    assert ledger.apply(sibling) == []
    assert ledger.apply(replace(first, kind="removed", token_ids=())) == [
        CacheRemoveEvent([2], "CPU")
    ]
    assert ledger.apply(replace(sibling, kind="removed", token_ids=())) == [
        CacheRemoveEvent([1, 3], "CPU")
    ]
    assert ledger.reset() == []


@pytest.mark.parametrize("changed", ["parent", "tokens"])
@pytest.mark.parametrize("duplicate_chunk", [False, True])
def test_conflicting_descriptor_does_not_mutate_contributions(
    changed: str, duplicate_chunk: bool
) -> None:
    ledger = KVEventLedger()
    first = record((1, 2), (10, 11, 20, 21))
    ledger.apply(first)
    candidate = first if duplicate_chunk else record((1, 3), (10, 11, 30, 31))
    candidate = (
        replace(candidate, parent_block_hash=99)
        if changed == "parent"
        else replace(candidate, token_ids=(99, *candidate.token_ids[1:]))
    )
    with pytest.raises(ValueError, match="conflicting native block"):
        ledger.apply(candidate)
    assert ledger.reset() == [CacheRemoveEvent([1, 2], "CPU")]
    assert ledger.reset() == []


def test_incomplete_chunk_does_not_create_a_contribution() -> None:
    ledger = KVEventLedger()
    with pytest.raises(ValueError, match="complete distinct blocks"):
        ledger.apply(record((1, 2), (10, 11, 20)))
    assert ledger.reset() == []


def test_reset_and_replay_emit_only_cpu_deltas() -> None:
    ledger = KVEventLedger()
    chunks = [record((1, 2), (10, 11, 20, 21)), record((1, 3), (10, 11, 30, 31))]
    original = [event for chunk in chunks for event in ledger.apply(chunk)]
    assert ledger.reset() == [CacheRemoveEvent([1, 2, 3], "CPU")]
    replay = [event for chunk in chunks for event in ledger.apply(chunk)]
    assert replay == original
    assert all(event.medium == "CPU" for event in original + replay)
    assert ledger.reset() == [CacheRemoveEvent([1, 2, 3], "CPU")]
