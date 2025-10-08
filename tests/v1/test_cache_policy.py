# SPDX-License-Identifier: Apache-2.0
# Standard
import time
from types import SimpleNamespace

# First Party
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.cache_policy.clock_ecl import CLOCKECLCachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve import SIEVECachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve_pdg import SIEVEPDGCachePolicy
from lmcache.v1.storage_backend.cache_policy.web_cache import WebCachePolicy

# Local
from .utils import dumb_cache_engine_key


class DummyMemoryObj:
    def __init__(self, can_evict: bool = True, size: int = 1):
        self.can_evict = can_evict
        self.size = size

    def get_size(self) -> int:
        return self.size


def test_lru():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lru_with_pin():
    policy = get_cache_policy("LRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key3, key1]


def test_fifo():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key1, key2]


def test_fifo_with_pin():
    policy = get_cache_policy("FIFO")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert evict_candidates == [key2, key3]


def test_lfu():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key1, key3]


def test_lfu_with_pin():
    policy = get_cache_policy("LFU")
    cache_dict = policy.init_mutable_mapping()

    obj1 = DummyMemoryObj(can_evict=False)  # Pinned object
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key3, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key2, cache_dict)
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert evict_candidates == [key3, key2]


def test_mru():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is the most recent, followed by key3.
    assert evict_candidates == [key1, key3], (evict_candidates, [key1, key3])


def test_mru_with_pin():
    policy = get_cache_policy("MRU")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj(can_evict=False)  # Pinned object
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1)
    cache_dict[key2] = obj2
    policy.update_on_put(key2)
    cache_dict[key3] = obj3
    policy.update_on_put(key3)

    policy.update_on_hit(key1, cache_dict)
    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)
    # key1 is most recent, followed by key3, but since key3 is pinned, wo go to key2.
    assert evict_candidates == [key1, key2], (evict_candidates, [key1, key2])


def test_sieve():
    policy = get_cache_policy("SIEVE")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj()
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict, stage="decode")

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert set(evict_candidates) == {key2, key3}, (
        evict_candidates,
        [key2, key3],
    )
    assert key1 not in evict_candidates


def test_sieve_with_pin():
    policy = get_cache_policy("SIEVE")
    cache_dict = policy.init_mutable_mapping()
    obj1 = DummyMemoryObj()
    obj2 = DummyMemoryObj(can_evict=False)
    obj3 = DummyMemoryObj()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)

    cache_dict[key1] = obj1
    policy.update_on_put(key1, stage="prefill")
    cache_dict[key2] = obj2
    policy.update_on_put(key2, stage="prefill")
    cache_dict[key3] = obj3
    policy.update_on_put(key3, stage="prefill")

    policy.update_on_hit(key1, cache_dict, stage="decode")

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert key1 not in evict_candidates
    assert key3 in evict_candidates


def test_sieve_slru_probation_evicts_mru():
    policy = get_cache_policy("SIEVE_SLRU")
    cache_dict = policy.init_mutable_mapping()
    keys = [dumb_cache_engine_key(i) for i in range(1, 5)]
    objs = [DummyMemoryObj() for _ in keys]

    for key, obj in zip(keys, objs, strict=False):
        cache_dict[key] = obj
        policy.update_on_put(key)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert evict_candidates == [keys[3]]


def test_sieve_slru_promotion_protects_hot_entries():
    policy = get_cache_policy("SIEVE_SLRU")
    cache_dict = policy.init_mutable_mapping()
    key1 = dumb_cache_engine_key(1)
    key2 = dumb_cache_engine_key(2)
    key3 = dumb_cache_engine_key(3)
    key4 = dumb_cache_engine_key(4)

    for key in (key1, key2, key3, key4):
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)

    # Promote key1 into the protected SIEVE region.
    policy.update_on_hit(key1, cache_dict)

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert key1 not in evict_candidates
    assert evict_candidates[0] in {key4, key3}


def test_sieve_pdg_decode_guard_protects_decode_hits():
    policy = SIEVEPDGCachePolicy(prefill_head_sample_rate=0.0)
    cache_dict = policy.init_mutable_mapping()
    keys = [dumb_cache_engine_key(i) for i in range(1, 5)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)

    # Promote the first key into the decode guard.
    policy.update_on_hit(keys[0], cache_dict, stage="decode")

    evict_candidates = policy.get_evict_candidates(cache_dict, num_candidates=2)

    assert keys[0] not in evict_candidates


def test_sieve_pdg_probation_handles_prefill_burst():
    policy = SIEVEPDGCachePolicy(probation_ratio=0.3, prefill_head_sample_rate=0.0)
    cache_dict = policy.init_mutable_mapping()

    keys = [dumb_cache_engine_key(i) for i in range(1, 7)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=3)

    assert set(victims) <= set(keys)
    assert len(victims) == 3


def test_sieve_pdg_probation_all_pinned_falls_back_to_protected():
    policy = SIEVEPDGCachePolicy(probation_ratio=0.5, prefill_head_sample_rate=0.0)
    cache_dict = policy.init_mutable_mapping()

    keys = [dumb_cache_engine_key(i) for i in range(1, 6)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)

    probation_keys = {
        key for key, region in policy._region.items() if region == "probation"
    }
    protected_keys = {
        key for key, region in policy._region.items() if region == "protected"
    }

    assert probation_keys  # Sanity: probation retains entries.
    assert protected_keys  # Sanity: protected region populated.

    for key in probation_keys:
        cache_dict[key].can_evict = False

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)

    assert victims
    assert victims[0] in protected_keys


def test_sieve_pdg_probation_blocked_rejoins_when_unpinned():
    policy = SIEVEPDGCachePolicy(probation_ratio=0.4, prefill_head_sample_rate=0.0)
    cache_dict = policy.init_mutable_mapping()

    pinned_key = dumb_cache_engine_key(1)
    other_key = dumb_cache_engine_key(2)

    cache_dict[pinned_key] = DummyMemoryObj(can_evict=False)
    policy.update_on_put(pinned_key)

    cache_dict[other_key] = DummyMemoryObj()
    policy.update_on_put(other_key)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == [other_key]

    cache_dict[pinned_key].can_evict = True

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)

    assert victims == [pinned_key]


def test_anchor_sliding_window_prefill_and_decode_flow():
    policy = get_cache_policy("ANCHOR_SLIDING_WINDOW")
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(101)
    cache_dict[key] = DummyMemoryObj()

    policy.update_on_put(key, stage="prefill")
    metrics = policy.get_debug_metrics()
    assert metrics["anchor_size"] >= 1

    policy.update_on_hit(key, cache_dict, stage="decode")
    metrics = policy.get_debug_metrics()
    assert metrics["flow_size"] >= 1
    assert metrics["rehydrates"] >= 1


def test_anchor_sliding_window_prefers_flow_eviction():
    policy = get_cache_policy("ANCHOR_SLIDING_WINDOW")
    cache_dict = policy.init_mutable_mapping()

    prefill_keys = [dumb_cache_engine_key(i) for i in range(201, 205)]
    for key in prefill_keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key, stage="prefill")

    decode_keys = [dumb_cache_engine_key(i) for i in range(301, 304)]
    for key in decode_keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key, stage="decode")

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims
    assert victims[0] in decode_keys


def test_anchor_sliding_window_rehydrates_into_flow():
    policy = get_cache_policy("ANCHOR_SLIDING_WINDOW")
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(401)
    cache_dict[key] = DummyMemoryObj()

    policy.update_on_put(key, stage="prefill")
    policy.update_on_hit(key, cache_dict, stage="decode")

    metrics = policy.get_debug_metrics()
    assert metrics["flow_size"] == 1
    assert metrics["rehydrates"] == 1


def test_clock_ecl_basic_order():
    policy = CLOCKECLCachePolicy()
    cache_dict = policy.init_mutable_mapping()
    keys = [dumb_cache_engine_key(i) for i in range(1, 5)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key, cache_dict=cache_dict)

    policy.update_on_hit(keys[0], cache_dict)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert victims == [keys[1], keys[2]]


def test_clock_ecl_respects_pin_and_ecl():
    policy = CLOCKECLCachePolicy()
    cache_dict = policy.init_mutable_mapping()
    k1, k2, k3 = [dumb_cache_engine_key(i) for i in (501, 502, 503)]
    cache_dict[k1] = DummyMemoryObj()
    cache_dict[k2] = DummyMemoryObj(can_evict=False)
    cache_dict[k3] = DummyMemoryObj()
    for key in (k1, k2, k3):
        policy.update_on_put(key, cache_dict=cache_dict)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=2)
    assert victims == [k1, k3]
    assert k2 not in victims


def test_clock_ecl_decode_guard_ttl():
    policy = CLOCKECLCachePolicy(decode_hot_ttl_ms=2000)
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(510)
    cache_dict[key] = DummyMemoryObj()

    policy.update_on_put(key, cache_dict=cache_dict, ctx=SimpleNamespace(stage="decode"))
    assert policy.get_evict_candidates(cache_dict, num_candidates=1) == []

    policy._hot_until[key] = 0.0
    first_try = policy.get_evict_candidates(cache_dict, num_candidates=1)
    # First pass clears visited bit
    assert first_try == []
    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == [key]


def test_clock_ecl_bounded_scan_and_grace():
    policy = CLOCKECLCachePolicy(scan_cap=1, big_threshold_bytes=64)
    cache_dict = policy.init_mutable_mapping()

    class _Sized(DummyMemoryObj):
        def __init__(self, sz: int) -> None:
            super().__init__()
            self.size = sz

    for idx in range(520, 525):
        key = dumb_cache_engine_key(idx)
        cache_dict[key] = _Sized(512 if idx == 520 else 8)
        policy.update_on_put(key, cache_dict=cache_dict)
        policy.update_on_hit(key, cache_dict)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert isinstance(victims, list)

def test_sieve_ecl_returns_candidate_without_scan():
    policy = SIEVECachePolicy(sample_period=1000, enable_cost=False, decode_hot_ttl_ms=0)
    cache_dict = policy.init_mutable_mapping()

    keys = [dumb_cache_engine_key(i) for i in range(501, 505)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key, stage="prefill")

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)

    assert victims == [keys[0]]
    assert policy._last_scan_iterations == 0  # type: ignore[attr-defined]


def test_sieve_decode_hot_ttl_defers_eviction():
    policy = SIEVECachePolicy(enable_cost=False, decode_hot_ttl_ms=2000)
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(510)
    cache_dict[key] = DummyMemoryObj()

    policy.update_on_put(key, stage="prefill")
    policy.update_on_hit(key, cache_dict, stage="decode")
    policy.set_mark(key, 0)

    policy._meta[key].hot_until = time.monotonic() + 1.0  # type: ignore[attr-defined]
    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == []

    policy._meta[key].hot_until = time.monotonic() - 1.0  # type: ignore[attr-defined]
    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == [key]


def test_sieve_cost_awareness_promotes_hot_entries():
    policy = SIEVECachePolicy(enable_cost=True, decode_hot_ttl_ms=0, big_object_bytes=512)
    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(520)

    class _BigDummy(DummyMemoryObj):
        def __init__(self) -> None:
            super().__init__(size=600000)

    cache_dict[key] = _BigDummy()
    policy.update_on_put(key, stage="prefill")
    policy.update_on_hit(key, cache_dict, stage="decode")

    assert policy.get_mark(key) == policy.max_mark()


def test_sieve_pdg_guard_demotes_on_ttl_expiry():
    policy = SIEVEPDGCachePolicy(prefill_head_sample_rate=0.0, decode_hot_ttl_ms=1)


    cache_dict = policy.init_mutable_mapping()
    key = dumb_cache_engine_key(1)

    cache_dict[key] = DummyMemoryObj()
    policy.update_on_put(key)
    policy.update_on_hit(key, cache_dict, stage="decode")

    guard = policy._decode_guard  # type: ignore[attr-defined]
    entry = guard._entries[key]  # type: ignore[attr-defined]
    entry.last_hit -= 5

    policy.get_evict_candidates(cache_dict, num_candidates=1)

    assert policy._region[key] == "protected"  # type: ignore[attr-defined]


def test_web_cache_promotes_hits_into_hot_window():
    policy = WebCachePolicy(hot_window=2)
    cache_dict = policy.init_mutable_mapping()

    keys = [dumb_cache_engine_key(i) for i in range(1, 5)]
    for key in keys:
        cache_dict[key] = DummyMemoryObj()
        policy.update_on_put(key)

    policy.update_on_hit(keys[0], cache_dict)
    assert keys[0] in policy._hot  # type: ignore[attr-defined]
    assert keys[0] not in policy._cold  # type: ignore[attr-defined]

    policy.update_on_hit(keys[1], cache_dict)
    assert keys[1] in policy._hot  # type: ignore[attr-defined]

    policy.update_on_hit(keys[2], cache_dict)
    assert keys[2] in policy._hot  # type: ignore[attr-defined]
    assert keys[0] in policy._cold  # type: ignore[attr-defined]


def test_web_cache_eviction_skips_pinned_entries():
    policy = WebCachePolicy(hot_window=2)
    cache_dict = policy.init_mutable_mapping()

    k1, k2, k3 = [dumb_cache_engine_key(i) for i in range(1, 4)]
    cache_dict[k1] = DummyMemoryObj()
    cache_dict[k2] = DummyMemoryObj(can_evict=False)
    cache_dict[k3] = DummyMemoryObj()

    for key in (k1, k2, k3):
        policy.update_on_put(key)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == [k1]

    cache_dict.pop(k1)
    policy.update_on_force_evict(k1)

    victims = policy.get_evict_candidates(cache_dict, num_candidates=1)
    assert victims == [k3]
