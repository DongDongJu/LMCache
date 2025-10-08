# SPDX-License-Identifier: Apache-2.0
"""Prefill/Decode-guarded SIEVE policy."""

from __future__ import annotations

from collections import OrderedDict
import random
import time
from typing import Any, Dict, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.decode_guard import (
    DecodeGuard,
    DecodeGuardEntry,
)
from lmcache.v1.storage_backend.cache_policy.sieve import SIEVECachePolicy

logger = init_logger(__name__)


class SIEVEPDGCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """SIEVE variant with a decode guard and probation window."""

    def __init__(
        self,
        probation_ratio: float = 0.08,
        decode_guard_ratio: float = 0.16,
        decode_hot_ttl_ms: int = 2000,
        prefill_head_sample_rate: float = 0.005,
        scan_cap: int = 128,
    ) -> None:
        self._probation_ratio = max(0.01, min(0.5, probation_ratio))
        self._decode_guard_ratio = max(0.05, min(0.5, decode_guard_ratio))
        self._decode_hot_ttl = max(0, decode_hot_ttl_ms) / 1000.0
        self._prefill_head_sample_rate = max(0.0, min(1.0, prefill_head_sample_rate))
        self._rng = random.Random()

        self._probation: "OrderedDict[CacheEngineKey, None]" = OrderedDict()
        self._probation_blocked: "OrderedDict[CacheEngineKey, float]" = OrderedDict()
        self._protected = SIEVECachePolicy(scan_cap=scan_cap)
        self._decode_guard = DecodeGuard(ttl_seconds=self._decode_hot_ttl)
        self._region: Dict[CacheEngineKey, str] = {}
        self._last_scan_iterations = 0
        self._dynamic_guard_ratio = self._decode_guard_ratio
        self._fifo_override_until = 0.0
        self._fifo_override_scan_cap = 8
        self._last_decode_hit_ratio = 0.0
        self._blocked_retry_interval = 0.25
        self._stats: Dict[str, int] = {
            "decode_hits": 0,
            "prefill_hits": 0,
            "guard_promotions": 0,
            "probation_scans": 0,
            "evict_calls": 0,
        }

        logger.info(
            "Initializing SIEVEPDGCachePolicy with probation_ratio=%.2f, "
            "decode_guard_ratio=%.2f",
            self._probation_ratio,
            self._decode_guard_ratio,
        )

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    # NOTE: stage defaults to decode for hits to match current runtime behaviour.
    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        stage: Optional[str] = None,
    ) -> None:
        now = time.monotonic()
        stage = stage or "decode"

        if stage == "prefill":
            self._handle_prefill_hit(key)
            self._stats["prefill_hits"] += 1
            return

        self._stats["decode_hits"] += 1
        cache_obj = cache_dict.get(key)
        size = 1
        if cache_obj is not None:
            size = getattr(cache_obj, "get_size", lambda: getattr(cache_obj, "size", 1))()
            if isinstance(size, tuple):  # defensive for unexpected returns
                size = size[0]
            if not isinstance(size, int):
                try:
                    size = int(size)
                except (TypeError, ValueError):
                    size = 1

        region = self._region.get(key)

        if region == "decode_guard":
            self._decode_guard.promote(
                key,
                size,
                now,
                self._target_decode_guard_size(),
                self._demote_guard_entry,
            )
            return

        if region == "probation":
            self._probation.pop(key, None)
        elif region == "probation_blocked":
            self._probation_blocked.pop(key, None)
        elif region == "protected":
            self._protected.update_on_force_evict(key)

        self._decode_guard.promote(
            key,
            size,
            now,
            self._target_decode_guard_size(),
            self._demote_guard_entry,
        )
        self._stats["guard_promotions"] += 1
        self._region[key] = "decode_guard"

    def update_on_put(
        self,
        key: CacheEngineKey,
        stage: Optional[str] = None,
    ) -> None:
        stage = stage or "prefill"

        self._evict_from_regions(key)

        if stage == "decode":
            # Rare but handle explicitly.
            self._decode_guard.promote(
                key,
                size=1,
                now=time.monotonic(),
                target_size=self._target_decode_guard_size(),
                demote_cb=self._demote_guard_entry,
            )
            self._stats["guard_promotions"] += 1
            self._region[key] = "decode_guard"
            return

        if self._should_sample_prefill_head():
            self._protected.admit_from_probation(
                key,
                visited=min(1, self._protected.max_mark()),
                insert_head=True,
            )
            self._region[key] = "protected"
        else:
            self._probation_blocked.pop(key, None)
            self._probation[key] = None
            self._probation.move_to_end(key)
            self._region[key] = "probation"
            self._rebalance_probation()

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        region = self._region.pop(key, None)
        if region == "decode_guard":
            self._decode_guard.remove(key)
        elif region == "probation":
            self._probation.pop(key, None)
        elif region == "probation_blocked":
            self._probation_blocked.pop(key, None)
        elif region == "protected":
            self._protected.update_on_force_evict(key)
            self._last_scan_iterations = getattr(
                self._protected, "_last_scan_iterations", 0
            )

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        if num_candidates <= 0:
            return []

        now = time.monotonic()
        self._decode_guard.expire_and_trim(
            now,
            self._target_decode_guard_size(),
            self._demote_guard_entry,
        )

        evict_keys: list[CacheEngineKey] = []
        self._stats["evict_calls"] += 1

        self._maybe_adjust_heuristics(now)
        self._release_blocked_probation(cache_dict, now)

        # 1) Probation queue (FIFO/MRU tail)
        probation_scan_limit = len(self._probation)
        if self._probation and now < self._fifo_override_until:
            probation_scan_limit = min(
                probation_scan_limit, self._fifo_override_scan_cap
            )
        attempts = 0
        while (
            self._probation
            and len(evict_keys) < num_candidates
            and attempts < probation_scan_limit
        ):
            key, _ = self._probation.popitem(last=False)
            attempts += 1
            cache_obj = cache_dict.get(key)
            if cache_obj is not None and not getattr(cache_obj, "can_evict", True):
                # Quarantine pinned entries until they become evictable again.
                self._probation_blocked[key] = now
                self._region[key] = "probation_blocked"
                continue
            self._region.pop(key, None)
            evict_keys.append(key)

        self._stats["probation_scans"] += attempts

        # 2) Protected SIEVE
        remaining = num_candidates - len(evict_keys)
        if remaining > 0:
            protected_candidates = self._protected.get_evict_candidates(
                cache_dict,
                remaining,
            )
            self._last_scan_iterations = getattr(
                self._protected, "_last_scan_iterations", 0
            )
            evict_keys.extend(protected_candidates)

        return evict_keys

    def contains(self, key: CacheEngineKey) -> bool:
        return key in self._region

    def size(self) -> int:
        return (
            len(self._probation)
            + len(self._probation_blocked)
            + len(self._decode_guard)
            + self._protected.size()
        )

    def get_debug_metrics(self) -> Dict[str, int]:
        metrics = {
            "probation_size": len(self._probation),
            "probation_blocked_size": len(self._probation_blocked),
            "protected_size": self._protected.size(),
            "last_scan_iterations": self._last_scan_iterations,
            "decode_hits": self._stats["decode_hits"],
            "prefill_hits": self._stats["prefill_hits"],
            "guard_promotions": self._stats["guard_promotions"],
            "probation_scans": self._stats["probation_scans"],
            "evict_calls": self._stats["evict_calls"],
            "dynamic_guard_ratio_percent": int(self._dynamic_guard_ratio * 100),
            "fifo_override_active": int(time.monotonic() < self._fifo_override_until),
        }
        total_hits = self._stats["decode_hits"] + self._stats["prefill_hits"]
        if total_hits:
            metrics["decode_hit_ratio"] = int(
                100 * self._stats["decode_hits"] / total_hits
            )
            self._last_decode_hit_ratio = self._stats["decode_hits"] / total_hits
        if self._stats["evict_calls"]:
            metrics["avg_probation_scans"] = int(
                self._stats["probation_scans"]
                / max(1, self._stats["evict_calls"])
            )
        metrics.update(self._decode_guard.snapshot())
        return metrics

    def _maybe_adjust_heuristics(self, now: float) -> None:
        if self._stats["evict_calls"] < 32:
            return

        total_hits = self._stats["decode_hits"] + self._stats["prefill_hits"]
        if total_hits == 0:
            return

        decode_ratio = self._stats["decode_hits"] / total_hits
        avg_scans = self._stats["probation_scans"] / max(1, self._stats["evict_calls"])

        scan_pressure = max(avg_scans, float(self._last_scan_iterations))

        if decode_ratio > 0.75:
            if scan_pressure > 2:
                self._dynamic_guard_ratio = min(0.5, self._dynamic_guard_ratio + 0.02)
            if scan_pressure > 4:
                self._fifo_override_until = max(self._fifo_override_until, now + 0.75)
        elif decode_ratio < 0.55 and scan_pressure < 1.5:
            self._dynamic_guard_ratio = max(0.05, self._dynamic_guard_ratio - 0.01)

        # dampen counters to keep them bounded
        self._stats["evict_calls"] = max(16, self._stats["evict_calls"] // 2)
        self._stats["probation_scans"] = max(0, self._stats["probation_scans"] // 2)

    def _release_blocked_probation(
        self, cache_dict: dict[CacheEngineKey, Any], now: float
    ) -> None:
        if not self._probation_blocked:
            return

        to_release: list[CacheEngineKey] = []
        to_remove: list[CacheEngineKey] = []
        for key, blocked_at in list(self._probation_blocked.items()):
            cache_obj = cache_dict.get(key)
            if cache_obj is None:
                to_remove.append(key)
                continue
            if getattr(cache_obj, "can_evict", True):
                to_release.append(key)
                continue
            if now - blocked_at >= self._blocked_retry_interval:
                self._probation_blocked[key] = now

        for key in to_remove:
            self._probation_blocked.pop(key, None)
            self._region.pop(key, None)

        for key in to_release:
            self._probation_blocked.pop(key, None)
            self._probation[key] = None
            self._probation.move_to_end(key)
            self._region[key] = "probation"

    def _handle_prefill_hit(self, key: CacheEngineKey) -> None:
        region = self._region.get(key)
        if region == "probation":
            self._probation.move_to_end(key)
            return
        if region == "probation_blocked":
            self._probation_blocked.move_to_end(key)
            return
        if region == "protected":
            mark = self._protected.get_mark(key)
            if mark is not None:
                self._protected.set_mark(key, max(1, mark))

    def _should_sample_prefill_head(self) -> bool:
        return self._rng.random() < self._prefill_head_sample_rate

    def _target_decode_guard_size(self) -> int:
        total = max(1, self.size())
        target = int(total * self._dynamic_guard_ratio)
        return max(1, target)

    def _target_probation_size(self) -> int:
        total = max(1, self.size())
        target = int(total * self._probation_ratio)
        return max(1, target)

    def _rebalance_probation(self) -> None:
        target = self._target_probation_size()
        total_probation = len(self._probation) + len(self._probation_blocked)
        while total_probation > target and self._probation:
            key, _ = self._probation.popitem(last=False)
            self._protected.admit_from_probation(
                key,
                visited=0,
                insert_head=False,
            )
            self._region[key] = "protected"
            total_probation -= 1

    def _demote_guard_entry(self, key: CacheEngineKey, entry: DecodeGuardEntry) -> None:
        # Guard demotions land at the head with a medium mark to retain protection.
        self._region[key] = "protected"
        self._protected.admit_from_probation(
            key,
            visited=min(1, self._protected.max_mark()),
            insert_head=True,
        )

    def _evict_from_regions(self, key: CacheEngineKey) -> None:
        region = self._region.pop(key, None)
        if region == "decode_guard":
            self._decode_guard.remove(key)
        elif region == "probation":
            self._probation.pop(key, None)
        elif region == "probation_blocked":
            self._probation_blocked.pop(key, None)
        elif region == "protected":
            self._protected.update_on_force_evict(key)
