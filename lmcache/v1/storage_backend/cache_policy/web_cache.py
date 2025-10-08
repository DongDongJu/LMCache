# SPDX-License-Identifier: Apache-2.0
"""Two-region web-style cache policy with hot MRU and cold FIFO queues."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy


class WebCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """Lightweight cache policy inspired by web caches.

    Maintains a small MRU "hot" window for recently accessed keys while keeping
    a larger FIFO "cold" queue for everything else. Hits promote entries into the
    hot window; evictions prefer draining the cold queue and fall back to the
    oldest hot entry if necessary. Pinned objects (``can_evict=False``) are
    rotated to the tail of their respective queues without blocking progress.
    """

    def __init__(self, hot_window: int = 64) -> None:
        self._hot_window = max(1, hot_window)
        self._hot: "OrderedDict[CacheEngineKey, None]" = OrderedDict()
        self._cold: "OrderedDict[CacheEngineKey, None]" = OrderedDict()

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        stage: Optional[str] = None,
    ) -> None:
        if key in self._hot:
            self._hot.move_to_end(key)
            return
        if key in self._cold:
            self._cold.pop(key, None)
            self._hot[key] = None
            self._hot.move_to_end(key)
            self._rebalance_hot()

    def update_on_put(
        self, key: CacheEngineKey, stage: str | None = None
    ) -> None:
        self._hot.pop(key, None)
        self._cold.pop(key, None)
        self._cold[key] = None

    def update_on_force_evict(self, key: CacheEngineKey) -> None:
        self._hot.pop(key, None)
        self._cold.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        if num_candidates <= 0:
            return []

        victims: list[CacheEngineKey] = []

        victims.extend(
            self._drain_queue(self._cold, cache_dict, num_candidates - len(victims))
        )
        if len(victims) < num_candidates:
            victims.extend(
                self._drain_queue(
                    self._hot, cache_dict, num_candidates - len(victims)
                )
            )

        return victims

    def contains(self, key: CacheEngineKey) -> bool:
        return key in self._hot or key in self._cold

    def size(self) -> int:
        return len(self._hot) + len(self._cold)

    def get_debug_metrics(self) -> Dict[str, int]:
        return {
            "hot_size": len(self._hot),
            "cold_size": len(self._cold),
            "hot_window": self._hot_window,
        }

    def _rebalance_hot(self) -> None:
        while len(self._hot) > self._hot_window:
            key, _ = self._hot.popitem(last=False)
            self._cold[key] = None

    def _drain_queue(
        self,
        queue: "OrderedDict[CacheEngineKey, None]",
        cache_dict: dict[CacheEngineKey, Any],
        limit: int,
    ) -> list[CacheEngineKey]:
        if limit <= 0 or not queue:
            return []
        victims: list[CacheEngineKey] = []
        attempts = len(queue)
        while queue and len(victims) < limit and attempts > 0:
            key, _ = queue.popitem(last=False)
            attempts -= 1
            cache_obj = cache_dict.get(key)
            if cache_obj is not None and not getattr(cache_obj, "can_evict", True):
                queue[key] = None
                continue
            victims.append(key)
        return victims
