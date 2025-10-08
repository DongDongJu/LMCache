# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve import SIEVECachePolicy

logger = init_logger(__name__)


class SIEVESLRUCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """Segmented SIEVE: MRU probation window + protected SIEVE region."""

    def __init__(self, probation_ratio: float = 0.08) -> None:
        self._probation_ratio = max(0.01, min(0.5, probation_ratio))
        self._probation: "OrderedDict[CacheEngineKey, None]" = OrderedDict()
        self._protected = SIEVECachePolicy()
        self._last_scan_iterations = 0
        logger.info(
            "Initializing SIEVESLRUCachePolicy with probation_ratio=%.2f",
            self._probation_ratio,
        )

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        stage: Optional[str] = None,
    ) -> None:
        if self._protected.contains(key):
            self._protected.update_on_hit(key, cache_dict)
            self._last_scan_iterations = getattr(
                self._protected, "_last_scan_iterations", 0
            )
            return

        if key in self._probation:
            self._probation.pop(key, None)
            self._protected.admit_from_probation(
                key,
                visited=self._protected.max_mark(),
                insert_head=True,
            )
            self._ensure_probation_budget()

    def update_on_put(
        self,
        key: CacheEngineKey,
        stage: Optional[str] = None,
    ) -> None:
        if self._protected.contains(key):
            self._protected.update_on_force_evict(key)
        self._probation[key] = None
        self._probation.move_to_end(key)
        self._ensure_probation_budget()

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        if key in self._probation:
            self._probation.pop(key, None)
            return
        if self._protected.contains(key):
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

        evict_keys: list[CacheEngineKey] = []
        self._last_scan_iterations = 0

        probation_candidates = list(reversed(self._probation))
        for key in probation_candidates:
            if len(evict_keys) >= num_candidates:
                break
            cache_obj = cache_dict.get(key)
            if cache_obj is not None and not cache_obj.can_evict:
                continue
            if key in self._probation:
                self._probation.pop(key)
            evict_keys.append(key)

        remaining = num_candidates - len(evict_keys)
        if remaining > 0:
            protected_candidates = self._protected.get_evict_candidates(
                cache_dict,
                remaining,
            )
            evict_keys.extend(protected_candidates)
            self._last_scan_iterations = getattr(
                self._protected, "_last_scan_iterations", 0
            )

        return evict_keys

    def _ensure_probation_budget(self) -> None:
        total = self._protected.size() + len(self._probation)
        if total == 0:
            return
        target = max(1, int(total * self._probation_ratio))
        while len(self._probation) > target:
            self._probation.popitem(last=False)
