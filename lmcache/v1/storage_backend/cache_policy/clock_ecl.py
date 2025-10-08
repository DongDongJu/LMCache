# SPDX-License-Identifier: Apache-2.0
"""CLOCK policy with Evictable Candidate List and decode guard TTL."""

from __future__ import annotations

# Standard
from collections import deque
import random
import time
from typing import Any, Deque, Dict, Optional

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


def _parse_bytes(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    multiplier = 1
    if text.endswith("kb"):
        multiplier = 1024
        text = text[:-2]
    elif text.endswith("mb"):
        multiplier = 1024**2
        text = text[:-2]
    elif text.endswith("gb"):
        multiplier = 1024**3
        text = text[:-2]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        logger.warning("Failed to parse byte quantity '%s'", value)
        return None


class CLOCKECLCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """CLOCK-style KV replacement with an Evictable Candidate List (ECL).

    Features:
      * Single queue (deque) maintains insertion order.
      * Two mark bits per entry: bit0=visited, bit1=decode guard priority.
      * Decode hits/puts set a short TTL to defer eviction.
      * Eviction first consumes the O(1) ECL; falls back to bounded demotion.
      * Optional BIP sampling and coarse cost bias for large entries.
    """

    VISITED_MASK = 0b001
    PRIORITY_MASK = 0b010
    GRACE_MASK = 0b100  # internal one-time grace for large entries

    def __init__(
        self,
        *,
        scan_cap: int = 32,
        decode_hot_ttl_ms: int = 1500,
        big_threshold_bytes: Optional[Any] = None,
        bip_sample_rate: float = 0.0,
    ) -> None:
        self._dq: Deque[CacheEngineKey] = deque()
        self._mark: Dict[CacheEngineKey, int] = {}
        self._hot_until: Dict[CacheEngineKey, float] = {}
        self._ecl: Deque[CacheEngineKey] = deque()
        self._ecl_members: set[CacheEngineKey] = set()

        self.scan_cap = max(1, int(scan_cap))
        self.decode_hot_ttl = max(0.0, decode_hot_ttl_ms) / 1000.0
        self.big_threshold_bytes = _parse_bytes(big_threshold_bytes)
        self.bip_sample_rate = max(0.0, min(1.0, float(bip_sample_rate)))

        self._rng = random.Random()

        logger.info(
            "Initializing CLOCKECLCachePolicy scan_cap=%d decode_hot_ttl_ms=%d",
            self.scan_cap,
            decode_hot_ttl_ms,
        )

    # ------------------------------------------------------------------
    # Base API
    # ------------------------------------------------------------------
    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_put(
        self,
        key: CacheEngineKey,
        cache_dict: Optional[dict[CacheEngineKey, Any]] = None,
        *,
        stage: Optional[str] = None,
        ctx: Optional[Any] = None,
    ) -> None:
        stage_value = self._resolve_stage(stage, ctx)

        # If the key already exists, detach before re-inserting.
        if key in self._mark:
            self._detach(key)

        if stage_value == "decode":
            self._dq.appendleft(key)
            self._mark[key] = self.VISITED_MASK | self.PRIORITY_MASK
            if self.decode_hot_ttl > 0.0:
                self._hot_until[key] = self._now() + self.decode_hot_ttl
            else:
                self._hot_until.pop(key, None)
            self._remove_from_ecl(key)
        else:
            insert_head = (
                self.bip_sample_rate > 0.0
                and self._rng.random() < self.bip_sample_rate
            )
            if insert_head:
                self._dq.appendleft(key)
                self._mark[key] = self.VISITED_MASK
            else:
                self._dq.append(key)
                self._mark[key] = 0
            self._hot_until.pop(key, None)
            if cache_dict is not None:
                self._enqueue_ecl_if_needed(key, cache_dict)

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        *,
        stage: Optional[str] = None,
        ctx: Optional[Any] = None,
    ) -> None:
        if key not in self._mark:
            return

        stage_value = self._resolve_stage(stage, ctx)
        if stage_value == "decode":
            self._mark[key] = (
                self._mark.get(key, 0) | self.VISITED_MASK | self.PRIORITY_MASK
            )
            if self.decode_hot_ttl > 0.0:
                self._hot_until[key] = self._now() + self.decode_hot_ttl
        else:
            self._mark[key] = self._mark.get(key, 0) | self.VISITED_MASK
            if key in self._hot_until:
                self._hot_until[key] = self._now()

        self._remove_from_ecl(key)

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        self._detach(key)
        self._mark.pop(key, None)
        self._hot_until.pop(key, None)
        self._remove_from_ecl(key)

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        if num_candidates <= 0 or not self._dq:
            return []

        victims: list[CacheEngineKey] = []
        now = self._now()

        # Fast path: pop from Evictable Candidate List.
        while self._ecl and len(victims) < num_candidates:
            key = self._ecl.popleft()
            if key not in self._ecl_members:
                continue
            self._ecl_members.discard(key)

            if key not in cache_dict:
                continue
            if self._mark.get(key, 0) & self.VISITED_MASK:
                continue
            if not self._is_evictable(key, cache_dict):
                continue
            if now < self._hot_until.get(key, 0.0):
                self._enqueue_ecl_if_needed(key, cache_dict)
                continue

            victims.append(key)

        if len(victims) >= num_candidates:
            return victims

        # Slow path: bounded demotion similar to CLOCK second chance.
        scans = 0
        max_scans = min(len(self._dq), self.scan_cap)
        while scans < max_scans and len(victims) < num_candidates and self._dq:
            key = self._dq[-1]
            mark = self._mark.get(key, 0)

            if not self._is_evictable(key, cache_dict):
                self._rotate_tail()
                scans += 1
                continue

            if now < self._hot_until.get(key, 0.0):
                self._rotate_tail()
                scans += 1
                continue

            if mark & self.VISITED_MASK:
                self._mark[key] = mark & ~self.VISITED_MASK
                self._enqueue_ecl_if_needed(key, cache_dict)
                self._rotate_tail()
                scans += 1
                continue

            if mark & self.PRIORITY_MASK:
                self._mark[key] = mark & ~self.PRIORITY_MASK
                self._enqueue_ecl_if_needed(key, cache_dict)
                self._rotate_tail()
                scans += 1
                continue

            if self.big_threshold_bytes is not None and not (mark & self.GRACE_MASK):
                size = self._safe_get_size(cache_dict.get(key))
                if size >= self.big_threshold_bytes:
                    self._mark[key] = (mark | self.GRACE_MASK | self.VISITED_MASK)
                    self._rotate_tail()
                    scans += 1
                    continue

            victims.append(key)
            self._rotate_tail(pop_only=True)
            scans += 1

        return victims

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.monotonic()

    def _resolve_stage(self, stage: Optional[str], ctx: Optional[Any]) -> str:
        value = stage
        if value is None and ctx is not None:
            value = getattr(ctx, "stage", None)
        if value is None:
            return "prefill"
        lowered = str(value).lower()
        if lowered.startswith("dec"):
            return "decode"
        if lowered.startswith("pre"):
            return "prefill"
        return "prefill"

    def _is_evictable(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> bool:
        obj = cache_dict.get(key)
        return getattr(obj, "can_evict", True)

    def _enqueue_ecl_if_needed(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> None:
        if key in self._ecl_members:
            return
        if self._mark.get(key, 0) & self.VISITED_MASK:
            return
        if not self._is_evictable(key, cache_dict):
            return
        self._ecl.append(key)
        self._ecl_members.add(key)

    def _remove_from_ecl(self, key: CacheEngineKey) -> None:
        self._ecl_members.discard(key)

    def _detach(self, key: CacheEngineKey) -> None:
        if key not in self._mark:
            return
        try:
            self._dq.remove(key)
        except ValueError:
            pass

    def _rotate_tail(self, *, pop_only: bool = False) -> None:
        if not self._dq:
            return
        key = self._dq.pop()
        if not pop_only:
            self._dq.appendleft(key)

    def _safe_get_size(self, cache_obj: Any) -> int:
        if cache_obj is None:
            return 0
        size = None
        if hasattr(cache_obj, "get_size"):
            size = cache_obj.get_size()
        elif hasattr(cache_obj, "size"):
            size = getattr(cache_obj, "size")
        if isinstance(size, tuple):
            size = size[0]
        try:
            return int(size) if size is not None else 0
        except (TypeError, ValueError):
            return 0
