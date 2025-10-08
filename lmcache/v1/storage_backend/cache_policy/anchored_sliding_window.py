# SPDX-License-Identifier: Apache-2.0
"""Anchored sliding-window cache policy."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


@dataclass
class _AnchorEntry:
    inserted_at: float
    last_refresh: float
    sampled: bool


@dataclass
class _FlowEntry:
    last_hit: float
    anchored: bool


class AnchoredSlidingWindowCachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """Hybrid policy that snapshots prefill anchors and tracks active decode flow."""

    def __init__(
        self,
        *,
        anchor_sample_stride: int = 4,
        anchor_ttl_seconds: float = 300.0,
        flow_max_entries: int = 512,
        flow_ttl_seconds: float = 60.0,
    ) -> None:
        self._anchors: "OrderedDict[CacheEngineKey, _AnchorEntry]" = OrderedDict()
        self._flow: "OrderedDict[CacheEngineKey, _FlowEntry]" = OrderedDict()
        self._prefill_counter = 0
        self._anchor_sample_stride = max(1, anchor_sample_stride)
        self._anchor_ttl = max(0.0, anchor_ttl_seconds)
        self._flow_max_entries = max(1, flow_max_entries)
        self._flow_ttl = max(0.0, flow_ttl_seconds)
        self._stats: dict[str, int] = {
            "prefill_hits": 0,
            "decode_hits": 0,
            "rehydrates": 0,
            "anchor_samples": 0,
            "anchor_skips": 0,
            "flow_trims": 0,
            "evict_requests": 0,
            "evict_selected": 0,
        }
        self._last_cache_size = 0
        logger.info(
            "Initializing AnchoredSlidingWindowCachePolicy with anchor_stride=%d,"
            " flow_max=%d",
            self._anchor_sample_stride,
            self._flow_max_entries,
        )

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        stage: Optional[str] = None,
    ) -> None:
        stage = self._normalize_stage(stage)
        now = time.monotonic()
        if stage == "prefill":
            self._stats["prefill_hits"] += 1
            self._touch_anchor(key, now)
            return

        self._stats["decode_hits"] += 1
        flow_entry = self._flow.get(key)
        if flow_entry is not None:
            flow_entry.last_hit = now
            self._flow.move_to_end(key)
            return

        if key in self._anchors:
            self._rehydrate_from_anchor(key, now)
        else:
            self._admit_flow(key, anchored=False, now=now)

    def update_on_put(
        self,
        key: CacheEngineKey,
        stage: Optional[str] = None,
    ) -> None:
        stage = self._normalize_stage(stage)
        now = time.monotonic()
        if stage == "decode":
            self._admit_flow(key, anchored=key in self._anchors, now=now)
            return

        self._flow.pop(key, None)
        self._prefill_counter += 1
        if self._prefill_counter % self._anchor_sample_stride == 0:
            self._anchors[key] = _AnchorEntry(now, now, sampled=True)
            self._anchors.move_to_end(key)
            self._stats["anchor_samples"] += 1
        else:
            existing = self._anchors.get(key)
            if existing is not None:
                existing.inserted_at = now
                existing.last_refresh = now
            else:
                self._anchors[key] = _AnchorEntry(now, now, sampled=False)
                self._anchors.move_to_end(key)
            self._stats["anchor_skips"] += 1

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        self._flow.pop(key, None)
        self._anchors.pop(key, None)

    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        if num_candidates <= 0:
            return []

        now = time.monotonic()
        self._last_cache_size = len(cache_dict)
        self._cleanup_expired(cache_dict, now)

        victims: list[CacheEngineKey] = []
        attempts = 0
        max_attempts = max(len(cache_dict) * 2, num_candidates * 3)

        while len(victims) < num_candidates and attempts < max_attempts:
            attempts += 1
            candidate = self._pop_flow_candidate(cache_dict)
            if candidate is None:
                candidate = self._pop_anchor_candidate(cache_dict)
            if candidate is None:
                candidate = self._pop_fallback_candidate(cache_dict)
            if candidate is None:
                break
            victims.append(candidate)

        self._stats["evict_requests"] += 1
        self._stats["evict_selected"] += len(victims)
        return victims

    def get_debug_metrics(self) -> dict[str, Any]:
        total_hits = self._stats["decode_hits"] + self._stats["prefill_hits"]
        decode_ratio = 0.0
        if total_hits:
            decode_ratio = self._stats["decode_hits"] / total_hits * 100.0
        return {
            "anchor_size": len(self._anchors),
            "flow_size": len(self._flow),
            "rehydrates": self._stats["rehydrates"],
            "decode_hit_ratio": round(decode_ratio, 2),
            "anchor_samples": self._stats["anchor_samples"],
            "anchor_skips": self._stats["anchor_skips"],
            "flow_trims": self._stats["flow_trims"],
            "evict_calls": self._stats["evict_requests"],
            "evict_selected": self._stats["evict_selected"],
        }

    def _normalize_stage(self, stage: Optional[str]) -> str:
        if stage is None:
            return "decode"
        stage_lower = stage.lower()
        if stage_lower not in {"prefill", "decode"}:
            return "decode"
        return stage_lower

    def _cleanup_expired(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        now: float,
    ) -> None:
        if self._flow_ttl > 0.0:
            for key in list(self._flow.keys()):
                entry = self._flow[key]
                if now - entry.last_hit > self._flow_ttl or key not in cache_dict:
                    self._flow.pop(key, None)

        if self._anchor_ttl > 0.0:
            for key in list(self._anchors.keys()):
                entry = self._anchors[key]
                if now - entry.last_refresh > self._anchor_ttl:
                    self._anchors.pop(key, None)
                elif key not in cache_dict:
                    self._anchors.pop(key, None)

    def _touch_anchor(self, key: CacheEngineKey, now: float) -> None:
        entry = self._anchors.get(key)
        if entry is None:
            self._anchors[key] = _AnchorEntry(now, now, sampled=False)
            self._anchors.move_to_end(key)
            return
        entry.last_refresh = now
        self._anchors.move_to_end(key)

    def _rehydrate_from_anchor(self, key: CacheEngineKey, now: float) -> None:
        anchor = self._anchors.get(key)
        if anchor is not None:
            anchor.last_refresh = now
        self._stats["rehydrates"] += 1
        self._admit_flow(key, anchored=True, now=now)

    def _admit_flow(
        self,
        key: CacheEngineKey,
        *,
        anchored: bool,
        now: float,
    ) -> None:
        self._flow[key] = _FlowEntry(last_hit=now, anchored=anchored)
        self._flow.move_to_end(key)
        self._enforce_flow_capacity()

    def _enforce_flow_capacity(self) -> None:
        while len(self._flow) > self._flow_max_entries:
            evicted_key, _ = self._flow.popitem(last=False)
            self._stats["flow_trims"] += 1
            # keep anchor metadata so it can be rehydrated later

    def _pop_flow_candidate(
        self,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> Optional[CacheEngineKey]:
        while self._flow:
            key, _ = self._flow.popitem(last=False)
            cache_obj = cache_dict.get(key)
            if cache_obj is None:
                self._anchors.pop(key, None)
                continue
            if not getattr(cache_obj, "can_evict", True):
                continue
            self._anchors.pop(key, None)
            return key
        return None

    def _pop_anchor_candidate(
        self,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> Optional[CacheEngineKey]:
        while self._anchors:
            key, _ = self._anchors.popitem(last=False)
            cache_obj = cache_dict.get(key)
            if cache_obj is None:
                continue
            if not getattr(cache_obj, "can_evict", True):
                continue
            self._flow.pop(key, None)
            return key
        return None

    def _pop_fallback_candidate(
        self,
        cache_dict: dict[CacheEngineKey, Any],
    ) -> Optional[CacheEngineKey]:
        for key, cache_obj in cache_dict.items():
            if key in self._anchors or key in self._flow:
                continue
            if not getattr(cache_obj, "can_evict", True):
                continue
            return key
        return None

