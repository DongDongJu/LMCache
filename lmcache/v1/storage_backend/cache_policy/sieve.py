# SPDX-License-Identifier: Apache-2.0
"""Cost-aware SIEVE policy with O(1) eviction candidates."""

from __future__ import annotations

# Standard
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

logger = init_logger(__name__)


class _SieveNode:
    __slots__ = ("key", "prev", "next", "visited")

    def __init__(self, key: CacheEngineKey, visited: int = 0) -> None:
        self.key: CacheEngineKey = key
        self.prev: Optional[_SieveNode] = None
        self.next: Optional[_SieveNode] = None
        self.visited: int = visited


@dataclass
class _NodeMeta:
    hot_until: float = 0.0
    last_hit: float = 0.0
    last_stage: str = "prefill"


class SIEVECachePolicy(BaseCachePolicy[dict[CacheEngineKey, Any]]):
    """Single-hand SIEVE policy enhanced with ECL, cost awareness, and decode TTL."""

    def __init__(
        self,
        sample_period: int = 100,
        max_mark: int = 2,
        scan_cap: Optional[int] = None,
        *,
        enable_ecl: bool = True,
        enable_cost: bool = True,
        decode_hot_ttl_ms: int = 1500,
        future_hit_delta_ms: int = 800,
        big_object_bytes: int = 512 * 1024,
        location_costs: Optional[dict[str, float]] = None,
    ) -> None:
        self._nodes: Dict[CacheEngineKey, _SieveNode] = {}
        self._head: Optional[_SieveNode] = None
        self._tail: Optional[_SieveNode] = None
        self._sample_period = max(1, sample_period)
        self._max_mark = max(1, max_mark)
        self._insertion_counter = 0
        self._last_scan_iterations = 0
        self._scan_cap = scan_cap if scan_cap is None or scan_cap > 0 else None

        self._enable_ecl = enable_ecl
        self._enable_cost = enable_cost
        self._decode_hot_ttl = max(0.0, decode_hot_ttl_ms) / 1000.0
        self._future_delta = max(0.0, future_hit_delta_ms) / 1000.0
        self._big_object_bytes = max(0, big_object_bytes)

        self._location_costs = {
            "gpu": 0.0,
            "cpu": 1.0,
            "disk": 2.0,
            "remote": 2.5,
        }
        if location_costs is not None:
            for loc, cost in location_costs.items():
                self._location_costs[str(loc).lower()] = float(cost)
        self._default_location_cost = 1.0

        self._ecl: deque[CacheEngineKey] = deque()
        self._ecl_members: set[CacheEngineKey] = set()
        self._meta: Dict[CacheEngineKey, _NodeMeta] = {}

        self._stats = {
            "ecl_victims": 0,
            "ttl_deferrals": 0,
            "scan_events": 0,
            "scan_iterations": 0,
        }

        logger.info(
            "Initializing SIEVECachePolicy (enable_ecl=%s, enable_cost=%s, decode_hot_ttl_ms=%d)",
            enable_ecl,
            enable_cost,
            decode_hot_ttl_ms,
        )

    def init_mutable_mapping(self) -> dict[CacheEngineKey, Any]:
        return {}

    # ------------------------------------------------------------------
    # Core policy updates
    # ------------------------------------------------------------------
    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: dict[CacheEngineKey, Any],
        stage: Optional[str] = None,
    ) -> None:
        node = self._nodes.get(key)
        if node is None:
            return

        stage_normalized = self._normalize_stage(stage)
        info = self._meta.setdefault(key, _NodeMeta())
        now = time.monotonic()
        delta = now - info.last_hit if info.last_hit > 0 else None
        info.last_hit = now
        info.last_stage = stage_normalized

        if stage_normalized == "decode":
            future_hint = self._extract_future_hint(key)
            if delta is not None and delta <= self._future_delta:
                future_hint = max(future_hint, 1)

            increment = 1 + future_hint
            if self._enable_cost:
                loc_cost, size_cost = self._cost_weights(cache_dict, key)
                increment += loc_cost + size_cost

            node.visited = min(self._max_mark, node.visited + int(round(increment)))
            if self._decode_hot_ttl > 0.0:
                info.hot_until = now + self._decode_hot_ttl
            if self._enable_ecl:
                self._remove_from_ecl(key)
        else:
            node.visited = min(self._max_mark, max(node.visited, 1))
            info.hot_until = 0.0
            if self._enable_ecl and node.visited == 0:
                self._maybe_enqueue_evictable(key)

    def update_on_put(
        self,
        key: CacheEngineKey,
        stage: Optional[str] = None,
    ) -> None:
        stage_normalized = self._normalize_stage(stage or "prefill")
        now = time.monotonic()

        existing = self._nodes.get(key)
        if existing is not None:
            self._detach(existing)

        if stage_normalized == "decode":
            insert_head = True
            visited = self._max_mark
        else:
            sampled_head = self._should_sample_head()
            insert_head = sampled_head
            visited = 1 if sampled_head else 0

        self._admit_new_node(key, visited, insert_head)

        info = self._meta.setdefault(key, _NodeMeta())
        info.last_stage = stage_normalized
        if stage_normalized == "decode":
            info.last_hit = now
            info.hot_until = now + self._decode_hot_ttl if self._decode_hot_ttl > 0 else 0.0
            if self._enable_ecl:
                self._remove_from_ecl(key)
        else:
            info.hot_until = 0.0

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        node = self._nodes.get(key)
        if node is None:
            return
        self._remove_node(node)

    # ------------------------------------------------------------------
    # Eviction candidates
    # ------------------------------------------------------------------
    def get_evict_candidates(
        self,
        cache_dict: dict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        if num_candidates <= 0 or not self._nodes:
            return []

        now = time.monotonic()
        victims: list[CacheEngineKey] = []
        scans = 0

        if self._enable_ecl and self._ecl:
            attempts = 0
            max_attempts = len(self._ecl) + len(self._nodes)
            while len(victims) < num_candidates and self._ecl and attempts < max_attempts:
                attempts += 1
                key = self._ecl.popleft()
                if key not in self._ecl_members:
                    continue
                self._ecl_members.discard(key)

                node = self._nodes.get(key)
                if node is None or node.visited != 0:
                    continue

                info = self._meta.get(key)
                if info is not None and info.hot_until > now:
                    self._stats["ttl_deferrals"] += 1
                    self._maybe_enqueue_evictable(key)
                    continue

                cache_obj = cache_dict.get(key)
                if cache_obj is not None and not getattr(cache_obj, "can_evict", True):
                    self._maybe_enqueue_evictable(key)
                    continue

                victims.append(key)
                self._stats["ecl_victims"] += 1
                self._remove_node(node)

        if len(victims) >= num_candidates:
            self._last_scan_iterations = 0
            return victims

        current = self._tail
        scan_cap = self._scan_cap if self._scan_cap is not None else len(self._nodes) * 2
        while (
            current is not None
            and len(victims) < num_candidates
            and scans < scan_cap
        ):
            key = current.key
            prev_node = current.prev
            cache_obj = cache_dict.get(key)
            info = self._meta.get(key)

            if cache_obj is not None and not getattr(cache_obj, "can_evict", True):
                current = prev_node
                scans += 1
                continue

            if info is not None and info.hot_until > now:
                self._stats["ttl_deferrals"] += 1
                current = prev_node
                scans += 1
                continue

            if current.visited > 0:
                current.visited = max(0, current.visited - 1)
                if self._enable_ecl and current.visited == 0:
                    self._maybe_enqueue_evictable(key)
                current = prev_node
                scans += 1
                continue

            victims.append(key)
            self._remove_node(current)
            current = prev_node
            scans += 1

        self._last_scan_iterations = scans
        if scans > 0:
            self._stats["scan_events"] += 1
            self._stats["scan_iterations"] += scans
        return victims

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def contains(self, key: CacheEngineKey) -> bool:
        return key in self._nodes

    def admit_from_probation(
        self,
        key: CacheEngineKey,
        visited: int,
        insert_head: bool = True,
    ) -> None:
        self._admit_new_node(key, visited, insert_head)
        info = self._meta.setdefault(key, _NodeMeta())
        info.last_stage = "prefill"
        info.hot_until = 0.0
        if self._enable_ecl and visited == 0:
            self._maybe_enqueue_evictable(key)

    def size(self) -> int:
        return len(self._nodes)

    def max_mark(self) -> int:
        return self._max_mark

    def get_mark(self, key: CacheEngineKey) -> Optional[int]:
        node = self._nodes.get(key)
        if node is None:
            return None
        return node.visited

    def set_mark(self, key: CacheEngineKey, visited: int) -> None:
        node = self._nodes.get(key)
        if node is None:
            return
        node.visited = max(0, min(self._max_mark, visited))
        if self._enable_ecl:
            if node.visited == 0:
                self._maybe_enqueue_evictable(key)
            else:
                self._remove_from_ecl(key)

    def get_debug_metrics(self) -> dict[str, Any]:
        avg_scan = 0.0
        if self._stats["scan_events"] > 0:
            avg_scan = self._stats["scan_iterations"] / self._stats["scan_events"]
        return {
            "list_size": len(self._nodes),
            "ecl_size": len(self._ecl_members) if self._enable_ecl else 0,
            "decode_hot_ttl_ms": int(self._decode_hot_ttl * 1000),
            "avg_probation_scans": round(avg_scan, 2),
            "last_scan_iterations": self._last_scan_iterations,
            "ecl_evictions": self._stats["ecl_victims"],
            "ttl_deferrals": self._stats["ttl_deferrals"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _admit_new_node(self, key: CacheEngineKey, visited: int, insert_head: bool) -> None:
        node = _SieveNode(key, visited=max(0, min(self._max_mark, visited)))
        self._nodes[key] = node
        if insert_head:
            self._insert_at_head(node)
        else:
            self._insert_at_tail(node)
        if self._enable_ecl and node.visited == 0:
            self._maybe_enqueue_evictable(key)

    def _insert_at_head(self, node: _SieveNode) -> None:
        node.prev = None
        node.next = self._head
        if self._head is not None:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _insert_at_tail(self, node: _SieveNode) -> None:
        node.next = None
        node.prev = self._tail
        if self._tail is not None:
            self._tail.next = node
        self._tail = node
        if self._head is None:
            self._head = node

    def _detach(self, node: _SieveNode) -> None:
        if node.prev is not None:
            node.prev.next = node.next
        else:
            self._head = node.next
        if node.next is not None:
            node.next.prev = node.prev
        else:
            self._tail = node.prev
        node.prev = None
        node.next = None
        if self._enable_ecl:
            self._remove_from_ecl(node.key)

    def _remove_node(self, node: _SieveNode) -> None:
        self._detach(node)
        self._nodes.pop(node.key, None)
        self._meta.pop(node.key, None)
        if self._enable_ecl:
            self._remove_from_ecl(node.key)

    def _should_sample_head(self) -> bool:
        self._insertion_counter += 1
        return self._insertion_counter % self._sample_period == 0

    def _normalize_stage(self, stage: Optional[str]) -> str:
        if stage is None:
            return "decode"
        lowered = stage.lower()
        if lowered.startswith("pre"):
            return "prefill"
        if lowered.startswith("dec"):
            return "decode"
        return "decode"

    def _maybe_enqueue_evictable(self, key: CacheEngineKey) -> None:
        if not self._enable_ecl or key in self._ecl_members:
            return
        self._ecl.append(key)
        self._ecl_members.add(key)

    def _remove_from_ecl(self, key: CacheEngineKey) -> None:
        self._ecl_members.discard(key)

    def _extract_future_hint(self, key: CacheEngineKey) -> int:
        configs = key.request_configs or {}
        hint = configs.get("lmcache.future_hint")
        if hint is None:
            hint = configs.get("lmcache.future_priority")
        if hint is None and key.tags is not None:
            for tag, value in key.tags:
                if tag == "future":
                    hint = value
                    break
        if isinstance(hint, (int, float)):
            return 1 if hint > 0 else 0
        if isinstance(hint, str):
            lowered = hint.lower()
            if lowered in {"1", "true", "yes", "y", "hot"}:
                return 1
        return 0

    def _cost_weights(self, cache_dict: dict[CacheEngineKey, Any], key: CacheEngineKey) -> tuple[int, int]:
        cache_obj = cache_dict.get(key)
        location = self._infer_location(cache_obj)
        loc_cost = self._location_costs.get(location, self._default_location_cost)
        size = self._safe_get_size(cache_obj)
        size_cost = 1.0 if size >= self._big_object_bytes else 0.0
        return int(round(loc_cost)), int(round(size_cost))

    def _safe_get_size(self, cache_obj: Any) -> int:
        if cache_obj is None:
            return 0
        size = None
        if hasattr(cache_obj, "get_size"):
            size = cache_obj.get_size()
        elif hasattr(cache_obj, "size"):
            size = cache_obj.size
        if isinstance(size, tuple):
            size = size[0]
        if size is None:
            return 0
        try:
            return max(0, int(size))
        except (TypeError, ValueError):
            return 0

    def _infer_location(self, cache_obj: Any) -> str:
        if cache_obj is None:
            return "unknown"
        if hasattr(cache_obj, "storage_location"):
            return str(getattr(cache_obj, "storage_location")).lower()
        cls_name = cache_obj.__class__.__name__.lower()
        if "disk" in cls_name:
            return "disk"
        tensor = getattr(cache_obj, "tensor", None)
        if tensor is not None and hasattr(tensor, "device"):
            try:
                return str(tensor.device.type).lower()
            except Exception:  # pragma: no cover - defensive
                return "gpu"
        metadata = getattr(cache_obj, "metadata", None)
        if metadata is not None:
            device = getattr(metadata, "device", None)
            if device is not None:
                try:
                    return str(device).lower()
                except Exception:  # pragma: no cover - defensive
                    return "cpu"
        return "cpu"
