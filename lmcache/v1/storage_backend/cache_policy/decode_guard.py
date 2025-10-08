# SPDX-License-Identifier: Apache-2.0
"""Lightweight decode guard buffer used by SIEVE-PDG."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict

# First Party
from lmcache.utils import CacheEngineKey


@dataclass
class DecodeGuardEntry:
    size: int
    last_hit: float
    lease: int


class DecodeGuard:
    def __init__(self, ttl_seconds: float, max_lease: int = 3) -> None:
        self._entries: "OrderedDict[CacheEngineKey, DecodeGuardEntry]" = OrderedDict()
        self._ttl = max(0.0, ttl_seconds)
        self._max_lease = max(1, max_lease)
        self._demotions = 0

    def __contains__(self, key: CacheEngineKey) -> bool:  # pragma: no cover - helper
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def demotions(self) -> int:
        return self._demotions

    def remove(self, key: CacheEngineKey) -> None:
        self._entries.pop(key, None)

    def promote(
        self,
        key: CacheEngineKey,
        size: int,
        now: float,
        target_size: int,
        demote_cb: Callable[[CacheEngineKey, DecodeGuardEntry], None],
    ) -> None:
        entry = self._entries.get(key)
        if entry is not None:
            entry.size = max(1, size)
            entry.last_hit = now
            entry.lease = min(self._max_lease, entry.lease + 1)
            self._entries.move_to_end(key)
        else:
            self._entries[key] = DecodeGuardEntry(
                size=max(1, size),
                last_hit=now,
                lease=1,
            )
        self._expire_and_trim(now, target_size, demote_cb)

    def expire_and_trim(
        self,
        now: float,
        target_size: int,
        demote_cb: Callable[[CacheEngineKey, DecodeGuardEntry], None],
    ) -> None:
        self._expire_and_trim(now, target_size, demote_cb)

    def occupancy_bytes(self) -> int:
        return sum(entry.size for entry in self._entries.values())

    def snapshot(self) -> Dict[str, int]:
        return {
            "decode_guard_size": len(self._entries),
            "decode_guard_bytes": self.occupancy_bytes(),
            "decode_guard_demotions": self._demotions,
        }

    def _expire_and_trim(
        self,
        now: float,
        target_size: int,
        demote_cb: Callable[[CacheEngineKey, DecodeGuardEntry], None],
    ) -> None:
        if target_size < 0:
            target_size = 0
        while self._entries:
            key, entry = next(iter(self._entries.items()))
            expired = self._ttl and now - entry.last_hit > self._ttl
            over_budget = len(self._entries) > target_size
            if not expired and not over_budget:
                break
            self._entries.popitem(last=False)
            self._demotions += 1
            demote_cb(key, entry)
