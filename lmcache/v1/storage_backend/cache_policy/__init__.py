# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict, Type

# First Party
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy
from lmcache.v1.storage_backend.cache_policy.clock_ecl import (
    CLOCKECLCachePolicy,
)
from lmcache.v1.storage_backend.cache_policy.anchored_sliding_window import (
    AnchoredSlidingWindowCachePolicy,
)
from lmcache.v1.storage_backend.cache_policy.fifo import FIFOCachePolicy
from lmcache.v1.storage_backend.cache_policy.lfu import LFUCachePolicy
from lmcache.v1.storage_backend.cache_policy.lru import LRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.mru import MRUCachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve import SIEVECachePolicy
from lmcache.v1.storage_backend.cache_policy.sieve_slru import (
    SIEVESLRUCachePolicy,
)
from lmcache.v1.storage_backend.cache_policy.sieve_pdg import (
    SIEVEPDGCachePolicy,
)
from lmcache.v1.storage_backend.cache_policy.web_cache import WebCachePolicy

# Cache policy mapping
POLICY_MAPPING: Dict[str, Type[BaseCachePolicy]] = {
    "ANCHOR_SLIDING_WINDOW": AnchoredSlidingWindowCachePolicy,
    "LRU": LRUCachePolicy,
    "LFU": LFUCachePolicy,
    "FIFO": FIFOCachePolicy,
    "MRU": MRUCachePolicy,
    "SIEVE": SIEVECachePolicy,
    "SIEVE_SLRU": SIEVESLRUCachePolicy,
    "SIEVE_PDG": SIEVEPDGCachePolicy,
    "WEB_CACHE": WebCachePolicy,
    "CLOCK_ECL": CLOCKECLCachePolicy,
}


def get_cache_policy(policy_name: str) -> BaseCachePolicy:
    """
    Factory function to get the cache policy instance based on the policy name.

    Args:
        policy_name: Name of the cache policy (case-insensitive, e.g., "LRU", "lru").

    Returns:
        Instance of the corresponding cache policy.

    Raises:
        ValueError: If the policy name is not supported.
    """
    if not policy_name:
        raise ValueError("Cache policy name cannot be empty")

    upper_policy_name = policy_name.upper()

    try:
        return POLICY_MAPPING[upper_policy_name]()
    except KeyError:
        raise ValueError(
            f"Unknown cache policy: {upper_policy_name}."
            f" Supported policies are: {list(POLICY_MAPPING.keys())}"
        ) from None
