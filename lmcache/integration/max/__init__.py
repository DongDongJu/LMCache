# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 LMCache Authors
"""LMCache integration for Modular MAX.

This module provides the LMCacheExternalBackend class that implements MAX's
external KV cache interface, enabling multi-tier caching (CPU + Disk).

Usage:
    Set environment variables before starting MAX:
    
    export LMCACHE_ENABLED=1
    export LMCACHE_LOCAL_CPU=true
    export LMCACHE_MAX_LOCAL_CPU_SIZE=4.0
    export LMCACHE_LOCAL_DISK="file:///tmp/lmcache/"
    export LMCACHE_MAX_LOCAL_DISK_SIZE=10.0
    export LMCACHE_CHUNK_SIZE=256
    
    max serve --model-path <model> --enable-prefix-caching
"""

from lmcache.integration.max.max_external_backend import (
    LMCacheExternalBackend,
    create_lmcache_backend,
    ExternalCacheLookupResult,
    ExternalCacheLoadResult,
    ExternalCacheStoreResult,
    ExternalCacheStats,
)

__all__ = [
    "LMCacheExternalBackend",
    "create_lmcache_backend",
    "ExternalCacheLookupResult",
    "ExternalCacheLoadResult",
    "ExternalCacheStoreResult",
    "ExternalCacheStats",
]
