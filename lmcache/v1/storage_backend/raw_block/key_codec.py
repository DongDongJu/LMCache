# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Literal
import hashlib

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey, parse_cache_key
from lmcache.v1.distributed.api import ObjectKey

RawBlockKeyNamespace = Literal["legacy", "object"]

_KEY_SEP = "@"
_PATH_SLASH_REPLACEMENT = "-SEP-"
_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class RawBlockKeySpec:
    encoded: str
    slot_identity: int


def object_key_to_string(key: ObjectKey) -> str:
    """Serialize an ObjectKey using the same reversible shape as FS L2."""
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}{_KEY_SEP}{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}"
    return base


def decode_object_key(encoded: str) -> ObjectKey:
    parts = encoded.split(_KEY_SEP)
    if len(parts) == 3:
        safe_model, kv_rank_str, chunk_hash_hex = parts
        cache_salt = ""
    elif len(parts) == 4:
        safe_model, kv_rank_str, chunk_hash_hex, cache_salt = parts
    else:
        raise ValueError(f"Invalid raw-block ObjectKey encoding: {encoded!r}")

    return ObjectKey(
        chunk_hash=bytes.fromhex(chunk_hash_hex),
        model_name=safe_model.replace(_PATH_SLASH_REPLACEMENT, "/"),
        kv_rank=int(kv_rank_str, 16),
        cache_salt=cache_salt,
    )


def encode_object_key(key: ObjectKey) -> RawBlockKeySpec:
    encoded = object_key_to_string(key)
    return RawBlockKeySpec(
        encoded=encoded,
        slot_identity=_object_slot_identity(encoded),
    )


def decode_legacy_key(encoded: str) -> CacheEngineKey | LayerCacheEngineKey:
    parsed = parse_cache_key(encoded)
    if not isinstance(parsed, (CacheEngineKey, LayerCacheEngineKey)):
        raise TypeError(
            "parse_cache_key returned unsupported key type "
            f"{type(parsed).__name__} for {encoded!r}"
        )
    return parsed


def encode_legacy_key(key: CacheEngineKey | LayerCacheEngineKey) -> RawBlockKeySpec:
    return RawBlockKeySpec(
        encoded=key.to_string(),
        slot_identity=int(key.chunk_hash) & _UINT64_MASK,
    )


def slot_identity_from_encoded_key(
    encoded: str,
    namespace: RawBlockKeyNamespace,
) -> int:
    if namespace == "legacy":
        key = decode_legacy_key(encoded)
        return int(key.chunk_hash) & _UINT64_MASK
    if namespace == "object":
        return _object_slot_identity(encoded)
    raise ValueError(f"Unsupported raw-block key namespace: {namespace!r}")


def _object_slot_identity(encoded: str) -> int:
    digest = hashlib.blake2b(encoded.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)
