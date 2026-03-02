# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.object_key_codec import (
    cache_key_to_object_key,
    object_key_to_cache_key,
    pack_object_key,
    unpack_object_key,
)


def test_pack_unpack_object_key_roundtrip() -> None:
    key = ObjectKey(
        chunk_hash=b"\x01\x02\x03\x04\x05\x06\x07\x08",
        model_name="llama-test",
        kv_rank=123,
    )

    payload = pack_object_key(key)
    unpacked = unpack_object_key(payload)

    assert unpacked.version == 1
    assert unpacked.model_name == key.model_name
    assert unpacked.kv_rank == key.kv_rank
    assert unpacked.chunk_hash == key.chunk_hash


def test_object_key_to_cache_key_is_deterministic() -> None:
    key = ObjectKey(
        chunk_hash=b"abcdef0123456789",
        model_name="model-a",
        kv_rank=7,
    )

    cache_key_a = object_key_to_cache_key(key)
    cache_key_b = object_key_to_cache_key(key)

    assert cache_key_a == cache_key_b
    assert cache_key_a.model_name == key.model_name
    assert cache_key_a.worker_id == key.kv_rank
    assert cache_key_a.dtype == torch.uint8


def test_cache_key_reverse_mapping() -> None:
    key = ObjectKey(
        chunk_hash=b"\xde\xad\xbe\xef",
        model_name="model-b",
        kv_rank=9,
    )

    cache_key = object_key_to_cache_key(key)
    restored = cache_key_to_object_key(cache_key)

    assert restored is not None
    assert restored.model_name == key.model_name
    assert restored.kv_rank == key.kv_rank
    assert restored.chunk_hash == key.chunk_hash
