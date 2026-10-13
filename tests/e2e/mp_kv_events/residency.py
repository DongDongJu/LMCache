# SPDX-License-Identifier: Apache-2.0
"""Exact physical L1 evidence, independent of engine/native CPU announcements."""

# Standard
import base64

ObjectIdentity = tuple[str, int, int, str, str]
_COMMITTED = {"l1.write.finished", "l1.write_finished_and_read_reserved"}


def encoded_key(key: dict) -> dict:
    """Normalize a raw ObjectKey from mp_trace into its public wire descriptor."""
    return {
        "model_name": key["model_name"],
        "kv_rank": key["kv_rank"],
        "object_group_id": key["object_group_id"],
        "cache_salt": key["cache_salt"],
        "chunk_hash_hex": base64.b64decode(
            key["chunk_hash"]["bytes_base64"], validate=True
        ).hex(),
    }


def identity(key: dict) -> ObjectIdentity:
    """Return every storage namespace dimension for one encoded ObjectKey."""
    return (
        key["model_name"],
        key["kv_rank"],
        key["object_group_id"],
        key["cache_salt"],
        key["chunk_hash_hex"],
    )


def l1_owners(records: list[dict]) -> dict[ObjectIdentity, set[str]]:
    """Project actual ordered L1 commit/eviction records into distinct owners.

    A reservation, MP_STORE_END, or native Stored record proves no residency.
    Duplicate commits are idempotent and removing one copy preserves other L1s.
    """
    owners: dict[ObjectIdentity, set[str]] = {}
    for record in records:
        event = record["event"]
        if event not in _COMMITTED and event != "l1.keys.evicted":
            continue
        metadata = record["metadata"]
        owner = metadata["l1_tag"]
        for raw in metadata["keys"]:
            key = identity(encoded_key(raw))
            if event in _COMMITTED:
                owners.setdefault(key, set()).add(owner)
            elif key in owners:
                owners[key].discard(owner)
                if not owners[key]:
                    del owners[key]
    return owners


def target_l1_keys(
    records: list[dict],
    tokens: list[int],
    chunk_size: int,
    model_name: str,
    world_size: int,
) -> list[dict]:
    """Resolve exact fixture chunks and all pinned dense-model KV ranks.

    Token-to-chunk mappings come from the existing MP_TOKENS trace, including
    absolute offsets; identities come from real L1 commit records. This oracle
    intentionally requires the pinned harness's one full-attention object group.
    """
    end = len(tokens) // chunk_size * chunk_size
    by_offset: dict[int, set[tuple[str, str | None]]] = {
        offset: set() for offset in range(0, end, chunk_size)
    }
    for record in records:
        if record["event"] != "mp.tokens":
            continue
        metadata = record["metadata"]
        for offset, token_chunk, chunk_hash, parent_hash in zip(
            metadata.get("token_offsets", []),
            metadata.get("token_chunks", []),
            metadata.get("chunk_hashes", []),
            metadata.get("parent_hashes", []),
            strict=True,
        ):
            if (
                offset in by_offset
                and token_chunk == tokens[offset : offset + chunk_size]
            ):
                by_offset[offset].add(
                    (
                        base64.b64decode(
                            chunk_hash["bytes_base64"], validate=True
                        ).hex(),
                        None
                        if parent_hash is None
                        else base64.b64decode(
                            parent_hash["bytes_base64"], validate=True
                        ).hex(),
                    )
                )
    # A shared suffix can have identical tokens and offsets in another request,
    # but its chained identity belongs to that request's different prefix.
    assert by_offset, "No complete fixture chunks"
    parent: str | None = None
    hashes: set[str] = set()
    for offset, candidates in by_offset.items():
        matching = {value for value, prior in candidates if prior == parent}
        assert len(matching) == 1, (
            "Missing or ambiguous exact token-to-LM-chunk evidence",
            offset,
            parent,
            candidates,
        )
        parent = next(iter(matching))
        hashes.add(parent)
    assert len(hashes) == len(by_offset), (
        "Distinct chained chunks unexpectedly share an identity"
    )
    known: dict[ObjectIdentity, dict] = {}
    for record in records:
        if record["event"] not in _COMMITTED:
            continue
        for raw in record["metadata"]["keys"]:
            key = encoded_key(raw)
            if (
                key["chunk_hash_hex"] in hashes
                and key["model_name"] == model_name
                and key["cache_salt"] == ""
                and key["kv_rank"] >> 24 == world_size
            ):
                known[identity(key)] = key
    for chunk_hash in hashes:
        shard_keys = [
            key for key in known.values() if key["chunk_hash_hex"] == chunk_hash
        ]
        assert len(shard_keys) == world_size, (
            "Incomplete required KV-rank evidence",
            shard_keys,
        )
        assert {key["kv_rank"] & 0xFF for key in shard_keys} == set(
            range(world_size)
        ), shard_keys
        assert {key["object_group_id"] for key in shard_keys} == {0}, shard_keys
    return list(known.values())


def assert_target_resident(records: list[dict], keys: list[dict]) -> None:
    """Require a committed readable owner for every specific fixture ObjectKey."""
    assert keys, "No exact fixture ObjectKeys were supplied"
    owners = l1_owners(records)
    missing = [key for key in keys if identity(key) not in owners]
    assert not missing, ("Fixture KV is not actually committed in L1", missing)


def assert_target_absent(records: list[dict], keys: list[dict]) -> None:
    """Require removal of every exact fixture object and all of its L1 copies."""
    assert keys, "No exact fixture ObjectKeys were supplied"
    owners = l1_owners(records)
    remaining = [key for key in keys if identity(key) in owners]
    assert not remaining, ("Fixture KV still has a committed L1 owner", remaining)


def target_evictions(records: list[dict], keys: list[dict]) -> list[dict]:
    """Return physical removal records touching the exact fixture key set."""
    target = {identity(key) for key in keys}
    return [
        record
        for record in records
        if record["event"] == "l1.keys.evicted"
        and any(
            identity(encoded_key(key)) in target for key in record["metadata"]["keys"]
        )
    ]
