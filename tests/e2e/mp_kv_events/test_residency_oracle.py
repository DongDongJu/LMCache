# SPDX-License-Identifier: Apache-2.0
"""Reject misleading native/aggregate state when exact physical KV is absent."""

# Standard
import base64
import unittest

# Third Party
from residency import (
    assert_target_absent,
    assert_target_resident,
    l1_owners,
    target_l1_keys,
)


def raw_key(chunk: bytes, rank: int, world: int = 2, model: str = "model") -> dict:
    """Create an ObjectKey wire fixture independently of the native-event binder."""
    return {
        "chunk_hash": {"bytes_base64": base64.b64encode(chunk).decode()},
        "model_name": model,
        "kv_rank": (world << 24) | (rank << 16) | (world << 8) | rank,
        "object_group_id": 0,
        "cache_salt": "",
    }


def event(kind: str, keys: list[dict], owner: str = "cpu0") -> dict:
    """Represent an actual storage notification's evidence schema."""
    return {"event": kind, "metadata": {"keys": keys, "l1_tag": owner}}


def bindings() -> dict:
    """Bind two exact, position-aware complete LMCache chunks."""
    return {
        "event": "mp.tokens",
        "metadata": {
            "chunk_hashes": [
                {"bytes_base64": base64.b64encode(value).decode()}
                for value in (b"a", b"b")
            ],
            "token_chunks": [[1, 2], [3, 4]],
            "token_offsets": [0, 2],
            "parent_hashes": [None, {"bytes_base64": base64.b64encode(b"a").decode()}],
        },
    }


class TestPhysicalResidencyOracle(unittest.TestCase):
    """Physical commit evidence must cover the exact full rank/chunk namespace."""

    def test_submission_and_aggregate_counts_cannot_prove_residency(self) -> None:
        records = [
            bindings(),
            {"event": "mp.store.end", "metadata": {"stored_count": 4}},
        ]
        self.assertEqual(l1_owners(records), {})
        with self.assertRaisesRegex(AssertionError, "Incomplete required"):
            target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)

    def test_all_required_ranks_must_have_committed(self) -> None:
        records = [
            bindings(),
            event("l1.write.finished", [raw_key(b"a", 0), raw_key(b"b", 0)]),
        ]
        with self.assertRaisesRegex(AssertionError, "Incomplete required"):
            target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)

    def test_removed_target_cannot_hide_behind_other_warm_objects(self) -> None:
        keys = [raw_key(chunk, rank) for chunk in (b"a", b"b") for rank in (0, 1)]
        records = [bindings(), event("l1.write.finished", keys)]
        target = target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)
        assert_target_resident(records, target)
        records.extend(
            [
                event("l1.keys.evicted", keys[:1]),
                event(
                    "l1.write.finished",
                    [raw_key(bytes([number]), 0) for number in range(10)],
                ),
            ]
        )
        self.assertGreater(len(l1_owners(records)), len(target))
        with self.assertRaisesRegex(AssertionError, "not actually committed"):
            assert_target_resident(records, target)

    def test_duplicate_commits_do_not_multiply_one_owner(self) -> None:
        keys = [raw_key(chunk, rank) for chunk in (b"a", b"b") for rank in (0, 1)]
        records = [
            bindings(),
            event("l1.write.finished", keys),
            event("l1.write.finished", keys),
        ]
        target = target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)
        records.append(event("l1.keys.evicted", keys))
        with self.assertRaisesRegex(AssertionError, "not actually committed"):
            assert_target_resident(records, target)

    def test_removing_one_of_two_real_copies_preserves_residency(self) -> None:
        keys = [raw_key(chunk, rank) for chunk in (b"a", b"b") for rank in (0, 1)]
        records = [
            bindings(),
            event("l1.write.finished", keys),
            event("l1.write.finished", keys, "cpu1"),
            event("l1.keys.evicted", keys),
        ]
        target = target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)
        assert_target_resident(records, target)

    def test_namespace_and_token_offsets_cannot_be_guessed(self) -> None:
        keys = [
            raw_key(chunk, rank, model="other")
            for chunk in (b"a", b"b")
            for rank in (0, 1)
        ]
        records = [bindings(), event("l1.write.finished", keys)]
        with self.assertRaisesRegex(AssertionError, "Incomplete required"):
            target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)

    def test_shared_suffix_from_another_request_keeps_its_parent_identity(self) -> None:
        keys = [raw_key(chunk, rank) for chunk in (b"a", b"b") for rank in (0, 1)]
        other = bindings()
        other["session_id"] = "another-request"
        other["metadata"]["token_chunks"][0] = [9, 2]
        other["metadata"]["chunk_hashes"] = [
            {"bytes_base64": base64.b64encode(value).decode()} for value in (b"x", b"y")
        ]
        other["metadata"]["parent_hashes"][1] = {
            "bytes_base64": base64.b64encode(b"x").decode()
        }
        records = [bindings(), other, event("l1.write.finished", keys)]
        target = target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)
        self.assertEqual(
            {key["chunk_hash_hex"] for key in target}, {b"a".hex(), b"b".hex()}
        )
        records[0]["metadata"]["parent_hashes"][1] = {
            "bytes_base64": base64.b64encode(b"x").decode()
        }
        with self.assertRaisesRegex(AssertionError, "Missing or ambiguous"):
            target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)

    def test_unrelated_eviction_or_remaining_copy_cannot_prove_target_absence(
        self,
    ) -> None:
        keys = [raw_key(chunk, rank) for chunk in (b"a", b"b") for rank in (0, 1)]
        records = [bindings(), event("l1.write.finished", keys)]
        target = target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)
        records.append(event("l1.keys.evicted", [raw_key(b"unrelated", 0)]))
        with self.assertRaisesRegex(AssertionError, "committed L1 owner"):
            assert_target_absent(records, target)
        records.extend(
            [event("l1.write.finished", keys, "cpu1"), event("l1.keys.evicted", keys)]
        )
        with self.assertRaisesRegex(AssertionError, "committed L1 owner"):
            assert_target_absent(records, target)
        records.append(event("l1.keys.evicted", keys, "cpu1"))
        assert_target_absent(records, target)

    def test_wrong_token_offsets_cannot_reconstruct_a_prefix(self) -> None:
        records = [bindings()]
        records[0]["metadata"]["token_offsets"] = [2, 0]
        with self.assertRaisesRegex(AssertionError, "Missing or ambiguous"):
            target_l1_keys(records, [1, 2, 3, 4, 5], 2, "model", 2)


if __name__ == "__main__":
    unittest.main()
