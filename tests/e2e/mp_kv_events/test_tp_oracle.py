# SPDX-License-Identifier: Apache-2.0
"""Final placement state cannot hide duplicate native TP publications."""

# Standard
import unittest

# Third Party
from tp_scenarios import require_native_publication_once


def publication(kind: str, hashes: list[int], medium: str = "CPU") -> dict:
    """Represent an independently captured native publication record."""
    return {
        "received_ns": 123,
        "payload": [1.0, [{"type": kind, "block_hashes": hashes, "medium": medium}], 0],
    }


class TestTPPublicationOracle(unittest.TestCase):
    def test_counts_each_target_once_and_excludes_gpu_and_barrier_hashes(self) -> None:
        target = {"-7", str(2**64 - 1)}
        records = [
            publication("BlockStored", [-7, 2**64 - 1], "GPU"),
            publication("BlockStored", [99]),
            publication("BlockStored", [-7, 2**64 - 1, 100], "CPU_PINNED"),
        ]
        evidence = require_native_publication_once(records, target, "BlockStored")
        self.assertEqual([row["target_hashes"] for row in evidence], [[-7, 2**64 - 1]])

    def test_duplicate_hashes_in_one_or_multiple_messages_are_rejected(self) -> None:
        for records in (
            [publication("BlockStored", [-7, -7])],
            [publication("BlockStored", [-7]), publication("BlockStored", [-7])],
        ):
            with (
                self.subTest(records=records),
                self.assertRaisesRegex(AssertionError, "exactly one"),
            ):
                require_native_publication_once(records, {"-7"}, "BlockStored")

    def test_missing_or_gpu_only_removal_does_not_satisfy_cpu_withdrawal(self) -> None:
        for records in ([], [publication("BlockRemoved", [-7], "GPU")]):
            with (
                self.subTest(records=records),
                self.assertRaisesRegex(AssertionError, "exactly one"),
            ):
                require_native_publication_once(records, {"-7"}, "BlockRemoved")

    def test_store_remove_store_cannot_pass_from_final_ledger_state(self) -> None:
        records = [
            publication("BlockStored", [-7]),
            publication("BlockRemoved", [-7]),
            publication("BlockStored", [-7]),
            publication("BlockRemoved", [-7]),
        ]
        for kind in ("BlockStored", "BlockRemoved"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(AssertionError, "exactly one"),
            ):
                require_native_publication_once(records, {"-7"}, kind)


if __name__ == "__main__":
    unittest.main()
