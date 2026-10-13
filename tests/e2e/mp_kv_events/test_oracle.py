# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the acceptance oracle's false-positive boundaries."""

# Standard
import base64
import unittest

# Third Party
from scenarios import NativeLedger


def batch(*events: dict) -> dict:
    """Represent an independently captured native publisher message."""
    return {"payload": [1.0, list(events), 0]}


def stored(
    medium: str, hashes: list[int], tokens: list[int], parent: int | None = None
) -> dict:
    """Build wire fixtures with native block size two, unrelated to LM chunks."""
    return {
        "type": "BlockStored",
        "medium": medium,
        "block_hashes": hashes,
        "token_ids": tokens,
        "block_size": 2,
        "parent_block_hash": parent,
    }


class TestNativeOracle(unittest.TestCase):
    """The oracle must preserve medium, exact token, and parent-chain boundaries."""

    def test_cpu_removal_preserves_gpu(self) -> None:
        """CPU withdrawal cannot pass by accidentally dropping both tiers."""
        ledger = NativeLedger(
            [
                batch(
                    stored("GPU", [7, 9], [1, 2, 3, 4]),
                    stored("CPU", [7, 9], [1, 2, 3, 4]),
                ),
                batch(
                    {"type": "BlockRemoved", "medium": "CPU", "block_hashes": [7, 9]}
                ),
            ]
        )
        self.assertEqual(len(ledger.prefix([1, 2, 3, 4], "GPU")), 2)
        self.assertFalse(ledger.prefix([1, 2, 3, 4], "CPU"))

    def test_wrong_parent_is_not_a_reusable_suffix(self) -> None:
        """Matching tokens alone cannot conceal a broken native parent chain."""
        ledger = NativeLedger(
            [batch(stored("CPU", [7], [1, 2]), stored("CPU", [9], [3, 4], parent=99))]
        )
        self.assertEqual(len(ledger.prefix([1, 2, 3, 4], "CPU")), 1)

    def test_subblock_identities_survive_remove_restore(self) -> None:
        """Store-remove-store order, signed IDs and subblocks remain exact."""
        ledger = NativeLedger(
            [
                batch(stored("CPU_PINNED", [-7, 9], [1, 2, 3, 4])),
                batch(
                    {
                        "type": "BlockRemoved",
                        "medium": "CPU_PINNED",
                        "block_hashes": [-7, 9],
                    }
                ),
                batch(stored("CPU_PINNED", [-7, 9], [1, 2, 3, 4])),
            ]
        )
        blocks = ledger.prefix([1, 2, 3, 4, 5], "CPU")
        self.assertEqual([block["hash"] for block in blocks], [-7, 9])
        self.assertEqual(blocks[1]["parent"], -7)

    def test_native_publication_loss_is_visible(self) -> None:
        """A missed native removal cannot leave an apparently valid CPU ledger."""
        first = batch(stored("CPU", [7], [1, 2]))
        third = batch(stored("CPU", [9], [3, 4], parent=7))
        for record, sequence in ((first, 0), (third, 2)):
            record["frames_base64"] = [
                "",
                base64.b64encode(sequence.to_bytes(8, "big")).decode(),
                "",
            ]
        with self.assertRaisesRegex(AssertionError, "publication gap"):
            NativeLedger([first, third])

    def test_incomplete_native_descriptor_rejected(self) -> None:
        """Malformed token geometry must fail instead of truncating the oracle."""
        with self.assertRaises(AssertionError):
            NativeLedger([batch(stored("CPU", [7, 9], [1, 2, 3]))])


if __name__ == "__main__":
    unittest.main()
