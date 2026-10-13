# SPDX-License-Identifier: Apache-2.0
"""Recovery cannot pass from final absence alone or GPU-scoped withdrawal."""

# Standard
import unittest

# Third Party
from recovery import assert_cpu_replay_order, require_loss_receipt


def transition(kind: str, medium: str = "CPU", block_hash: int = -7) -> dict:
    """Represent independent engine publication for one opaque native ID."""
    return {
        "payload": [
            1.0,
            [{"type": kind, "medium": medium, "block_hashes": [block_hash]}],
            0,
        ]
    }


class TestRecoveryOracle(unittest.TestCase):
    """Require the whole invalidation/replay transition for each original ID."""

    def test_invalidation_and_ordered_replay_are_required(self) -> None:
        records = [
            transition("BlockRemoved"),
            transition("BlockStored"),
            transition("BlockRemoved"),
        ]
        self.assertEqual(
            assert_cpu_replay_order(records, {"-7"}),
            {"-7": ["removed", "stored", "removed"]},
        )

    def test_final_absence_and_duplicate_delivery_do_not_suffice(self) -> None:
        for kinds in (
            [],
            ["BlockStored", "BlockRemoved"],
            ["BlockRemoved", "BlockStored"],
            ["BlockRemoved", "BlockStored", "BlockStored", "BlockRemoved"],
        ):
            with self.subTest(kinds=kinds):
                with self.assertRaisesRegex(AssertionError, "order changed"):
                    assert_cpu_replay_order(
                        [transition(kind) for kind in kinds], {"-7"}
                    )

    def test_gpu_removal_cannot_replace_cpu_invalidation(self) -> None:
        records = [
            transition("BlockRemoved", "GPU"),
            transition("BlockStored"),
            transition("BlockRemoved"),
        ]
        with self.assertRaisesRegex(AssertionError, "order changed"):
            assert_cpu_replay_order(records, {"-7"})

    def test_unrelated_placement_cannot_satisfy_fixture_replay(self) -> None:
        records = [
            transition("BlockRemoved", block_hash=99),
            transition("BlockStored", block_hash=99),
            transition("BlockRemoved", block_hash=99),
        ]
        with self.assertRaisesRegex(AssertionError, "order changed"):
            assert_cpu_replay_order(records, {"-7"})

    def test_overflow_requires_explicit_loss_then_matching_cursor(self) -> None:
        loss = "MP_KV_SUBSCRIBER reset=events_lost\n"
        received = "MP_KV_SUBSCRIBER received_ns=100 incarnation=7 cursor=30 events=8\n"
        self.assertEqual(require_loss_receipt(loss + received, 7, 30)["cursor"], 30)
        for log in (
            "MP_KV_SUBSCRIBER reset=disconnect\n" + received,
            received + loss,
            loss + received.replace("incarnation=7", "incarnation=8"),
            loss + received.replace("cursor=30", "cursor=29"),
        ):
            with self.subTest(log=log), self.assertRaises(AssertionError):
                require_loss_receipt(log, 7, 30)


if __name__ == "__main__":
    unittest.main()
