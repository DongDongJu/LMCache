# SPDX-License-Identifier: Apache-2.0
"""Performance arms require matching real workloads and complete inference."""

# Standard
from copy import deepcopy
import unittest

# Third Party
from overhead import validate_matched_arms


def arm(enabled: bool) -> dict:
    """Provide twenty complete cold/repeat pairs with explicit runtime pins."""
    return {
        "status": "PASS",
        "events_enabled": enabled,
        "gpu": "gpu-uuid",
        "source": {"sha": "source"},
        "runner_source_sha256": {"overhead.py": "runner-sha"},
        "workload": "same",
        "samples_per_phase": 20,
        "deployment": {
            "engine": "vllm",
            "tp": 1,
            "transport": "zmq",
            "chunk_size": 16,
            "native_block_size": 16,
            "event_log_size": 4096,
            "node": "node",
            "image": "sha256:image",
            "external_runtime": None,
            "lock": {"version": 1},
        },
        "rows": [
            {
                "phase": phase,
                "tokens": [index, 198],
                "saw_done": True,
                "completion_tokens": 1,
                "output": "x",
                "response": [
                    {"choices": [{"index": 0, "text": "x", "finish_reason": "length"}]},
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": 1,
                            "total_tokens": 3,
                        },
                    },
                ],
            }
            for index in range(20)
            for phase in ("cold", "repeat")
        ],
    }


class TestOverheadOracle(unittest.TestCase):
    def test_identical_completed_workloads_are_comparable(self) -> None:
        validate_matched_arms(arm(True), arm(False))

    def test_runtime_pin_changes_are_rejected(self) -> None:
        for field in (
            "native_block_size",
            "event_log_size",
            "node",
            "image",
            "external_runtime",
        ):
            disabled = arm(False)
            disabled["deployment"][field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_matched_arms(arm(True), disabled)

    def test_same_image_cannot_hide_host_driver_changes(self) -> None:
        disabled = arm(False)
        disabled["runner_source_sha256"]["overhead.py"] = "different-runner"
        with self.assertRaisesRegex(ValueError, "runner_source_sha256"):
            validate_matched_arms(arm(True), disabled)

    def test_workload_labels_cannot_hide_different_tokens_or_output(self) -> None:
        for field, value in (("tokens", [99, 198]), ("output", "different")):
            disabled = arm(False)
            for row in disabled["rows"][:2]:
                row[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                validate_matched_arms(arm(True), disabled)

    def test_partial_response_and_missing_samples_do_not_pass(self) -> None:
        disabled = arm(False)
        disabled["rows"][0]["response"] = []
        with self.assertRaises(AssertionError):
            validate_matched_arms(arm(True), disabled)
        disabled = arm(False)
        disabled["rows"].pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_matched_arms(arm(True), disabled)
        enabled = arm(True)
        disabled = deepcopy(enabled)
        disabled["events_enabled"] = False
        for row in disabled["rows"]:
            row["saw_done"] = False
        with self.assertRaisesRegex(AssertionError, "DONE"):
            validate_matched_arms(enabled, disabled)


if __name__ == "__main__":
    unittest.main()
