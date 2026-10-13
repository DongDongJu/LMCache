# SPDX-License-Identifier: Apache-2.0
"""Incomplete or failed streams cannot become successful cache-load evidence."""

# Standard
from copy import deepcopy
import unittest

# Third Party
from response_oracle import validate_inference_response


def completed() -> list[dict]:
    """Use the terminal/usage shape captured from the real Dynamo vLLM stream."""
    return [
        {
            "choices": [{"text": "   ", "index": 0, "finish_reason": "length"}],
            "usage": None,
        },
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 513,
                "completion_tokens": 1,
                "total_tokens": 514,
            },
        },
    ]


class TestInferenceResponseOracle(unittest.TestCase):
    def test_real_terminal_shape_and_eos_are_valid(self) -> None:
        for reason in ("length", "stop"):
            chunks = completed()
            chunks[0]["choices"][0]["finish_reason"] = reason
            self.assertEqual(
                validate_inference_response(chunks, saw_done=True, max_tokens=1), 1
            )

    def test_partial_text_error_and_missing_terminal_fail(self) -> None:
        for reason in (None, "error: worker failed", "abort"):
            chunks = completed()
            chunks[0]["choices"][0]["finish_reason"] = reason
            with self.subTest(reason=reason), self.assertRaises(AssertionError):
                validate_inference_response(chunks, saw_done=True, max_tokens=1)
        chunks = completed() + [{"error": {"message": "worker failed"}}]
        with self.assertRaisesRegex(AssertionError, "error payload"):
            validate_inference_response(chunks, saw_done=True, max_tokens=1)

    def test_missing_done_or_usage_cannot_pass_from_matching_text(self) -> None:
        with self.assertRaisesRegex(AssertionError, "DONE"):
            validate_inference_response(completed(), saw_done=False, max_tokens=1)
        with self.assertRaisesRegex(AssertionError, "token usage"):
            validate_inference_response(completed()[:1], saw_done=True, max_tokens=1)

    def test_bad_usage_and_duplicate_completion_are_rejected(self) -> None:
        for count in (0, 2, True, None):
            chunks = completed()
            chunks[1]["usage"]["completion_tokens"] = count
            with self.subTest(count=count), self.assertRaises(AssertionError):
                validate_inference_response(chunks, saw_done=True, max_tokens=1)
        chunks = completed()
        chunks.append(deepcopy(chunks[0]))
        with self.assertRaisesRegex(AssertionError, "finish normally"):
            validate_inference_response(chunks, saw_done=True, max_tokens=1)


if __name__ == "__main__":
    unittest.main()
