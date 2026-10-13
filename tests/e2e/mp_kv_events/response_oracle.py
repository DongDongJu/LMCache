# SPDX-License-Identifier: Apache-2.0
"""Reject failed or truncated inference streams before using their evidence."""


def validate_inference_response(
    chunks: list[dict], *, saw_done: bool, max_tokens: int
) -> int:
    """Require a complete single-choice OpenAI stream and return generated tokens.

    Query-only routing replies are intentionally outside this inference contract.
    """
    assert chunks and saw_done, "Inference stream ended without its DONE marker"
    assert not any(chunk.get("error") for chunk in chunks), (
        "Inference stream contains an error payload"
    )
    choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
    assert choices and all(choice.get("index") == 0 for choice in choices), (
        "Expected one inference choice with index zero"
    )
    terminal = [choice for choice in choices if choice.get("finish_reason") is not None]
    assert len(terminal) == 1 and terminal[0]["finish_reason"] in ("stop", "length"), (
        "Inference stream did not finish normally",
        terminal,
    )
    assert choices[-1] is terminal[0], "Inference emitted content after completion"
    usages = [chunk["usage"] for chunk in chunks if chunk.get("usage") is not None]
    assert usages, "Inference stream omitted final token usage"
    counts = []
    for usage in usages:
        prompt = usage.get("prompt_tokens")
        completed = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        assert all(type(value) is int for value in (prompt, completed, total)), (
            "Inference token usage must contain integer counts",
            usage,
        )
        assert (
            prompt > 0 and 0 < completed <= max_tokens and total == prompt + completed
        ), ("Invalid inference token usage", usage)
        counts.append((prompt, completed, total))
    assert len(set(counts)) == 1, "Inference reported conflicting final token usage"
    return counts[-1][1]
