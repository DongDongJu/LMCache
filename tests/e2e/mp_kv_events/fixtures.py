# SPDX-License-Identifier: Apache-2.0
"""Deterministic token fixtures shared by the MP event scenarios."""

# Qwen tokenizer revision 7ae557604adf67be50417f59c2c2f167def9a775:
# "<|im_end|>\n<|im_start|>user\nReply YES.<|im_end|>\n"
# "<|im_start|>assistant\n". Keep the last newline outside the complete chunks.
# This strong continuation avoids near-tied whitespace logits from meaningless
# repeated-token prompts across native prefill and cached-token attention.
CONTINUATION = [
    151645,
    198,
    151644,
    872,
    198,
    20841,
    14080,
    13,
    151645,
    198,
    151644,
    77091,
    198,
]
# Same pinned tokenizer: "This is a deterministic cache routing test. "
FILLER = [1986, 374, 264, 72349, 6500, 29058, 1273, 13, 220]


def fixture(index: int, chunks: int = 2, chunk_size: int = 256) -> list[int]:
    """Return deterministic valid Qwen vocabulary IDs with a terminal token.

    Each fixture has a unique first token, preventing accidental common prefixes.
    Fixed natural-language padding keeps continuations stable across fixture IDs.
    The final token gives the engine its required recomputation boundary.
    """
    padding = chunks * chunk_size + 1 - len(CONTINUATION)
    if padding < 1:
        raise ValueError("Fixture requires room for its unique token and continuation")
    filler = (FILLER * ((padding - 1 + len(FILLER) - 1) // len(FILLER)))[: padding - 1]
    return [1000 + index] + filler + CONTINUATION
