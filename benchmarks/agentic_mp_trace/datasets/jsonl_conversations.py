# SPDX-License-Identifier: Apache-2.0

"""Generic JSONL conversation adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class JSONLConversationsAdapter(DatasetAdapter):
    adapter_name = "jsonl_conversations"
    default_storage_class = "mixed_agent"

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages = record.get("messages")
        if isinstance(messages, list) and all(
            isinstance(item, dict) and "role" in item and "content" in item
            for item in messages
        ):
            return [
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in messages
            ]
        prompt = (
            record.get("prompt")
            or record.get("prompt_text")
            or _flatten_text(record)
        )
        return [{"role": "user", "content": str(prompt)}]
