# SPDX-License-Identifier: Apache-2.0

"""ToolBench offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class ToolBenchAdapter(DatasetAdapter):
    adapter_name = "toolbench"
    default_storage_class = "shared_prefix_heavy"
    default_system_prompt = "You are a tool-using assistant."

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        tools = _flatten_text(
            record.get("tools")
            or record.get("apis")
            or record.get("tool_descriptions")
            or record.get("api_list"),
            limit=18000,
        )
        task = _flatten_text(
            record.get("query")
            or record.get("instruction")
            or record.get("task")
            or record,
            limit=12000,
        )
        return [
            {
                "role": "system",
                "content": f"{self.default_system_prompt}\n\nAvailable tools:\n{tools}",
            },
            {"role": "user", "content": task},
        ]

