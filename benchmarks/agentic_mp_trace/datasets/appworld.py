# SPDX-License-Identifier: Apache-2.0

"""AppWorld offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class AppWorldAdapter(DatasetAdapter):
    adapter_name = "appworld"
    default_storage_class = "tool_agent"
    default_system_prompt = (
        "You are an API and coding agent operating simulated day-to-day apps."
    )

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        apis = _flatten_text(record.get("api_docs") or record.get("apis"), limit=16000)
        task = _flatten_text(
            record.get("instruction") or record.get("task") or record,
            limit=12000,
        )
        state = _flatten_text(record.get("initial_state") or record.get("state"))
        return [
            {
                "role": "system",
                "content": f"{self.default_system_prompt}\n\nAvailable APIs:\n{apis}",
            },
            {
                "role": "user",
                "content": f"Instruction:\n{task}\n\nInitial state:\n{state}",
            },
        ]

