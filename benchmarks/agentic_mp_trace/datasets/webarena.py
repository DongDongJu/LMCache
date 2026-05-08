# SPDX-License-Identifier: Apache-2.0

"""WebArena offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class WebArenaAdapter(DatasetAdapter):
    adapter_name = "webarena"
    default_storage_class = "browser_agent"
    default_system_prompt = (
        "You are a browser agent. Use the observation, URL, action history, "
        "and task instruction to choose the next browser action."
    )

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        task = _flatten_text(
            record.get("intent")
            or record.get("task")
            or record.get("instruction")
            or record,
            limit=10000,
        )
        observation = _flatten_text(
            record.get("observation")
            or record.get("trajectory")
            or record.get("actions")
            or record.get("html"),
            limit=14000,
        )
        url = record.get("start_url") or record.get("url") or ""
        return [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Task instruction:\n{task}\n\nStart URL: {url}\n\n"
                    f"Observation/action history:\n{observation}"
                ),
            },
        ]

