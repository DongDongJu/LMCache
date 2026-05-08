# SPDX-License-Identifier: Apache-2.0

"""Mind2Web offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class Mind2WebAdapter(DatasetAdapter):
    adapter_name = "mind2web"
    default_storage_class = "browser_agent"
    default_system_prompt = (
        "You are a web automation agent. Interpret HTML, prior actions, "
        "and the user task to decide the next action."
    )

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        task = _flatten_text(
            record.get("confirmed_task")
            or record.get("task")
            or record.get("annotation")
            or record,
            limit=10000,
        )
        html = _flatten_text(
            record.get("cleaned_html") or record.get("raw_html") or record.get("html"),
            limit=18000,
        )
        actions = _flatten_text(
            record.get("action_reprs") or record.get("actions"),
            limit=8000,
        )
        website = record.get("website") or record.get("domain") or ""
        return [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Website/domain: {website}\nTask:\n{task}\n\n"
                    f"HTML excerpt:\n{html}\n\nAction history:\n{actions}"
                ),
            },
        ]

