# SPDX-License-Identifier: Apache-2.0

"""SWE-bench offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class SWEBenchAdapter(DatasetAdapter):
    adapter_name = "swe_bench"
    default_storage_class = "coding_agent"
    default_system_prompt = (
        "You are a coding agent. Resolve the issue, inspect relevant files, "
        "reason about tests, and propose a patch."
    )

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        repo = record.get("repo") or record.get("repository") or "unknown"
        commit = record.get("base_commit") or record.get("commit") or "unknown"
        issue = _flatten_text(
            record.get("problem_statement")
            or record.get("issue")
            or record.get("description")
            or record,
            limit=16000,
        )
        hints = _flatten_text(record.get("hints_text") or record.get("hints"))
        snippets = _flatten_text(
            record.get("patch")
            or record.get("test_patch")
            or record.get("FAIL_TO_PASS")
            or record.get("files"),
            limit=12000,
        )
        return [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Repository: {repo}\nBase commit: {commit}\n\n"
                    f"Issue:\n{issue}\n\nHints:\n{hints}\n\n"
                    f"Retrieved files / tests / snippets:\n{snippets}"
                ),
            },
        ]

