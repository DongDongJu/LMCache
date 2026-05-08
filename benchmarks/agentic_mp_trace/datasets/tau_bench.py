# SPDX-License-Identifier: Apache-2.0

"""tau-bench/tau2-bench offline transcript adapter."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter, _flatten_text


class TauBenchAdapter(DatasetAdapter):
    adapter_name = "tau_bench"
    default_storage_class = "tool_agent"
    default_system_prompt = (
        "You are a customer-service tool agent. Follow the domain policy, "
        "use the available tools, and keep state across turns."
    )

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        policy = _flatten_text(
            record.get("policy")
            or record.get("domain_policy")
            or record.get("instructions")
            or job.get("subset"),
            limit=8000,
        )
        tools = _flatten_text(
            record.get("tools") or record.get("apis") or record.get("actions"),
            limit=8000,
        )
        user = _flatten_text(
            record.get("user_request")
            or record.get("request")
            or record.get("task")
            or record,
            limit=10000,
        )
        return [
            {
                "role": "system",
                "content": (
                    f"{self.default_system_prompt}\n\nPolicy:\n{policy}\n\n"
                    f"Tools:\n{tools}"
                ),
            },
            {"role": "user", "content": user},
        ]

