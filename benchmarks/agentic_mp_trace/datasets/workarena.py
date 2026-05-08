# SPDX-License-Identifier: Apache-2.0

"""WorkArena offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.webarena import WebArenaAdapter


class WorkArenaAdapter(WebArenaAdapter):
    adapter_name = "workarena"
    default_storage_class = "workplace_agent"
    gated = True
    default_system_prompt = (
        "You are a workplace browser agent operating ServiceNow-style tasks."
    )

