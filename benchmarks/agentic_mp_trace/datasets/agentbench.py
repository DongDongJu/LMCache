# SPDX-License-Identifier: Apache-2.0

"""AgentBench offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter


class AgentBenchAdapter(DatasetAdapter):
    adapter_name = "agentbench"
    default_storage_class = "tool_agent"
    default_system_prompt = (
        "You are an interactive agent operating benchmark environments "
        "such as OS, database, knowledge graph, web shopping, and browsing."
    )

