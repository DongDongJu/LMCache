# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter registry for agentic MP trace recording."""

# First Party
from benchmarks.agentic_mp_trace.datasets.agentbench import AgentBenchAdapter
from benchmarks.agentic_mp_trace.datasets.appworld import AppWorldAdapter
from benchmarks.agentic_mp_trace.datasets.base import (
    AdapterLoadResult,
    AgenticRequest,
    DatasetAdapter,
)
from benchmarks.agentic_mp_trace.datasets.browsergym import BrowserGymAdapter
from benchmarks.agentic_mp_trace.datasets.gaia import GAIAAdapter
from benchmarks.agentic_mp_trace.datasets.jsonl_conversations import (
    JSONLConversationsAdapter,
)
from benchmarks.agentic_mp_trace.datasets.mind2web import Mind2WebAdapter
from benchmarks.agentic_mp_trace.datasets.osworld import OSWorldAdapter
from benchmarks.agentic_mp_trace.datasets.swe_bench import SWEBenchAdapter
from benchmarks.agentic_mp_trace.datasets.tau_bench import TauBenchAdapter
from benchmarks.agentic_mp_trace.datasets.the_agent_company import (
    TheAgentCompanyAdapter,
)
from benchmarks.agentic_mp_trace.datasets.toolbench import ToolBenchAdapter
from benchmarks.agentic_mp_trace.datasets.webarena import WebArenaAdapter
from benchmarks.agentic_mp_trace.datasets.workarena import WorkArenaAdapter


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "agentbench": AgentBenchAdapter,
    "appworld": AppWorldAdapter,
    "browsergym": BrowserGymAdapter,
    "gaia": GAIAAdapter,
    "jsonl_conversations": JSONLConversationsAdapter,
    "mind2web": Mind2WebAdapter,
    "osworld": OSWorldAdapter,
    "swe_bench": SWEBenchAdapter,
    "tau_bench": TauBenchAdapter,
    "the_agent_company": TheAgentCompanyAdapter,
    "toolbench": ToolBenchAdapter,
    "webarena": WebArenaAdapter,
    "workarena": WorkArenaAdapter,
}


def get_adapter(name: str) -> DatasetAdapter:
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        raise ValueError(f"unknown dataset adapter: {name}")
    return adapter_cls()


__all__ = [
    "ADAPTERS",
    "AdapterLoadResult",
    "AgenticRequest",
    "DatasetAdapter",
    "get_adapter",
]

