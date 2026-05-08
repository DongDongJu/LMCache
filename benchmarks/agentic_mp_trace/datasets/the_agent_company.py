# SPDX-License-Identifier: Apache-2.0

"""TheAgentCompany offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter


class TheAgentCompanyAdapter(DatasetAdapter):
    adapter_name = "the_agent_company"
    default_storage_class = "workplace_agent"
    default_system_prompt = (
        "You are a digital coworker handling software-company tasks involving "
        "web browsing, code, programs, and coworker communication."
    )

