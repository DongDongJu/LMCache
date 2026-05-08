# SPDX-License-Identifier: Apache-2.0

"""GAIA offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter


class GAIAAdapter(DatasetAdapter):
    adapter_name = "gaia"
    default_storage_class = "general_assistant"
    gated = True
    default_system_prompt = (
        "You are a general AI assistant with access to tools and web-search "
        "style observations. Do not expose private benchmark answers in logs."
    )

