# SPDX-License-Identifier: Apache-2.0

"""OSWorld offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.base import DatasetAdapter


class OSWorldAdapter(DatasetAdapter):
    adapter_name = "osworld"
    default_storage_class = "desktop_agent"
    default_system_prompt = (
        "You are a desktop computer-use agent. Use observations and action "
        "history to complete the open-ended task."
    )

