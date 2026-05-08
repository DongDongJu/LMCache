# SPDX-License-Identifier: Apache-2.0

"""BrowserGym offline transcript adapter."""

# First Party
from benchmarks.agentic_mp_trace.datasets.webarena import WebArenaAdapter


class BrowserGymAdapter(WebArenaAdapter):
    adapter_name = "browsergym"

