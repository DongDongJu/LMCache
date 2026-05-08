# SPDX-License-Identifier: Apache-2.0

"""Measurement helpers for agentic replay summaries."""

# Future
from __future__ import annotations

# Standard
from typing import Any


def annotate_target(summary: dict[str, Any], *, target_bytes: int) -> dict[str, Any]:
    host_delta = summary.get("host_write_bytes_delta")
    reached = host_delta is not None and int(host_delta) >= int(target_bytes)
    summary["target_host_write_bytes"] = int(target_bytes)
    summary["target_host_write_bytes_reached"] = bool(reached)
    summary["waf_available"] = summary.get("waf") is not None
    if not summary["waf_available"]:
        summary["waf_unavailable_reason"] = summary.get("waf_status")
    return summary

