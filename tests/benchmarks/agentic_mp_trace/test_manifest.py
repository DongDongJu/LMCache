# SPDX-License-Identifier: Apache-2.0

# Standard
import json


def test_manifest_schema_lists_required_trace_fields():
    with open("benchmarks/agentic_mp_trace/trace_manifest.schema.json") as file_obj:
        schema = json.load(file_obj)
    required = set(schema["properties"]["traces"]["items"]["required"])
    assert "trace_id" in required
    assert "dataset" in required
    assert "trace_stats" in required

