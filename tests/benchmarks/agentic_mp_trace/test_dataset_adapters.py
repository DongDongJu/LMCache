# SPDX-License-Identifier: Apache-2.0

# Standard
import json

# First Party
from benchmarks.agentic_mp_trace.datasets import get_adapter


def test_jsonl_conversation_adapter(tmp_path):
    path = tmp_path / "conversations.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "conv1",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
            }
        )
        + "\n"
    )
    adapter = get_adapter("jsonl_conversations")
    result = adapter.load_requests(
        dataset_name="local",
        catalog_entry={
            "adapter": "jsonl_conversations",
            "repo_url": "file://local",
            "storage_classes": ["mixed_agent"],
        },
        job={
            "name": "job",
            "path": str(path),
            "num_tasks": 1,
            "turns_per_task": 1,
            "storage_class": "mixed_agent",
        },
    )
    assert not result.skipped
    assert result.requests[0].messages[1]["content"] == "hello"


def test_tau_adapter_builds_policy_and_tool_prompt(tmp_path):
    path = tmp_path / "tau.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "t1",
                    "policy": "refund policy",
                    "tools": [{"name": "lookup_order"}],
                    "user_request": "I need a refund",
                }
            ]
        )
    )
    result = get_adapter("tau_bench").load_requests(
        dataset_name="tau",
        catalog_entry={
            "adapter": "tau_bench",
            "repo_url": "https://github.com/sierra-research/tau2-bench",
            "storage_classes": ["tool_agent"],
        },
        job={
            "name": "tau_job",
            "path": str(path),
            "num_tasks": 1,
            "turns_per_task": 2,
            "storage_class": "tool_agent",
        },
    )
    assert len(result.requests) == 2
    assert "refund policy" in result.requests[0].messages[0]["content"]


def test_gated_dataset_skips_without_access():
    result = get_adapter("gaia").load_requests(
        dataset_name="gaia",
        catalog_entry={
            "adapter": "gaia",
            "hf_url": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
            "storage_classes": ["general_assistant"],
        },
        job={
            "name": "gaia_job",
            "path": "/does/not/exist",
            "num_tasks": 1,
            "turns_per_task": 1,
            "storage_class": "general_assistant",
        },
    )
    assert result.skipped
    assert "gated" in result.warnings[0]

