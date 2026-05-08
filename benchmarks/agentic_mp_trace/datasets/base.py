# SPDX-License-Identifier: Apache-2.0

"""Dataset adapter primitives for agentic MP trace recording."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os


@dataclass
class AgenticRequest:
    request_id: str
    session_id: str
    dataset_name: str
    dataset_source_url: str
    task_id: str
    turn_index: int
    messages: list[dict[str, str]]
    prompt_text: str | None
    max_tokens: int
    temperature: float
    scheduled_time_s: float | None
    reuse_group: str | None
    expected_input_tokens: int | None
    storage_class: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdapterLoadResult:
    requests: list[AgenticRequest]
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


def _flatten_text(value: Any, *, limit: int = 16000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_flatten_text(item, limit=limit) for item in value[:12]]
        return "\n".join(part for part in parts if part)[:limit]
    if isinstance(value, dict):
        preferred = [
            "instruction",
            "task",
            "question",
            "problem_statement",
            "confirmed_task",
            "intent",
            "description",
            "goal",
            "query",
            "prompt",
            "messages",
        ]
        parts = []
        for key in preferred:
            if key in value:
                text = _flatten_text(value[key], limit=limit)
                if text:
                    parts.append(f"{key}: {text}")
        if not parts:
            for key, item in list(value.items())[:12]:
                text = _flatten_text(item, limit=limit // 2)
                if text:
                    parts.append(f"{key}: {text}")
        return "\n".join(parts)[:limit]
    return str(value)[:limit]


def _record_id(record: dict[str, Any], fallback: int) -> str:
    for key in ("id", "task_id", "instance_id", "qid", "question_id", "name"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return f"task_{fallback:05d}"


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in (
            "data",
            "examples",
            "instances",
            "tasks",
            "rows",
            "records",
            "trajectories",
        ):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    return []


def read_dataset_records(path: str | None, *, limit: int | None = None) -> list[dict]:
    if not path:
        return []
    root = Path(os.path.expanduser(os.path.expandvars(path)))
    if not root.exists():
        return []
    files: list[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(
            item
            for item in root.rglob("*")
            if item.suffix.lower() in {".json", ".jsonl", ".ndjson"}
        )

    records: list[dict[str, Any]] = []
    for file_path in files:
        if limit is not None and len(records) >= limit:
            break
        if file_path.suffix.lower() in {".jsonl", ".ndjson"}:
            with open(file_path) as file_obj:
                for line in file_obj:
                    if limit is not None and len(records) >= limit:
                        break
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if isinstance(item, dict):
                        item.setdefault("_source_file", os.fspath(file_path))
                        records.append(item)
        else:
            with open(file_path) as file_obj:
                payload = json.load(file_obj)
            for item in _extract_records(payload):
                if limit is not None and len(records) >= limit:
                    break
                item.setdefault("_source_file", os.fspath(file_path))
                records.append(item)
    return records


class DatasetAdapter:
    adapter_name = "base"
    default_storage_class = "tool_agent"
    default_system_prompt = "You are an agentic assistant."
    gated = False

    def load_requests(
        self,
        *,
        dataset_name: str,
        catalog_entry: dict[str, Any],
        job: dict[str, Any],
        strict_dataset_access: bool = False,
    ) -> AdapterLoadResult:
        if self.gated and not bool(job.get("access_granted", False)):
            message = (
                f"{dataset_name}: dataset access is gated; set "
                "access_granted: true after accepting upstream terms"
            )
            if strict_dataset_access:
                raise PermissionError(message)
            return AdapterLoadResult([], [message], skipped=True)

        records = read_dataset_records(job.get("path"), limit=int(job["num_tasks"]))
        warnings = []
        if not records:
            warnings.append(
                f"{dataset_name}: no local records found at {job.get('path')!r}"
            )
            return AdapterLoadResult([], warnings, skipped=True)

        requests: list[AgenticRequest] = []
        for index, record in enumerate(records[: int(job["num_tasks"])]):
            requests.extend(
                self._record_to_requests(
                    dataset_name=dataset_name,
                    catalog_entry=catalog_entry,
                    job=job,
                    record=record,
                    index=index,
                )
            )
        return AdapterLoadResult(requests, warnings, skipped=False)

    def _record_to_requests(
        self,
        *,
        dataset_name: str,
        catalog_entry: dict[str, Any],
        job: dict[str, Any],
        record: dict[str, Any],
        index: int,
    ) -> list[AgenticRequest]:
        task_id = _record_id(record, index)
        turns = max(1, int(job.get("turns_per_task", 1)))
        storage_class = str(
            job.get("storage_class")
            or catalog_entry.get("storage_classes", [self.default_storage_class])[0]
        )
        source_url = str(
            catalog_entry.get("repo_url")
            or catalog_entry.get("hf_url")
            or catalog_entry.get("homepage_url")
            or ""
        )
        data_url = str(
            catalog_entry.get("hf_url")
            or catalog_entry.get("resource_page_url")
            or catalog_entry.get("data_url")
            or source_url
        )
        requests = []
        base_messages = self.build_messages(record=record, job=job)
        for turn_index in range(turns):
            messages = self.messages_for_turn(base_messages, record, turn_index)
            requests.append(
                AgenticRequest(
                    request_id=(
                        f"{dataset_name}.{job['name']}.{task_id}.turn_{turn_index:03d}"
                    ),
                    session_id=f"{dataset_name}.{job['name']}.{task_id}",
                    dataset_name=dataset_name,
                    dataset_source_url=source_url,
                    task_id=task_id,
                    turn_index=turn_index,
                    messages=messages,
                    prompt_text=None,
                    max_tokens=int(job.get("max_tokens", 256)),
                    temperature=float(job.get("temperature", 0.0)),
                    scheduled_time_s=None,
                    reuse_group=str(job.get("reuse_group", dataset_name)),
                    expected_input_tokens=None,
                    storage_class=storage_class,
                    metadata={
                        "adapter": self.adapter_name,
                        "subset": job.get("subset"),
                        "split": job.get("split"),
                        "source": job.get("source"),
                        "data_url": data_url,
                        "source_file": record.get("_source_file"),
                    },
                )
            )
        return requests

    def build_messages(
        self,
        *,
        record: dict[str, Any],
        job: dict[str, Any],
    ) -> list[dict[str, str]]:
        body = _flatten_text(record)
        return [
            {"role": "system", "content": self.default_system_prompt},
            {"role": "user", "content": body},
        ]

    def messages_for_turn(
        self,
        base_messages: list[dict[str, str]],
        record: dict[str, Any],
        turn_index: int,
    ) -> list[dict[str, str]]:
        if turn_index == 0:
            return list(base_messages)
        observation = _flatten_text(
            record.get("observation")
            or record.get("observations")
            or record.get("actions")
            or record,
            limit=6000,
        )
        return [
            *base_messages,
            {
                "role": "assistant",
                "content": f"Investigation step {turn_index}: choose the next action.",
            },
            {
                "role": "user",
                "content": f"Tool or environment observation:\n{observation}",
            },
        ]

