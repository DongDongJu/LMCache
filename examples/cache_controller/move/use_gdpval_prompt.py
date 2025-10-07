#!/usr/bin/env python3
"""Send a GDPVal dataset prompt to a vLLM endpoint and print token ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASET_DIR = "/xfs1/alex/dataset/openai_gdpval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(DEFAULT_DATASET_DIR),
        help="Directory containing the downloaded openai/gdpval dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Row index within the dataset to use (ignored when task-id is set).",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        help="Optional task_id to select a specific dataset row.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Model name passed through to the vLLM endpoint.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Hostname for the vLLM server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the vLLM server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate in the completion request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature for the completion request.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout (seconds) for requests to the vLLM server.",
    )
    parser.add_argument(
        "--save-token-file",
        type=Path,
        help="Optional path to save the returned token list as JSON.",
    )
    return parser.parse_args()


def select_entry(args: argparse.Namespace) -> dict[str, Any]:
    dataset = load_dataset(str(args.dataset_dir), split="train")

    if args.task_id is not None:
        try:
            entry = next(row for row in dataset if row["task_id"] == args.task_id)
        except StopIteration as exc:  # pragma: no cover - simple CLI error path
            raise SystemExit(f"task_id {args.task_id!r} not found in dataset") from exc
        return entry

    if not (0 <= args.index < len(dataset)):
        raise SystemExit(f"index {args.index} is out of bounds for {len(dataset)} rows")
    return dataset[args.index]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> None:
    args = parse_args()
    entry = select_entry(args)
    prompt = entry["prompt"]

    base_url = f"http://{args.host}:{args.port}"
    completion_payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    print("Sending completion request...")
    completion_response = post_json(
        f"{base_url}/v1/completions", completion_payload, args.timeout
    )
    choice = completion_response.get("choices", [{}])[0]
    print("Completion response:")
    print(choice.get("text", "<no text>"))

    tokenize_payload = {"model": args.model, "prompt": prompt}
    print("\nRequesting tokenization...")
    token_response = post_json(f"{base_url}/tokenize", tokenize_payload, args.timeout)
    tokens = token_response.get("tokens", [])
    print("Token IDs:")
    print(tokens)

    if args.save_token_file is not None:
        args.save_token_file.write_text(json.dumps(tokens, indent=2))
        print(f"Tokens saved to {args.save_token_file}")

    print("\nContext metadata:")
    print(json.dumps({
        "task_id": entry["task_id"],
        "sector": entry.get("sector"),
        "occupation": entry.get("occupation"),
    }, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
