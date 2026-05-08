# SPDX-License-Identifier: Apache-2.0

"""Config helpers shared by agentic MP trace scripts."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any
import json
import os
import re

# Third Party
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any, *, strict: bool = False) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name in os.environ:
                return os.environ[name]
            if strict:
                raise ValueError(f"environment variable {name} is not set")
            return match.group(0)

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item, strict=strict) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item, strict=strict) for key, item in value.items()}
    return value


def load_yaml_config(path: str | Path, *, strict_env: bool = False) -> dict[str, Any]:
    with open(path) as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    return expand_env(payload, strict=strict_env)


def write_yaml(path: str | Path, payload: Any) -> None:
    with open(path, "w") as file_obj:
        yaml.safe_dump(payload, file_obj, sort_keys=False)


def write_json(path: str | Path, payload: Any) -> None:
    with open(path, "w") as file_obj:
        json.dump(payload, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def write_text(path: str | Path, text: str) -> None:
    with open(path, "w") as file_obj:
        file_obj.write(text)

