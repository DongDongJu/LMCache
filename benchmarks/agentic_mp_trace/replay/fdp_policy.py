# SPDX-License-Identifier: Apache-2.0

"""FDP RUH policy resolution for agentic replay classes."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import expand_ruh_ids


def validate_ruh_ids(ids: list[int], *, ruh_count: int) -> list[int]:
    if len(set(ids)) != len(ids):
        raise ValueError("RUH ID lists must not contain duplicates")
    for ruh_id in ids:
        if ruh_id < 0 or ruh_id >= ruh_count:
            raise ValueError(
                f"RUH ID {ruh_id} is outside configured ruh_count={ruh_count}"
            )
    return ids


def resolve_policy(
    *,
    mode_cfg: dict[str, Any],
    storage_class: str,
    ruh_count: int,
) -> tuple[bool, list[int], list[int]]:
    use_fdp = bool(mode_cfg.get("use_fdp", False))
    if not use_fdp:
        return False, [], []

    data_spec = mode_cfg.get("default_data_ruhs")
    metadata_spec = mode_cfg.get("default_metadata_ruhs")
    class_cfg = mode_cfg.get("classes", {}).get(storage_class)
    if class_cfg is not None:
        data_spec = class_cfg.get("data_ruhs", data_spec)
        metadata_spec = class_cfg.get("metadata_ruhs", metadata_spec)
    if data_spec is None or metadata_spec is None:
        raise ValueError(f"missing RUH policy for storage class {storage_class!r}")

    data_ruhs = validate_ruh_ids(expand_ruh_ids(data_spec), ruh_count=ruh_count)
    metadata_ruhs = validate_ruh_ids(
        expand_ruh_ids(metadata_spec),
        ruh_count=ruh_count,
    )
    return True, data_ruhs, metadata_ruhs


def validate_replay_modes(replay_cfg: dict[str, Any]) -> None:
    ruh_count = int(replay_cfg.get("ruh_count", 4))
    if ruh_count < 1:
        raise ValueError("replay.ruh_count must be positive")
    for mode_name, mode_cfg in replay_cfg.get("modes", {}).items():
        if not bool(mode_cfg.get("use_fdp", False)):
            continue
        for key in ("default_data_ruhs", "default_metadata_ruhs"):
            if key in mode_cfg:
                validate_ruh_ids(expand_ruh_ids(mode_cfg[key]), ruh_count=ruh_count)
        for class_cfg in mode_cfg.get("classes", {}).values():
            validate_ruh_ids(
                expand_ruh_ids(class_cfg.get("data_ruhs", [])),
                ruh_count=ruh_count,
            )
            validate_ruh_ids(
                expand_ruh_ids(class_cfg.get("metadata_ruhs", [])),
                ruh_count=ruh_count,
            )

