# SPDX-License-Identifier: Apache-2.0

"""Replay heterogeneous agentic traces into one raw-block/FDP namespace."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import argparse
import os
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

# First Party
from benchmarks.agentic_mp_trace.config import (  # noqa: E402
    load_yaml_config,
    write_json,
    write_yaml,
)
from benchmarks.agentic_mp_trace.replay.command_builder import (  # noqa: E402
    MODE_ALIASES,
    auto_iterations,
    build_fdp_replay_config,
    target_host_write_bytes,
)
from benchmarks.agentic_mp_trace.replay.measurement import (  # noqa: E402
    annotate_target,
)
from benchmarks.fdp_waf_stress.run_fdp_waf_stress import (  # noqa: E402
    build_summary_md,
    main as fdp_main,
)


VALID_MODES = ("no_fdp", "fdp_mixed", "fdp_separated")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=VALID_MODES)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=2)
    parser.add_argument("--duration-seconds", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id", default="agentic001")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_yaml_config(args.config)
    manifest = load_yaml_config(args.trace_manifest)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    fdp_config = build_fdp_replay_config(
        config=config,
        manifest=manifest,
        mode=args.mode,
    )
    target_bytes = target_host_write_bytes(
        fdp_config=fdp_config,
        replay_cfg=config["replay"],
    )
    iterations = args.iterations
    if args.duration_seconds is None:
        iterations = auto_iterations(
            requested_iterations=args.iterations,
            fdp_config=fdp_config,
            replay_cfg=config["replay"],
        )

    replay_config_path = os.path.join(output_dir, "replay_config.resolved.yaml")
    replay_manifest_path = os.path.join(output_dir, "replay_manifest.resolved.yaml")
    fdp_config_path = os.path.join(output_dir, "_fdp_replay_config.yaml")
    write_yaml(replay_config_path, fdp_config)
    write_yaml(replay_manifest_path, manifest)
    write_yaml(fdp_config_path, fdp_config)

    fdp_args = [
        "--config",
        fdp_config_path,
        "--mode",
        MODE_ALIASES[args.mode],
        "--warmup-iterations",
        str(args.warmup_iterations),
        "--output-dir",
        output_dir,
        "--run-id",
        args.run_id,
    ]
    if args.duration_seconds is not None:
        fdp_args.extend(["--duration-seconds", str(args.duration_seconds)])
    else:
        fdp_args.extend(["--iterations", str(iterations)])
    if args.dry_run:
        fdp_args.append("--dry-run")

    result = fdp_main(fdp_args)
    if not args.dry_run:
        summary_path = os.path.join(output_dir, "summary.json")
        summary = load_yaml_config(summary_path)
        summary["agentic_mode"] = args.mode
        summary["requested_iterations"] = args.iterations
        summary["target_host_write_multiplier"] = int(
            config["replay"].get("target_host_write_multiplier", 5)
        )
        annotate_target(summary, target_bytes=target_bytes)
        write_json(summary_path, summary)
        with open(os.path.join(output_dir, "summary.md"), "w") as file_obj:
            file_obj.write(build_summary_md(summary))
    return result


if __name__ == "__main__":
    sys.exit(main())
