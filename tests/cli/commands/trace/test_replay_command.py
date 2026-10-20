# SPDX-License-Identifier: Apache-2.0
"""``lmcache trace replay`` refuses a storage configuration with a shared L1."""

# Standard
from pathlib import Path
import argparse
import json

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace.replay_command import ReplayCommand, run_trace_replay


def test_replay_refuses_a_shared_l1(tmp_path: Path) -> None:
    spec = {
        "type": "DRAM",
        "tag": "pool",
        "size_gb": 1,
        "path": str(tmp_path / "region"),
        "shared": {"orchestrator": "127.0.0.1:7700", "region_id": "pool-a"},
    }
    parser = argparse.ArgumentParser()
    ReplayCommand().add_arguments(parser)
    args = parser.parse_args(
        [
            str(tmp_path / "missing.lct"),  # never opened: the config is refused first
            "--eviction-policy",
            "noop",
            "--l1-manager",
            json.dumps(spec),
        ]
    )
    with pytest.raises(ValueError, match="shared L1"):
        run_trace_replay(args)
