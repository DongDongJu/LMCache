# SPDX-License-Identifier: Apache-2.0
"""Install the companion L2 plugins and run their tests against this LMCache.

Set LMCACHE_L2_PLUGINS_PATH to test a local checkout. Otherwise the pinned
revision below is checked out in a temporary directory. LMCache and its test
dependencies must already be installed in the current Python environment.
"""

# Standard
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile

# shortcut: use the migration fork until the filesystem adapter lands upstream.
PLUGIN_REPOSITORY = "https://github.com/DongDongJu/lmcache_l2_plugins.git"
PLUGIN_REVISION = "ff6030afeef0f5a39d0e1302b908a201fb0fc82b"


def main() -> None:
    """Install a local or pinned plugin checkout, then run its test suite.

    Installation uses ``--no-deps`` to preserve the LMCache and device runtime
    under test. Package installation, checkout, and test failures propagate.
    """
    with tempfile.TemporaryDirectory(prefix="lmcache-l2-plugins-") as temporary:
        local = os.environ.get("LMCACHE_L2_PLUGINS_PATH")
        if local:
            source = Path(local).resolve()
            print(f"Testing L2 plugin checkout: {source}", flush=True)
        else:
            source = Path(temporary) / "plugins"
            print(
                f"Testing L2 plugins: {PLUGIN_REPOSITORY}@{PLUGIN_REVISION}",
                flush=True,
            )
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--quiet",
                    "--no-checkout",
                    PLUGIN_REPOSITORY,
                    str(source),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "checkout", "--quiet", "--detach", PLUGIN_REVISION],
                cwd=source,
                check=True,
            )

        uv = shutil.which("uv")
        installer = (
            [uv, "pip", "install", "--python", sys.executable, "--reinstall"]
            if uv
            else [sys.executable, "-m", "pip", "install", "--force-reinstall"]
        )
        subprocess.run([*installer, "--no-deps", str(source)], check=True)
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests"],
            cwd=source,
            env={**os.environ, "LMCACHE_TRACK_USAGE": "false"},
            check=True,
        )


if __name__ == "__main__":
    main()
