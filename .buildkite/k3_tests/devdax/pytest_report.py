# SPDX-License-Identifier: Apache-2.0
"""Record actual collection, independently of pytest's process exit status."""

# Standard
from pathlib import Path
import json
import os

# Third Party
import pytest


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write collected node IDs for the selected suite to its report path."""
    Path(os.environ["LMCACHE_DEVDAX_COLLECTION"]).write_text(
        json.dumps([item.nodeid for item in session.items]) + "\n"
    )
