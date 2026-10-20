# SPDX-License-Identifier: Apache-2.0
"""CPU ownership tests; real registration is covered by test_shared_l1_servers."""

# Third Party
import pytest

# First Party
from lmcache.v1.platform import current_device_spec


@pytest.fixture(autouse=True)
def cpu_mapping_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU mapping tests independent of GPU registration support."""
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: True)
    monkeypatch.setattr(current_device_spec, "unpin_memory", lambda *args: True)
