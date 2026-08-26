# SPDX-License-Identifier: Apache-2.0
"""AST-based architecture boundary tests.

The MP Memory Coordinator must stay a separate process that talks to the MP
Coordinator over HTTP only. These tests walk every module of the package
and reject any import of ``lmcache.v1.mp_coordinator`` (its context,
managers, registries, EventGate, persistence, or schemas). They also prove
that production code never imports the E2E test packages.
"""

# Standard
from pathlib import Path
import ast

# First Party
import lmcache

PACKAGE_ROOT = Path(lmcache.__file__).resolve().parent
MEMORY_COORDINATOR_ROOT = PACKAGE_ROOT / "v1" / "mp_memory_coordinator"
FORBIDDEN_PREFIXES = ("lmcache.v1.mp_coordinator",)
TEST_ONLY_PREFIXES = ("tests.", "tests.e2e", "mock_memory_allocation_service")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # Relative import: resolve against the package.
                package = ".".join(path.relative_to(PACKAGE_ROOT.parent).parts[:-1])
                base = (
                    package.rsplit(".", node.level - 1)[0]
                    if node.level > 1
                    else package
                )
                module = f"{base}.{module}" if module else base
            names.append(module)
    return names


def _package_modules() -> list[Path]:
    modules = sorted(MEMORY_COORDINATOR_ROOT.rglob("*.py"))
    assert modules, "package not found"
    return modules


def test_package_never_imports_mp_coordinator() -> None:
    offenders: list[str] = []
    for module in _package_modules():
        for name in _imports(module):
            if name.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{module.relative_to(PACKAGE_ROOT.parent)}: {name}")
    assert offenders == [], "\n".join(offenders)


def test_package_never_imports_test_packages() -> None:
    offenders: list[str] = []
    for module in _package_modules():
        for name in _imports(module):
            if name.startswith(TEST_ONLY_PREFIXES):
                offenders.append(f"{module.relative_to(PACKAGE_ROOT.parent)}: {name}")
    assert offenders == [], "\n".join(offenders)


def test_cli_command_module_never_imports_mp_coordinator() -> None:
    command = PACKAGE_ROOT / "cli" / "commands" / "mp_memory_coordinator.py"
    if not command.exists():
        return
    assert not [n for n in _imports(command) if n.startswith(FORBIDDEN_PREFIXES)]


def test_no_production_module_imports_the_e2e_mock_packages() -> None:
    offenders: list[str] = []
    for module in sorted(PACKAGE_ROOT.rglob("*.py")):
        for name in _imports(module):
            if name.startswith(TEST_ONLY_PREFIXES):
                offenders.append(f"{module.relative_to(PACKAGE_ROOT.parent)}: {name}")
    assert offenders == [], "\n".join(offenders)


def test_boundary_detector_catches_a_violation(tmp_path: Path) -> None:
    """The detector itself must see plain, aliased, and from-imports."""
    sample = tmp_path / "bad.py"
    sample.write_text(
        "import lmcache.v1.mp_coordinator.registry as r\n"
        "from lmcache.v1.mp_coordinator.schemas import RegisterRequest\n"
        "import os\n"
    )
    found = [n for n in _imports(sample) if n.startswith(FORBIDDEN_PREFIXES)]
    assert len(found) == 2
