# SPDX-License-Identifier: Apache-2.0
"""``lmcache mp-memory-coordinator`` — run the standalone MP Memory Coordinator.

Usage::

    lmcache mp-memory-coordinator --config /etc/lmcache/mp-memory-coordinator.yaml
    lmcache mp-memory-coordinator --config cfg.yaml --adopt allocations.yaml

The process is configured from one YAML file (see
:class:`lmcache.v1.mp_memory_coordinator.config.MPMemoryCoordinatorConfig`).
``--adopt`` runs the explicit one-time adoption of the listed allocations
into the journal and exits without starting the control loop.
"""

# Standard
from pathlib import Path
import argparse
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand


class MPMemoryCoordinatorCommand(BaseCommand):
    """CLI command that runs the MP Memory Coordinator process."""

    def name(self) -> str:
        """Return the subcommand name.

        Returns:
            ``"mp-memory-coordinator"``.
        """
        return "mp-memory-coordinator"

    def help(self) -> str:
        """Return short help text.

        Returns:
            Help string shown by ``lmcache -h``.
        """
        return "Run the standalone MP Memory Coordinator (DAX capacity rebalancing)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the command's arguments.

        Args:
            parser: The ``ArgumentParser`` for this subcommand.
        """
        parser.add_argument(
            "--config",
            required=True,
            type=Path,
            help="YAML configuration file (see the design doc for keys).",
        )
        parser.add_argument(
            "--adopt",
            type=Path,
            default=None,
            help=(
                "Operator-approved allowlist of existing runtime allocations to "
                "adopt into the journal once, then exit. Optional: the running "
                "coordinator discovers allocator-assigned devices on its own."
            ),
        )
        parser.add_argument(
            "--check",
            action="store_true",
            help="Validate the configuration and exit without starting.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Load the configuration and run (or adopt, or check).

        Args:
            args: Parsed CLI arguments.

        Raises:
            SystemExit: With status 2 on a configuration error, 1 when the
                full installation is missing.
        """
        try:
            # First Party
            from lmcache.v1.mp_memory_coordinator.config import load_config
        except ImportError:
            print(
                "The 'lmcache mp-memory-coordinator' command requires the full "
                "lmcache installation.\nInstall with: pip install lmcache",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            config = load_config(args.config)
        except (OSError, ValueError) as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            sys.exit(2)
        if args.check:
            print(f"configuration OK: {args.config}")
            return
        if args.adopt is not None:
            # First Party
            from lmcache.v1.mp_memory_coordinator.app import run_adoption

            run_adoption(config, args.adopt)
            return
        # First Party
        from lmcache.v1.mp_memory_coordinator.app import run_memory_coordinator

        run_memory_coordinator(config)
