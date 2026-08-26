# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point running the public and admin listeners together.

Example::

    uv run python -m \\
      tests.e2e.mp_memory_coordinator.mock_memory_allocation_service \\
      --fixture tests/e2e/mp_memory_coordinator/fixtures/two_server_local_dax.yaml \\
      --public-host 127.0.0.1 --public-port 18080 \\
      --admin-host 127.0.0.1 --admin-port 19090
"""

# Standard
from pathlib import Path
import argparse
import asyncio
import logging
import sys

# Third Party
from fastapi import FastAPI
import uvicorn

# Local
from .app import build_apps, create_state
from .faults import BarrierRegistry, FaultRegistry


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        Namespace with ``fixture``, ``public_host``, ``public_port``,
        ``admin_host``, ``admin_port`` and ``state_file`` (``""`` if unset).
    """
    parser = argparse.ArgumentParser(
        prog="mock_memory_allocation_service",
        description="Strict development mock of the outside Memory Allocation API.",
    )
    parser.add_argument("--fixture", required=True, help="topology fixture YAML")
    parser.add_argument("--public-host", default="127.0.0.1")
    parser.add_argument("--public-port", type=int, default=8080)
    parser.add_argument("--admin-host", default="127.0.0.1")
    parser.add_argument("--admin-port", type=int, default=9090)
    parser.add_argument(
        "--state-file",
        default="",
        help="JSON file persisted after every mutation and loaded on restart",
    )
    return parser.parse_args(argv)


async def serve_both(
    public_app: FastAPI,
    admin_app: FastAPI,
    public_host: str,
    public_port: int,
    admin_host: str,
    admin_port: int,
) -> None:
    """Serve both applications with uvicorn on one event loop until shutdown.

    When either server stops (signal or error) the other is asked to exit so
    the process never lingers with a single listener.

    Args:
        public_app: Application bound to the public listener.
        admin_app: Application bound to the admin listener.
        public_host: Bind address of the public listener.
        public_port: Bind port of the public listener.
        admin_host: Bind address of the admin listener.
        admin_port: Bind port of the admin listener.
    """
    public_server = uvicorn.Server(
        uvicorn.Config(public_app, host=public_host, port=public_port, log_level="info")
    )
    admin_server = uvicorn.Server(
        uvicorn.Config(admin_app, host=admin_host, port=admin_port, log_level="info")
    )

    async def serve_then_stop_sibling(
        server: uvicorn.Server, sibling: uvicorn.Server
    ) -> None:
        try:
            await server.serve()
        finally:
            sibling.should_exit = True

    await asyncio.gather(
        serve_then_stop_sibling(public_server, admin_server),
        serve_then_stop_sibling(admin_server, public_server),
    )


def main(argv: list[str]) -> None:
    """Build the state and applications from ``argv`` and serve until shutdown.

    Args:
        argv: Command-line arguments without the program name.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    state_file = Path(args.state_file) if args.state_file else None
    state = create_state(Path(args.fixture), state_file)
    public_app, admin_app = build_apps(state, FaultRegistry(), BarrierRegistry())
    asyncio.run(
        serve_both(
            public_app,
            admin_app,
            args.public_host,
            args.public_port,
            args.admin_host,
            args.admin_port,
        )
    )


if __name__ == "__main__":
    main(sys.argv[1:])
