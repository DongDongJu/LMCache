# SPDX-License-Identifier: Apache-2.0
"""Run the scenario server: six uvicorn listeners in one asyncio loop.

Example (from the repository root)::

    python -m tests.e2e.mp_memory_coordinator.scenario_server \\
        --fixture tests/e2e/mp_memory_coordinator/fixtures/two_server_local_dax.yaml

Listeners: coordinator, donor MP, donor MP alternate, receiver MP, receiver
MP alternate, admin. Advertised values (what ``/instances`` reports) default
to the bind values and can be overridden by flags or environment variables
for Kubernetes, where the advertised ip is the pod IP.

This module deliberately uses the standard ``logging`` module rather than
``lmcache.logging``: the Docker image ships without the ``lmcache`` package.
"""

# Standard
from pathlib import Path
import argparse
import asyncio
import logging
import os
import sys

# Third Party
from fastapi import FastAPI
import uvicorn

# Local
from .app import build_apps
from .state import DONOR_ID, RECEIVER_ID, InstanceEndpoint, ScenarioState

ALT_PORT_OFFSET = 100
logger = logging.getLogger("scenario_server")


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``.

    Returns:
        The parsed value or ``default`` when unset or empty.
    """
    raw = os.environ.get(name, "")
    return int(raw) if raw else default


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments without the program name.

    Returns:
        Namespace with ``fixture``, ``host``, ``advertise_ip``, the six bind
        ports and the four advertised MP ports (0 = same as bind port).
    """
    parser = argparse.ArgumentParser(
        prog="scenario_server",
        description="Fake MP Coordinator + fake donor/receiver MP servers.",
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--advertise-ip",
        default=os.environ.get("SCENARIO_ADVERTISE_IP", "127.0.0.1"),
        help="ip reported by /instances (env SCENARIO_ADVERTISE_IP)",
    )
    parser.add_argument("--coordinator-port", type=int, default=9300)
    parser.add_argument("--donor-port", type=int, default=8081)
    parser.add_argument("--receiver-port", type=int, default=8082)
    parser.add_argument("--admin-port", type=int, default=9091)
    parser.add_argument(
        "--donor-alt-port", type=int, default=0, help="default donor-port+100"
    )
    parser.add_argument(
        "--receiver-alt-port", type=int, default=0, help="default receiver-port+100"
    )
    parser.add_argument(
        "--advertise-donor-port",
        type=int,
        default=_env_int("SCENARIO_DONOR_HTTP_PORT", 0),
        help="http_port reported for mp-donor (env SCENARIO_DONOR_HTTP_PORT)",
    )
    parser.add_argument(
        "--advertise-receiver-port",
        type=int,
        default=_env_int("SCENARIO_RECEIVER_HTTP_PORT", 0),
        help="http_port reported for mp-receiver (env SCENARIO_RECEIVER_HTTP_PORT)",
    )
    parser.add_argument(
        "--advertise-donor-alt-port",
        type=int,
        default=_env_int("SCENARIO_DONOR_ALT_HTTP_PORT", 0),
        help="alt http_port for mp-donor (env SCENARIO_DONOR_ALT_HTTP_PORT)",
    )
    parser.add_argument(
        "--advertise-receiver-alt-port",
        type=int,
        default=_env_int("SCENARIO_RECEIVER_ALT_HTTP_PORT", 0),
        help="alt http_port for mp-receiver (env SCENARIO_RECEIVER_ALT_HTTP_PORT)",
    )
    return parser.parse_args(argv)


def build_endpoints(args: argparse.Namespace) -> dict[str, InstanceEndpoint]:
    """Derive the advertised endpoints from parsed arguments.

    Args:
        args: Result of :func:`parse_args`.

    Returns:
        Advertised endpoint per instance id.
    """
    donor_alt = args.donor_alt_port or args.donor_port + ALT_PORT_OFFSET
    receiver_alt = args.receiver_alt_port or args.receiver_port + ALT_PORT_OFFSET
    return {
        DONOR_ID: InstanceEndpoint(
            ip=args.advertise_ip,
            http_port=args.advertise_donor_port or args.donor_port,
            alt_http_port=args.advertise_donor_alt_port or donor_alt,
        ),
        RECEIVER_ID: InstanceEndpoint(
            ip=args.advertise_ip,
            http_port=args.advertise_receiver_port or args.receiver_port,
            alt_http_port=args.advertise_receiver_alt_port or receiver_alt,
        ),
    }


def _server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    """Create a quiet uvicorn server for one listener.

    Returns:
        The unstarted server.
    """
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", access_log=False
    )
    return uvicorn.Server(config)


async def serve_all(servers: list[uvicorn.Server]) -> None:
    """Serve every listener until the first one stops, then stop the rest.

    A signal or a bind failure on any listener therefore brings the whole
    process down instead of leaving a half-alive server.

    Args:
        servers: Configured servers.

    Raises:
        Exception: Whatever made the first listener stop, after the others
            have been shut down.
    """
    tasks = [asyncio.create_task(server.serve()) for server in servers]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for server in servers:
        server.should_exit = True
    await asyncio.gather(*pending, return_exceptions=True)
    for task in done:
        task.result()


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Arguments without the program name.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    args = parse_args(argv)
    state = ScenarioState(args.fixture, build_endpoints(args))
    apps = build_apps(state)
    donor_alt = args.donor_alt_port or args.donor_port + ALT_PORT_OFFSET
    receiver_alt = args.receiver_alt_port or args.receiver_port + ALT_PORT_OFFSET
    servers = [
        _server(apps.coordinator, args.host, args.coordinator_port),
        _server(apps.donor, args.host, args.donor_port),
        _server(apps.donor, args.host, donor_alt),
        _server(apps.receiver, args.host, args.receiver_port),
        _server(apps.receiver, args.host, receiver_alt),
        _server(apps.admin, args.host, args.admin_port),
    ]
    logger.info(
        "listening on %s: coordinator=%d donor=%d/%d receiver=%d/%d admin=%d",
        args.host,
        args.coordinator_port,
        args.donor_port,
        donor_alt,
        args.receiver_port,
        receiver_alt,
        args.admin_port,
    )
    asyncio.run(serve_all(servers))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
