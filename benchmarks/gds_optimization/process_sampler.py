# SPDX-License-Identifier: Apache-2.0
"""Read CPU and I/O counters only for this controller's named child processes."""

# Standard
from pathlib import Path
from typing import cast
import json
import os
import time

IO_FIELDS = (
    "rchar",
    "wchar",
    "syscr",
    "syscw",
    "read_bytes",
    "write_bytes",
    "cancelled_write_bytes",
)


def process_identity(pid: int) -> dict[str, object]:
    """Record PID/start ticks/boot ID, or retain an early-exit probe limitation."""
    record: dict[str, object] = {
        "pid": pid,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }
    try:
        record["start_ticks"] = int(
            Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        )
    except FileNotFoundError:
        record["start_ticks"] = None
        record["start_ticks_reason"] = "process exited before /proc identity probe"
    return record


def parse_io(text: str) -> dict[str, int]:
    """Parse the seven requested process I/O counters; missing is not zero."""
    fields = dict(line.split(":", 1) for line in text.splitlines() if ":" in line)
    if any(name not in fields for name in IO_FIELDS):
        raise ValueError("Required /proc I/O counters are missing")
    result = {name: int(fields[name]) for name in IO_FIELDS}
    if any(value < 0 for value in result.values()):
        raise ValueError("Negative process I/O counter")
    return result


def sample_cpu(
    output: Path,
    driver_pid: int,
    identities: dict[str, dict[str, object]],
    proc_root: Path = Path("/proc"),
) -> dict[str, object]:
    """Read CPU/I/O counters for server/client PIDs named by the owned MP driver.

    PID identity is pinned on first observation after verifying its parent.
    Missing/changed processes are explicit null samples, never zero CPU use.
    """
    begin, wall_begin = time.monotonic_ns(), time.time_ns()
    discovery_errors = []
    changed = set()
    for path in [output / "commands.json", *sorted(output.glob("rank*/commands.json"))]:
        try:
            manifest = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as error:
            discovery_errors.append({"path": str(path), "reason": str(error)})
            continue
        ranks = manifest.get("ranks", [manifest])
        for rank in ranks:
            if "rank" not in rank:
                continue
            for role in ("server", "client"):
                pid = rank.get(role + "_pid")
                if pid is None:
                    continue
                key = f"rank{rank['rank']}-{role}"
                if key in identities and identities[key]["pid"] != pid:
                    changed.add(key)
                identities.setdefault(
                    key, {"rank": rank["rank"], "role": role, "pid": pid}
                )
    boot = (proc_root / "sys/kernel/random/boot_id").read_text().strip()
    processes = []
    for key, identity in sorted(identities.items()):
        read_begin = time.monotonic_ns()
        row = {
            "key": key,
            "rank": identity["rank"],
            "role": identity["role"],
            "pid": identity["pid"],
            "boot_id": boot,
            "utime_ticks": None,
            "stime_ticks": None,
            "start_ticks": None,
            "io_status": "unavailable",
            "io_counters": None,
            "io_reason": "CPU/PID identity unavailable",
        }
        try:
            if key in changed:
                raise ValueError("manifest PID changed")
            if identity.get("invalid_reason"):
                raise ValueError(str(identity["invalid_reason"]))
            fields = (
                (proc_root / str(identity["pid"]) / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()
            )
            parent, user, system, start = (int(fields[n]) for n in (1, 11, 12, 19))
            if parent != driver_pid:
                raise ValueError("PID is not a direct child of the owned MP driver")
            if user < 0 or system < 0:
                raise ValueError("negative CPU counter")
            if "start_ticks" in identity:
                if start != identity["start_ticks"] or boot != identity["boot_id"]:
                    raise ValueError("PID/start-ticks/boot identity changed")
                if user < cast(int, identity["utime_ticks"]) or system < cast(
                    int, identity["stime_ticks"]
                ):
                    raise ValueError("CPU counter decreased")
            identity.update(
                start_ticks=start, boot_id=boot, utime_ticks=user, stime_ticks=system
            )
            row.update(
                status="sampled",
                utime_ticks=user,
                stime_ticks=system,
                start_ticks=start,
            )
            row["io_read_begin_monotonic_ns"] = time.monotonic_ns()
            row["io_read_begin_wall_ns"] = time.time_ns()
            try:
                if identity.get("io_invalid_reason"):
                    raise ValueError(str(identity["io_invalid_reason"]))
                io = parse_io((proc_root / str(identity["pid"]) / "io").read_text())
                after = (
                    (proc_root / str(identity["pid"]) / "stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()
                )
                if int(after[19]) != start or int(after[1]) != driver_pid:
                    raise ValueError("PID identity changed during I/O counter read")
                previous = cast(dict[str, int] | None, identity.get("io_counters"))
                if previous is not None and any(
                    io[name] < previous[name] for name in IO_FIELDS
                ):
                    raise ValueError("Process I/O counter decreased")
                identity["io_counters"] = io
                row.update(io_status="sampled", io_counters=io, io_reason=None)
            except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
                row["io_reason"] = str(error)
            except (ValueError, IndexError) as error:
                identity["io_invalid_reason"] = str(error)
                row["io_reason"] = str(error)
            row["io_read_end_monotonic_ns"] = time.monotonic_ns()
            row["io_read_end_wall_ns"] = time.time_ns()
        except (FileNotFoundError, PermissionError, ProcessLookupError) as error:
            row.update(status="unavailable", reason=str(error))
        except (ValueError, IndexError) as error:
            identity["invalid_reason"] = str(error)
            row.update(status="invalid", reason=str(error))
        row.update(
            read_begin_monotonic_ns=read_begin,
            read_end_monotonic_ns=time.monotonic_ns(),
        )
        processes.append(row)
    return {
        "read_begin_monotonic_ns": begin,
        "read_end_monotonic_ns": time.monotonic_ns(),
        "read_begin_wall_ns": wall_begin,
        "read_end_wall_ns": time.time_ns(),
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
        "driver_pid": driver_pid,
        "processes": processes,
        "discovery_errors": discovery_errors,
    }
