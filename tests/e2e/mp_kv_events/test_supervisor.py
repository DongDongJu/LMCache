# SPDX-License-Identifier: Apache-2.0
"""Guard against cross-arm evidence reuse and unverified launcher replacement."""

# Standard
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import hashlib
import json
import signal
import unittest

# Third Party
from worker import Supervisor


class ReloadTests(unittest.TestCase):
    """Event-mode reload must preserve the old arm and begin empty raw evidence."""

    def test_checksum_mismatch_preserves_running_arm(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"runtime_files_sha256": {"worker.py": "source"}}
            path = root / "launcher.json"
            path.write_text(json.dumps(config))
            supervisor = Supervisor(config, root / "logs")
            with patch.object(supervisor, "close") as close:
                with self.assertRaises(ValueError):
                    supervisor.reload(path, "not-the-projected-checksum")
                close.assert_not_called()

    def test_raw_evidence_is_archived_before_fresh_processes(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"runtime_files_sha256": {"worker.py": "source"}, "mode": "on"}
            updated = {**config, "mode": "off"}
            path = root / "launcher.json"
            path.write_text(json.dumps(updated))
            supervisor = Supervisor(config, root / "logs")
            (supervisor.logs / "native-events.jsonl").write_text("old event\n")
            (supervisor.logs / "provenance.json").write_text(
                json.dumps(
                    {"source": "same image", "gpu": "same UUID", "launcher": config}
                )
            )
            starts = []

            def start(name: str) -> None:
                self.assertFalse((supervisor.logs / "native-events.jsonl").exists())
                starts.append(name)

            with (
                patch.object(supervisor, "close"),
                patch.object(supervisor, "start", start),
            ):
                supervisor.reload(path, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(starts, ["mp", "observer", "engine"])
            provenance = json.loads((supervisor.logs / "provenance.json").read_text())
            self.assertEqual(provenance["gpu"], "same UUID")
            self.assertEqual(provenance["launcher"], updated)
            archive = Path(provenance["event_mode_reload"]["previous_artifacts"])
            self.assertEqual(
                (archive / "native-events.jsonl").read_text(), "old event\n"
            )


class HeartbeatRestartTests(unittest.TestCase):
    """MP restart must observe new health loss and always restore the server."""

    def test_only_new_health_transitions_satisfy_outage_gate(self) -> None:
        with TemporaryDirectory() as temp:
            supervisor = Supervisor({}, Path(temp))
            engine_log = supervisor.logs / "engine.log"
            message = "LMCache server is unhealthy — entering degraded mode\n"
            engine_log.write_text(message * 2)

            def stop(name: str) -> None:
                with engine_log.open("a") as stream:
                    stream.write(message * 2)

            with (
                patch.object(supervisor, "stop", stop),
                patch.object(supervisor, "start") as start,
            ):
                supervisor.restart("mp", wait_for_unhealthy_heartbeats=2)
                start.assert_called_once_with("mp")
            record = json.loads((supervisor.logs / "lifecycle.jsonl").read_text())
            self.assertEqual(record["count"], 2)
            self.assertGreater(record["engine_log_offset"], 0)

    def test_failed_outage_gate_still_starts_mp(self) -> None:
        with TemporaryDirectory() as temp:
            supervisor = Supervisor({}, Path(temp))
            (supervisor.logs / "engine.log").write_text("")
            with (
                patch.object(supervisor, "stop"),
                patch.object(supervisor, "start") as start,
                patch("worker.time.monotonic", side_effect=[0, 61]),
            ):
                with self.assertRaises(TimeoutError):
                    supervisor.restart("mp", wait_for_unhealthy_heartbeats=2)
                start.assert_called_once_with("mp")


class PauseTests(unittest.TestCase):
    """Process pause must leave discovery alive and avoid recycled process IDs."""

    def test_only_owned_inference_children_receive_signals(self) -> None:
        with TemporaryDirectory() as temp:
            supervisor = Supervisor({}, Path(temp))
            engine = Mock(pid=100)
            engine.poll.return_value = None
            supervisor.processes["engine"] = engine
            parent = Mock()
            child = Mock(pid=200)
            child.create_time.return_value = 123.0
            child.name.return_value = "VLLM::EngineCore"
            child.cmdline.return_value = ["VLLM::EngineCore"]
            parent.children.return_value = [child]
            with patch(
                "worker.psutil.Process",
                side_effect=lambda pid: parent if pid == 100 else child,
            ):
                supervisor.pause(True)
                supervisor.pause(False)
            parent.send_signal.assert_not_called()
            self.assertEqual(
                [call.args[0] for call in child.send_signal.call_args_list],
                [signal.SIGSTOP, signal.SIGCONT],
            )
            rows = [
                json.loads(line)
                for line in (supervisor.logs / "lifecycle.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(rows[0]["children"][0]["created"], 123.0)
            self.assertEqual(rows[0]["discovery_parent_pid"], 100)

    def test_recycled_child_pid_is_not_signaled(self) -> None:
        with TemporaryDirectory() as temp:
            supervisor = Supervisor({}, Path(temp))
            supervisor.paused_processes = [{"pid": 200, "created": 123.0}]
            child = Mock()
            child.create_time.return_value = 456.0
            with patch("worker.psutil.Process", return_value=child):
                with self.assertRaises(RuntimeError):
                    supervisor._resume_descendants(strict=True)
            child.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
