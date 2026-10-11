# SPDX-License-Identifier: Apache-2.0
"""CPU-only resource, failure and shutdown contracts of the GDS worker pool."""

# Standard
from collections.abc import Callable
from concurrent.futures import Future, wait
from unittest.mock import patch
import gc
import threading
import time
import unittest
import weakref

# First Party
from lmcache.v1.gpu_connector import gds_workers
from lmcache.v1.gpu_connector.gds_workers import GDSWorkerPool


class GDSWorkerPoolTests(unittest.TestCase):
    def test_invalid_limits_and_oversized_request(self) -> None:
        """Invalid admission limits never enqueue work."""
        for workers, budget in [(0, 8), (-1, 8), (1, 0), (1, -1)]:
            with self.subTest(workers=workers, budget=budget):
                with self.assertRaises(ValueError):
                    GDSWorkerPool(workers, budget)
        pool = GDSWorkerPool(1, 8)
        self.addCleanup(pool.close)
        for size in [0, -1, 9]:
            with self.assertRaises(ValueError):
                pool.submit(lambda: None, size)
        self.assertEqual(pool.snapshot()["inflight_operations"], 0)

    def test_operation_and_byte_bounds_across_callers(self) -> None:
        """Independent callers must obey both shared admission bounds."""
        for workers, budget, size in [(1, 100, 6), (2, 10, 6), (2, 100, 6)]:
            with self.subTest(workers=workers, budget=budget):
                self._check_bounds(workers, budget, size)

    def test_callable_failure_propagates_and_worker_can_be_reused(self) -> None:
        """An operation failure releases its reservation without killing a worker."""
        pool = GDSWorkerPool(1, 8)
        self.addCleanup(pool.close)

        def fail() -> None:
            raise OSError("delayed I/O failure")

        failed = pool.submit(fail, 8)
        with self.assertRaisesRegex(OSError, "delayed I/O failure"):
            failed.result(timeout=3)
        pool.submit(lambda: None, 8).result(timeout=3)
        self.assertEqual(pool.snapshot()["inflight_bytes"], 0)

    def test_cancelled_queued_work_notifies_wait_and_releases_capacity(self) -> None:
        """Cancellation releases queued work and reaches Future.wait's done set."""
        gate = threading.Event()
        real_thread = threading.Thread

        def delayed_thread(
            *, target: Callable[[], None], name: str, daemon: bool
        ) -> threading.Thread:
            def delayed() -> None:
                gate.wait()
                target()

            return real_thread(target=delayed, name=name, daemon=daemon)

        # Hold the worker before it consumes queued work; no scheduling race.
        with patch.object(gds_workers.threading, "Thread", delayed_thread):
            pool = GDSWorkerPool(1, 8)
        ran = threading.Event()
        try:
            cancelled = pool.submit(ran.set, 8)
            self.assertTrue(cancelled.cancel())
            done, pending = wait([cancelled], timeout=0.1)
            self.assertEqual(done, {cancelled})
            self.assertFalse(pending)
            self.assertEqual(pool.snapshot()["inflight_operations"], 0)
            replacement = pool.submit(lambda: None, 8)
            gate.set()
            replacement.result(timeout=3)
            self.assertFalse(ran.is_set())
        finally:
            gate.set()
            pool.close()

    def test_close_wakes_blocked_submitter_and_drains_work(self) -> None:
        """Closing rejects blocked callers and waits for accepted operations."""
        pool = GDSWorkerPool(1, 8)
        entered = threading.Event()
        release = threading.Event()
        attempting = threading.Event()
        rejected = threading.Event()
        closed = [threading.Event(), threading.Event()]

        def hold() -> None:
            entered.set()
            release.wait()

        def submit_blocked() -> None:
            attempting.set()
            try:
                pool.submit(lambda: None, 8)
            except RuntimeError:
                rejected.set()

        def close(index: int) -> None:
            pool.close()
            closed[index].set()

        future = pool.submit(hold, 8)
        submitter = threading.Thread(target=submit_blocked)
        closers = [threading.Thread(target=close, args=(i,)) for i in range(2)]
        try:
            self.assertTrue(entered.wait(3))
            submitter.start()
            self.assertTrue(attempting.wait(3))
            for thread in closers:
                thread.start()
            self.assertTrue(rejected.wait(3))
            self.assertFalse(any(event.is_set() for event in closed))
            self.assertTrue(pool.snapshot()["closing"])
            self.assertFalse(pool.snapshot()["closed"])
            release.set()
            future.result(timeout=3)
            for event in closed:
                self.assertTrue(event.wait(3))
        finally:
            release.set()
            for thread in [submitter, *closers]:
                if thread.ident is not None:
                    thread.join(timeout=3)
            pool.close()
        self.assertTrue(pool.snapshot()["closed"])
        with self.assertRaises(RuntimeError):
            pool.submit(lambda: None, 8)

    def test_completed_future_does_not_retain_operation_resources(self) -> None:
        """Keeping a completed Future must not pin the callable's buffers."""

        class Payload:
            touched = False

        pool = GDSWorkerPool(1, 8)
        self.addCleanup(pool.close)
        payload = Payload()
        reference = weakref.ref(payload)

        def operation(value: Payload = payload) -> None:
            value.touched = True

        future = pool.submit(operation, 8)
        del operation, payload
        future.result(timeout=3)
        deadline = time.monotonic() + 3
        while reference() is not None and time.monotonic() < deadline:
            gc.collect()
            time.sleep(0.001)
        self.assertIsNone(reference())

    def _check_bounds(self, workers: int, budget: int, size: int) -> None:
        """Exercise one shared admission bound with an already running operation."""
        pool = GDSWorkerPool(workers, budget)
        release = threading.Event()
        entered = threading.Event()
        attempted = threading.Event()
        admitted = threading.Event()
        second: list[Future[None]] = []

        def hold() -> None:
            entered.set()
            release.wait()

        def submit_second() -> None:
            attempted.set()
            second.append(pool.submit(lambda: None, size))
            admitted.set()

        first = pool.submit(hold, size)
        thread = threading.Thread(target=submit_second)
        try:
            self.assertTrue(entered.wait(3))
            thread.start()
            self.assertTrue(attempted.wait(3))
            if workers == 1 or size * 2 > budget:
                self.assertFalse(admitted.wait(0.1))
            else:
                self.assertTrue(admitted.wait(3))
                second[0].result(timeout=3)
                self.assertFalse(first.done())
            snapshot = pool.snapshot()
            self.assertEqual(snapshot["inflight_operations"], 1)
            self.assertEqual(snapshot["inflight_bytes"], size)
            release.set()
            first.result(timeout=3)
            self.assertTrue(admitted.wait(3))
            second[0].result(timeout=3)
        finally:
            release.set()
            if thread.ident is not None:
                thread.join(timeout=3)
            pool.close()
        snapshot = pool.snapshot()
        self.assertLessEqual(snapshot["peak_inflight_operations"], workers)
        self.assertLessEqual(snapshot["peak_inflight_bytes"], budget)
        self.assertEqual(snapshot["inflight_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
