# SPDX-License-Identifier: Apache-2.0
"""Bounded ordinary worker threads for explicitly completed GDS operations."""

# Standard
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass(eq=False)
class _Operation:
    operation: Callable[[], None] | None
    nbytes: int
    future: Future[None]
    released: bool = False


class GDSWorkerPool:
    """Bound pending/running operations and bytes across all submitting callers.

    An operation must finish using its resources before returning. In particular,
    CUDA operations must perform their owning stream's completion wait inside
    the callable. The pool has no CUDA dependency and does not infer DMA readiness.
    Callers must close the pool before freeing shared buffers or file handles.
    """

    def __init__(self, workers: int, max_inflight_bytes: int) -> None:
        """Start a fixed pool with at most ``workers`` accepted operations.

        Raises:
            ValueError: Either resource limit is nonpositive.
            RuntimeError: A worker thread cannot be started.
        """
        if workers <= 0 or max_inflight_bytes <= 0:
            raise ValueError("GDS worker and byte limits must be positive")
        self._workers = workers
        self._max_inflight_bytes = max_inflight_bytes
        self._condition = threading.Condition()
        self._close_lock = threading.Lock()
        self._queue: deque[_Operation] = deque()
        self._threads: list[threading.Thread] = []
        self._inflight_operations = 0
        self._inflight_bytes = 0
        self._peak_operations = 0
        self._peak_bytes = 0
        self._closing = False
        self._closed = False
        try:
            for index in range(workers):
                thread = threading.Thread(
                    target=self._run,
                    name=f"lmcache-gds-{index}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
        except BaseException:
            logger.exception("Could not start GDS worker threads")
            self.close()
            raise

    def submit(self, operation: Callable[[], None], nbytes: int) -> Future[None]:
        """Admit work after bounded operation/byte capacity becomes available.

        Waiting submitters consume no queue entry or byte reservation. Closing
        wakes them and rejects new work. Do not submit work that waits for work
        submitted recursively to this same bounded pool.

        Args:
            operation: Callable that completes all of its resource use.
            nbytes: Positive bytes held in flight by this operation.

        Returns:
            A future carrying completion, cancellation or the callable's error.

        Raises:
            ValueError: The size is nonpositive or exceeds the byte budget.
            RuntimeError: Closing has started.
        """
        if nbytes <= 0 or nbytes > self._max_inflight_bytes:
            raise ValueError("GDS operation size is outside the in-flight byte budget")
        with self._condition:
            while not self._closing and (
                self._inflight_operations >= self._workers
                or self._inflight_bytes + nbytes > self._max_inflight_bytes
            ):
                self._condition.wait()
            if self._closing:
                raise RuntimeError("GDS worker pool is closing")
            future: Future[None] = Future()
            item = _Operation(operation, nbytes, future)
            self._inflight_operations += 1
            self._inflight_bytes += nbytes
            try:
                future.add_done_callback(partial(self._release, item))
                self._queue.append(item)
            except BaseException:
                logger.exception("Could not queue a GDS worker operation")
                future.cancel()
                self._release(item, future)
                raise
            self._peak_operations = max(
                self._peak_operations, self._inflight_operations
            )
            self._peak_bytes = max(self._peak_bytes, self._inflight_bytes)
            self._condition.notify_all()
            return future

    def close(self) -> None:
        """Reject new work, wake blocked submitters, and drain accepted work.

        Repeated and concurrent closes are safe. Running work is not cancelled.

        Raises:
            RuntimeError: A worker attempts to join its own pool.
        """
        if threading.current_thread() in self._threads:
            raise RuntimeError("A GDS worker cannot close its own pool")
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        with self._close_lock:
            for thread in self._threads:
                thread.join()
            with self._condition:
                self._closed = True
                self._condition.notify_all()

    def snapshot(self) -> dict[str, int | bool]:
        """Return one consistent snapshot of actual use, peaks and lifecycle."""
        with self._condition:
            return {
                "workers": self._workers,
                "max_inflight_operations": self._workers,
                "max_inflight_bytes": self._max_inflight_bytes,
                "inflight_operations": self._inflight_operations,
                "inflight_bytes": self._inflight_bytes,
                "peak_inflight_operations": self._peak_operations,
                "peak_inflight_bytes": self._peak_bytes,
                "closing": self._closing,
                "closed": self._closed,
            }

    def _release(self, item: _Operation, future: Future[None]) -> None:
        """Release exactly once, also removing cancelled work still in the queue."""
        with self._condition:
            if item.released:
                return
            item.released = True
            if future.cancelled() and item in self._queue:
                self._queue.remove(item)
                # Removed work has no worker left to notify Future.wait waiters.
                future.set_running_or_notify_cancel()
            item.operation = None
            self._inflight_operations -= 1
            self._inflight_bytes -= item.nbytes
            self._condition.notify_all()

    def _run(self) -> None:
        """Consume admitted work until closing has drained the queue."""
        while True:
            with self._condition:
                while not self._queue:
                    if self._closing:
                        return
                    self._condition.wait()
                item = self._queue.popleft()
            if not item.future.set_running_or_notify_cancel():
                del item
                continue
            operation = item.operation
            assert operation is not None
            try:
                operation()
            except BaseException as error:
                logger.exception("GDS worker operation failed")
                self._release(item, item.future)
                item.future.set_exception(error)
            else:
                self._release(item, item.future)
                item.future.set_result(None)
            finally:
                del operation, item
