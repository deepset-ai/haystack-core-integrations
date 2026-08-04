# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import concurrent.futures
import threading
from collections.abc import Coroutine
from typing import Any


class AsyncExecutor:
    """
    Thread-safe event loop executor for running async code from sync contexts.

    Mirage's :meth:`Workspace.execute` is a coroutine, but Haystack `Tool` functions are synchronous.
    This singleton owns a dedicated event loop running on a daemon thread, so blocking sync code (the
    tool's `function`) can drive mirage coroutines without clashing with any event loop the caller is
    already running.
    """

    _singleton_instance: "AsyncExecutor | None" = None
    _singleton_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "AsyncExecutor":
        """Get or create the global singleton executor instance."""
        with cls._singleton_lock:
            if cls._singleton_instance is None:
                cls._singleton_instance = cls()
            return cls._singleton_instance

    def __init__(self) -> None:
        """Initialize a dedicated event loop running on a daemon thread."""
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread = threading.Thread(target=self._run_loop, daemon=True)
        self._started = threading.Event()
        self._thread.start()
        if not self._started.wait(timeout=5):
            message = "AsyncExecutor failed to start background event loop"
            raise RuntimeError(message)

    def _run_loop(self) -> None:
        """Run the event loop forever on the background thread."""
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()

    def run(self, coro: Coroutine[Any, Any, Any], timeout: float | None = None) -> Any:
        """
        Run a coroutine on the background event loop and block until it completes.

        :param coro: Coroutine to execute.
        :param timeout: Optional timeout in seconds.
        :returns: The result of the coroutine.
        :raises TimeoutError: If execution exceeds the timeout.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            message = f"Operation timed out after {timeout} seconds"
            raise TimeoutError(message) from e

    def shutdown(self, timeout: float = 2) -> None:
        """
        Shut down the background event loop and thread.

        :param timeout: Timeout in seconds for shutting down the event loop.
        """

        def _stop_loop() -> None:
            self._loop.stop()

        asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop).result(timeout=timeout)
        self._loop.call_soon_threadsafe(_stop_loop)
        self._thread.join(timeout=timeout)
