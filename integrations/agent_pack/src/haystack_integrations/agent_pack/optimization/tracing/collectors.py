# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Installing trace capture around Agent runs."""

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Any

from haystack import tracing
from haystack.components.agents import Agent
from haystack.tracing import Tracer

from haystack_integrations.agent_pack.optimization.tracing.dataclasses import (
    DEFAULT_TRACE_CAPTURE_LIMITS,
    CapturedAgentRun,
    TraceCaptureLimits,
)
from haystack_integrations.agent_pack.optimization.tracing.stores import LocalTraceStore
from haystack_integrations.agent_pack.optimization.tracing.tracers import CapturedRun, RunCaptureTracer, current_run

_installation_lock = RLock()


@dataclass
class _InstallationState:
    """Tracks which collector currently owns the process-wide tracer."""

    collector: "LocalTraceCollector | None" = None


_installation_state = _InstallationState()


class LocalTraceCollector:
    """
    Install a run-scoped tracer and collect completed runs into a local store.

    """

    def __init__(
        self,
        store: LocalTraceStore | None = None,
        *,
        capture_content: bool = True,
        limits: TraceCaptureLimits = DEFAULT_TRACE_CAPTURE_LIMITS,
    ) -> None:
        """
        Create a collector.

        :param store: Where finished artifacts are written. A fresh in-memory store is used when omitted.
        :param capture_content: Whether prompts, documents, and tool payloads are recorded locally. Required to
            derive replay inputs and reference outputs. This does not change the process-wide content tracing
            setting, so an already-installed tracer keeps exporting exactly what it exported before.
        :param limits: Bounds applied to captured values.
        """
        self.store = store or LocalTraceStore()
        self.capture_content = capture_content
        self.limits = limits
        self._installation_depth = 0
        self._previous_tracer: Tracer | None = None
        self._capture_tracer: RunCaptureTracer | None = None

    @contextmanager
    def install(self) -> Iterator["LocalTraceCollector"]:
        """
        Install this collector process-wide, delegating to the previously configured tracer.

        :returns: A context manager yielding this collector, restoring the previous tracer on exit.
        :raises RuntimeError: If a different collector is already installed in this process.
        """
        with _installation_lock:
            if _installation_state.collector not in (None, self):
                msg = "Another LocalTraceCollector is already installed in this process."
                raise RuntimeError(msg)
            if self._installation_depth == 0:
                self._previous_tracer = tracing.tracer.actual_tracer
                self._capture_tracer = RunCaptureTracer(
                    delegate=self._previous_tracer, capture_content=self.capture_content, limits=self.limits
                )
                tracing.enable_tracing(self._capture_tracer)
                _installation_state.collector = self
            self._installation_depth += 1
        try:
            yield self
        finally:
            with _installation_lock:
                self._installation_depth -= 1
                if self._installation_depth == 0:
                    if tracing.tracer.actual_tracer is self._capture_tracer and self._previous_tracer is not None:
                        tracing.enable_tracing(self._previous_tracer)
                    self._capture_tracer = None
                    self._previous_tracer = None
                    _installation_state.collector = None

    @contextmanager
    def capture_run(self) -> Iterator[CapturedRun]:
        """
        Capture one run, persist its artifact, and re-raise application failures.

        :returns: A context manager yielding the run being captured. The artifact is added to the store on exit,
            whether the run succeeded or raised.
        """
        capture = CapturedRun()
        token = current_run.set(capture)
        try:
            with self.install():
                try:
                    yield capture
                except Exception as error:
                    capture.status = "failed"
                    capture.failure = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "stacktrace": traceback.format_exc().splitlines(),
                    }
                    raise
                finally:
                    capture.finish()
                    self.store.add(artifact=capture.to_artifact())
        finally:
            current_run.reset(token)


class TraceCapturingAgentRunner:
    """
    Convenience runner that captures synchronous and asynchronous Agent executions.
    """

    def __init__(self, collector: LocalTraceCollector | None = None) -> None:
        """
        Create a runner.

        :param collector: The collector to capture with. A fresh in-memory collector is used when omitted.
        """
        self.collector = collector or LocalTraceCollector()

    def run(self, agent: Agent, **run_kwargs: Any) -> CapturedAgentRun:
        """
        Run an Agent synchronously and return its result plus trace artifact.

        :param agent: The Agent to run.
        :param run_kwargs: Keyword arguments forwarded to `Agent.run`.
        :returns: The run result and the captured trace.
        """
        with self.collector.capture_run() as capture:
            result = agent.run(**run_kwargs)
        return CapturedAgentRun(result=result, trace=capture.to_artifact())

    async def run_async(self, agent: Agent, **run_kwargs: Any) -> CapturedAgentRun:
        """
        Run an Agent asynchronously and return its result plus trace artifact.

        :param agent: The Agent to run.
        :param run_kwargs: Keyword arguments forwarded to `Agent.run_async`.
        :returns: The run result and the captured trace.
        """
        with self.collector.capture_run() as capture:
            result = await agent.run_async(**run_kwargs)
        return CapturedAgentRun(result=result, trace=capture.to_artifact())
