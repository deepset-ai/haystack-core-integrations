# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Local, machine-readable trace capture for Agent Pack optimization campaigns."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import uuid4

from haystack import tracing
from haystack.components.agents import Agent
from haystack.tracing import Span, Tracer

TRACE_SCHEMA_VERSION = "haystack-trace/v1"


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _serialize_trace_value(value: Any) -> Any:
    """Convert a Haystack trace value into JSON-safe structured data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return _serialize_trace_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _serialize_trace_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_trace_value(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _serialize_trace_value(to_dict())
        except Exception:  # noqa: S110 - tracing must never break the application being observed
            pass
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _serialize_trace_value(getattr(value, item.name)) for item in fields(value)}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


@dataclass(frozen=True)
class TraceArtifact:
    """A locally captured run in the platform-compatible ``haystack-trace/v1`` shape."""

    run_id: str
    started_at: str
    finished_at: str
    duration_ms: float
    status: Literal["success", "failed"]
    traces: tuple[dict[str, Any], ...]
    failure: dict[str, Any] | None = None
    logs: tuple[dict[str, Any], ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON-compatible trace artifact envelope."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "traces": [_serialize_trace_value(span) for span in self.traces],
            "logs": [_serialize_trace_value(record) for record in self.logs],
            "failure": _serialize_trace_value(self.failure),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceArtifact:
        """Create and validate an artifact loaded from local storage."""
        if data.get("schema_version") != TRACE_SCHEMA_VERSION:
            msg = f"Unsupported trace schema: {data.get('schema_version')!r}."
            raise ValueError(msg)
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            duration_ms=float(data["duration_ms"]),
            status=data["status"],
            traces=tuple(data.get("traces") or ()),
            logs=tuple(data.get("logs") or ()),
            failure=data.get("failure"),
        )


@dataclass(frozen=True)
class TraceSelection:
    """Selection criteria shared by local and future remote trace sources."""

    run_ids: frozenset[str] | None = None
    status: Literal["success", "failed"] | None = "success"
    limit: int | None = None

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 1:
            msg = "TraceSelection.limit must be at least 1."
            raise ValueError(msg)


class TraceSource(Protocol):
    """Source of platform-shaped Haystack trace artifacts."""

    def list(self, selection: TraceSelection | None = None) -> list[TraceArtifact]:
        """Return artifacts matching the selection."""
        ...


class LocalTraceStore(TraceSource):
    """In-memory trace source with optional JSON persistence in a local directory."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self._artifacts: dict[str, TraceArtifact] = {}
        self._lock = RLock()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for path in self.directory.glob("*.json"):
                artifact = TraceArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
                self._artifacts[artifact.run_id] = artifact

    def add(self, artifact: TraceArtifact) -> None:
        """Store or replace an artifact by run ID."""
        with self._lock:
            self._artifacts[artifact.run_id] = artifact
            if self.directory is None:
                return
            target = self.directory / f"{artifact.run_id}.json"
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(artifact.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, target)

    def get(self, run_id: str) -> TraceArtifact:
        """Return one artifact by run ID."""
        with self._lock:
            try:
                return self._artifacts[run_id]
            except KeyError as error:
                msg = f"Unknown trace run ID: {run_id}."
                raise KeyError(msg) from error

    def list(self, selection: TraceSelection | None = None) -> list[TraceArtifact]:
        """Return matching artifacts, newest first."""
        selection = selection or TraceSelection()
        with self._lock:
            artifacts = sorted(self._artifacts.values(), key=lambda artifact: artifact.started_at, reverse=True)
        if selection.run_ids is not None:
            artifacts = [artifact for artifact in artifacts if artifact.run_id in selection.run_ids]
        if selection.status is not None:
            artifacts = [artifact for artifact in artifacts if artifact.status == selection.status]
        return artifacts[: selection.limit] if selection.limit is not None else artifacts


@dataclass
class _CapturedSpan(Span):
    operation_name: str
    tags: dict[str, Any]
    parent_span_id: str | None
    span_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: str = field(default_factory=_utc_now)
    _started_perf: float = field(default_factory=perf_counter)
    end_time: str | None = None
    duration_ms: float | None = None

    def set_tag(self, key: str, value: Any) -> None:
        """Set a structured, JSON-safe span tag."""
        self.tags[key] = _serialize_trace_value(value)

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Return identifiers suitable for correlating log records."""
        return {"span_id": self.span_id, "parent_span_id": self.parent_span_id}

    def finish(self) -> None:
        self.end_time = _utc_now()
        self.duration_ms = round((perf_counter() - self._started_perf) * 1000, 3)

    def to_record(self) -> dict[str, Any]:
        component = (
            self.tags.get("haystack.component.name")
            or self.tags.get("component.name")
            or self.tags.get("component")
            or self.tags.get("haystack.tool.name")
        )
        if component is None and self.operation_name.startswith("haystack.agent.step"):
            component = self.operation_name.rpartition(".")[2] or "agent"
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "component": component,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
        }


@dataclass
class _CombinedSpan(Span):
    captured: _CapturedSpan
    delegated: Span

    def set_tag(self, key: str, value: Any) -> None:
        """Forward a tag to both tracers."""
        self.captured.set_tag(key, value)
        self.delegated.set_tag(key, value)

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """Merge local and delegated correlation identifiers."""
        return {**self.delegated.get_correlation_data_for_logs(), **self.captured.get_correlation_data_for_logs()}


@dataclass
class CapturedRun:
    """Mutable run capture finalized into a ``TraceArtifact`` on context exit."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_utc_now)
    _started_perf: float = field(default_factory=perf_counter)
    finished_at: str | None = None
    duration_ms: float | None = None
    status: Literal["success", "failed"] = "success"
    spans: list[_CapturedSpan] = field(default_factory=list)
    failure: dict[str, Any] | None = None

    def finish(self) -> None:
        """Finalize run timing."""
        self.finished_at = _utc_now()
        self.duration_ms = round((perf_counter() - self._started_perf) * 1000, 3)

    def to_artifact(self) -> TraceArtifact:
        """Return the finalized platform-shaped artifact."""
        if self.finished_at is None or self.duration_ms is None:
            msg = "The run capture has not finished yet."
            raise RuntimeError(msg)
        return TraceArtifact(
            run_id=self.run_id,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_ms=self.duration_ms,
            status=self.status,
            traces=tuple(span.to_record() for span in self.spans),
            failure=self.failure,
        )


_current_run: ContextVar[CapturedRun | None] = ContextVar("agent_pack_trace_capture", default=None)
_active_spans: ContextVar[tuple[_CombinedSpan, ...]] = ContextVar("agent_pack_active_spans", default=())


class RunCaptureTracer(Tracer):
    """Tracer that captures spans for the current run and delegates every span to an existing tracer."""

    def __init__(self, delegate: Tracer) -> None:
        self.delegate = delegate

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Capture and delegate a span."""
        capture = _current_run.get()
        if capture is None:
            with self.delegate.trace(operation_name, tags=tags, parent_span=parent_span) as delegated:
                yield delegated
            return

        local_parent = parent_span.captured if isinstance(parent_span, _CombinedSpan) else None
        if local_parent is None and _active_spans.get():
            local_parent = _active_spans.get()[-1].captured
        delegated_parent = parent_span.delegated if isinstance(parent_span, _CombinedSpan) else parent_span
        captured = _CapturedSpan(
            operation_name=operation_name,
            tags=_serialize_trace_value(dict(tags or {})),
            parent_span_id=local_parent.span_id if local_parent is not None else None,
        )
        capture.spans.append(captured)
        with self.delegate.trace(operation_name, tags=tags, parent_span=delegated_parent) as delegated:
            combined = _CombinedSpan(captured=captured, delegated=delegated)
            previous = _active_spans.get()
            _active_spans.set((*previous, combined))
            try:
                yield combined
            except Exception as error:
                captured.set_tag("error", True)
                captured.set_tag("error.type", type(error).__name__)
                captured.set_tag("error.message", str(error))
                raise
            finally:
                captured.finish()
                _active_spans.set(previous)

    def current_span(self) -> Span | None:
        """Return the active combined span, falling back to the delegated tracer."""
        active = _active_spans.get()
        return active[-1] if active else self.delegate.current_span()


_installation_lock = RLock()


@dataclass
class _InstallationState:
    collector: LocalTraceCollector | None = None


_installation_state = _InstallationState()


class LocalTraceCollector:
    """Install a run-scoped tracer and collect completed runs into a local store."""

    def __init__(self, store: LocalTraceStore | None = None, *, content_tracing: bool = True) -> None:
        self.store = store or LocalTraceStore()
        self.content_tracing = content_tracing
        self._installation_depth = 0
        self._previous_tracer: Tracer | None = None
        self._previous_content_tracing = False
        self._capture_tracer: RunCaptureTracer | None = None

    @contextmanager
    def install(self) -> Iterator[LocalTraceCollector]:
        """Install this collector process-wide, delegating to the previously configured tracer."""
        with _installation_lock:
            if _installation_state.collector not in (None, self):
                msg = "Another LocalTraceCollector is already installed in this process."
                raise RuntimeError(msg)
            if self._installation_depth == 0:
                self._previous_tracer = tracing.tracer.actual_tracer
                self._previous_content_tracing = tracing.tracer.is_content_tracing_enabled
                self._capture_tracer = RunCaptureTracer(delegate=self._previous_tracer)
                tracing.enable_tracing(self._capture_tracer)
                tracing.tracer.is_content_tracing_enabled = self.content_tracing
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
                    tracing.tracer.is_content_tracing_enabled = self._previous_content_tracing
                    self._capture_tracer = None
                    self._previous_tracer = None
                    _installation_state.collector = None

    @contextmanager
    def capture_run(self) -> Iterator[CapturedRun]:
        """Capture one run, persist its artifact, and re-raise application failures."""
        capture = CapturedRun()
        previous = _current_run.get()
        with self.install():
            _current_run.set(capture)
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
                _current_run.set(previous)
                self.store.add(capture.to_artifact())


@dataclass(frozen=True)
class CapturedAgentRun:
    """Result and trace artifact produced by ``TraceCapturingAgentRunner``."""

    result: dict[str, Any]
    trace: TraceArtifact


class TraceCapturingAgentRunner:
    """Convenience runner that captures synchronous and asynchronous Agent executions."""

    def __init__(self, collector: LocalTraceCollector | None = None) -> None:
        self.collector = collector or LocalTraceCollector()

    def run(self, agent: Agent, **run_kwargs: Any) -> CapturedAgentRun:
        """Run an Agent synchronously and return its result plus trace artifact."""
        with self.collector.capture_run() as capture:
            result = agent.run(**run_kwargs)
        return CapturedAgentRun(result=result, trace=capture.to_artifact())

    async def run_async(self, agent: Agent, **run_kwargs: Any) -> CapturedAgentRun:
        """Run an Agent asynchronously and return its result plus trace artifact."""
        with self.collector.capture_run() as capture:
            result = await agent.run_async(**run_kwargs)
        return CapturedAgentRun(result=result, trace=capture.to_artifact())


def extract_agent_replay_inputs(artifact: TraceArtifact) -> dict[str, Any]:
    """Extract the root Agent inputs required to replay a successful reference trace."""
    for span in artifact.traces:
        if span.get("operation_name") == "haystack.agent.run" and span.get("parent_span_id") is None:
            inputs = span.get("tags", {}).get("haystack.agent.input")
            if isinstance(inputs, dict):
                return inputs
    msg = (
        f"Trace {artifact.run_id} has no captured haystack.agent.input. "
        "Capture the reference run with content tracing enabled."
    )
    raise ValueError(msg)


def extract_agent_reference_output(artifact: TraceArtifact) -> dict[str, Any]:
    """Extract the root Agent output from a successful reference trace."""
    for span in artifact.traces:
        if span.get("operation_name") == "haystack.agent.run" and span.get("parent_span_id") is None:
            output = span.get("tags", {}).get("haystack.agent.output")
            if isinstance(output, dict):
                return output
    msg = (
        f"Trace {artifact.run_id} has no captured haystack.agent.output. "
        "Capture the reference run with content tracing enabled."
    )
    raise ValueError(msg)
