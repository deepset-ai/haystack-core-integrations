# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""The capturing tracer, its spans, and the run they accumulate into."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from haystack.tracing import Span, Tracer

from haystack_integrations.agent_pack.optimization.tracing.dataclasses import (
    DEFAULT_TRACE_CAPTURE_LIMITS,
    TraceArtifact,
    TraceCaptureLimits,
)
from haystack_integrations.agent_pack.optimization.tracing.serialization import _serialize_trace_value


def _utc_now() -> str:
    """
    Return the current time as an ISO 8601 string in UTC.

    :returns: The current UTC timestamp.
    """
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class _CapturedSpan(Span):
    """A span recorded into the local artifact."""

    operation_name: str
    tags: dict[str, Any]
    parent_span_id: str | None
    capture_content: bool = True
    limits: TraceCaptureLimits = DEFAULT_TRACE_CAPTURE_LIMITS
    span_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: str = field(default_factory=_utc_now)
    _started_perf: float = field(default_factory=perf_counter)
    end_time: str | None = None
    duration_ms: float | None = None

    def set_tag(self, key: str, value: Any) -> None:
        """
        Set a bounded, JSON-safe span tag.

        :param key: The tag name.
        :param value: The tag value.
        """
        self.tags[key] = _serialize_trace_value(value=value, limits=self.limits)

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Capture a content tag locally, independently of the process-wide content tracing setting.

        Haystack's default implementation is gated on the global `tracer.is_content_tracing_enabled`. Capture needs
        content to derive replay inputs and reference outputs, but flipping that global would also start routing
        prompts and documents to whatever tracer was already installed. Overriding here keeps the decision local.

        :param key: The tag name.
        :param value: The content to record.
        """
        if self.capture_content:
            self.set_tag(key=key, value=value)

    def finish(self) -> None:
        """Record the span's end time and duration."""
        self.end_time = _utc_now()
        self.duration_ms = round((perf_counter() - self._started_perf) * 1000, 3)

    def _plain_tag(self, key: str) -> Any:
        """Read a captured tag's data without deserializing it, for the plain string tags used to name a span."""
        payload = self.tags.get(key)
        return payload.get("serialized_data") if isinstance(payload, dict) else payload

    def to_record(self) -> dict[str, Any]:
        """
        Convert the span into the record shape stored in a trace artifact.

        :returns: A dictionary describing the span, its place in the hierarchy, its timing, and its tags.
        """
        component = (
            self._plain_tag(key="haystack.component.name")
            or self._plain_tag(key="component.name")
            or self._plain_tag(key="component")
            or self._plain_tag(key="haystack.tool.name")
        )
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
    """A span that records locally and forwards to the tracer that was already installed."""

    captured: _CapturedSpan
    delegated: Span

    def set_tag(self, key: str, value: Any) -> None:
        """
        Forward a tag to both tracers.

        :param key: The tag name.
        :param value: The tag value.
        """
        self.captured.set_tag(key=key, value=value)
        self.delegated.set_tag(key, value)

    def set_content_tag(self, key: str, value: Any) -> None:
        """
        Capture content locally, and forward it to the delegate only if the delegate would have taken it anyway.

        `Span.set_content_tag` on the delegate self-gates on the global content tracing setting, so a user running
        Langfuse or OpenTelemetry does not start exporting prompts and documents just because capture is installed.

        :param key: The tag name.
        :param value: The content to record.
        """
        self.captured.set_content_tag(key=key, value=value)
        self.delegated.set_content_tag(key, value)

    def raw_span(self) -> Any:
        """
        Return the delegate's underlying span so tracer-specific integrations keep working during capture.

        :returns: The delegated tracer's raw span object.
        """
        return self.delegated.raw_span()

    def get_correlation_data_for_logs(self) -> dict[str, Any]:
        """
        Return the delegate's correlation identifiers, which is what application logs are correlated against.

        :returns: The delegated tracer's correlation data.
        """
        return self.delegated.get_correlation_data_for_logs()


@dataclass
class CapturedRun:
    """
    Mutable run capture finalized into a `TraceArtifact` on context exit.

    :param run_id: Identifier assigned to this run.
    :param started_at: ISO 8601 timestamp of when capture began.
    :param finished_at: ISO 8601 timestamp of when capture ended.
    :param duration_ms: Wall-clock duration of the run.
    :param status: Whether the run succeeded or raised.
    :param spans: The spans recorded so far.
    :param failure: Type, message, and stacktrace of the failure, for a failed run.
    """

    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=_utc_now)
    _started_perf: float = field(default_factory=perf_counter)
    finished_at: str | None = None
    duration_ms: float | None = None
    status: Literal["success", "failed"] = "success"
    spans: list[_CapturedSpan] = field(default_factory=list)
    failure: dict[str, Any] | None = None
    _lock: Any = field(default_factory=RLock, repr=False, compare=False)

    def add_span(self, span: _CapturedSpan) -> None:
        """
        Record a span for this run. Safe to call from concurrently executing tool calls.

        :param span: The span to record.
        """
        with self._lock:
            self.spans.append(span)

    def finish(self) -> None:
        """Record the run's end time and duration."""
        self.finished_at = _utc_now()
        self.duration_ms = round((perf_counter() - self._started_perf) * 1000, 3)

    def to_artifact(self) -> "TraceArtifact":
        """
        Return the finalized artifact.

        :returns: The captured run as a `haystack-trace/v1` artifact.
        :raises RuntimeError: If the run has not finished yet.
        """
        if self.finished_at is None or self.duration_ms is None:
            msg = "The run capture has not finished yet."
            raise RuntimeError(msg)
        with self._lock:
            records = tuple(span.to_record() for span in self.spans)
        return TraceArtifact(
            run_id=self.run_id,
            started_at=self.started_at,
            finished_at=self.finished_at,
            duration_ms=self.duration_ms,
            status=self.status,
            traces=records,
            failure=self.failure,
        )


_current_run: ContextVar[CapturedRun | None] = ContextVar("agent_pack_trace_capture", default=None)
_active_spans: ContextVar[tuple[_CombinedSpan, ...]] = ContextVar("agent_pack_active_spans", default=())


class RunCaptureTracer(Tracer):
    """
    Tracer that captures spans for the current run and delegates every span to an existing tracer.

    Only runs started through `LocalTraceCollector.capture_run` are captured; every other span is passed straight
    through, so installing this tracer never changes what an unrelated part of the application records.
    """

    def __init__(
        self,
        delegate: Tracer,
        *,
        capture_content: bool = True,
        limits: TraceCaptureLimits = DEFAULT_TRACE_CAPTURE_LIMITS,
    ) -> None:
        """
        Create a capturing tracer.

        :param delegate: The tracer that was installed before capture, which keeps receiving every span.
        :param capture_content: Whether content tags are recorded locally.
        :param limits: Bounds applied to captured values.
        """
        self.delegate = delegate
        self.capture_content = capture_content
        self.limits = limits

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """
        Capture and delegate a span.

        :param operation_name: Name of the traced operation.
        :param tags: Tags to set when the span opens.
        :param parent_span: The span this one nests under.
        :returns: A context manager yielding the span to instrument.
        """
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
            tags={key: _serialize_trace_value(value=value, limits=self.limits) for key, value in (tags or {}).items()},
            parent_span_id=local_parent.span_id if local_parent is not None else None,
            capture_content=self.capture_content,
            limits=self.limits,
        )
        capture.add_span(span=captured)
        with self.delegate.trace(operation_name, tags=tags, parent_span=delegated_parent) as delegated:
            combined = _CombinedSpan(captured=captured, delegated=delegated)
            previous = _active_spans.get()
            _active_spans.set((*previous, combined))
            try:
                yield combined
            except Exception as error:
                captured.set_tag(key="error", value=True)
                captured.set_tag(key="error.type", value=type(error).__name__)
                captured.set_tag(key="error.message", value=str(error))
                raise
            finally:
                captured.finish()
                _active_spans.set(previous)

    def current_span(self) -> Span | None:
        """
        Return the active span.

        :returns: The innermost combined span, or the delegated tracer's current span when none is open.
        """
        active = _active_spans.get()
        return active[-1] if active else self.delegate.current_span()
