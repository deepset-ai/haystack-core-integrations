# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from _thread import LockType
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from haystack import tracing
from haystack.tracing import Span, Tracer

from .span_records import (
    EVAL_CASE_SPAN,
    EvalCaseUsage,
    ReportedUsage,
    SpanRecord,
    eval_case_usage_from_records,
    get_component_name,
    is_generator_span,
    measure_output,
)

# The tags a component's output arrives under.
USAGE_OUTPUT_TAGS = ("haystack.component.output", "haystack.agent.step.llm.output")


@dataclass
class SpanRecords:
    """
    Every span record collected under one eval case span.

    :param records: The records, in the order their spans ended.
    :param lock: Guards appends, since a run's components may be traced from several threads or tasks.
    """

    records: list[SpanRecord] = field(default_factory=list)
    lock: LockType = field(default_factory=Lock, repr=False)

    def add(self, record: SpanRecord) -> None:
        """
        Collect one closed span's record.

        :param record: What the span reported.
        """
        with self.lock:
            self.records.append(record)


class _HarnessSpan(Span):
    def __init__(self, collected: SpanRecords | None, record: SpanRecord) -> None:
        """
        Create a span that reports into one eval case.

        :param collected: Where this span's record goes when it closes, or `None` when the span happened
            outside any eval case and nothing it reports is kept.
        :param record: What this span will fill in as it runs.
        """
        self.collected = collected
        self.record = record

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def set_content_tag(self, key: str, value: Any) -> None:
        """Measure one component output and discard it, so no content is retained and none has to be enabled."""
        if self.collected is None or key not in USAGE_OUTPUT_TAGS or not isinstance(value, dict):
            return
        if not self.record.is_generator_span:
            # A generator reports what it spent, not how much reached the next stage, so measuring its
            # replies would only produce a size the measurement throws away.
            self.record.output_sizes, self.record.output_texts = measure_output(value=value)
        else:
            self.record.reported_output = True
            self.record.reported_usage = [
                ReportedUsage(model=reply.meta.get("model"), tokens=reply.meta.get("usage") or {})
                for reply in value.get("replies") or []
            ]


class HarnessTracer(Tracer):
    """
    Record what every span under an eval case reported, for `eval_case_usage_from_records` to make a measurement of.

    Three things are taken from the spans a run emits and nothing else is kept: the token usage a generator
    reports, how many items every other component emitted, and a capped sample of the sockets that emitted
    short strings.
    """

    def __init__(self) -> None:
        """Initialize task-local span context."""
        self._span: ContextVar[_HarnessSpan | None] = ContextVar("harness_span", default=None)

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Follow explicit parents as well as context propagated into async worker threads."""
        parent = parent_span if isinstance(parent_span, _HarnessSpan) else self.current_span()
        # An eval case span opens a fresh collection; every span under it reports into that one.
        inherited = parent.collected if parent is not None else None
        collected = SpanRecords() if operation_name == EVAL_CASE_SPAN else inherited
        span = _HarnessSpan(
            collected=collected,
            record=SpanRecord(
                parent_span_id=parent.record.span_id if parent is not None else None,
                component_name=get_component_name(tags=tags or {}),
                is_generator_span=is_generator_span(operation_name=operation_name, tags=tags or {}),
            ),
        )
        token = self._span.set(span)
        try:
            yield span
        finally:
            self._span.reset(token)
            # The eval case span collects the records rather than becoming one of them.
            if collected is not None and operation_name != EVAL_CASE_SPAN:
                collected.add(record=span.record)

    def current_span(self) -> _HarnessSpan | None:
        """Return the current span for Haystack's explicit thread-parent propagation."""
        return self._span.get()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Own global tracing for evaluation, disabling it afterward even on failure."""
        tracing.enable_tracing(self)
        try:
            yield
        finally:
            tracing.disable_tracing()


def usage_from_span(span: Span) -> EvalCaseUsage:
    """
    Return what a harness measured under one eval case span.

    :param span: The span a harness opened with `EVAL_CASE_SPAN`.
    :returns: The measurement, or an empty one when a HarnessTracer was not the active tracer.
    """
    if isinstance(span, _HarnessSpan) and span.collected is not None:
        return eval_case_usage_from_records(records=span.collected.records)
    return EvalCaseUsage()
