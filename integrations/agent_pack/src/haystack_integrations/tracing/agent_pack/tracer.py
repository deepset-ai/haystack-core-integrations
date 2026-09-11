# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
from typing import Any

from haystack import tracing
from haystack.tracing import Span, Tracer

from .span_records import (
    EVAL_CASE_SPAN,
    EvalCaseSummary,
    ReportedUsage,
    SpanRecord,
    SpanRecords,
    _eval_case_summary_from_records,
)

# The tags a component's output arrives under.
USAGE_OUTPUT_TAGS = ("haystack.component.output", "haystack.agent.step.llm.output")

# How many of a socket's strings to keep, and how much of each.
MAX_RECORDED_TEXTS = 8
MAX_RECORDED_TEXT_CHARS = 120


def _capped(text: str) -> str:
    """
    Cut one recorded string to its allowance, marking the cut so a reader knows there was more.

    :param text: The string a component emitted.
    :returns: The string, ending in an ellipsis when anything was dropped.
    """
    return text if len(text) <= MAX_RECORDED_TEXT_CHARS else f"{text[:MAX_RECORDED_TEXT_CHARS]}..."


def _measure_output(value: dict[str, Any]) -> tuple[dict[str, int], dict[str, list[str]]]:
    """
    Measure how much a component emitted on each socket, and sample whatever it emitted as short strings.

    :param value: The component's output, by socket.
    :returns: How many items each socket carried, and a capped sample of the sockets carrying only strings.
    """
    sizes: dict[str, int] = {}
    texts: dict[str, list[str]] = {}
    for socket, emitted in value.items():
        # A socket carrying one string or one object emits one item, not none: a router or a prompt builder
        # belongs in the chain as much as a retriever does. A string is characters, not items, so it counts once.
        items = list(emitted) if isinstance(emitted, (list, tuple)) else [emitted]
        sizes[socket] = len(items)
        # Documents and messages are counted and dropped, so nothing long is retained by accident.
        if items and all(isinstance(item, str) for item in items):
            texts[socket] = [_capped(text=item) for item in items[:MAX_RECORDED_TEXTS]]
    return sizes, texts


def _is_generator_span(operation_name: str, tags: dict[str, Any]) -> bool:
    """
    Decide whether a span is a model call, whose token usage is what it reports.

    :param operation_name: The span's operation name.
    :param tags: The span's tags.
    :returns: True for an agent step, a component-held generator, or a generator run as a pipeline component.
    """
    return operation_name in ("haystack.agent.step.llm", "haystack.chat_generator.run") or (
        operation_name == "haystack.component.run"
        and str(tags.get("haystack.component.type", "")).endswith("ChatGenerator")
    )


class _HarnessSpan(Span):
    def __init__(self, collected: SpanRecords | None, record: SpanRecord) -> None:
        """
        Create a span that reports into one eval case.

        :param collected: Where this span's record goes when it closes, or `None` when the span happened
            outside any eval case, in which case its record is dropped instead.
        :param record: What this span will fill in as it runs.
        """
        self.collected = collected
        self.record = record

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def set_content_tag(self, key: str, value: Any) -> None:
        """Measure one component output and discard it, so no content is retained and none has to be enabled."""
        if key not in USAGE_OUTPUT_TAGS or not isinstance(value, dict):
            return
        if not self.record.is_generator_span:
            # A generator reports what it spent, not how much reached the next stage, so measuring its
            # replies would only produce a size the summary throws away.
            sizes, texts = _measure_output(value=value)
            self.record = replace(self.record, output_sizes=sizes, output_texts=texts)
        else:
            self.record = replace(
                self.record,
                reported_output=True,
                reported_usage=[
                    ReportedUsage(model=reply.meta.get("model"), tokens=reply.meta.get("usage") or {})
                    for reply in value.get("replies") or []
                ],
            )


class HarnessTracer(Tracer):
    """A tracer that collects spans for one eval case, and discards everything else."""

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
                component_name=str((tags or {}).get("haystack.component.name") or "") or None,
                is_generator_span=_is_generator_span(operation_name=operation_name, tags=tags or {}),
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


def _eval_case_summary_from_span(span: Span) -> EvalCaseSummary:
    """
    Return the summary of what ran under one eval case span.

    :param span: The span a harness opened with `EVAL_CASE_SPAN`.
    :returns: The summary, or an empty one when a HarnessTracer was not the active tracer.
    """
    if isinstance(span, _HarnessSpan) and span.collected is not None:
        return _eval_case_summary_from_records(records=span.collected.records)
    return EvalCaseSummary()
