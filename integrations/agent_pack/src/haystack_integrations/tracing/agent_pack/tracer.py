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
from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric
from haystack.tracing import Span, Tracer

from haystack_integrations.agent_pack.evaluation.dataclasses import ModelTokenUsage

EVAL_CASE_SPAN = "haystack.harness.eval_case"
MAX_RECORDED_TEXTS = 8
MAX_RECORDED_TEXT_CHARS = 120


def _capped(text: str) -> str:
    """
    Cut one recorded string to its allowance, marking the cut so a reader knows there was more.

    :param text: The string a component emitted.
    :returns: The string, ending in an ellipsis when anything was dropped.
    """
    return text if len(text) <= MAX_RECORDED_TEXT_CHARS else f"{text[:MAX_RECORDED_TEXT_CHARS]}..."


@dataclass
class EvalCaseUsage:
    """
    What one eval case spent and what reached each of its stages.

    :param models: Token usage attributed to each model the eval case called, keyed by model identifier.
    :param outputs: How many items each component emitted, by component name and output socket.
    :param texts: A sample of whatever each component emitted as text, by component name and output socket,
        capped at `MAX_RECORDED_TEXTS` entries of `MAX_RECORDED_TEXT_CHARS`.
    :param complete: Whether every model call reported token usage. False means the total token usage is underestimated.
    :param calls: How many model calls the eval case made.
    :param lock: Guards the counters, since eval cases can be run in parallel.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    complete: bool = True
    calls: int = 0
    lock: LockType = field(default_factory=Lock, repr=False)


class _HarnessSpan(Span):
    def __init__(
        self, eval_case_usage: EvalCaseUsage | None, is_generator_span: bool, component_name: str | None = None
    ) -> None:
        """
        Create a span that records into one eval case.

        :param eval_case_usage: Where this span records, or `None` when the span happened outside any eval case and
            nothing it reports is kept.
        :param is_generator_span: Whether this span is a model call, whose token usage is recorded. Every other span is
            measured by how much it emitted instead.
        :param component_name: The component the span belongs to, which names its entry in `outputs` and `texts`.
            `None` for a span that is not a component run, such as an agent step or a hook.
        """
        self.eval_case_usage = eval_case_usage
        self.is_generator_span = is_generator_span
        self.component_name = component_name
        # Set once this span reports its token usage.
        self.recorded = False

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def _record_outputs(self, value: Any) -> None:
        """
        Record how much one component emitted, and a capped sample of whatever it emitted as text.

        Generators are skipped: a reply count is always one and says nothing about how much reached the next
        stage, and the generators held inside other components share the name their owner gave them, so counting
        them would collide two stages under one entry.
        """
        if (
            self.eval_case_usage is None
            or self.component_name is None
            or self.is_generator_span
            or not isinstance(value, dict)
        ):
            return
        # Only sequences are measurable, and only a sequence of nothing but strings is worth sampling. Documents
        # and messages are counted and dropped, so nothing long is retained by accident.
        emitted = {socket: items for socket, items in value.items() if isinstance(items, (list, tuple))}
        sizes = {socket: len(items) for socket, items in emitted.items()}
        texts = {
            socket: [_capped(text=item) for item in items[:MAX_RECORDED_TEXTS]]
            for socket, items in emitted.items()
            if items and all(isinstance(item, str) for item in items)
        }
        if not sizes:
            return
        with self.eval_case_usage.lock:
            self.eval_case_usage.outputs[self.component_name] = sizes
            if texts:
                self.eval_case_usage.texts[self.component_name] = texts

    def set_content_tag(self, key: str, value: Any) -> None:
        """Extract usage and output sizes from one component output without enabling content logging."""
        # Record output lengths and truncated text samples for every component
        if key == "haystack.component.output":
            self._record_outputs(value)

        if (
            # We only care if a generator was detected
            not self.is_generator_span
            # Empty eval_case_usage means the span is outside any eval case
            or self.eval_case_usage is None
            # Only these operation names are known to report token usage in their output
            or key not in ("haystack.component.output", "haystack.agent.step.llm.output")
        ):
            return

        # Record token usage from every generator that reports it
        self.recorded = True
        with self.eval_case_usage.lock:
            self.eval_case_usage.calls += 1
            replies = value.get("replies", []) if isinstance(value, dict) else []
            if not replies:
                self.eval_case_usage.complete = False
            for reply in replies:
                model = reply.meta.get("model")
                tokens = reply.meta.get("usage") or {}
                if not isinstance(model, str) or not all(
                    any(isinstance(tokens.get(key), (int, float)) for key in keys)
                    for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)
                ):
                    self.eval_case_usage.complete = False
                    continue
                current = self.eval_case_usage.models.get(model, ModelTokenUsage())
                self.eval_case_usage.models[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + _first_numeric(usage=tokens, keys=_INPUT_TOKEN_KEYS),
                    output_tokens=current.output_tokens + _first_numeric(usage=tokens, keys=_OUTPUT_TOKEN_KEYS),
                )


class HarnessTracer(Tracer):
    """
    Collect what one eval case spent and how much reached each of its stages.

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
        # An eval case span opens a fresh collection; every span under it records into that one.
        inherited = parent.eval_case_usage if parent is not None else None
        eval_case_usage = EvalCaseUsage() if operation_name == EVAL_CASE_SPAN else inherited
        is_generator_span = operation_name in ("haystack.agent.step.llm", "haystack.chat_generator.run") or (
            operation_name == "haystack.component.run"
            and str((tags or {}).get("haystack.component.type", "")).endswith("ChatGenerator")
        )
        span = _HarnessSpan(
            eval_case_usage=eval_case_usage,
            is_generator_span=is_generator_span,
            component_name=str((tags or {}).get("haystack.component.name") or "") or None,
        )
        token = self._span.set(span)
        try:
            yield span
        finally:
            if is_generator_span and not span.recorded and eval_case_usage is not None:
                eval_case_usage.complete = False
            self._span.reset(token)

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
    Return what a harness collected under one eval case span.

    :param span: The span a harness opened with `EVAL_CASE_SPAN`.
    :returns: What was recorded, or an empty record when a HarnessTracer was not the active tracer.
    """
    if isinstance(span, _HarnessSpan) and span.eval_case_usage is not None:
        return span.eval_case_usage
    return EvalCaseUsage()
