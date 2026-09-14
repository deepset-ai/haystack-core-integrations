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
from uuid import uuid4

from haystack import tracing
from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric
from haystack.tracing import Span, Tracer

from .dataclasses import EvalCaseSummary, ModelTokenUsage, ReportedUsage

# The span a harness opens around one eval case. Everything traced under it belongs to that eval case.
EVAL_CASE_SPAN = "haystack.harness.eval_case"

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


def _reported_tokens(usage: Any) -> ModelTokenUsage | None:
    """
    Read the token counts one LLM call reported, whatever keys it used.

    :param usage: What a reply reported under its usage key, which crosses an untyped meta dict.
    :returns: The counts, or `None` when either the input or the output count is missing or not a number.
    """
    if not isinstance(usage, dict):
        return None
    # Providers key token counts differently, so any known input key and any known output key will do.
    if not all(
        any(isinstance(usage.get(key), (int, float)) for key in keys)
        for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)
    ):
        return None
    return ModelTokenUsage(
        input_tokens=_first_numeric(usage=usage, keys=_INPUT_TOKEN_KEYS),
        output_tokens=_first_numeric(usage=usage, keys=_OUTPUT_TOKEN_KEYS),
    )


def _is_generator_span(operation_name: str, tags: dict[str, Any]) -> bool:
    """
    Decide whether a span is an LLM call, whose token usage is what it reports.

    :param operation_name: The span's operation name.
    :param tags: The span's tags.
    :returns: True for an agent step, a component-held generator, or a generator run as a pipeline component.
    """
    return operation_name in ("haystack.agent.step.llm", "haystack.chat_generator.run") or (
        operation_name == "haystack.component.run"
        and str(tags.get("haystack.component.type", "")).endswith("ChatGenerator")
    )


@dataclass
class CollectedSpans:
    """
    Every span collected under one eval case span.

    :param spans: The spans, in the order they ended.
    :param lock: Guards appends, since a run's components may be traced from several threads or tasks.
    """

    spans: list["HarnessSpan"] = field(default_factory=list)
    lock: LockType = field(default_factory=Lock, repr=False)

    def add(self, span: "HarnessSpan") -> None:
        """
        Collect one closed span.

        :param span: The span that just ended.
        """
        with self.lock:
            self.spans.append(span)

    def summarize(self) -> EvalCaseSummary:
        """
        Summarize what the collected spans reported.

        :returns: The eval case's token usage and per-stage output sizes.
        """
        models: dict[str, ModelTokenUsage] = {}
        outputs: dict[str, dict[str, int]] = {}
        texts: dict[str, dict[str, list[str]]] = {}
        all_tokens_reported = True
        for span in self.spans:
            # A generator run as a pipeline component is a stage like any other; the spans a component opens
            # inside itself are named for the attribute holding them, so they never overwrite their owner.
            if span.component_name is not None and span.output_sizes:
                outputs[span.component_name] = span.output_sizes
                if span.output_texts:
                    texts[span.component_name] = span.output_texts

            # An LLM call that reported no usage at all spent tokens nobody can account for.
            if span.is_generator_span and not span.reported_usage:
                all_tokens_reported = False

            for entry in span.reported_usage:
                # Usage nobody can attribute to a model, or missing either count, cannot be priced.
                if entry.model is None or entry.tokens is None:
                    all_tokens_reported = False
                    continue
                current = models.get(entry.model, ModelTokenUsage())
                models[entry.model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + entry.tokens.input_tokens,
                    output_tokens=current.output_tokens + entry.tokens.output_tokens,
                )
        return EvalCaseSummary(models=models, outputs=outputs, texts=texts, all_tokens_reported=all_tokens_reported)


class HarnessSpan(Span):
    """A span that keeps how much its component emitted and what its LLM call reported, and discards the rest."""

    def __init__(
        self,
        collected: CollectedSpans | None = None,
        parent_span_id: str | None = None,
        component_name: str | None = None,
        is_generator_span: bool = False,
    ) -> None:
        """
        Create a span that reports into one eval case.

        :param collected: Where this span goes when it closes, or `None` when it ran outside any eval case,
            in which case it is dropped instead.
        :param parent_span_id: The span this one ran under, or `None` for the root of a collection.
        :param component_name: The component the span belongs to, or `None` for a span that is not a component run.
        :param is_generator_span: Whether the span is an LLM call, whose token usage is what it reports.
        """
        self.collected = collected
        self.span_id = str(uuid4())
        self.parent_span_id = parent_span_id
        self.component_name = component_name
        self.is_generator_span = is_generator_span
        self.output_sizes: dict[str, int] = {}
        self.output_texts: dict[str, list[str]] = {}
        self.reported_usage: list[ReportedUsage] = []

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def set_content_tag(self, key: str, value: Any) -> None:
        """Measure one component output and discard it, so no content is retained and none has to be enabled."""
        if key not in USAGE_OUTPUT_TAGS or not isinstance(value, dict):
            return
        self.output_sizes, self.output_texts = _measure_output(value=value)
        if self.is_generator_span:
            self.reported_usage = [
                ReportedUsage(model=reply.meta.get("model"), tokens=_reported_tokens(usage=reply.meta.get("usage")))
                for reply in value.get("replies") or []
            ]


class HarnessTracer(Tracer):
    """A tracer that collects spans for one eval case, and discards everything else."""

    def __init__(self) -> None:
        """Initialize task-local span context."""
        self._span: ContextVar[HarnessSpan | None] = ContextVar("harness_span", default=None)

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Follow explicit parents as well as context propagated into async worker threads."""
        parent = parent_span if isinstance(parent_span, HarnessSpan) else self.current_span()
        # An eval case span opens a fresh collection; every span under it reports into that one.
        inherited = parent.collected if parent is not None else None
        collected = CollectedSpans() if operation_name == EVAL_CASE_SPAN else inherited
        span = HarnessSpan(
            collected=collected,
            parent_span_id=parent.span_id if parent is not None else None,
            component_name=str((tags or {}).get("haystack.component.name") or "") or None,
            is_generator_span=_is_generator_span(operation_name=operation_name, tags=tags or {}),
        )
        token = self._span.set(span)
        try:
            yield span
        finally:
            self._span.reset(token)
            # The eval case span collects the spans rather than becoming one of them.
            if collected is not None and operation_name != EVAL_CASE_SPAN:
                collected.add(span=span)

    def current_span(self) -> HarnessSpan | None:
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
