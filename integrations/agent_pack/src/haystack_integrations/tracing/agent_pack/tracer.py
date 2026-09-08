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
from haystack.dataclasses import ChatMessage
from haystack.tracing import Span, Tracer

from haystack_integrations.agent_pack.dataclasses import ModelTokenUsage


@dataclass
class EvalCaseUsage:
    """
    Only usage totals and output sizes are retained; prompts, replies and documents are discarded.

    `outputs` is how many items each component emitted, by component name and output socket. A configuration is
    a chain of stages, and what a score means depends on how much reached each of them: a pipeline pooling six
    searches deduplicates them into a candidate set whose size no configuration value states, so widening the
    search and widening what survives it cannot be told apart from the inputs and the final answer alone.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    complete: bool = True
    calls: int = 0
    lock: LockType = field(default_factory=Lock, repr=False)


class _HarnessSpan(Span):
    def __init__(self, usage: EvalCaseUsage | None, generator: bool, component: str | None = None) -> None:
        self.usage = usage
        self.generator = generator
        self.component = component
        self.recorded = False

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def _record_output_sizes(self, value: Any) -> None:
        """
        Count what one component emitted, keeping the sizes and discarding the items.

        Generators are skipped: a reply count is always one and says nothing about how much reached the next
        stage, and the generators held inside other components share the name their owner gave them, so counting
        them would collide two stages under one entry.
        """
        if self.usage is None or self.component is None or self.generator or not isinstance(value, dict):
            return
        sizes = {socket: len(items) for socket, items in value.items() if isinstance(items, (list, tuple))}
        if sizes:
            with self.usage.lock:
                self.usage.outputs[self.component] = sizes

    def set_content_tag(self, key: str, value: Any) -> None:
        """Extract usage and output sizes from one component output without enabling content logging."""
        if key == "haystack.component.output":
            self._record_output_sizes(value)
        if (
            not self.generator
            or self.usage is None
            or self.recorded
            or key not in ("haystack.component.output", "haystack.agent.step.llm.output")
        ):
            return
        self.recorded = True
        with self.usage.lock:
            self.usage.calls += 1
            replies = value.get("replies", []) if isinstance(value, dict) else []
            if not replies:
                self.usage.complete = False
            for reply in replies:
                if not isinstance(reply, ChatMessage):
                    self.usage.complete = False
                    continue
                model = reply.meta.get("model")
                tokens = reply.meta.get("usage") or {}
                if not isinstance(model, str) or not all(
                    any(isinstance(tokens.get(key), (int, float)) for key in keys)
                    for keys in (_INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS)
                ):
                    self.usage.complete = False
                    continue
                current = self.usage.models.get(model, ModelTokenUsage())
                self.usage.models[model] = ModelTokenUsage(
                    input_tokens=current.input_tokens + _first_numeric(tokens, _INPUT_TOKEN_KEYS),
                    output_tokens=current.output_tokens + _first_numeric(tokens, _OUTPUT_TOKEN_KEYS),
                )


class HarnessTracer(Tracer):
    """
    Collect what one eval case spent and how much reached each of its stages.

    Two things are taken from the spans a run emits and nothing else is kept: the token usage a generator
    reports, and how many items every other component emitted. Prompts, replies and documents are discarded as
    they pass, so no content is retained and content tracing never has to be enabled.
    """

    def __init__(self) -> None:
        """Initialize task-local case and span context."""
        self._case: ContextVar[EvalCaseUsage | None] = ContextVar("harness_usage", default=None)
        self._span: ContextVar[_HarnessSpan | None] = ContextVar("harness_span", default=None)

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Follow explicit parents as well as context propagated into async worker threads."""
        parent = parent_span if isinstance(parent_span, _HarnessSpan) else self.current_span()
        usage = parent.usage if parent is not None else self._case.get()
        generator = operation_name in ("haystack.chat_generator.run", "haystack.agent.step.llm") or (
            operation_name == "haystack.component.run"
            and str((tags or {}).get("haystack.component.type", "")).endswith("ChatGenerator")
        )
        span = _HarnessSpan(usage, generator, str((tags or {}).get("haystack.component.name") or "") or None)
        token = self._span.set(span)
        try:
            yield span
        finally:
            if generator and not span.recorded and usage is not None:
                usage.complete = False
            self._span.reset(token)

    def current_span(self) -> _HarnessSpan | None:
        """Return the current span for Haystack's explicit thread-parent propagation."""
        return self._span.get()

    @contextmanager
    def case(self) -> Iterator[EvalCaseUsage]:
        """Collect one eval case independently of concurrently running cases."""
        usage = EvalCaseUsage()
        token = self._case.set(usage)
        try:
            yield usage
        finally:
            self._case.reset(token)

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Own global tracing for evaluation, disabling it afterward even on failure."""
        tracing.enable_tracing(self)
        try:
            yield
        finally:
            tracing.disable_tracing()
