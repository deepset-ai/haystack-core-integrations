# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""In-memory model usage collection for a harness that owns tracing during evaluation."""

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
class CaseUsage:
    """Only usage totals are retained; prompts and replies are discarded."""

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    complete: bool = True
    calls: int = 0
    lock: LockType = field(default_factory=Lock, repr=False)


class _UsageSpan(Span):
    def __init__(self, usage: CaseUsage | None, generator: bool) -> None:
        self.usage = usage
        self.generator = generator
        self.recorded = False

    def set_tag(self, key: str, value: Any) -> None:
        """Discard ordinary trace tags."""

    def set_content_tag(self, key: str, value: Any) -> None:
        """Extract usage from one generator output without enabling content logging."""
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


class UsageTracer(Tracer):
    """Collect generator-call spans only, ignoring aggregate Agent and Pipeline outputs."""

    def __init__(self) -> None:
        """Initialize task-local case and span context."""
        self._case: ContextVar[CaseUsage | None] = ContextVar("harness_usage", default=None)
        self._span: ContextVar[_UsageSpan | None] = ContextVar("harness_span", default=None)

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Follow explicit parents as well as context propagated into async worker threads."""
        parent = parent_span if isinstance(parent_span, _UsageSpan) else self.current_span()
        usage = parent.usage if parent is not None else self._case.get()
        generator = operation_name in ("haystack.chat_generator.run", "haystack.agent.step.llm") or (
            operation_name == "haystack.component.run"
            and str((tags or {}).get("haystack.component.type", "")).endswith("ChatGenerator")
        )
        span = _UsageSpan(usage, generator)
        token = self._span.set(span)
        try:
            yield span
        finally:
            if generator and not span.recorded and usage is not None:
                usage.complete = False
            self._span.reset(token)

    def current_span(self) -> _UsageSpan | None:
        """Return the current span for Haystack's explicit thread-parent propagation."""
        return self._span.get()

    @contextmanager
    def case(self) -> Iterator[CaseUsage]:
        """Collect one evaluation case independently of concurrently running cases."""
        usage = CaseUsage()
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
