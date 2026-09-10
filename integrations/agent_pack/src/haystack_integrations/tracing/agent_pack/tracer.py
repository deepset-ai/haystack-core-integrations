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

from haystack_integrations.agent_pack.evaluation.dataclasses import ModelTokenUsage

# How many of a socket's strings to keep, and how much of each. A query expansion is worth reading back; an
# unbounded one, or a socket carrying document text, must not reach whoever reads the measurement.
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

    Usage totals, output sizes, and a capped sample of short strings are retained; prompts, replies and
    documents are discarded.

    `outputs` is how many items each component emitted, by component name and output socket. A configuration is
    a chain of stages, and what a score means depends on how much reached each of them: a pipeline pooling six
    searches deduplicates them into a candidate set whose size no configuration value states, so widening the
    search and widening what survives it cannot be told apart from the inputs and the final answer alone.

    `texts` samples the sockets that emitted short strings, capped at `MAX_RECORDED_TEXTS` entries of
    `MAX_RECORDED_TEXT_CHARS` each. It is the one thing kept verbatim, because a count cannot say it: a stage
    that rewrites the question decides what the search can possibly find, and whether four rewrites decomposed
    the question or restated it is the difference between a configuration worth keeping and one worth undoing.
    """

    models: dict[str, ModelTokenUsage] = field(default_factory=dict)
    outputs: dict[str, dict[str, int]] = field(default_factory=dict)
    texts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    complete: bool = True
    calls: int = 0
    hook_calls: int = 0
    lock: LockType = field(default_factory=Lock, repr=False)


class _HarnessSpan(Span):
    def __init__(
        self, usage: EvalCaseUsage | None, generator: bool, component: str | None = None, hook: bool = False
    ) -> None:
        self.usage = usage
        self.generator = generator
        self.component = component
        self.hook = hook
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
        if self.usage is None or self.component is None or self.generator or not isinstance(value, dict):
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
        with self.usage.lock:
            self.usage.outputs[self.component] = sizes
            if texts:
                self.usage.texts[self.component] = texts

    def set_content_tag(self, key: str, value: Any) -> None:
        """Extract usage and output sizes from one component output without enabling content logging."""
        if key == "haystack.component.output":
            self._record_outputs(value)
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
            # A hook's own model call is not one of the Agent's steps and leaves nothing in its output, so the
            # span it runs under is the only record that it happened.
            self.usage.hook_calls += self.hook
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

    Three things are taken from the spans a run emits and nothing else is kept: the token usage a generator
    reports, how many items every other component emitted, and a capped sample of the sockets that emitted
    short strings. Prompts, replies and documents are discarded as they pass, so the only content retained is
    that sample and content tracing never has to be enabled.
    """

    def __init__(self) -> None:
        """Initialize task-local eval case and span context."""
        self._case: ContextVar[EvalCaseUsage | None] = ContextVar("harness_usage", default=None)
        self._span: ContextVar[_HarnessSpan | None] = ContextVar("harness_span", default=None)

    @contextmanager
    def trace(
        self, operation_name: str, tags: dict[str, Any] | None = None, parent_span: Span | None = None
    ) -> Iterator[Span]:
        """Follow explicit parents as well as context propagated into async worker threads."""
        parent = parent_span if isinstance(parent_span, _HarnessSpan) else self.current_span()
        usage = parent.usage if parent is not None else self._case.get()
        agent_step = operation_name == "haystack.agent.step.llm"
        generator = (
            agent_step
            or operation_name == "haystack.chat_generator.run"
            or (
                operation_name == "haystack.component.run"
                and str((tags or {}).get("haystack.component.type", "")).endswith("ChatGenerator")
            )
        )
        # Hooks run their own models outside the step loop, and the flag descends so a generator nested under
        # a hook span is recognized as the hook's own call.
        hook = operation_name == "haystack.agent.hook" or (parent.hook if parent is not None else False)
        span = _HarnessSpan(usage, generator, str((tags or {}).get("haystack.component.name") or "") or None, hook)
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
    def eval_case(self) -> Iterator[EvalCaseUsage]:
        """Collect one eval case independently of concurrently running eval cases."""
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
