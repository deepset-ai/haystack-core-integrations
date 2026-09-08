# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Conversation-aware tracing for applications that drive Haystack from their own loop."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from haystack import logging
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, TraceFlags

from haystack_integrations.tracing.rhesis.tracer import tracing_context_var
from rhesis.telemetry.constants import ConversationContext
from rhesis.telemetry.context import get_root_trace_id, set_root_trace_id

logger = logging.getLogger(__name__)

DEFAULT_TURN_SPAN_NAME = "function.haystack.turn"

_SPAN_ATTRS = ConversationContext.SpanAttributes
_MAX_IO = ConversationContext.MAX_IO_LENGTH


def _conversation_parent_context(trace_id: str) -> Any:
    """
    Build a synthetic parent so a turn inherits the conversation's trace id.

    OpenTelemetry mints a fresh trace id for every parentless span, which would scatter one
    conversation across a trace per turn. Attaching a non-recording parent carrying the
    conversation's trace id makes the new span join it instead. The parent's span id is the
    agreed placeholder that the Rhesis exporter strips, so the turn is still stored as a root
    span — the same approach the Rhesis SDK uses for turns it serves itself.

    :param trace_id: 32-character hex trace id of the conversation's first turn.
    :returns: An OTel context carrying the synthetic parent, or ``None`` if the id is unusable.
    """
    try:
        span_context = SpanContext(
            trace_id=int(trace_id, 16),
            span_id=ConversationContext.SYNTHETIC_PARENT_SPAN_ID,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
    except (TypeError, ValueError):
        logger.warning("Invalid conversation trace id '{trace_id}'; starting a new trace", trace_id=trace_id)
        return None
    return trace.set_span_in_context(NonRecordingSpan(span_context))


class ConversationTurn:
    """
    A single conversation turn, yielded by :meth:`RhesisTracing.turn`.

    Assign :attr:`output` with the reply the user actually sees. Only the application knows
    what that is — it may be a tool result or a value held in agent state rather than the last
    assistant message — so it cannot be inferred from the span tree.
    """

    def __init__(self, span: Span | None = None) -> None:
        self._span = span
        self._output = ""

    @property
    def span(self) -> Span | None:
        """The underlying OTel span, or ``None`` when tracing is disabled."""
        return self._span

    @property
    def output(self) -> str:
        """The reply recorded for this turn."""
        return self._output

    @output.setter
    def output(self, reply: str) -> None:
        self._output = reply or ""
        if self._span is None or not self._output:
            return
        self._span.set_attribute(_SPAN_ATTRS.CONVERSATION_OUTPUT, self._output[:_MAX_IO])


class RhesisTracing:
    """
    Enable Rhesis tracing for an application that runs Haystack from its own loop.

    :class:`RhesisConnector` covers the common case: add it to a pipeline and every run is
    traced. An application that owns its loop — a chat REPL, a batch script, a server handling
    one turn per request — needs two things a component inside the pipeline cannot provide:
    tracing switched on without a pipeline to attach to, and a span wrapping a whole pipeline
    run so a conversation turn has a root of its own.

    Without that root span, the Haystack pipeline span claims the turn and reports the
    serialized pipeline input and output as the conversation text.

    ``HAYSTACK_CONTENT_TRACING_ENABLED`` must still be set to ``"true"`` before Haystack is
    imported, exactly as when using the connector.

    ### Usage example

    ```python
    import os

    os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

    from haystack_integrations.tracing.rhesis import RhesisTracing

    tracing = RhesisTracing("My Assistant")
    tracing.start_conversation("conversation-1")

    for message in ["Hello", "Tell me more"]:
        with tracing.turn(message) as turn:
            result = pipeline.run(...)
            turn.output = result["llm"]["replies"][0].text

    tracing.flush()
    ```
    """

    def __init__(
        self,
        name: str,
        *,
        enabled: bool = True,
        turn_span_name: str = DEFAULT_TURN_SPAN_NAME,
        **connector_kwargs: Any,
    ) -> None:
        """
        Enable tracing, or fall back to a no-op when Rhesis is not configured.

        Construction never raises on a missing or rejected configuration: an application should
        run untraced rather than fail to start. Check :attr:`enabled` to report it.

        This is the opposite of :class:`~haystack_integrations.components.connectors.rhesis.RhesisConnector`,
        which raises when no API key resolves, and deliberately so. The connector is a component the
        user put in a pipeline; failing loudly there is the honest signal that the thing they wired
        up will not do its job. Here tracing wraps an application's own loop and is not in its data
        path, so the same failure should cost the application nothing.

        :param name: Trace name shown in the Rhesis UI.
        :param enabled: Set to ``False`` to build a no-op instance, so an application can gate
            tracing on its own policy without branching around every call.
        :param turn_span_name: Span name for each conversation turn root.
        :param connector_kwargs: Forwarded to :class:`RhesisConnector` (``api_key``,
            ``base_url``, ``project_id``, ``environment``, ``frontend_url``, ``span_handler``).
        """
        self.name = name
        self.turn_span_name = turn_span_name
        self._conversation_trace_id: str | None = None
        self._tracer = None

        if not enabled:
            logger.info("Rhesis tracing disabled by the caller.")
            return
        if not os.getenv("RHESIS_API_KEY") and "api_key" not in connector_kwargs:
            logger.info("RHESIS_API_KEY is not set; Rhesis tracing is disabled.")
            return

        # Imported here: the connector package imports this one, so a module-level import
        # would be circular.
        from haystack_integrations.components.connectors.rhesis import RhesisConnector  # noqa: PLC0415

        try:
            # Constructing the connector builds the Rhesis tracer provider and registers the tracer
            # with Haystack. Only the Haystack registration is process-wide; the OTel provider
            # belongs to the connector, which is why the turn spans below borrow its tracer. The
            # component is never added to a pipeline.
            self._tracer = RhesisConnector(name, **connector_kwargs).tracer
        except Exception as exc:  # tracing must never break the application
            logger.warning("Could not enable Rhesis tracing: {error}", error=exc)

    @property
    def enabled(self) -> bool:
        """Whether tracing was successfully enabled."""
        return self._tracer is not None

    def start_conversation(self, conversation_id: str, **invocation_context: Any) -> None:
        """
        Group the turns that follow into one conversation, sharing one trace.

        Calling this again starts a new conversation: the next turn opens a new trace and later
        turns join it.

        :param conversation_id: Identifier grouping the turns, shown as the conversation in Rhesis.
        :param invocation_context: Extra metadata for the root span, as
            :meth:`RhesisConnector.run` accepts (test run identifiers, tags, …).
        """
        if not self.enabled:
            return
        tracing_context_var.set({"session_id": conversation_id, **invocation_context})
        self._conversation_trace_id = None

    @contextmanager
    def turn(self, user_input: str) -> Iterator[ConversationTurn]:
        """
        Open the root span for one conversation turn.

        Run the turn's work inside the block and assign the reply to
        :attr:`ConversationTurn.output`. Every turn after the first joins the first one's trace,
        so a conversation reads as one trace rather than one per exchange.

        Yields an inert turn when tracing is disabled, so callers need no branching.

        :param user_input: The user's message, recorded as the turn's conversation input.
        """
        tracer = self._tracer
        if tracer is None:
            yield ConversationTurn()
            return

        conversation_id = (tracing_context_var.get({}) or {}).get("session_id")
        parent_context = (
            _conversation_parent_context(self._conversation_trace_id) if self._conversation_trace_id else None
        )
        # Opened through the connector's own tracer rather than ``trace.get_tracer()``: the connector
        # builds a provider it keeps to itself instead of installing it as the OpenTelemetry global,
        # so the global tracer here would be a no-op and every turn root would silently vanish while
        # the Haystack spans nested inside it were still exported. Sharing the tracer also means a
        # turn is flushed by the same provider as its children.
        otel_tracer = tracer.telemetry.otel_tracer
        previous_root = get_root_trace_id()

        with otel_tracer.start_as_current_span(self.turn_span_name, context=parent_context) as span:
            span.set_attribute(_SPAN_ATTRS.IS_TURN_ROOT, True)
            if conversation_id:
                span.set_attribute(_SPAN_ATTRS.CONVERSATION_ID, conversation_id)
            if user_input:
                span.set_attribute(_SPAN_ATTRS.CONVERSATION_INPUT, user_input[:_MAX_IO])

            trace_id = format(span.get_span_context().trace_id, "032x")
            self._conversation_trace_id = trace_id
            # Marks the turn as owned here, so the Haystack root span nests inside it instead of
            # claiming the turn and restating its input and output.
            set_root_trace_id(trace_id)
            try:
                yield ConversationTurn(span)
            finally:
                set_root_trace_id(previous_root)

    def flush(self) -> None:
        """Flush pending spans. Call before exit; batched spans are otherwise lost."""
        if self._tracer is not None:
            self._tracer.flush()


__all__ = ["DEFAULT_TURN_SPAN_NAME", "ConversationTurn", "RhesisTracing"]
