# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the conversation-aware tracing entry point."""

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.telemetry.constants import ConversationContext
from rhesis.telemetry.context import get_root_trace_id

from haystack_integrations.tracing.rhesis import ConversationTurn, RhesisTracing
from haystack_integrations.tracing.rhesis.tracer import tracing_context_var

ATTRS = ConversationContext.SpanAttributes


class _StubTelemetry:
    """
    Stands in for ``RhesisTelemetry``, collecting spans in memory instead of exporting them.

    A turn span is opened through the connector's own ``otel_tracer``, not ``trace.get_tracer()``:
    the connector's provider is private to it and never installed as the OpenTelemetry global. So the
    stub has to carry a real provider — patching the global would test a path the code no longer uses,
    and would pass just as happily if turn spans had stopped being recorded altogether.
    """

    def __init__(self) -> None:
        self.provider = TracerProvider()
        self.exporter = InMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.otel_tracer = self.provider.get_tracer("test")


@pytest.fixture
def telemetry():
    """The stand-in telemetry the stub connector below hands to ``RhesisTracing``."""
    return _StubTelemetry()


@pytest.fixture
def exporter(telemetry):
    """The in-memory spans, with no backend involved."""
    return telemetry.exporter


@pytest.fixture
def tracing(telemetry, monkeypatch):
    """A RhesisTracing whose connector construction is stubbed out."""

    class _StubTracer:
        def __init__(self) -> None:
            self.flushed = 0
            self.telemetry = telemetry

        def flush(self) -> None:
            self.flushed += 1

    class _StubConnector:
        def __init__(self, name: str, **kwargs: Any) -> None:
            self.name = name
            self.tracer = _StubTracer()

    monkeypatch.setenv("RHESIS_API_KEY", "test-key")
    import haystack_integrations.components.connectors.rhesis as connector_module  # noqa: PLC0415

    monkeypatch.setattr(connector_module, "RhesisConnector", _StubConnector)
    instance = RhesisTracing("Test App")
    assert instance.enabled
    return instance


def _turn_roots(exporter) -> list[Any]:
    return [s for s in exporter.get_finished_spans() if ATTRS.IS_TURN_ROOT in s.attributes]


class TestRhesisTracingSetup:
    def test_disabled_without_api_key(self, monkeypatch):
        monkeypatch.delenv("RHESIS_API_KEY", raising=False)
        instance = RhesisTracing("Test App")
        assert instance.enabled is False

    def test_disabled_when_the_connector_rejects_the_configuration(self, monkeypatch):
        """An application must start even when tracing cannot be configured."""

        def _explode(*args: Any, **kwargs: Any) -> None:
            msg = "bad configuration"
            raise ValueError(msg)

        monkeypatch.setenv("RHESIS_API_KEY", "test-key")
        import haystack_integrations.components.connectors.rhesis as connector_module  # noqa: PLC0415

        monkeypatch.setattr(connector_module, "RhesisConnector", _explode)
        instance = RhesisTracing("Test App")
        assert instance.enabled is False

    def test_disabled_turn_yields_an_inert_handle(self, monkeypatch):
        monkeypatch.delenv("RHESIS_API_KEY", raising=False)
        instance = RhesisTracing("Test App")
        with instance.turn("hello") as turn:
            turn.output = "hi"
        assert turn.span is None
        assert turn.output == "hi"
        instance.flush()  # must not raise

    def test_flush_delegates_to_the_tracer(self, tracing):
        tracing.flush()
        assert tracing._tracer.flushed == 1


class TestConversationTurn:
    def test_turn_records_input_and_output(self, tracing, exporter):
        tracing.start_conversation("conv-1")
        with tracing.turn("What is the capital of Germany?") as turn:
            turn.output = "Berlin."

        (root,) = _turn_roots(exporter)
        assert root.name == tracing.turn_span_name
        assert root.attributes[ATTRS.CONVERSATION_INPUT] == "What is the capital of Germany?"
        assert root.attributes[ATTRS.CONVERSATION_OUTPUT] == "Berlin."
        assert root.attributes[ATTRS.CONVERSATION_ID] == "conv-1"

    def test_turn_without_output_records_only_input(self, tracing, exporter):
        tracing.start_conversation("conv-1")
        with tracing.turn("hello"):
            pass

        (root,) = _turn_roots(exporter)
        assert root.attributes[ATTRS.CONVERSATION_INPUT] == "hello"
        assert ATTRS.CONVERSATION_OUTPUT not in root.attributes

    def test_long_io_is_truncated(self, tracing, exporter):
        oversized = "x" * (ConversationContext.MAX_IO_LENGTH + 500)
        tracing.start_conversation("conv-1")
        with tracing.turn(oversized) as turn:
            turn.output = oversized

        (root,) = _turn_roots(exporter)
        assert len(root.attributes[ATTRS.CONVERSATION_INPUT]) == ConversationContext.MAX_IO_LENGTH
        assert len(root.attributes[ATTRS.CONVERSATION_OUTPUT]) == ConversationContext.MAX_IO_LENGTH

    def test_turn_owns_the_root_trace_id_and_restores_it(self, tracing):
        """The Haystack tracer reads this ContextVar to decide it does not own the turn."""
        before = get_root_trace_id()
        with tracing.turn("hello"):
            assert get_root_trace_id() is not None
        assert get_root_trace_id() == before

    def test_start_conversation_carries_extra_invocation_context(self, tracing):
        tracing.start_conversation("conv-1", test_run_id="run-7")
        assert tracing_context_var.get({}) == {"session_id": "conv-1", "test_run_id": "run-7"}


class TestConversationTraceContinuity:
    def test_turns_of_one_conversation_share_a_trace(self, tracing, exporter):
        tracing.start_conversation("conv-1")
        for message in ["first", "second", "third"]:
            with tracing.turn(message) as turn:
                turn.output = f"reply to {message}"

        roots = _turn_roots(exporter)
        assert len(roots) == 3
        assert len({format(s.context.trace_id, "032x") for s in roots}) == 1

    def test_turns_after_the_first_hang_off_the_synthetic_parent(self, tracing, exporter):
        """The placeholder parent is what carries the trace id; the exporter strips it later."""
        tracing.start_conversation("conv-1")
        for message in ["first", "second"]:
            with tracing.turn(message):
                pass

        first, second = _turn_roots(exporter)
        assert first.parent is None
        assert second.parent is not None
        assert second.parent.span_id == ConversationContext.SYNTHETIC_PARENT_SPAN_ID

    def test_starting_a_new_conversation_starts_a_new_trace(self, tracing, exporter):
        tracing.start_conversation("conv-1")
        with tracing.turn("first"):
            pass
        tracing.start_conversation("conv-2")
        with tracing.turn("second"):
            pass

        first, second = _turn_roots(exporter)
        assert first.context.trace_id != second.context.trace_id
        assert second.parent is None

    def test_work_inside_a_turn_joins_the_turn_trace(self, tracing, telemetry, exporter):
        """Spans opened during the turn nest under it, so a conversation is one trace."""
        tracing.start_conversation("conv-1")
        with tracing.turn("first") as turn:
            telemetry.otel_tracer.start_span("inner").end()
            expected = turn.span.get_span_context().trace_id

        inner = next(s for s in exporter.get_finished_spans() if s.name == "inner")
        assert inner.context.trace_id == expected


def test_conversation_turn_handles_a_missing_span():
    turn = ConversationTurn()
    turn.output = "reply"
    assert turn.output == "reply"
    assert turn.span is None
