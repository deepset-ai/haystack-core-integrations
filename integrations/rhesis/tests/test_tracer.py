# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import ChatMessage, ToolCall
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rhesis.sdk.telemetry.attributes import AIAttributes
from rhesis.sdk.telemetry.context import set_root_trace_id
from rhesis.telemetry.constants import ConversationContext

from haystack_integrations.tracing.rhesis.tracer import (
    DefaultSpanHandler,
    RhesisSpan,
    RhesisTelemetry,
    RhesisTracer,
    SpanContext,
    _sanitize_usage_data,
    span_stack_var,
    trace_id_var,
    tracing_context_var,
)

_AGENT_INPUT_KEY = "haystack.agent.input"
_AGENT_OUTPUT_KEY = "haystack.agent.output"

_PIPELINE_INPUT_KEY = "haystack.pipeline.input_data"
_PIPELINE_OUTPUT_KEY = "haystack.pipeline.output_data"

_COMPONENT_OUTPUT_KEY = "haystack.component.output"
_COMPONENT_INPUT_KEY = "haystack.component.input"
_COMPONENT_NAME_KEY = "haystack.component.name"


class RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.status = None
        self.exceptions: list[Any] = []
        self.name = "span"
        self._ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.events.append((name, attributes or {}))

    def set_status(self, status: Any) -> None:
        self.status = status

    def record_exception(self, exc: BaseException) -> None:
        self.exceptions.append(exc)

    def get_span_context(self) -> Any:
        mock_ctx = Mock()
        mock_ctx.trace_id = 0xABC123
        mock_ctx.span_id = 0xDEF456
        return mock_ctx

    def update_name(self, name: str) -> None:
        self.name = name

    def end(self) -> None:
        self._ended = True


class RecordingContextManager:
    def __init__(self, span: RecordingSpan | None = None) -> None:
        self._span = span or RecordingSpan()

    def __enter__(self) -> RecordingSpan:
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


@pytest.fixture
def sdk_endpoint_span_open():
    """Simulate a Rhesis SDK root span (``@endpoint``) already tracing this call."""
    set_root_trace_id("a" * 32)
    try:
        yield
    finally:
        set_root_trace_id(None)


def _make_telemetry() -> RhesisTelemetry:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_tracer = provider.get_tracer("test")
    return RhesisTelemetry(
        provider=provider,
        otel_tracer=otel_tracer,
        project_id="proj-1",
        environment="test",
        base_url="http://localhost:8080",
        frontend_url="http://localhost:3000",
    )


class TestSanitizeUsageData:
    def test_openai_usage(self):
        result = _sanitize_usage_data({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
        assert result[AIAttributes.LLM_TOKENS_INPUT] == 10
        assert result[AIAttributes.LLM_TOKENS_OUTPUT] == 5
        assert result[AIAttributes.LLM_TOKENS_TOTAL] == 15

    def test_invalid_usage(self):
        assert _sanitize_usage_data({}) == {}
        assert _sanitize_usage_data(None) == {}


class TestRhesisSpan:
    def test_set_tag(self):
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span.set_tag("haystack.component.type", "OpenAIChatGenerator")
        assert recording.attributes["haystack.component.type"] == "OpenAIChatGenerator"
        assert span.get_data()["haystack.component.type"] == "OpenAIChatGenerator"

    def test_set_content_tag_gated(self):
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        with patch("haystack_integrations.tracing.rhesis.tracer.proxy_tracer.is_content_tracing_enabled", False):
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("hi")]})
        assert not recording.events

    def test_set_content_tag_messages(self):
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        with patch("haystack_integrations.tracing.rhesis.tracer.proxy_tracer.is_content_tracing_enabled", True):
            span.set_content_tag("haystack.component.input", {"messages": [ChatMessage.from_user("hello")]})
        assert recording.events
        assert recording.events[0][0] == "ai.prompt"

    def test_correlation_data(self):
        span = RhesisSpan(RecordingContextManager())
        data = span.get_correlation_data_for_logs()
        assert "trace_id" in data
        assert "span_id" in data


class TestDefaultSpanHandler:
    def test_create_root_pipeline_span(self):
        telemetry = _make_telemetry()
        handler = DefaultSpanHandler()
        handler.init_tracer(telemetry)

        with patch.object(telemetry.otel_tracer, "start_as_current_span") as mock_start:
            mock_start.return_value = RecordingContextManager()
            context = SpanContext(
                name="haystack.pipeline.run",
                operation_name="haystack.pipeline.run",
                component_type=None,
                tags={},
                parent_span=None,
                trace_name="My Pipeline",
                is_root=True,
            )
            token = tracing_context_var.set({"session_id": "sess-1"})
            try:
                span = handler.create_span(context)
                assert isinstance(span, RhesisSpan)
                mock_start.assert_called_once()
                assert mock_start.call_args.kwargs["name"] == "function.haystack.pipeline.run"
            finally:
                tracing_context_var.reset(token)

    def test_root_pipeline_span_claims_turn_when_standalone(self):
        telemetry = _make_telemetry()
        handler = DefaultSpanHandler()
        handler.init_tracer(telemetry)

        with patch.object(telemetry.otel_tracer, "start_as_current_span") as mock_start:
            recording = RecordingSpan()
            mock_start.return_value = RecordingContextManager(recording)
            context = SpanContext(
                name="haystack.pipeline.run",
                operation_name="haystack.pipeline.run",
                component_type=None,
                tags={},
                parent_span=None,
                is_root=True,
            )
            token = tracing_context_var.set({"session_id": "sess-1"})
            try:
                span = handler.create_span(context)
            finally:
                tracing_context_var.reset(token)

        assert span.owns_conversation_turn is True
        assert recording.attributes[ConversationContext.SpanAttributes.IS_TURN_ROOT] is True
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_ID] == "sess-1"

    def test_root_pipeline_span_yields_turn_to_sdk_endpoint(self, sdk_endpoint_span_open):
        telemetry = _make_telemetry()
        handler = DefaultSpanHandler()
        handler.init_tracer(telemetry)

        with patch.object(telemetry.otel_tracer, "start_as_current_span") as mock_start:
            recording = RecordingSpan()
            mock_start.return_value = RecordingContextManager(recording)
            context = SpanContext(
                name="haystack.pipeline.run",
                operation_name="haystack.pipeline.run",
                component_type=None,
                tags={},
                parent_span=None,
                is_root=True,
            )
            token = tracing_context_var.set({"session_id": "sess-1"})
            try:
                span = handler.create_span(context)
            finally:
                tracing_context_var.reset(token)

        # No turn-root flag: the exporter would strip this span's real parent and the Haystack
        # subtree would surface as a second conversation turn.
        assert span.owns_conversation_turn is False
        assert ConversationContext.SpanAttributes.IS_TURN_ROOT not in recording.attributes
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_ID] == "sess-1"
        assert recording.attributes[AIAttributes.SESSION_ID] == "sess-1"

    def test_handle_pipeline_io(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _PIPELINE_INPUT_KEY: {"chat": {"messages": [ChatMessage.from_user("hello")]}},
            _PIPELINE_OUTPUT_KEY: {"chat": {"messages": [ChatMessage.from_assistant("world")]}},
        }
        handler.handle(span, component_type=None)
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_INPUT] == "hello"
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_OUTPUT] == "world"

    def test_handle_pipeline_io_skipped_when_sdk_owns_turn(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span.is_root = True
        span.owns_conversation_turn = False
        span._data = {
            "haystack.pipeline.input_data": {"query": "hello"},
            "haystack.pipeline.output_data": {"answer": "world"},
        }
        token = tracing_context_var.set({"session_id": "sess-1"})
        try:
            handler.handle(span, component_type=None)
        finally:
            tracing_context_var.reset(token)

        # The SDK endpoint span carries the mapped user message and reply; restating them here as raw
        # pipeline dumps is what produced duplicate, unreadable turns.
        assert ConversationContext.SpanAttributes.CONVERSATION_INPUT not in recording.attributes
        assert ConversationContext.SpanAttributes.CONVERSATION_OUTPUT not in recording.attributes
        assert ConversationContext.SpanAttributes.IS_TURN_ROOT not in recording.attributes

    def test_handle_agent_conversation_io(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span.is_root = True
        span._data = {
            _AGENT_INPUT_KEY: {"messages": [ChatMessage.from_user("Plan a trip to Rome")]},
            _AGENT_OUTPUT_KEY: {"messages": [ChatMessage.from_assistant("Here is your Rome itinerary.")]},
        }
        token = tracing_context_var.set({"session_id": "sess-agent"})
        stack_token = span_stack_var.set([RhesisSpan(RecordingContextManager(RecordingSpan())), span])
        try:
            handler.handle(span, component_type="Agent")
        finally:
            span_stack_var.reset(stack_token)
            tracing_context_var.reset(token)

        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_INPUT] == ("Plan a trip to Rome")
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_OUTPUT] == (
            "Here is your Rome itinerary."
        )
        assert recording.attributes[ConversationContext.SpanAttributes.IS_TURN_ROOT] is True
        assert recording.attributes[ConversationContext.SpanAttributes.CONVERSATION_ID] == "sess-agent"

    def test_handle_pipeline_conversation_io_extracts_message_text(self):
        """A pipeline root owning the turn must publish text, not a serialized payload."""
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span.is_root = True
        span._data = {
            _PIPELINE_INPUT_KEY: {
                "chat": {
                    "messages": [ChatMessage.from_user("Plan a trip to Rome")],
                    "slots": {"city": None},
                }
            },
            _PIPELINE_OUTPUT_KEY: {
                "chat": {
                    "messages": [
                        ChatMessage.from_assistant("Here is your Rome itinerary."),
                    ],
                    "tool_call_counts": {"search": 1},
                }
            },
        }
        token = tracing_context_var.set({"session_id": "sess-pipeline"})
        try:
            handler.handle(span, component_type=None)
        finally:
            tracing_context_var.reset(token)

        attrs = ConversationContext.SpanAttributes
        assert recording.attributes[attrs.CONVERSATION_INPUT] == "Plan a trip to Rome"
        assert recording.attributes[attrs.CONVERSATION_OUTPUT] == "Here is your Rome itinerary."
        assert recording.attributes[attrs.IS_TURN_ROOT] is True
        assert recording.attributes[attrs.CONVERSATION_ID] == "sess-pipeline"

    def test_handle_pipeline_conversation_io_prefers_last_message(self):
        """An Agent inside the pipeline reports its closing turn as `last_message`."""
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _PIPELINE_INPUT_KEY: {"agent": {"messages": [ChatMessage.from_user("hello")]}},
            _PIPELINE_OUTPUT_KEY: {"agent": {"last_message": ChatMessage.from_assistant("Hi there.")}},
        }
        token = tracing_context_var.set({"session_id": "sess-last"})
        try:
            handler.handle(span, component_type=None)
        finally:
            tracing_context_var.reset(token)

        attrs = ConversationContext.SpanAttributes
        assert recording.attributes[attrs.CONVERSATION_INPUT] == "hello"
        assert recording.attributes[attrs.CONVERSATION_OUTPUT] == "Hi there."

    def test_handle_pipeline_conversation_io_skips_tool_call_only_turns(self):
        """An assistant message that only requests tools has no text worth showing."""
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _PIPELINE_INPUT_KEY: {"agent": {"messages": [ChatMessage.from_user("hello")]}},
            _PIPELINE_OUTPUT_KEY: {
                "agent": {
                    "messages": [
                        ChatMessage.from_assistant("The real reply."),
                        ChatMessage.from_assistant(tool_calls=[ToolCall(id="1", tool_name="search", arguments={})]),
                    ]
                }
            },
        }
        token = tracing_context_var.set({"session_id": "sess-tools"})
        try:
            handler.handle(span, component_type=None)
        finally:
            tracing_context_var.reset(token)

        attrs = ConversationContext.SpanAttributes
        assert recording.attributes[attrs.CONVERSATION_OUTPUT] == "The real reply."

    def test_handle_pipeline_conversation_io_stamps_nothing_without_messages(self):
        """No chat history means no conversation text: a dict dump is never a valid rendering."""
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _PIPELINE_INPUT_KEY: {"ranker": {"query": "rome", "top_k": 5}},
            _PIPELINE_OUTPUT_KEY: {"ranker": {"documents": ["a", "b"]}},
        }
        token = tracing_context_var.set({"session_id": "sess-nodocs"})
        try:
            handler.handle(span, component_type=None)
        finally:
            tracing_context_var.reset(token)

        attrs = ConversationContext.SpanAttributes
        assert attrs.CONVERSATION_INPUT not in recording.attributes
        assert attrs.CONVERSATION_OUTPUT not in recording.attributes

    def test_handle_agent_conversation_io_skipped_when_sdk_owns_turn(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span.owns_conversation_turn = False
        span._data = {
            _AGENT_INPUT_KEY: {"messages": [ChatMessage.from_user("Plan a trip to Rome")]},
            _AGENT_OUTPUT_KEY: {"messages": [ChatMessage.from_assistant("Here is your Rome itinerary.")]},
        }
        token = tracing_context_var.set({"session_id": "sess-agent"})
        stack_token = span_stack_var.set([RhesisSpan(RecordingContextManager(RecordingSpan())), span])
        try:
            handler.handle(span, component_type="Agent")
        finally:
            span_stack_var.reset(stack_token)
            tracing_context_var.reset(token)

        assert ConversationContext.SpanAttributes.CONVERSATION_INPUT not in recording.attributes
        assert ConversationContext.SpanAttributes.CONVERSATION_OUTPUT not in recording.attributes
        assert ConversationContext.SpanAttributes.IS_TURN_ROOT not in recording.attributes

    def test_handle_chat_generator(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _COMPONENT_OUTPUT_KEY: {
                "replies": [
                    ChatMessage.from_assistant(
                        "answer",
                        meta={"model": "gpt-4o-mini", "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
                    )
                ]
            }
        }
        handler.handle(span, component_type="OpenAIChatGenerator")
        assert recording.attributes[AIAttributes.MODEL_NAME] == "gpt-4o-mini"
        assert recording.attributes[AIAttributes.LLM_TOKENS_INPUT] == 3

    def test_handle_agent_step_llm_promotes_model_and_tokens(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording), operation_name="haystack.agent.step.llm")
        span._data = {
            "haystack.agent.step.llm.output": {
                "replies": [
                    ChatMessage.from_assistant(
                        "answer",
                        meta={"model": "gemini-flash", "usage": {"prompt_tokens": 4, "completion_tokens": 6}},
                    )
                ]
            }
        }
        handler.handle(span, component_type=None)
        assert recording.attributes[AIAttributes.MODEL_NAME] == "gemini-flash"
        assert recording.attributes[AIAttributes.LLM_TOKENS_INPUT] == 4

    def test_create_agent_step_tool_span_sets_tool_name(self):
        telemetry = _make_telemetry()
        handler = DefaultSpanHandler()
        handler.init_tracer(telemetry)

        with patch.object(telemetry.otel_tracer, "start_as_current_span") as mock_start:
            recording = RecordingSpan()
            mock_start.return_value = RecordingContextManager(recording)
            context = SpanContext(
                name="haystack.agent.step.tool",
                operation_name="haystack.agent.step.tool",
                component_type=None,
                tags={"haystack.tool.name": "research", "haystack.tool.description": "Delegate research"},
                parent_span=None,
                is_root=False,
            )
            span = handler.create_span(context)
            assert isinstance(span, RhesisSpan)
            assert mock_start.call_args.kwargs["name"] == "ai.tool.invoke"
            assert recording.attributes[AIAttributes.TOOL_NAME] == "research"
            assert recording.attributes[AIAttributes.TOOL_TYPE] == "haystack"
            assert recording.attributes[AIAttributes.OPERATION_TYPE] == AIAttributes.OPERATION_TOOL_INVOKE

    def test_nested_agent_promotes_parent_tool_to_handoff(self):
        telemetry = _make_telemetry()
        handler = DefaultSpanHandler()
        handler.init_tracer(telemetry)

        tool_recording = RecordingSpan()
        tool_recording.name = "ai.tool.invoke"
        tool_span = RhesisSpan(RecordingContextManager(tool_recording), operation_name="haystack.agent.step.tool")
        tool_span._data = {"haystack.tool.name": "research"}
        tool_recording.attributes[AIAttributes.TOOL_NAME] = "research"

        component_recording = RecordingSpan()
        component_span = RhesisSpan(
            RecordingContextManager(component_recording), operation_name="haystack.component.run"
        )
        component_span._data = {"haystack.component.name": "coordinator"}

        stack_token = span_stack_var.set([component_span, tool_span])
        try:
            with patch.object(telemetry.otel_tracer, "start_as_current_span") as mock_start:
                agent_recording = RecordingSpan()
                mock_start.return_value = RecordingContextManager(agent_recording)
                context = SpanContext(
                    name="haystack.agent.run",
                    operation_name="haystack.agent.run",
                    component_type=None,
                    tags={},
                    parent_span=tool_span,
                    is_root=False,
                )
                handler.create_span(context)
                assert tool_recording.name == "ai.agent.handoff"
                assert tool_recording.attributes[AIAttributes.OPERATION_TYPE] == AIAttributes.OPERATION_AGENT_HANDOFF
                assert tool_recording.attributes[AIAttributes.AGENT_HANDOFF_TO] == "research"
                assert tool_recording.attributes[AIAttributes.AGENT_HANDOFF_FROM] == "coordinator"
                assert agent_recording.attributes[AIAttributes.AGENT_NAME] == "research"
        finally:
            span_stack_var.reset(stack_token)

    def test_handle_tool_invoker_rename(self):
        handler = DefaultSpanHandler()
        recording = RecordingSpan()
        span = RhesisSpan(RecordingContextManager(recording))
        span._data = {
            _COMPONENT_NAME_KEY: "tools",
            _COMPONENT_INPUT_KEY: {
                "messages": [
                    ChatMessage.from_assistant(
                        tool_calls=[ToolCall(id="1", tool_name="search", arguments={})],
                    ),
                    ChatMessage.from_assistant(
                        tool_calls=[
                            ToolCall(id="2", tool_name="search", arguments={}),
                            ToolCall(id="3", tool_name="calc", arguments={}),
                        ],
                    ),
                ]
            },
        }
        handler.handle(span, component_type="ToolInvoker")
        assert "search (x2)" in recording.name
        assert "calc" in recording.name


class TestRhesisTracer:
    def test_span_stack_isolation(self):
        telemetry = _make_telemetry()
        tracer = RhesisTracer(telemetry=telemetry, name="test")
        tracer.enforce_flush = False

        seen: list[str] = []

        def fake_create(context: SpanContext) -> RhesisSpan:
            recording = RecordingSpan()
            recording.name = context.name
            return RhesisSpan(RecordingContextManager(recording))

        tracer._span_handler.create_span = fake_create  # type: ignore[method-assign]
        tracer._span_handler.handle = lambda span, component_type: seen.append(component_type)  # type: ignore

        with tracer.trace("outer", tags={"haystack.component.name": "outer"}):
            assert tracer.current_span() is not None
            with tracer.trace("inner", tags={"haystack.component.name": "inner", "haystack.component.type": "Gen"}):
                assert tracer.current_span() is not None
            assert tracer.current_span() is not None
        assert tracer.current_span() is None

    def test_exception_records_error_and_reraises(self):
        telemetry = _make_telemetry()
        tracer = RhesisTracer(telemetry=telemetry, name="test")
        tracer.enforce_flush = False
        recording = RecordingSpan()

        tracer._span_handler.create_span = lambda ctx: RhesisSpan(RecordingContextManager(recording))  # type: ignore
        tracer._span_handler.handle = lambda span, ct: None  # type: ignore

        with pytest.raises(RuntimeError, match="boom"):
            with tracer.trace("op"):
                raise RuntimeError("boom")

        assert recording.exceptions
        assert recording.attributes.get(AIAttributes.ERROR_TYPE) == "RuntimeError"

    def test_enforce_flush(self):
        telemetry = _make_telemetry()
        telemetry.flush = Mock()
        tracer = RhesisTracer(telemetry=telemetry, name="test")
        tracer.enforce_flush = True
        tracer._span_handler.create_span = lambda ctx: RhesisSpan(RecordingContextManager())  # type: ignore
        tracer._span_handler.handle = lambda span, ct: None  # type: ignore

        with tracer.trace("op"):
            pass
        telemetry.flush.assert_called_once()

    def test_no_flush_when_disabled(self):
        telemetry = _make_telemetry()
        telemetry.flush = Mock()
        tracer = RhesisTracer(telemetry=telemetry, name="test")
        tracer.enforce_flush = False
        tracer._span_handler.create_span = lambda ctx: RhesisSpan(RecordingContextManager())  # type: ignore
        tracer._span_handler.handle = lambda span, ct: None  # type: ignore

        with tracer.trace("op"):
            pass
        telemetry.flush.assert_not_called()

    def test_get_trace_url(self):
        telemetry = _make_telemetry()
        tracer = RhesisTracer(telemetry=telemetry, name="test")
        token = trace_id_var.set("abc123")
        try:
            url = tracer.get_trace_url()
        finally:
            trace_id_var.reset(token)
        assert "open_trace=abc123" in url
        assert "project_id=proj-1" in url

    def test_trace_id_is_context_local_across_concurrent_root_spans(self):
        """
        Two concurrent root spans through *one* tracer each report their own trace id.

        The connector installs a single tracer process-wide, so instance state here would let one
        request return another request's trace id and deep link. test_concurrent_span_stacks below
        uses two tracer instances and so cannot reach this case.
        """
        telemetry = _make_telemetry()
        tracer = RhesisTracer(telemetry=telemetry, name="shared")
        tracer.enforce_flush = False

        async def run_trace(label: str) -> tuple[str, str]:
            with tracer.trace(label, tags={"haystack.component.name": label}) as span:
                await asyncio.sleep(0.01)
                # Read it back the way RhesisConnector.run() does, while the root span is open.
                observed = tracer.get_trace_id()
                actual = format(span.raw_span().get_span_context().trace_id, "032x")
                return observed, actual

        async def main() -> list[tuple[str, str]]:
            return await asyncio.gather(run_trace("a"), run_trace("b"))

        (observed_a, actual_a), (observed_b, actual_b) = asyncio.run(main())

        assert observed_a == actual_a
        assert observed_b == actual_b
        assert observed_a != observed_b
        # And nothing leaks out of the runs that produced it.
        assert tracer.get_trace_id() == ""

    def test_concurrent_span_stacks(self):
        telemetry = _make_telemetry()
        tracer_a = RhesisTracer(telemetry=telemetry, name="a")
        tracer_b = RhesisTracer(telemetry=telemetry, name="b")
        tracer_a.enforce_flush = False
        tracer_b.enforce_flush = False
        tracer_a._span_handler.create_span = lambda ctx: RhesisSpan(RecordingContextManager())  # type: ignore
        tracer_b._span_handler.create_span = lambda ctx: RhesisSpan(RecordingContextManager())  # type: ignore
        tracer_a._span_handler.handle = lambda span, ct: None  # type: ignore
        tracer_b._span_handler.handle = lambda span, ct: None  # type: ignore

        async def run_trace(tr: RhesisTracer, label: str) -> str | None:
            with tr.trace(label, tags={"haystack.component.name": label}):
                await asyncio.sleep(0.01)
                current = tr.current_span()
                return current.raw_span().name if current else None

        async def main() -> tuple[str | None, str | None]:
            return await asyncio.gather(run_trace(tracer_a, "a"), run_trace(tracer_b, "b"))

        result_a, result_b = asyncio.run(main())
        assert tracer_a.current_span() is None
        assert tracer_b.current_span() is None
