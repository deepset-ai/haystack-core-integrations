# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime
import logging
import sys
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack.dataclasses import ChatMessage, ToolCall

from haystack_integrations.tracing.langfuse.tracer import (
    _COMPONENT_OUTPUT_KEY,
    DefaultSpanHandler,
    LangfuseSpan,
    LangfuseTracer,
    SpanContext,
    _sanitize_usage_data,
)


# Mock functions for Langfuse v3 API
def mock_get_client():
    mock_client = Mock()
    mock_client.start_as_current_span = Mock(return_value=MockContextManager())
    mock_client.start_as_current_observation = Mock(return_value=MockContextManager())
    mock_client.get_current_trace_id = Mock(return_value="mock_trace_id_123")
    return mock_client


class MockContextManager:
    """Mock context manager that simulates Langfuse v3 context managers"""

    def __init__(self, name="mock_span"):
        self._span = MockSpan(name)

    def __enter__(self):
        return self._span

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockSpan:
    def __init__(self, name="mock_span"):
        self._data = {}
        self.operation_name = name
        self._name = name
        # Make update a Mock so we can assert on it, but also make it actually work
        self.update = Mock(side_effect=self._update_data)

    def _update_data(self, **kwargs):
        """Helper method to actually update _data when update is called"""
        self._data.update(kwargs)

    def raw_span(self):
        return self

    def span(self, name=None):
        # Return a new mock span for child spans
        return MockSpan(name=name or "child_span")

    def update_trace(self, **kwargs):
        # v3 API method for updating trace-level data
        self._data.update(kwargs)

    def generation(self, name=None):
        # Return a new mock span for generation spans
        return MockSpan(name=name or "generation_span")

    def end(self):
        pass


class MockTracer:
    def trace(self, name, **kwargs):  # noqa: ARG002
        # Return a unique mock span for each trace call
        return MockSpan(name=name)

    def flush(self):
        pass


class MockLangfuseClient:
    """Mock Langfuse client that has all the required methods"""

    def __init__(self):
        self._mock_context_manager = MockContextManager()

    def start_as_current_span(self, _name=None, **_kwargs):
        return self._mock_context_manager

    def start_as_current_observation(self, _name=None, _as_type=None, **_kwargs):
        return self._mock_context_manager

    def get_current_trace_id(self):
        return "mock_trace_id_123"

    def get_current_observation_id(self):
        return "mock_observation_id_123"

    def flush(self):
        pass


class CustomSpanHandler(DefaultSpanHandler):
    def handle(self, span: LangfuseSpan, component_type: Optional[str]) -> None:
        if component_type == "OpenAIChatGenerator":
            output = span.get_data().get(_COMPONENT_OUTPUT_KEY, {})
            replies = output.get("replies", [])
            if len(replies[0].text) > 10:
                span.raw_span().update(level="WARNING", status_message="Response too long (> 10 chars)")


class TestLangfuseSpan:
    #  LangfuseSpan can be initialized with a span object
    def test_initialized_with_span_object(self):
        mock_context_manager = MockContextManager()
        span = LangfuseSpan(mock_context_manager)
        assert span.raw_span() == mock_context_manager._span

    #  set_tag method can update metadata of the span object
    def test_set_tag_updates_metadata(self):
        mock_context_manager = MockContextManager()
        span = LangfuseSpan(mock_context_manager)

        span.set_tag("key", "value")
        mock_context_manager._span.update.assert_called_once_with(metadata={"key": "value"})
        assert span._data["key"] == "value"

    #  set_content_tag method can update input and output of the span object
    def test_set_content_tag_updates_input_and_output(self):
        mock_context_manager = MockContextManager()

        span = LangfuseSpan(mock_context_manager)
        span.set_content_tag("test.input", "input_value")
        # Check that the span.update method was called with input parameter
        mock_context_manager._span.update.assert_called_with(input="input_value")

        mock_context_manager._span.update.reset_mock()
        span.set_content_tag("test.output", "output_value")
        # Check that the span.update method was called with output parameter
        mock_context_manager._span.update.assert_called_with(output="output_value")

    # set_content_tag method can update input and output of the span object with messages/replies
    def test_set_content_tag_updates_input_and_output_with_messages(self):
        mock_context_manager = MockContextManager()

        # test message input
        span = LangfuseSpan(mock_context_manager)
        span.set_content_tag("key.input", {"messages": [ChatMessage.from_user("message")]})
        assert mock_context_manager._span.update.call_count == 1
        # check we converted ChatMessage to OpenAI format
        assert mock_context_manager._span.update.call_args_list[0][1] == {
            "input": [{"role": "user", "content": "message"}]
        }
        # test replies ChatMessage list
        mock_context_manager._span.update.reset_mock()
        span.set_content_tag("key.output", {"replies": [ChatMessage.from_system("reply")]})
        assert mock_context_manager._span.update.call_count == 1
        # check we converted ChatMessage to OpenAI format
        assert mock_context_manager._span.update.call_args_list[0][1] == {
            "output": [{"role": "system", "content": "reply"}]
        }

        # test replies string list
        mock_context_manager._span.update.reset_mock()
        span.set_content_tag("key.output", {"replies": ["reply1", "reply2"]})
        assert mock_context_manager._span.update.call_count == 1
        # check we handle properly string list replies
        assert mock_context_manager._span.update.call_args_list[0][1] == {"output": ["reply1", "reply2"]}


class TestSpanContext:
    def test_post_init(self):
        with pytest.raises(ValueError):
            SpanContext(name=None, operation_name="operation_name", component_type=None, tags={}, parent_span=None)
        with pytest.raises(ValueError):
            SpanContext(name="name", operation_name=None, component_type=None, tags={}, parent_span=None)
        with pytest.raises(ValueError):
            SpanContext(
                name="name",
                operation_name="operation_name",
                component_type=None,
                tags={},
                parent_span=None,
                trace_name=None,
            )


class TestSanitizeUsageData:
    def test_anthropic_usage_flattens_and_filters(self):
        """Test Anthropic's nested dict with None and strings gets flattened and filtered"""
        usage = {
            "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "server_tool_use": None,  # Should be filtered
            "service_tier": "standard",  # Should be filtered
            "prompt_tokens": 25,
            "completion_tokens": 449,
        }
        result = _sanitize_usage_data(usage)
        assert result == {
            "cache_creation.ephemeral_1h_input_tokens": 0,
            "cache_creation.ephemeral_5m_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "prompt_tokens": 25,
            "completion_tokens": 449,
        }

    def test_openai_usage_preserved(self):
        """Test OpenAI/Cohere flat dict with only numeric values works unchanged"""
        usage = {"prompt_tokens": 29, "completion_tokens": 267, "total_tokens": 296}
        result = _sanitize_usage_data(usage)
        assert result == {"prompt_tokens": 29, "completion_tokens": 267, "total_tokens": 296}

    def test_empty_and_invalid_input(self):
        """Test edge cases return empty dict"""
        assert _sanitize_usage_data({}) == {}
        assert _sanitize_usage_data(None) == {}
        assert _sanitize_usage_data({"only_strings": "value", "only_none": None}) == {}


class TestDefaultSpanHandler:
    def test_handle_generator(self):
        mock_span = Mock()
        mock_span.raw_span.return_value = mock_span
        mock_span.get_data.return_value = {
            "haystack.component.type": "OpenAIGenerator",
            "haystack.component.output": {"replies": ["This the LLM's response"], "meta": [{"model": "test_model"}]},
        }

        handler = DefaultSpanHandler()
        handler.handle(mock_span, component_type="OpenAIGenerator")

        assert mock_span.update.call_count == 1
        assert mock_span.update.call_args_list[0][1] == {"usage_details": None, "model": "test_model"}

    def test_handle_chat_generator(self):
        mock_span = Mock()
        mock_span.raw_span.return_value = mock_span
        mock_span.get_data.return_value = {
            "haystack.component.type": "OpenAIChatGenerator",
            "haystack.component.output": {
                "replies": [
                    ChatMessage.from_assistant(
                        "This the LLM's response",
                        meta={"model": "test_model", "completion_start_time": "2021-07-27T16:02:08.012345"},
                    )
                ]
            },
        }

        handler = DefaultSpanHandler()
        handler.handle(mock_span, component_type="OpenAIChatGenerator")

        assert mock_span.update.call_count == 1
        assert mock_span.update.call_args_list[0][1] == {
            "usage_details": None,
            "model": "test_model",
            "completion_start_time": datetime.datetime(  # noqa: DTZ001
                2021, 7, 27, 16, 2, 8, 12345
            ),
        }

    def test_handle_bad_completion_start_time(self, caplog):
        mock_span = Mock()
        mock_span.raw_span.return_value = mock_span
        mock_span.get_data.return_value = {
            "haystack.component.type": "OpenAIChatGenerator",
            "haystack.component.output": {
                "replies": [
                    ChatMessage.from_assistant(
                        "This the LLM's response",
                        meta={"model": "test_model", "completion_start_time": "2021-07-32"},
                    )
                ]
            },
        }

        handler = DefaultSpanHandler()
        with caplog.at_level(logging.ERROR):
            handler.handle(mock_span, component_type="OpenAIChatGenerator")
            assert "Failed to parse completion_start_time" in caplog.text

        assert mock_span.update.call_count == 1
        assert mock_span.update.call_args_list[0][1] == {
            "usage_details": None,
            "model": "test_model",
            "completion_start_time": None,
        }

    def test_create_span_custom_chat_generator(self):
        """Test that custom chat generators create 'generation' span type."""
        mock_client = Mock()
        mock_client.start_as_current_span = Mock(return_value=MockContextManager())
        mock_client.start_as_current_observation = Mock(return_value=MockContextManager())

        handler = DefaultSpanHandler()
        handler.init_tracer(mock_client)

        context = SpanContext(
            name="MistralChatGenerator",
            operation_name="haystack.component.run",
            component_type="MistralChatGenerator",
            tags={},
            parent_span=LangfuseSpan(mock_client.start_as_current_span()),
        )

        span = handler.create_span(context)
        assert isinstance(span, LangfuseSpan)
        mock_client.start_as_current_observation.assert_called_once_with(
            name="MistralChatGenerator", as_type="generation"
        )

    def test_create_span_custom_generator(self):
        """Test that custom generators create 'generation' span type."""
        mock_client = Mock()
        mock_client.start_as_current_span = Mock(return_value=MockContextManager())
        mock_client.start_as_current_observation = Mock(return_value=MockContextManager())

        handler = DefaultSpanHandler()
        handler.init_tracer(mock_client)

        context = SpanContext(
            name="CustomAPIGenerator",
            operation_name="haystack.component.run",
            component_type="CustomAPIGenerator",
            tags={},
            parent_span=LangfuseSpan(mock_client.start_as_current_span()),
        )

        span = handler.create_span(context)
        assert isinstance(span, LangfuseSpan)
        mock_client.start_as_current_observation.assert_called_once_with(
            name="CustomAPIGenerator", as_type="generation"
        )

    def test_create_span_retriever(self):
        """Test that retrievers create 'retriever' span type."""
        mock_client = Mock()
        mock_client.start_as_current_span = Mock(return_value=MockContextManager())
        mock_client.start_as_current_observation = Mock(return_value=MockContextManager())

        handler = DefaultSpanHandler()
        handler.init_tracer(mock_client)

        context = SpanContext(
            name="InMemoryBM25Retriever",
            operation_name="haystack.component.run",
            component_type="InMemoryBM25Retriever",
            tags={},
            parent_span=LangfuseSpan(mock_client.start_as_current_span()),
        )

        span = handler.create_span(context)
        assert isinstance(span, LangfuseSpan)
        mock_client.start_as_current_observation.assert_called_once_with(
            name="InMemoryBM25Retriever", as_type="retriever"
        )

    def test_create_span_embedder(self):
        """Test that embedders create 'embedding' span type."""
        mock_client = Mock()
        mock_client.start_as_current_span = Mock(return_value=MockContextManager())
        mock_client.start_as_current_observation = Mock(return_value=MockContextManager())

        handler = DefaultSpanHandler()
        handler.init_tracer(mock_client)

        context = SpanContext(
            name="SentenceTransformersDocumentEmbedder",
            operation_name="haystack.component.run",
            component_type="SentenceTransformersDocumentEmbedder",
            tags={},
            parent_span=LangfuseSpan(mock_client.start_as_current_span()),
        )

        span = handler.create_span(context)
        assert isinstance(span, LangfuseSpan)
        mock_client.start_as_current_observation.assert_called_once_with(
            name="SentenceTransformersDocumentEmbedder", as_type="embedding"
        )

    def test_create_span_non_component(self):
        """Test that non-matching components create regular spans."""
        mock_client = Mock()
        mock_client.start_as_current_span = Mock(return_value=MockContextManager())
        mock_client.start_as_current_observation = Mock(return_value=MockContextManager())

        handler = DefaultSpanHandler()
        handler.init_tracer(mock_client)

        context = SpanContext(
            name="DocumentJoiner",
            operation_name="haystack.component.run",
            component_type="DocumentJoiner",
            tags={},
            parent_span=LangfuseSpan(mock_client.start_as_current_span()),
        )

        span = handler.create_span(context)
        assert isinstance(span, LangfuseSpan)
        # Non-matching components should use start_as_current_span, not start_as_current_observation
        mock_client.start_as_current_observation.assert_not_called()
        # Verify start_as_current_span was called for the actual span creation (not just parent)
        assert mock_client.start_as_current_span.call_count == 2  # Once for parent, once for the span


class TestCustomSpanHandler:
    def test_handle(self):
        mock_span = Mock()
        mock_span.raw_span.return_value = mock_span
        mock_span.get_data.return_value = {
            "haystack.component.type": "OpenAIChatGenerator",
            "haystack.component.output": {
                "replies": [
                    ChatMessage.from_assistant(
                        "This the LLM's response",
                        meta={"model": "test_model", "completion_start_time": "2021-07-32"},
                    )
                ]
            },
        }

        handler = CustomSpanHandler()
        handler.handle(mock_span, component_type="OpenAIChatGenerator")

        assert mock_span.update.call_count == 1
        assert mock_span.update.call_args_list[0][1] == {
            "level": "WARNING",
            "status_message": "Response too long (> 10 chars)",
        }


class TestLangfuseTracer:
    def test_initialization(self):
        langfuse_instance = Mock()
        tracer = LangfuseTracer(tracer=langfuse_instance, name="Haystack", public=True)
        assert tracer._tracer == langfuse_instance
        # Check behavioral state instead of internal _context list
        assert tracer.current_span() is None
        assert tracer._name == "Haystack"
        assert tracer._public

    def test_create_new_span(self):
        mock_raw_span = MagicMock()
        mock_raw_span.operation_name = "operation_name"
        mock_raw_span.metadata = {"tag1": "value1", "tag2": "value2"}

        with patch("haystack_integrations.tracing.langfuse.tracer.LangfuseSpan") as mock_langfuse_span:
            mock_span_instance = mock_langfuse_span.return_value
            mock_span_instance.raw_span.return_value = mock_raw_span

            mock_context_manager = MockContextManager()
            mock_context_manager._span = mock_raw_span
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value = mock_context_manager

            tracer = LangfuseTracer(tracer=mock_tracer, name="Haystack", public=False)

            # check that the trace method is called on the tracer instance with the provided operation name and tags
            with tracer.trace("operation_name", tags={"tag1": "value1", "tag2": "value2"}) as span:
                # Check that there is a current active span during tracing
                assert tracer.current_span() is not None, "There should be an active span during tracing"
                assert tracer.current_span() == span, "The current span should be the active span"
                assert span.raw_span().operation_name == "operation_name"
                assert span.raw_span().metadata == {"tag1": "value1", "tag2": "value2"}

            # Check that the span is cleaned up after tracing
            assert tracer.current_span() is None, "There should be no active span after tracing completes"

    # check that update method is called on the span instance with the provided key value pairs
    def test_update_span_with_pipeline_input_output_data(self):
        with patch("haystack_integrations.tracing.langfuse.tracer.langfuse.get_client"):
            tracer = LangfuseTracer(tracer=MockLangfuseClient(), name="Haystack", public=False)
            with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
                assert span.raw_span()._data["metadata"] == {"haystack.pipeline.input_data": "hello"}

            with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.output_data": "bye"}) as span:
                assert span.raw_span()._data["metadata"] == {"haystack.pipeline.output_data": "bye"}

    def test_trace_generation(self):
        with patch("haystack_integrations.tracing.langfuse.tracer.langfuse.get_client"):
            tracer = LangfuseTracer(tracer=MockLangfuseClient(), name="Haystack", public=False)
            tags = {
                "haystack.component.type": "OpenAIChatGenerator",
                "haystack.component.output": {
                    "replies": [
                        ChatMessage.from_assistant(
                            "", meta={"completion_start_time": "2021-07-27T16:02:08.012345", "model": "test_model"}
                        )
                    ]
                },
            }
            with tracer.trace(operation_name="operation_name", tags=tags) as span:
                ...
            assert span.raw_span()._data["usage_details"] is None
            assert span.raw_span()._data["model"] == "test_model"
            assert span.raw_span()._data["completion_start_time"] == datetime.datetime(2021, 7, 27, 16, 2, 8, 12345)  # noqa: DTZ001

    def test_handle_tool_invoker(self):
        """
        Test that the ToolInvoker span name is updated correctly with the tool names invoked for better UI/UX
        """
        mock_span = Mock()
        mock_span.raw_span.return_value = mock_span

        # Simulate data for the ToolInvoker component
        span_data = {
            "haystack.component.name": "tool_invoker",
            "haystack.component.type": "ToolInvoker",
            "haystack.component.input": {
                "messages": [
                    # Create a chat message with tool calls
                    ChatMessage.from_assistant(
                        text="Calling tools",
                        tool_calls=[
                            ToolCall(tool_name="search_tool", arguments={"query": "test"}),
                            ToolCall(tool_name="search_tool", arguments={"query": "another test"}),
                            ToolCall(tool_name="weather_tool", arguments={"location": "Berlin"}),
                        ],
                    )
                ]
            },
        }

        mock_span.get_data.return_value = span_data

        handler = DefaultSpanHandler()
        handler.handle(mock_span, component_type="ToolInvoker")

        assert mock_span.update.call_count >= 1
        name_update_call = None
        for call in mock_span.update.call_args_list:
            if "name" in call[1]:
                name_update_call = call
                break

        assert name_update_call is not None, "No call to update the span name was made"
        updated_name = name_update_call[1]["name"]

        # verify the format of the updated span name to be: `original_component_name - [list_of_tool_names]`
        assert updated_name != "tool_invoker", "Expected 'tool_invoker` to be upddated with tool names"
        assert " - " in updated_name, f"Expected ' - ' in {updated_name}"
        assert "[" in updated_name, f"Expected '[' in {updated_name}"
        assert "]" in updated_name, f"Expected ']' in {updated_name}"
        assert "tool_invoker" in updated_name, f"Expected 'tool_invoker' in {updated_name}"
        assert "search_tool (x2)" in updated_name, f"Expected 'search_tool (x2)' in {updated_name}"
        assert "weather_tool" in updated_name, f"Expected 'weather_tool' in {updated_name}"

    def test_trace_generation_invalid_start_time(self):
        with patch("haystack_integrations.tracing.langfuse.tracer.langfuse.get_client"):
            tracer = LangfuseTracer(tracer=MockLangfuseClient(), name="Haystack", public=False)
            tags = {
                "haystack.component.type": "OpenAIChatGenerator",
                "haystack.component.output": {
                    "replies": [
                        ChatMessage.from_assistant("", meta={"completion_start_time": "foobar", "model": "test_model"}),
                    ]
                },
            }
            with tracer.trace(operation_name="operation_name", tags=tags) as span:
                ...
            assert span.raw_span()._data["usage_details"] is None
            assert span.raw_span()._data["model"] == "test_model"
            assert span.raw_span()._data["completion_start_time"] is None

    def test_update_span_gets_flushed_by_default(self):
        tracer_mock = MockLangfuseClient()
        tracer_mock.flush = Mock()  # Make flush a mock for assertions

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}):
            pass

        tracer_mock.flush.assert_called_once()

    def test_update_span_flush_disable(self, monkeypatch):
        monkeypatch.setenv("HAYSTACK_LANGFUSE_ENFORCE_FLUSH", "false")
        tracer_mock = MockLangfuseClient()
        tracer_mock.flush = Mock()  # Make flush a mock for assertions

        # Re-import LangfuseTracer to ensure it picks up the new environment variable
        from haystack_integrations.tracing.langfuse.tracer import LangfuseTracer  # noqa: PLC0415

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}):
            pass

        tracer_mock.flush.assert_not_called()

    def test_context_is_empty_after_tracing(self):
        tracer_mock = MockLangfuseClient()

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}):
            pass

        # Check behavioral state instead of internal _context list
        assert tracer.current_span() is None

    def test_init_with_tracing_disabled(self, monkeypatch, caplog):
        # Clear haystack modules because ProxyTracer is initialized whenever haystack is imported
        modules_to_clear = [name for name in sys.modules if name.startswith("haystack")]
        for name in modules_to_clear:
            sys.modules.pop(name, None)

        # Re-import LangfuseTracer and instantiate it with tracing disabled
        with caplog.at_level(logging.WARNING):
            monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "false")
            from haystack_integrations.tracing.langfuse import LangfuseTracer  # noqa: PLC0415

            LangfuseTracer(tracer=MockLangfuseClient(), name="Haystack", public=False)
            assert "tracing is disabled" in caplog.text

    def test_async_concurrency_span_isolation(self):
        """
        Test that concurrent async traces maintain isolated span contexts.

        This test verifies that the context-local span stack prevents cross-request
        span interleaving in concurrent environments like FastAPI servers.
        """
        tracer = LangfuseTracer(tracer=MockLangfuseClient(), name="Haystack", public=False)

        # Track spans from each task for verification
        task1_spans = []
        task2_spans = []

        async def trace_task(task_id: str, spans_list: list):
            """Simulate a request with nested tracing operations"""
            with tracer.trace(f"outer_operation_{task_id}") as outer_span:
                spans_list.append(("outer", outer_span, tracer.current_span()))

                # Simulate some async work
                await asyncio.sleep(0.01)

                with tracer.trace(f"inner_operation_{task_id}") as inner_span:
                    spans_list.append(("inner", inner_span, tracer.current_span()))

                    # Simulate more async work
                    await asyncio.sleep(0.01)

                    # Verify nested relationship within this task
                    assert tracer.current_span() == inner_span

                # After inner span, outer should be current again
                spans_list.append(("after_inner", None, tracer.current_span()))
                assert tracer.current_span() == outer_span

            # After all spans, should be None
            spans_list.append(("after_outer", None, tracer.current_span()))
            assert tracer.current_span() is None

        async def run_concurrent_traces():
            """Run two concurrent tracing tasks"""
            await asyncio.gather(trace_task("task1", task1_spans), trace_task("task2", task2_spans))

        # Run the concurrent test
        asyncio.run(run_concurrent_traces())

        # Verify both tasks completed successfully
        assert len(task1_spans) == 4
        assert len(task2_spans) == 4

        # Verify each task had proper span isolation
        # Task 1 spans should be different from Task 2 spans
        task1_outer = task1_spans[0][1]  # outer span from task1
        task2_outer = task2_spans[0][1]  # outer span from task2
        assert task1_outer != task2_outer

        task1_inner = task1_spans[1][1]  # inner span from task1
        task2_inner = task2_spans[1][1]  # inner span from task2
        assert task1_inner != task2_inner

        # Verify proper nesting within each task
        # Task 1: outer -> inner -> outer -> None
        assert task1_spans[0][2] == task1_outer  # current_span during outer
        assert task1_spans[1][2] == task1_inner  # current_span during inner
        assert task1_spans[2][2] == task1_outer  # current_span after inner
        assert task1_spans[3][2] is None  # current_span after outer

        # Task 2: outer -> inner -> outer -> None
        assert task2_spans[0][2] == task2_outer  # current_span during outer
        assert task2_spans[1][2] == task2_inner  # current_span during inner
        assert task2_spans[2][2] == task2_outer  # current_span after inner
        assert task2_spans[3][2] is None  # current_span after outer
