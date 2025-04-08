# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import sys
from unittest.mock import MagicMock, Mock, patch
from typing import Optional

import pytest
from haystack.dataclasses import ChatMessage
from haystack_integrations.tracing.langfuse.tracer import LangfuseTracer, LangfuseSpan, SpanContext, DefaultSpanHandler
from haystack_integrations.tracing.langfuse.tracer import _COMPONENT_OUTPUT_KEY


class MockSpan:
    def __init__(self):
        self._data = {}
        self._span = self
        self.operation_name = "operation_name"

    def raw_span(self):
        return self

    def span(self, name=None):
        # assert correct operation name passed to the span
        assert name == "operation_name"
        return self

    def update(self, **kwargs):
        self._data.update(kwargs)

    def generation(self, name=None):
        return self

    def end(self):
        pass


class MockTracer:

    def trace(self, name, **kwargs):
        return MockSpan()

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
        mock_span = Mock()
        span = LangfuseSpan(mock_span)
        assert span.raw_span() == mock_span

    #  set_tag method can update metadata of the span object
    def test_set_tag_updates_metadata(self):
        mock_span = Mock()
        span = LangfuseSpan(mock_span)

        span.set_tag("key", "value")
        mock_span.update.assert_called_once_with(metadata={"key": "value"})
        assert span._data["key"] == "value"

    #  set_content_tag method can update input and output of the span object
    def test_set_content_tag_updates_input_and_output(self):
        mock_span = Mock()

        span = LangfuseSpan(mock_span)
        span.set_content_tag("input_key", "input_value")
        assert span._data["input_key"] == "input_value"

        mock_span.reset_mock()
        span.set_content_tag("output_key", "output_value")
        assert span._data["output_key"] == "output_value"

    # set_content_tag method can update input and output of the span object with messages/replies
    def test_set_content_tag_updates_input_and_output_with_messages(self):
        mock_span = Mock()

        # test message input
        span = LangfuseSpan(mock_span)
        span.set_content_tag("key.input", {"messages": [ChatMessage.from_user("message")]})
        assert mock_span.update.call_count == 1
        # check we converted ChatMessage to OpenAI format
        assert mock_span.update.call_args_list[0][1] == {"input": [{"role": "user", "content": "message"}]}
        assert span._data["key.input"] == {"messages": [ChatMessage.from_user("message")]}

        # test replies ChatMessage list
        mock_span.reset_mock()
        span.set_content_tag("key.output", {"replies": [ChatMessage.from_system("reply")]})
        assert mock_span.update.call_count == 1
        # check we converted ChatMessage to OpenAI format
        assert mock_span.update.call_args_list[0][1] == {"output": [{"role": "system", "content": "reply"}]}
        assert span._data["key.output"] == {"replies": [ChatMessage.from_system("reply")]}

        # test replies string list
        mock_span.reset_mock()
        span.set_content_tag("key.output", {"replies": ["reply1", "reply2"]})
        assert mock_span.update.call_count == 1
        # check we handle properly string list replies
        assert mock_span.update.call_args_list[0][1] == {"output": ["reply1", "reply2"]}
        assert span._data["key.output"] == {"replies": ["reply1", "reply2"]}


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
        assert mock_span.update.call_args_list[0][1] == {"usage": None, "model": "test_model"}

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
            "usage": None,
            "model": "test_model",
            "completion_start_time": datetime.datetime(2021, 7, 27, 16, 2, 8, 12345),
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
            "usage": None,
            "model": "test_model",
            "completion_start_time": None,
        }


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
        assert tracer._context == []
        assert tracer._name == "Haystack"
        assert tracer._public

    def test_create_new_span(self):
        mock_raw_span = MagicMock()
        mock_raw_span.operation_name = "operation_name"
        mock_raw_span.metadata = {"tag1": "value1", "tag2": "value2"}

        with patch("haystack_integrations.tracing.langfuse.tracer.LangfuseSpan") as MockLangfuseSpan:
            mock_span_instance = MockLangfuseSpan.return_value
            mock_span_instance.raw_span.return_value = mock_raw_span

            mock_context_manager = MagicMock()
            mock_context_manager.__enter__.return_value = mock_span_instance

            mock_tracer = MagicMock()
            mock_tracer.trace.return_value = mock_context_manager

            tracer = LangfuseTracer(tracer=mock_tracer, name="Haystack", public=False)

            # check that the trace method is called on the tracer instance with the provided operation name and tags
            with tracer.trace("operation_name", tags={"tag1": "value1", "tag2": "value2"}) as span:
                assert len(tracer._context) == 1, "The trace span should have been added to the the root context span"
                assert span.raw_span().operation_name == "operation_name"
                assert span.raw_span().metadata == {"tag1": "value1", "tag2": "value2"}

            assert (
                len(tracer._context) == 0
            ), "The trace span should have been popped, and the root span is closed as well"

    # check that update method is called on the span instance with the provided key value pairs
    def test_update_span_with_pipeline_input_output_data(self):
        tracer = LangfuseTracer(tracer=MockTracer(), name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
            assert span.raw_span()._data["metadata"] == {"haystack.pipeline.input_data": "hello"}

        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.output_data": "bye"}) as span:
            assert span.raw_span()._data["metadata"] == {"haystack.pipeline.output_data": "bye"}

    def test_trace_generation(self):
        tracer = LangfuseTracer(tracer=MockTracer(), name="Haystack", public=False)
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
        assert span.raw_span()._data["usage"] is None
        assert span.raw_span()._data["model"] == "test_model"
        assert span.raw_span()._data["completion_start_time"] == datetime.datetime(2021, 7, 27, 16, 2, 8, 12345)

    def test_trace_generation_invalid_start_time(self):
        tracer = LangfuseTracer(tracer=MockTracer(), name="Haystack", public=False)
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
        assert span.raw_span()._data["usage"] is None
        assert span.raw_span()._data["model"] == "test_model"
        assert span.raw_span()._data["completion_start_time"] is None

    def test_update_span_gets_flushed_by_default(self):
        tracer_mock = Mock()

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
            pass

        tracer_mock.flush.assert_called_once()

    def test_update_span_flush_disable(self, monkeypatch):
        monkeypatch.setenv("HAYSTACK_LANGFUSE_ENFORCE_FLUSH", "false")
        tracer_mock = Mock()

        from haystack_integrations.tracing.langfuse.tracer import LangfuseTracer

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
            pass

        tracer_mock.flush.assert_not_called()

    def test_context_is_empty_after_tracing(self):
        tracer_mock = Mock()

        tracer = LangfuseTracer(tracer=tracer_mock, name="Haystack", public=False)
        with tracer.trace(operation_name="operation_name", tags={"haystack.pipeline.input_data": "hello"}) as span:
            pass

        assert tracer._context == []

    def test_init_with_tracing_disabled(self, monkeypatch, caplog):
        # Clear haystack modules because ProxyTracer is initialized whenever haystack is imported
        modules_to_clear = [name for name in sys.modules if name.startswith('haystack')]
        for name in modules_to_clear:
            sys.modules.pop(name, None)

        # Re-import LangfuseTracer and instantiate it with tracing disabled
        with caplog.at_level(logging.WARNING):
            monkeypatch.setenv("HAYSTACK_CONTENT_TRACING_ENABLED", "false")
            from haystack_integrations.tracing.langfuse import LangfuseTracer

            LangfuseTracer(tracer=MockTracer(), name="Haystack", public=False)
            assert "tracing is disabled" in caplog.text
