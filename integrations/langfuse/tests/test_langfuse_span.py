import os

os.environ["HAYSTACK_CONTENT_TRACING_ENABLED"] = "true"

from unittest.mock import Mock
from haystack.dataclasses import ChatMessage
from haystack_integrations.tracing.langfuse.tracer import LangfuseSpan


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
