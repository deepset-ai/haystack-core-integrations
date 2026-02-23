import base64
import json

import pytest
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    SignatureDelta,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FileContent,
    ImageContent,
    StreamingChunk,
    TextContent,
    ToolCall,
    ToolCallDelta,
)
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
    AnthropicChatGenerator,
)
from haystack_integrations.components.generators.anthropic.chat.utils import (
    FINISH_REASON_MAPPING,
    _convert_anthropic_chunk_to_streaming_chunk,
    _convert_chat_completion_to_chat_message,
    _convert_file_content_to_anthropic_format,
    _convert_image_content_to_anthropic_format,
    _convert_messages_to_anthropic_format,
    _finalize_reasoning_group,
)


class TestUtils:
    def test_convert_chat_completion_to_chat_message(self, mock_chat_completion):
        """
        Test converting Anthropic chat completion to ChatMessage
        """
        chat_completion = mock_chat_completion.return_value

        chat_message = _convert_chat_completion_to_chat_message(chat_completion, ignore_tools_thinking_messages=True)
        assert chat_message.text == "Hello, world!"
        assert chat_message.role == "assistant"
        assert chat_message.meta["model"] == "claude-sonnet-4-5"
        assert "usage" in chat_message.meta
        assert chat_message.meta["usage"]["prompt_tokens"] == 57
        assert chat_message.meta["usage"]["completion_tokens"] == 40

    def test_convert_chat_completion_to_chat_message_with_reasoning_and_tool_call(self):
        """
        Test converting Anthropic chat completion to ChatMessage
        """
        chat_completion = Message(
            id="msg_01MZF",
            content=[
                ThinkingBlock(signature="sign1", thinking="User has asked 2 questions", type="thinking"),
                TextBlock(citations=None, text="I'll provide the answers!", type="text"),
                ToolUseBlock(
                    id="toolu_01XEkx", input={"expression": "7 * (4 + 2)"}, name="calculator", type="tool_use"
                ),
            ],
            model="claude-sonnet-4-5",
            role="assistant",
            stop_reason="tool_use",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=507, output_tokens=219),
        )
        chat_message = _convert_chat_completion_to_chat_message(chat_completion, ignore_tools_thinking_messages=False)
        assert chat_message.text == "I'll provide the answers!"
        assert chat_message.reasoning.reasoning_text == "User has asked 2 questions"
        assert chat_message.reasoning.extra == {
            "reasoning_contents": [
                {"reasoning_content": {"reasoning_text": {"text": "User has asked 2 questions", "signature": "sign1"}}}
            ]
        }
        assert chat_message.meta["model"] == "claude-sonnet-4-5"
        assert chat_message.meta["finish_reason"] == "tool_calls"
        assert "usage" in chat_message.meta
        assert chat_message.meta["usage"]["prompt_tokens"] == 507
        assert chat_message.meta["usage"]["completion_tokens"] == 219

    def test_convert_anthropic_completion_chunks_with_multiple_tool_calls_and_reasoning_to_streaming_chunks(self):
        """
        Test converting Anthropic stream events with tools to Haystack StreamingChunks
        """
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        component_info = ComponentInfo.from_component(component)

        raw_chunks = []

        # Test message_start chunk
        message_start_chunk = RawMessageStartEvent(
            message=Message(
                id="msg_01ApGaijiGeLtxWLCKUKELfT",
                content=[],
                model="claude-sonnet-4-5",
                role="assistant",
                stop_reason=None,
                stop_sequence=None,
                type="message",
                usage=Usage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    input_tokens=393,
                    output_tokens=3,
                    server_tool_use=None,
                    service_tier="standard",
                ),
            ),
            type="message_start",
        )
        raw_chunks.append(message_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta["message"]["model"] == message_start_chunk.message.model
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""

        # Test content_block_start for reasoning
        reasoning_block_start_chunk = RawContentBlockStartEvent(
            content_block=ThinkingBlock(type="thinking", signature="", thinking=""), index=0, type="content_block_start"
        )
        raw_chunks.append(reasoning_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 0
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""
        assert streaming_chunk.reasoning is not None
        assert streaming_chunk.reasoning.reasoning_text == ""

        # Test content_block_delta for reasoning
        reasoning_delta_chunk = RawContentBlockDeltaEvent(
            delta=ThinkingDelta(thinking="The user is asking 2 questions.", type="thinking_delta"),
            index=0,
            type="content_block_delta",
        )
        raw_chunks.append(reasoning_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_delta_chunk.model_dump()
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 0
        assert streaming_chunk.tool_calls is None
        assert not streaming_chunk.start
        assert streaming_chunk.reasoning is not None
        assert streaming_chunk.reasoning.reasoning_text == "The user is asking 2 questions."

        # Test content_block_delta for reasoning signature
        reasoning_signature_delta_chunk = RawContentBlockDeltaEvent(
            delta=SignatureDelta(signature="1234567890", type="signature_delta"),
            index=0,
            type="content_block_delta",
        )
        raw_chunks.append(reasoning_signature_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_signature_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_signature_delta_chunk.model_dump()
        assert streaming_chunk.content == ""
        assert streaming_chunk.reasoning is not None
        assert streaming_chunk.reasoning.extra.get("signature") == "1234567890"

        # Test content_block_start for text
        text_block_start_chunk = RawContentBlockStartEvent(
            content_block=TextBlock(citations=None, text="", type="text"), index=1, type="content_block_start"
        )
        raw_chunks.append(text_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            text_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == text_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 1
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""

        # Test content_block_delta with text_delta
        text_delta_chunk = RawContentBlockDeltaEvent(
            delta=TextDelta(text="I'll calculate the factorial of 5", type="text_delta"),
            index=1,
            type="content_block_delta",
        )
        raw_chunks.append(text_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            text_delta_chunk, component_info=component_info, tool_call_index=0
        )

        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == text_delta_chunk.model_dump()
        assert streaming_chunk.content == text_delta_chunk.delta.text
        assert not streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 1
        assert streaming_chunk.tool_calls is None

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Test content_block_start for tool_use
        tool_block_start_chunk = RawContentBlockStartEvent(
            content_block=ToolUseBlock(
                id="toolu_011dE5KDKxSh6hi85EnRKZT3", input={}, name="calculator", type="tool_use"
            ),
            index=2,
            type="content_block_start",
        )
        raw_chunks.append(tool_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            tool_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == tool_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=0, id="toolu_011dE5KDKxSh6hi85EnRKZT3", tool_name="calculator", arguments=None)
        ]
        assert streaming_chunk.content == ""

        # Test content_block_delta with input_json_delta (empty)
        empty_json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json="", type="input_json_delta"), index=2, type="content_block_delta"
        )
        raw_chunks.append(empty_json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            empty_json_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == empty_json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=0, arguments=empty_json_delta_chunk.delta.partial_json)
        ]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert not streaming_chunk.start

        # Test content_block_delta with input_json_delta (with content)
        json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json='{"expression": 5 ', type="input_json_delta"),
            index=2,
            type="content_block_delta",
        )
        raw_chunks.append(json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            json_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [ToolCallDelta(index=0, arguments=json_delta_chunk.delta.partial_json)]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert not streaming_chunk.start

        # Test message_delta chunk
        message_delta_chunk = RawMessageDeltaEvent(
            delta=Delta(stop_reason="tool_use", stop_sequence=None),
            type="message_delta",
            usage=MessageDeltaUsage(
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                input_tokens=None,
                output_tokens=77,
                server_tool_use=None,
            ),
        )
        raw_chunks.append(message_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == message_delta_chunk.model_dump()
        assert streaming_chunk.finish_reason == FINISH_REASON_MAPPING.get(message_delta_chunk.delta.stop_reason)
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""
        assert not streaming_chunk.start

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Test content_block_start for second tool_call
        tool_block_start_chunk = RawContentBlockStartEvent(
            content_block=ToolUseBlock(
                id="toolu_011dE5KDKxSh6hi85EnRKZT4", input={}, name="factorial", type="tool_use"
            ),
            index=3,
            type="content_block_start",
        )
        raw_chunks.append(tool_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            tool_block_start_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == tool_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=1, id="toolu_011dE5KDKxSh6hi85EnRKZT4", tool_name="factorial", arguments=None)
        ]
        assert streaming_chunk.content == ""

        # Test content_block_delta with input_json_delta (empty)
        empty_json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json="", type="input_json_delta"), index=3, type="content_block_delta"
        )
        raw_chunks.append(empty_json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            empty_json_delta_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == empty_json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=1, arguments=empty_json_delta_chunk.delta.partial_json)
        ]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert not streaming_chunk.start

        # Test content_block_delta with input_json_delta (with content)
        json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json='{"expression": 5 ', type="input_json_delta"),
            index=3,
            type="content_block_delta",
        )
        raw_chunks.append(json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            json_delta_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [ToolCallDelta(index=1, arguments=json_delta_chunk.delta.partial_json)]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert not streaming_chunk.start

        # Test message_delta chunk
        message_delta_chunk = RawMessageDeltaEvent(
            delta=Delta(stop_reason="tool_use", stop_sequence=None),
            type="message_delta",
            usage=MessageDeltaUsage(
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                input_tokens=None,
                output_tokens=77,
                server_tool_use=None,
            ),
        )
        raw_chunks.append(message_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == message_delta_chunk.model_dump()
        assert streaming_chunk.finish_reason == FINISH_REASON_MAPPING.get(message_delta_chunk.delta.stop_reason)
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""
        assert not streaming_chunk.start

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Then a message_stop chunk
        # message_stop_chunk = RawMessageStopEvent(type="message_stop")
        # but we don't stream it

        generator = AnthropicChatGenerator(Secret.from_token("test-api-key"))
        message = generator._process_response(raw_chunks)
        assert message["replies"][0].meta["usage"] == {
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "input_tokens": 393,
            "output_tokens": 77,
            "server_tool_use": None,
        }

    def test_convert_streaming_chunks_to_chat_message_with_multiple_tool_calls(self):
        """
        Test converting streaming chunks to a chat message with tool calls
        """
        # Create a sequence of streaming chunks that simulate Anthropic's response
        chunks = [
            # Message start with input tokens
            StreamingChunk(
                content="",
                meta={
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 0},
                    },
                },
                component_info=ComponentInfo.from_component(self),
                start=True,
            ),
            # Initial text content
            StreamingChunk(
                content="",
                meta={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                component_info=ComponentInfo.from_component(self),
                index=0,
                start=True,
            ),
            StreamingChunk(
                content="Let me check",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Let me check"},
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
            ),
            StreamingChunk(
                content=" the weather",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " the weather"},
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id="toolu_123", tool_name="weather", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments='{"city":')],
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments='"Paris"}')],
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "toolu_224", "name": "factorial", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id="toolu_224", tool_name="factorial", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"expression":'},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id=None, tool_name=None, arguments='{"expression":')],
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": " 5 }"},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id=None, tool_name=None, arguments=" 5 }")],
            ),
            # Final message delta
            StreamingChunk(
                content="",
                meta={
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_calls", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
                component_info=ComponentInfo.from_component(self),
                finish_reason="tool_calls",
            ),
        ]

        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Cannot test creating ReasoningContent from StreamingChunk objects
        # because reasoning is added outside _convert_streaming_chunks_to_chat_message

        # Verify the message content
        assert message.text == "Let me check the weather"
        # Verify tool calls
        assert len(message.tool_calls) == 2
        tool_call = message.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        tool_call = message.tool_calls[1]
        assert tool_call.id == "toolu_224"
        assert tool_call.tool_name == "factorial"
        assert tool_call.arguments == {"expression": 5}

        # Verify meta information
        assert message._meta["index"] == 0
        assert message._meta["finish_reason"] == "tool_calls"
        assert message._meta["usage"] == {"output_tokens": 40}

    def test_convert_streaming_chunks_to_chat_message_tool_call_with_empty_arguments(self):
        """
        Test converting streaming chunks with an empty tool call arguments
        """

        # Create a sequence of streaming chunks that simulate Anthropic's response
        chunks = [
            # Message start with input tokens
            StreamingChunk(
                content="",
                meta={
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-5",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 0},
                    },
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
                start=True,
            ),
            # Initial text content
            StreamingChunk(
                content="",
                meta={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                component_info=ComponentInfo.from_component(self),
                index=1,
                start=True,
            ),
            StreamingChunk(
                content="Let me check",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Let me check"},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
            ),
            StreamingChunk(
                content=" the weather",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " the weather"},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id="toolu_123", tool_name="weather", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments="")],
            ),
            # Final message delta
            StreamingChunk(
                content="",
                meta={
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_calls", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                finish_reason="tool_calls",
            ),
        ]

        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Verify the message content
        assert message.text == "Let me check the weather"

        # Verify tool calls
        assert len(message.tool_calls) == 1
        tool_call = message.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {}

        # Verify meta information
        assert message._meta["index"] == 0
        assert message._meta["finish_reason"] == "tool_calls"
        assert message._meta["usage"] == {"output_tokens": 40}

    def test_convert_image_content_to_anthropic_format_with_unsupported_mime_type(self):
        """Test that an ImageContent with unsupported mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/bmp")  # Unsupported format

        with pytest.raises(ValueError, match="Unsupported image format: image/bmp"):
            _convert_image_content_to_anthropic_format(image_content)

    def test_convert_image_content_to_anthropic_format_with_none_mime_type(self):
        """Test that an ImageContent with None mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Manually set mime_type to None to test the validation
        image_content.mime_type = None

        with pytest.raises(ValueError, match="Unsupported image format: None"):
            _convert_image_content_to_anthropic_format(image_content)

    def test_convert_file_content_to_anthropic_format_with_unsupported_mime_type(self):
        base64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        file_content = FileContent(base64_data=base64_data, mime_type="image/png")

        with pytest.raises(ValueError, match="Unsupported file format: image/png"):
            _convert_file_content_to_anthropic_format(file_content)

    def test_convert_message_to_anthropic_format_from_system(self):
        messages = [ChatMessage.from_system("You are good assistant")]
        assert _convert_messages_to_anthropic_format(messages) == (
            [{"type": "text", "text": "You are good assistant"}],
            [],
        )

    def test_convert_message_to_anthropic_format_from_user(self):
        messages = [ChatMessage.from_user("I have a question")]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [{"role": "user", "content": [{"type": "text", "text": "I have a question"}]}],
        )

    def test_convert_message_to_anthropic_format_from_assistant(self):
        messages = [ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [{"role": "assistant", "content": [{"type": "text", "text": "I have an answer"}]}],
        )

        messages = [
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}}],
                }
            ],
        )

        messages = [
            ChatMessage.from_assistant(
                text="",  # this should not happen, but we should handle it without errors
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})],
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}}],
                }
            ],
        )

        messages = [
            ChatMessage.from_assistant(
                text="For that I'll need to check the weather",
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})],
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "For that I'll need to check the weather"},
                        {"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}},
                    ],
                }
            ],
        )

    def test_convert_message_to_anthropic_format_from_tool(self):
        tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
        messages = [
            ChatMessage.from_tool(
                tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            )
        ]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "123",
                            "content": [{"type": "text", "text": '{"weather": "sunny", "temperature": "25"}'}],
                            "is_error": False,
                        }
                    ],
                }
            ],
        )

        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )

        tool_result = [
            TextContent("Here's the retrieved image"),
            ImageContent(base64_image=base64_image, mime_type="image/png"),
        ]
        messages = [
            ChatMessage.from_tool(
                tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            )
        ]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "123",
                            "content": [
                                {"type": "text", "text": "Here's the retrieved image"},
                                {
                                    "type": "image",
                                    "source": {"type": "base64", "media_type": "image/png", "data": base64_image},
                                },
                            ],
                            "is_error": False,
                        }
                    ],
                }
            ],
        )

    def test_convert_message_to_anthropic_format_complex(self):
        """
        Test that the AnthropicChatGenerator can convert a complex sequence of ChatMessages to Anthropic format.
        In particular, we check that different tool results are packed in a single dictionary with role=user.
        """

        messages = [
            ChatMessage.from_system("You are good assistant"),
            ChatMessage.from_user("What's the weather like in Paris? And how much is 2+2?"),
            ChatMessage.from_assistant(
                text="",
                tool_calls=[
                    ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}),
                    ToolCall(id="456", tool_name="math", arguments={"expression": "2+2"}),
                ],
            ),
            ChatMessage.from_tool(
                tool_result="22° C", origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            ),
            ChatMessage.from_tool(
                tool_result="4", origin=ToolCall(id="456", tool_name="math", arguments={"expression": "2+2"})
            ),
        ]

        system_messages, non_system_messages = _convert_messages_to_anthropic_format(messages)

        assert system_messages == [{"type": "text", "text": "You are good assistant"}]
        assert non_system_messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather like in Paris? And how much is 2+2?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}},
                    {"type": "tool_use", "id": "456", "name": "math", "input": {"expression": "2+2"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": [{"type": "text", "text": "22° C"}],
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "456",
                        "content": [{"type": "text", "text": "4"}],
                        "is_error": False,
                    },
                ],
            },
        ]

    def test_convert_message_to_anthropic_format_with_image(self):
        """Test that a ChatMessage with ImageContent is converted to Anthropic format correctly."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        _, non_system_messages = _convert_messages_to_anthropic_format([message])

        assert len(non_system_messages) == 1
        anthropic_message = non_system_messages[0]
        assert anthropic_message["role"] == "user"
        assert len(anthropic_message["content"]) == 2

        # Check text and image blocks
        assert anthropic_message["content"][0]["type"] == "text"
        assert anthropic_message["content"][0]["text"] == "What's in this image?"
        assert anthropic_message["content"][1]["type"] == "image"
        assert anthropic_message["content"][1]["source"]["type"] == "base64"
        assert anthropic_message["content"][1]["source"]["media_type"] == "image/png"
        assert anthropic_message["content"][1]["source"]["data"] == base64_image

    def test_convert_message_to_anthropic_format_with_file_content(self, test_files_path):
        pdf_path = test_files_path / "sample_pdf_3.pdf"
        with open(pdf_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode("utf-8")

        extra = {"context": "This document contains a table", "title": "A nice PDF"}
        file_content = FileContent(base64_data=base64_data, mime_type="application/pdf", extra=extra)
        message = ChatMessage.from_user(content_parts=["Describe this document", file_content])

        _, non_system_messages = _convert_messages_to_anthropic_format([message])
        assert non_system_messages == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this document"},
                    {
                        "type": "document",
                        "source": {"type": "base64", "media_type": "application/pdf", "data": base64_data},
                        "context": "This document contains a table",
                        "title": "A nice PDF",
                    },
                ],
            }
        ]

    def test_convert_message_to_anthropic_invalid(self):
        """
        Test that the AnthropicChatGenerator component fails to convert an invalid ChatMessage to Anthropic format.
        """
        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

        tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

        base64_data = "JVBERi0xLjEKMSAwIG9iago8PC9UeXBlL0NhdGFsb2c+PgplbmRvYmoKdHJhaWxlcgo8PC9Sb290IDEgMCBSPj4KJSVFT0Y="
        file_content = FileContent(base64_data=base64_data, mime_type="application/pdf")
        message = ChatMessage.from_assistant()
        message._content = [file_content]
        with pytest.raises(ValueError, match="File content is only supported for user messages"):
            _convert_messages_to_anthropic_format([message])

    def test_finalize_reasoning_group_with_thinking_text(self):
        """Test that _finalize_reasoning_group appends a reasoning_text entry."""
        formatted: list = []
        _finalize_reasoning_group(formatted, "The user is asking about weather.", "sig123", None)
        assert len(formatted) == 1
        assert formatted[0] == {
            "reasoning_content": {
                "reasoning_text": {"text": "The user is asking about weather.", "signature": "sig123"},
            }
        }

    def test_finalize_reasoning_group_with_redacted_thinking(self):
        """Test that _finalize_reasoning_group appends a redacted_thinking entry."""
        formatted: list = []
        _finalize_reasoning_group(formatted, "", None, "redacted_data_abc")
        assert len(formatted) == 1
        assert formatted[0] == {"reasoning_content": {"redacted_thinking": "redacted_data_abc"}}

    def test_finalize_reasoning_group_with_both(self):
        """Test that _finalize_reasoning_group appends both reasoning_text and redacted_thinking entries."""
        formatted: list = []
        _finalize_reasoning_group(formatted, "Some thinking.", "sig456", "redacted_xyz")
        assert len(formatted) == 2
        assert formatted[0] == {
            "reasoning_content": {
                "reasoning_text": {"text": "Some thinking.", "signature": "sig456"},
            }
        }
        assert formatted[1] == {"reasoning_content": {"redacted_thinking": "redacted_xyz"}}

    def test_finalize_reasoning_group_with_empty_inputs(self):
        """Test that _finalize_reasoning_group does nothing when all inputs are empty."""
        formatted: list = []
        _finalize_reasoning_group(formatted, "", None, None)
        assert len(formatted) == 0
