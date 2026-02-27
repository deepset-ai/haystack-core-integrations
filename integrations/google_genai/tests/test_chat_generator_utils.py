# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
from unittest.mock import Mock

import pytest
from google.genai import types
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FileContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCall,
)

from haystack_integrations.components.generators.google_genai.chat.chat_generator import (
    GoogleGenAIChatGenerator,
)
from haystack_integrations.components.generators.google_genai.chat.utils import (
    _aggregate_streaming_chunks_with_reasoning,
    _convert_google_chunk_to_streaming_chunk,
    _convert_google_genai_response_to_chatmessage,
    _convert_message_to_google_genai_format,
    _convert_usage_metadata_to_serializable,
    _process_thinking_config,
)


def test_process_thinking_budget():
    """Test the _process_thinking_config function with different thinking_budget values."""

    # Test valid thinking_budget values
    generation_kwargs = {"thinking_budget": 1024, "temperature": 0.7}
    result = _process_thinking_config(generation_kwargs)

    # thinking_budget should be moved to thinking_config
    assert "thinking_budget" not in result
    assert "thinking_config" in result
    assert result["thinking_config"].thinking_budget == 1024
    assert result["thinking_config"].include_thoughts is True
    # Other kwargs should be preserved
    assert result["temperature"] == 0.7

    # Test dynamic allocation (-1)
    generation_kwargs = {"thinking_budget": -1}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == -1
    assert result["thinking_config"].include_thoughts is True

    # Test zero (disable thinking)
    generation_kwargs = {"thinking_budget": 0}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == 0
    assert result["thinking_config"].include_thoughts is False

    # Test large value
    generation_kwargs = {"thinking_budget": 24576}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == 24576
    assert result["thinking_config"].include_thoughts is True

    # Test when thinking_budget is not present
    generation_kwargs = {"temperature": 0.5}
    result = _process_thinking_config(generation_kwargs)
    assert result == generation_kwargs  # No changes

    # Test invalid type (should fall back to dynamic)
    generation_kwargs = {"thinking_budget": "invalid", "temperature": 0.5}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == -1  # Dynamic allocation
    assert result["temperature"] == 0.5


def test_process_thinking_level():
    """Test the _process_thinking_config function with different thinking_level values."""

    # Test valid thinking_level values
    generation_kwargs = {"thinking_level": "high", "temperature": 0.7}
    result = _process_thinking_config(generation_kwargs)

    # thinking_level should be moved to thinking_config
    assert "thinking_level" not in result
    assert "thinking_config" in result
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.HIGH
    assert result["thinking_config"].include_thoughts is True
    # Other kwargs should be preserved
    assert result["temperature"] == 0.7

    # Test THINKING_LEVEL_LOW in upper case
    generation_kwargs = {"thinking_level": "LOW"}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.LOW
    assert result["thinking_config"].include_thoughts is True

    # Test MINIMAL (should disable include_thoughts)
    generation_kwargs = {"thinking_level": "MINIMAL"}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.MINIMAL
    assert result["thinking_config"].include_thoughts is False

    # Test THINKING_LEVEL_UNSPECIFIED (invalid value falls back)
    generation_kwargs = {"thinking_level": "test"}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
    assert result["thinking_config"].include_thoughts is True

    # Test when thinking_level is not present
    generation_kwargs = {"temperature": 0.5}
    result = _process_thinking_config(generation_kwargs)
    assert result == generation_kwargs  # No changes

    # Test invalid type (should fall back to THINKING_LEVEL_UNSPECIFIED)
    generation_kwargs = {"thinking_level": 123, "temperature": 0.5}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
    assert result["thinking_config"].include_thoughts is True
    assert result["temperature"] == 0.5


def test_process_thinking_config_explicit_include_thoughts():
    """Test that explicit include_thoughts in generation_kwargs overrides the auto-derived value."""
    # thinking_budget=0 normally means include_thoughts=False, but user explicitly sets True
    generation_kwargs = {"thinking_budget": 0, "include_thoughts": True}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == 0
    assert result["thinking_config"].include_thoughts is True
    assert "include_thoughts" not in result  # should be popped from top-level kwargs

    # thinking_budget=1024 normally means include_thoughts=True, but user explicitly sets False
    generation_kwargs = {"thinking_budget": 1024, "include_thoughts": False}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_budget == 1024
    assert result["thinking_config"].include_thoughts is False
    assert "include_thoughts" not in result

    # thinking_level="high" normally means include_thoughts=True, but user explicitly sets False
    generation_kwargs = {"thinking_level": "high", "include_thoughts": False}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.HIGH
    assert result["thinking_config"].include_thoughts is False
    assert "include_thoughts" not in result

    # thinking_level="minimal" normally means include_thoughts=False, but user explicitly sets True
    generation_kwargs = {"thinking_level": "minimal", "include_thoughts": True}
    result = _process_thinking_config(generation_kwargs)
    assert result["thinking_config"].thinking_level == types.ThinkingLevel.MINIMAL
    assert result["thinking_config"].include_thoughts is True
    assert "include_thoughts" not in result

    # include_thoughts alone (no thinking_budget or thinking_level) should just be popped and ignored
    generation_kwargs = {"include_thoughts": True, "temperature": 0.5}
    result = _process_thinking_config(generation_kwargs)
    assert "include_thoughts" not in result
    assert "thinking_config" not in result
    assert result == {"temperature": 0.5}


class TestStreamingChunkConversion:
    def test_convert_google_chunk_to_streaming_chunk_text_only(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        mock_chunk = Mock()
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_chunk.candidates = [mock_candidate]

        mock_content = Mock()
        mock_content.parts = []
        mock_part = Mock()
        mock_part.text = "Hello, world!"
        mock_part.function_call = None
        # Explicitly set thought=False to simulate a regular (non-thought) part
        mock_part.thought = False
        mock_content.parts.append(mock_part)
        mock_candidate.content = mock_content

        chunk = _convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk,
            index=0,
            component_info=component_info,
            model="gemini-2.5-flash",
        )

        assert chunk.content == "Hello, world!"
        assert chunk.tool_calls == []
        assert chunk.finish_reason == "stop"
        assert chunk.index == 0
        assert "received_at" in chunk.meta
        assert chunk.component_info == component_info

    def test_convert_google_chunk_to_streaming_chunk_tool_call(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        mock_chunk = Mock()
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_chunk.candidates = [mock_candidate]

        mock_content = Mock()
        mock_content.parts = []
        mock_part = Mock()
        mock_part.text = None
        mock_function_call = Mock()
        mock_function_call.name = "weather"
        mock_function_call.args = {"city": "Paris"}
        mock_function_call.id = "call_123"
        mock_part.function_call = mock_function_call
        mock_content.parts.append(mock_part)
        mock_candidate.content = mock_content

        chunk = _convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info, model="gemini-2.5-flash"
        )

        assert chunk.content == ""
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].tool_name == "weather"
        assert chunk.tool_calls[0].arguments == '{"city": "Paris"}'
        assert chunk.tool_calls[0].id == "call_123"
        assert chunk.finish_reason == "stop"
        assert chunk.index == 0
        assert "received_at" in chunk.meta
        assert chunk.component_info == component_info

    def test_convert_google_chunk_to_streaming_chunk_mixed_content(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        mock_chunk = Mock()
        mock_candidate = Mock()
        mock_candidate.finish_reason = "STOP"
        mock_chunk.candidates = [mock_candidate]

        mock_content = Mock()
        mock_content.parts = []

        mock_text_part = Mock()
        mock_text_part.text = "I'll check the weather for you."
        mock_text_part.function_call = None
        # Explicitly set thought=False to simulate a regular (non-thought) part
        mock_text_part.thought = False
        mock_content.parts.append(mock_text_part)

        mock_tool_part = Mock()
        mock_tool_part.text = None
        mock_function_call = Mock()
        mock_function_call.name = "weather"
        mock_function_call.args = {"city": "London"}
        mock_function_call.id = "call_456"
        mock_tool_part.function_call = mock_function_call
        mock_content.parts.append(mock_tool_part)

        mock_candidate.content = mock_content

        chunk = _convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info, model="gemini-2.5-flash"
        )

        # When both text and tool calls are present, tool calls are prioritized
        assert chunk.content == ""
        assert chunk.tool_calls is not None
        assert len(chunk.tool_calls) == 1
        assert chunk.tool_calls[0].tool_name == "weather"
        assert chunk.tool_calls[0].arguments == '{"city": "London"}'
        assert chunk.finish_reason == "stop"
        assert chunk.component_info == component_info

    def test_convert_google_chunk_to_streaming_chunk_empty_parts(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        mock_chunk = Mock()
        mock_candidate = Mock()
        mock_content = Mock()
        mock_content.parts = []
        mock_candidate.content = mock_content
        mock_chunk.candidates = [mock_candidate]

        chunk = _convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info, model="gemini-2.5-flash"
        )

        assert chunk.content == ""
        assert chunk.tool_calls == []
        assert chunk.component_info == component_info

    def test_convert_google_chunk_to_streaming_chunk_real_example(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        # Chunk 1: Text only
        chunk1_parts = [
            types.Part(
                text="I'll get the weather information for Paris and Berlin", function_call=None, function_response=None
            )
        ]
        chunk1_content = types.Content(role="model", parts=chunk1_parts)
        chunk1_candidate = types.Candidate(
            content=chunk1_content,
            finish_reason=None,
            index=None,
            safety_ratings=None,
            citation_metadata=None,
            grounding_metadata=None,
            finish_message=None,
            token_count=None,
            logprobs_result=None,
            avg_logprobs=None,
            url_context_metadata=None,
        )
        chunk1_usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=217, candidates_token_count=None, total_token_count=217
        )
        chunk1 = types.GenerateContentResponse(
            candidates=[chunk1_candidate],
            usage_metadata=chunk1_usage,
            model_version="gemini-2.5-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk1 = _convert_google_chunk_to_streaming_chunk(
            chunk=chunk1, index=0, component_info=component_info, model="gemini-2.5-flash"
        )
        assert streaming_chunk1.content == "I'll get the weather information for Paris and Berlin"
        assert streaming_chunk1.tool_calls == []
        assert streaming_chunk1.finish_reason is None
        assert streaming_chunk1.index == 0
        assert "received_at" in streaming_chunk1.meta
        assert streaming_chunk1.meta["model"] == "gemini-2.5-flash"
        assert "usage" in streaming_chunk1.meta
        assert streaming_chunk1.meta["usage"]["prompt_tokens"] == 217
        assert streaming_chunk1.meta["usage"]["completion_tokens"] is None
        assert streaming_chunk1.meta["usage"]["total_tokens"] == 217
        assert streaming_chunk1.component_info == component_info

        # Chunk 2: Text only
        chunk2_parts = [
            types.Part(text=" and present it in a structured format.", function_call=None, function_response=None)
        ]
        chunk2_content = types.Content(role="model", parts=chunk2_parts)
        chunk2_candidate = types.Candidate(
            content=chunk2_content,
            finish_reason=None,
            index=None,
            safety_ratings=None,
            citation_metadata=None,
            grounding_metadata=None,
            finish_message=None,
            token_count=None,
            logprobs_result=None,
            avg_logprobs=None,
            url_context_metadata=None,
        )
        chunk2_usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=217, candidates_token_count=None, total_token_count=217
        )
        chunk2 = types.GenerateContentResponse(
            candidates=[chunk2_candidate],
            usage_metadata=chunk2_usage,
            model_version="gemini-2.5-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk2 = _convert_google_chunk_to_streaming_chunk(
            chunk=chunk2, index=1, component_info=component_info, model="gemini-2.5-flash"
        )
        assert streaming_chunk2.content == " and present it in a structured format."
        assert streaming_chunk2.tool_calls == []
        assert streaming_chunk2.finish_reason is None
        assert streaming_chunk2.index == 1
        assert "received_at" in streaming_chunk2.meta
        assert streaming_chunk2.meta["model"] == "gemini-2.5-flash"
        assert "usage" in streaming_chunk2.meta
        assert streaming_chunk2.meta["usage"]["prompt_tokens"] == 217
        assert streaming_chunk2.meta["usage"]["completion_tokens"] is None
        assert streaming_chunk2.meta["usage"]["total_tokens"] == 217
        assert streaming_chunk2.component_info == component_info

        # Chunk 3: Multiple tool calls (6 function calls) for 2 cities with 3 tools each
        fc1 = types.FunctionCall(id=None, name="get_weather", args={"city": "Paris"})
        fc2 = types.FunctionCall(id=None, name="get_population", args={"city": "Paris"})
        fc3 = types.FunctionCall(id=None, name="get_time", args={"city": "Paris"})
        fc4 = types.FunctionCall(id=None, name="get_weather", args={"city": "Berlin"})
        fc5 = types.FunctionCall(id=None, name="get_population", args={"city": "Berlin"})
        fc6 = types.FunctionCall(id=None, name="get_time", args={"city": "Berlin"})

        parts = [
            types.Part(text=None, function_call=fc1, function_response=None),
            types.Part(text=None, function_call=fc2, function_response=None),
            types.Part(text=None, function_call=fc3, function_response=None),
            types.Part(text=None, function_call=fc4, function_response=None),
            types.Part(text=None, function_call=fc5, function_response=None),
            types.Part(text=None, function_call=fc6, function_response=None),
        ]

        content = types.Content(role="model", parts=parts)
        candidate = types.Candidate(
            content=content,
            finish_reason=types.FinishReason.STOP,
            index=None,
            safety_ratings=None,
            citation_metadata=None,
            grounding_metadata=None,
            finish_message=None,
            token_count=None,
            logprobs_result=None,
            avg_logprobs=None,
            url_context_metadata=None,
        )

        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=144, candidates_token_count=121, total_token_count=265
        )
        chunk = types.GenerateContentResponse(
            candidates=[candidate],
            usage_metadata=usage_metadata,
            model_version="gemini-2.5-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk = _convert_google_chunk_to_streaming_chunk(
            chunk=chunk, index=2, component_info=component_info, model="gemini-2.5-flash"
        )
        assert streaming_chunk.content == ""
        assert streaming_chunk.tool_calls is not None
        assert len(streaming_chunk.tool_calls) == 6
        assert streaming_chunk.finish_reason == "stop"
        assert streaming_chunk.index == 2
        assert "received_at" in streaming_chunk.meta
        assert streaming_chunk.meta["model"] == "gemini-2.5-flash"
        assert streaming_chunk.component_info == component_info
        assert "usage" in streaming_chunk.meta
        assert streaming_chunk.meta["usage"]["prompt_tokens"] == 144
        assert streaming_chunk.meta["usage"]["completion_tokens"] == 121
        assert streaming_chunk.meta["usage"]["total_tokens"] == 265

        assert streaming_chunk.tool_calls[0].tool_name == "get_weather"
        assert streaming_chunk.tool_calls[0].arguments == '{"city": "Paris"}'
        assert streaming_chunk.tool_calls[0].id is None
        assert streaming_chunk.tool_calls[0].index == 0

        assert streaming_chunk.tool_calls[1].tool_name == "get_population"
        assert streaming_chunk.tool_calls[1].arguments == '{"city": "Paris"}'
        assert streaming_chunk.tool_calls[1].id is None
        assert streaming_chunk.tool_calls[1].index == 1

        assert streaming_chunk.tool_calls[2].tool_name == "get_time"
        assert streaming_chunk.tool_calls[2].arguments == '{"city": "Paris"}'
        assert streaming_chunk.tool_calls[2].id is None
        assert streaming_chunk.tool_calls[2].index == 2

        assert streaming_chunk.tool_calls[3].tool_name == "get_weather"
        assert streaming_chunk.tool_calls[3].arguments == '{"city": "Berlin"}'
        assert streaming_chunk.tool_calls[3].id is None
        assert streaming_chunk.tool_calls[3].index == 3

        assert streaming_chunk.tool_calls[4].tool_name == "get_population"
        assert streaming_chunk.tool_calls[4].arguments == '{"city": "Berlin"}'
        assert streaming_chunk.tool_calls[4].id is None
        assert streaming_chunk.tool_calls[4].index == 4

        assert streaming_chunk.tool_calls[5].tool_name == "get_time"
        assert streaming_chunk.tool_calls[5].arguments == '{"city": "Berlin"}'
        assert streaming_chunk.tool_calls[5].id is None
        assert streaming_chunk.tool_calls[5].index == 5

    def test_aggregate_streaming_chunks_with_reasoning(self):
        """Test the _aggregate_streaming_chunks_with_reasoning function for reasoning content aggregation."""

        # Create mock streaming chunks with reasoning content
        chunk1 = Mock()
        chunk1.content = "Hello"
        chunk1.tool_calls = []
        chunk1.meta = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        chunk1.reasoning = None

        chunk2 = Mock()
        chunk2.content = " world"
        chunk2.tool_calls = []
        chunk2.meta = {"usage": {"prompt_tokens": 10, "completion_tokens": 8}}
        chunk2.reasoning = None

        # Mock the final chunk with reasoning
        final_chunk = Mock()
        final_chunk.content = ""
        final_chunk.tool_calls = []
        final_chunk.meta = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 13, "thoughts_token_count": 5},
            "model": "gemini-2.5-pro",
        }
        final_chunk.reasoning = ReasoningContent(reasoning_text="I should greet the user politely")

        # Add reasoning deltas to the final chunk meta (this is how the real method works)
        final_chunk.meta["reasoning_deltas"] = [{"type": "reasoning", "content": "I should greet the user politely"}]

        # Test aggregation
        result = _aggregate_streaming_chunks_with_reasoning([chunk1, chunk2, final_chunk])

        # Verify the aggregated message
        assert result.text == "Hello world"
        assert result.tool_calls == []
        assert result.reasoning is not None
        assert result.reasoning.reasoning_text == "I should greet the user politely"
        assert result.meta["usage"]["prompt_tokens"] == 10
        assert result.meta["usage"]["completion_tokens"] == 13
        assert result.meta["usage"]["thoughts_token_count"] == 5
        assert result.meta["model"] == "gemini-2.5-pro"


class TestConvertMessageToGoogleGenAI:
    def test_convert_message_to_google_genai_format_complex(self):
        """
        Test that the GoogleGenAIChatGenerator can convert a complex sequence of ChatMessages to Google GenAI format.
        In particular, we check that different tool results are handled properly in sequence.
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

        # Test system message handling (should be handled separately in Google GenAI)
        system_message = messages[0]
        assert system_message.is_from(ChatRole.SYSTEM)

        # Test user message conversion
        user_message = messages[1]
        google_content = _convert_message_to_google_genai_format(user_message)
        assert google_content.role == "user"
        assert len(google_content.parts) == 1
        assert google_content.parts[0].text == "What's the weather like in Paris? And how much is 2+2?"

        # Test assistant message with tool calls
        assistant_message = messages[2]
        google_content = _convert_message_to_google_genai_format(assistant_message)
        assert google_content.role == "model"
        assert len(google_content.parts) == 2
        assert google_content.parts[0].function_call.name == "weather"
        assert google_content.parts[0].function_call.args == {"city": "Paris"}
        assert google_content.parts[1].function_call.name == "math"
        assert google_content.parts[1].function_call.args == {"expression": "2+2"}

        # Test tool result messages
        tool_result_1 = messages[3]
        google_content = _convert_message_to_google_genai_format(tool_result_1)
        assert google_content.role == "user"
        assert len(google_content.parts) == 1
        assert google_content.parts[0].function_response.name == "weather"
        assert google_content.parts[0].function_response.response == {"result": "22° C"}

        tool_result_2 = messages[4]
        google_content = _convert_message_to_google_genai_format(tool_result_2)
        assert google_content.role == "user"
        assert len(google_content.parts) == 1
        assert google_content.parts[0].function_response.name == "math"
        assert google_content.parts[0].function_response.response == {"result": "4"}

    def test_convert_message_to_google_genai_format_with_multiple_images(self, test_files_path):
        """Test converting a message with multiple images in mixed content to Google GenAI format."""
        apple_path = test_files_path / "apple.jpg"
        banana_path = test_files_path / "banana.png"

        apple_content = ImageContent.from_file_path(apple_path, size=(100, 100))
        banana_content = ImageContent.from_file_path(banana_path, size=(100, 100))

        message = ChatMessage.from_user(
            content_parts=[
                "Compare these fruits. First:",
                apple_content,
                "Second:",
                banana_content,
                "Which is healthier?",
            ]
        )

        google_content = _convert_message_to_google_genai_format(message)

        assert google_content.role == "user"
        assert len(google_content.parts) == 5

        # Verify the exact order is preserved
        assert google_content.parts[0].text == "Compare these fruits. First:"

        # First image (apple)
        assert hasattr(google_content.parts[1], "inline_data")
        assert google_content.parts[1].inline_data.mime_type == "image/jpeg"
        assert google_content.parts[1].inline_data.data is not None

        assert google_content.parts[2].text == "Second:"

        # Second image (banana)
        assert hasattr(google_content.parts[3], "inline_data")
        assert google_content.parts[3].inline_data.mime_type == "image/png"
        assert google_content.parts[3].inline_data.data is not None

        assert google_content.parts[4].text == "Which is healthier?"

    def test_convert_message_to_google_genai_format_image_with_minimal_text(self, test_files_path):
        """Test converting a message with minimal text and image to Google GenAI format."""
        apple_path = test_files_path / "apple.jpg"
        apple_content = ImageContent.from_file_path(apple_path, size=(100, 100))

        # Haystack requires at least one textual part for user messages, so we use minimal text
        message = ChatMessage.from_user(content_parts=["", apple_content])

        google_content = _convert_message_to_google_genai_format(message)

        assert google_content.role == "user"
        # Empty text should be filtered out by our implementation, leaving only the image
        assert len(google_content.parts) == 1

        # Should only have the image part (empty text filtered out)
        assert hasattr(google_content.parts[0], "inline_data")
        assert google_content.parts[0].inline_data.mime_type == "image/jpeg"
        assert google_content.parts[0].inline_data.data is not None

    def test_convert_message_to_google_genai_file_content(self, test_files_path):
        file_path = test_files_path / "sample_pdf_3.pdf"
        file_content = FileContent.from_file_path(file_path)
        message = ChatMessage.from_user(content_parts=["Describe this document:", file_content])
        google_content = _convert_message_to_google_genai_format(message)
        assert google_content.role == "user"
        assert len(google_content.parts) == 2

        assert google_content.parts[0].text == "Describe this document:"
        assert hasattr(google_content.parts[1], "inline_data")
        assert google_content.parts[1].inline_data.mime_type == "application/pdf"
        assert google_content.parts[1].inline_data.data is not None

    def test_convert_message_to_google_genai_file_content_in_assistant_message(self, test_files_path):
        file_path = test_files_path / "sample_pdf_3.pdf"
        file_content = FileContent.from_file_path(file_path)
        message = ChatMessage.from_assistant("This is a document")
        message._content.append(file_content)

        with pytest.raises(ValueError, match="FileContent is only supported for user messages"):
            _convert_message_to_google_genai_format(message)

    def test_convert_message_to_google_genai_format_with_thought_signatures(self):
        """Test converting an assistant message with thought signatures for multi-turn context preservation."""

        # Create an assistant message with tool calls and thought signatures in meta
        tool_call = ToolCall(id="call_123", tool_name="weather", arguments={"city": "Paris"})

        # Thought signatures are stored in meta when thinking is enabled with tools
        # They must be base64 encoded as per the API requirements
        thought_signatures = [
            {
                "part_index": 0,
                "signature": base64.b64encode(b"encrypted_mock_thought_signature_1").decode("utf-8"),
                "has_text": True,
                "has_function_call": False,
                "is_thought": False,
            },
            {
                "part_index": 1,
                "signature": base64.b64encode(b"encrypted_mock_thought_signature_2").decode("utf-8"),
                "has_text": False,
                "has_function_call": True,
                "is_thought": False,
            },
        ]

        message = ChatMessage.from_assistant(
            text="I'll check the weather for you",
            tool_calls=[tool_call],
            meta={"thought_signatures": thought_signatures},
        )

        google_content = _convert_message_to_google_genai_format(message)

        assert google_content.role == "model"
        assert len(google_content.parts) == 2

        # First part should have text and its thought signature
        assert google_content.parts[0].text == "I'll check the weather for you"
        # thought_signature is returned as bytes from the API
        assert google_content.parts[0].thought_signature == b"encrypted_mock_thought_signature_1"

        # Second part should have function call and its thought signature
        assert google_content.parts[1].function_call.name == "weather"
        assert google_content.parts[1].function_call.args == {"city": "Paris"}
        assert google_content.parts[1].thought_signature == b"encrypted_mock_thought_signature_2"

    def test_convert_message_to_google_genai_format_with_reasoning_content(self):
        """Test that ReasoningContent is properly skipped during conversion."""
        # ReasoningContent is for human transparency only, not sent to the API
        reasoning = ReasoningContent(reasoning_text="Of Life, the Universe and Everything...")

        # Create a message with both text and reasoning content
        message = ChatMessage.from_assistant(text="Forty-two", reasoning=reasoning)

        google_content = _convert_message_to_google_genai_format(message)

        assert google_content.role == "model"
        assert len(google_content.parts) == 1

        # Only the text should be included, reasoning content should be skipped
        assert google_content.parts[0].text == "Forty-two"
        # Verify no thought part was created (reasoning is not sent to API)
        assert not hasattr(google_content.parts[0], "thought") or not google_content.parts[0].thought

    def test_convert_message_to_google_genai_format_tool_message(self):
        tool_call = ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_tool(tool_result="22° C", origin=tool_call)
        google_content = _convert_message_to_google_genai_format(message)
        assert google_content.role == "user"
        assert len(google_content.parts) == 1
        assert isinstance(google_content.parts[0].function_response, types.FunctionResponse)
        assert google_content.parts[0].function_response.id == "123"
        assert google_content.parts[0].function_response.name == "weather"
        assert google_content.parts[0].function_response.response == {"result": "22° C"}
        assert google_content.parts[0].function_response.parts is None

    def test_convert_message_to_google_genai_format_image_in_tool_result(self):
        tool_call = ToolCall(id="123", tool_name="image_retriever", arguments={})

        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
        tool_result = [TextContent("Here is the image"), ImageContent(base64_image=base64_str, mime_type="image/jpeg")]

        message = ChatMessage.from_tool(tool_result=tool_result, origin=tool_call)
        google_content = _convert_message_to_google_genai_format(message)
        assert google_content.role == "user"
        assert len(google_content.parts) == 1
        assert isinstance(google_content.parts[0].function_response, types.FunctionResponse)
        assert google_content.parts[0].function_response.id == "123"
        assert google_content.parts[0].function_response.name == "image_retriever"
        assert google_content.parts[0].function_response.response == {"result": ""}
        assert len(google_content.parts[0].function_response.parts) == 2
        assert isinstance(google_content.parts[0].function_response.parts[0], types.FunctionResponsePart)
        assert google_content.parts[0].function_response.parts[0].inline_data.mime_type == "text/plain"
        assert google_content.parts[0].function_response.parts[0].inline_data.data == b"Here is the image"
        assert isinstance(google_content.parts[0].function_response.parts[1], types.FunctionResponsePart)
        assert google_content.parts[0].function_response.parts[1].inline_data.mime_type == "image/jpeg"
        assert google_content.parts[0].function_response.parts[1].inline_data.data == base64.b64decode(base64_str)
        assert len(google_content.parts[0].function_response.parts[1].inline_data.data) > 0

    def test_convert_message_to_google_genai_format_invalid_tool_result_type(self):
        tool_call = ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_tool(tool_result=256, origin=tool_call)
        with pytest.raises(ValueError, match="Unsupported content type in tool call result"):
            _convert_message_to_google_genai_format(message)

        message = ChatMessage.from_tool(tool_result=[TextContent("This is supported"), 256], origin=tool_call)
        with pytest.raises(
            ValueError,
            match=(
                r"Unsupported content type in tool call result list. "
                "Only TextContent and ImageContent are supported."
            ),
        ):
            _convert_message_to_google_genai_format(message)


class TestConvertGoogleGenAIToMessage:
    def test_convert_google_genai_response_to_chatmessage_parses_cached_tokens(self):
        """
        When the API response includes cached_content_token_count in usage_metadata,
        it is parsed into meta['usage'].
        """

        # Minimal candidate with one text part
        mock_part = Mock()
        mock_part.text = "Four."
        mock_part.function_call = None
        mock_part.thought = False
        mock_part.thought_signature = None
        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        # Usage metadata including cached tokens (as returned by API when cache is used)
        mock_usage = Mock()
        mock_usage.prompt_token_count = 1000
        mock_usage.candidates_token_count = 5
        mock_usage.total_token_count = 1005
        mock_usage.cached_content_token_count = 800
        mock_usage.thoughts_token_count = None

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        message = _convert_google_genai_response_to_chatmessage(mock_response, "gemini-2.5-flash")

        assert message.meta is not None
        assert "usage" in message.meta
        usage = message.meta["usage"]
        assert usage["prompt_tokens"] == 1000
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 1005
        assert usage["cached_content_token_count"] == 800

    def test_convert_usage_metadata_to_serializable(self):
        """_convert_usage_metadata_to_serializable builds a serializable dict from a UsageMetadata object."""
        assert _convert_usage_metadata_to_serializable(None) == {}
        assert _convert_usage_metadata_to_serializable(False) == {}

        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100,
            candidates_token_count=5,
            total_token_count=105,
        )
        result = _convert_usage_metadata_to_serializable(usage_metadata)
        assert result["prompt_token_count"] == 100
        assert result["candidates_token_count"] == 5
        assert result["total_token_count"] == 105
        assert len(result) == 3

        # Serialization of zero and composite types (ModalityTokenCount, lists)
        modality_token_count = types.ModalityTokenCount(modality=types.Modality.TEXT, tokenCount=100)
        usage_with_details = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=0,
            candidates_tokens_details=[modality_token_count],
        )
        result2 = _convert_usage_metadata_to_serializable(usage_with_details)
        assert result2["prompt_token_count"] == 0
        assert result2["candidates_tokens_details"] == [{"modality": "TEXT", "token_count": 100}]
