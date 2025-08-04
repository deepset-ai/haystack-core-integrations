#!/usr/bin/env python3

import asyncio
import os
from unittest.mock import Mock

import pytest
from google.genai import types
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, ComponentInfo, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.google_genai.chat.chat_generator import (
    GoogleGenAIChatGenerator,
    _convert_message_to_google_genai_format,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: 22°C, sunny"


@pytest.fixture
def tools():
    return [
        Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
    ]


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
        mock_content.parts.append(mock_part)
        mock_candidate.content = mock_content

        chunk = component._convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk,
            index=0,
            component_info=component_info,
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

        chunk = component._convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info
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

        chunk = component._convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info
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

        chunk = component._convert_google_chunk_to_streaming_chunk(
            chunk=mock_chunk, index=0, component_info=component_info
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
            model_version="gemini-2.0-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk1 = component._convert_google_chunk_to_streaming_chunk(
            chunk=chunk1, index=0, component_info=component_info
        )
        assert streaming_chunk1.content == "I'll get the weather information for Paris and Berlin"
        assert streaming_chunk1.tool_calls == []
        assert streaming_chunk1.finish_reason is None
        assert streaming_chunk1.index == 0
        assert "received_at" in streaming_chunk1.meta
        assert streaming_chunk1.meta["model"] == "gemini-2.0-flash"
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
            model_version="gemini-2.0-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk2 = component._convert_google_chunk_to_streaming_chunk(
            chunk=chunk2, index=1, component_info=component_info
        )
        assert streaming_chunk2.content == " and present it in a structured format."
        assert streaming_chunk2.tool_calls == []
        assert streaming_chunk2.finish_reason is None
        assert streaming_chunk2.index == 1
        assert "received_at" in streaming_chunk2.meta
        assert streaming_chunk2.meta["model"] == "gemini-2.0-flash"
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
            model_version="gemini-2.0-flash",
            response_id=None,
            create_time=None,
            prompt_feedback=None,
            automatic_function_calling_history=None,
            parsed=None,
        )

        streaming_chunk = component._convert_google_chunk_to_streaming_chunk(
            chunk=chunk, index=2, component_info=component_info
        )
        assert streaming_chunk.content == ""
        assert streaming_chunk.tool_calls is not None
        assert len(streaming_chunk.tool_calls) == 6
        assert streaming_chunk.finish_reason == "stop"
        assert streaming_chunk.index == 2
        assert "received_at" in streaming_chunk.meta
        assert streaming_chunk.meta["model"] == "gemini-2.0-flash"
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


class TestGoogleGenAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        assert component._model == "gemini-2.0-flash"
        assert component._generation_kwargs == {}
        assert component._safety_settings == []
        assert component._streaming_callback is None
        assert component._tools is None
        assert component._api_key is not None
        assert component._api_key.resolve_value() == "test-api-key"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError):
            GoogleGenAIChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            GoogleGenAIChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=weather)
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-from-env")
        component = GoogleGenAIChatGenerator(
            api_key=Secret.from_token("test-api-key-from-env"),
            model="gemini-2.0-flash",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"temperature": 0.5, "max_output_tokens": 100},
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            tools=[tool],
        )
        assert component._model == "gemini-2.0-flash"
        assert component._streaming_callback is print_streaming_chunk
        assert component._generation_kwargs == {"temperature": 0.5, "max_output_tokens": 100}
        assert component._safety_settings == [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        assert component._tools == [tool]

    def test_init_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        assert generator._tools == toolset

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        data = generator.to_dict()

        assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"
        assert "tools" in data["init_parameters"]["tools"]["data"]
        assert len(data["init_parameters"]["tools"]["data"]["tools"]) == len(tools)

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", tools=toolset)
        data = component.to_dict()

        deserialized_component = GoogleGenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component._tools, Toolset)
        assert len(deserialized_component._tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component._tools)

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

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.0-flash" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        component = GoogleGenAIChatGenerator()
        component_info = ComponentInfo.from_component(component)

        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""
                assert chunk.component_info == component_info

        callback = Callback()

        results = component.run(
            messages=[ChatMessage.from_user("What's the capital of France?")], streaming_callback=callback
        )

        assert len(results["replies"]) == 1
        assert callback.counter > 0, "No streaming chunks received"
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the GoogleGenAIChatGenerator component can run with tools and streaming.
        """
        component = GoogleGenAIChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run([ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert tool_message.tool_calls is not None, "Tool message has no tool calls"
        assert len(tool_message.tool_calls) == 1, "Tool message has multiple tool calls"
        # Google Gen AI (gemini-2.0-flash and gemini-2.5-pro-preview-05-06) does not provide ids for tool calls although
        # it is in the response schema, revisit in future to see if there are changes and id is provided
        # assert tool_message.tool_calls[0].id is not None, "Tool call has no id"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}

        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"
        assert tool_message.meta["finish_reason"] == "stop"

        tool_call = tool_message.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_toolset(self, tools):
        """Test that GoogleGenAIChatGenerator can run with a Toolset."""
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)

        messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        results = component.run(messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Check if tool calls were made
        assert message.tool_calls is not None, "Message has no tool calls"
        assert len(message.tool_calls) == 1, "Message has multiple tool calls and it should only have one"
        tool_call = message.tool_calls[0]
        # Google Gen AI (gemini-2.0-flash and gemini-2.5-pro-preview-05-06) does not provide ids for tool calls although
        # it is in the response schema, revisit in future to see if there are changes and id is provided
        # assert tool_call.id is not None, "Tool call has no id"
        assert message.meta["finish_reason"] == "stop"

        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

        # Test full conversation with tool result
        tool_result_message = ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call)
        follow_up_messages = [*messages, message, tool_result_message]
        final_results = component.run(follow_up_messages)

        assert len(final_results["replies"]) == 1
        final_message = final_results["replies"][0]
        assert final_message.text
        assert "paris" in final_message.text.lower() or "weather" in final_message.text.lower(), (
            "Response does not contain Paris or weather"
        )

    @pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY", None),
        reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_parallel_tools(self, tools):
        """
        Integration test that the GoogleGenAIChatGenerator component can run with parallel tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = GoogleGenAIChatGenerator(tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Google GenAI should make tool calls for both cities
        assert len(message.tool_calls) == 2
        tool_call_paris = (
            message.tool_calls[0] if message.tool_calls[0].arguments["city"] == "Paris" else message.tool_calls[1]
        )
        assert isinstance(tool_call_paris, ToolCall)
        assert tool_call_paris.tool_name == "weather"
        assert tool_call_paris.arguments["city"] == "Paris"
        assert message.meta["finish_reason"] == "stop"

        tool_call_berlin = message.tool_calls[1]
        assert isinstance(tool_call_berlin, ToolCall)
        assert tool_call_berlin.tool_name == "weather"
        assert tool_call_berlin.arguments["city"] == "Berlin"
        assert message.meta["finish_reason"] == "stop"

        # Google GenAI expects results from both tools in separate messages
        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call_paris),
            ChatMessage.from_tool(tool_result="15°C, cloudy", origin=tool_call_berlin),
        ]

        # Response from the model contains results from both tools
        results = component.run(new_messages)
        message = results["replies"][0]

        assert not message.tool_calls, "Message has tool calls and it should not have any"
        assert len(message.text) > 0, "Message has no text"
        assert message.text and ("paris" in message.text.lower() or "berlin" in message.text.lower())
        # Check that the response mentions both temperature readings
        assert "22" in message.text or "15" in message.text


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncGoogleGenAIChatGenerator:
    """Test class for async functionality of GoogleGenAIChatGenerator."""

    async def test_live_run_async(self) -> None:
        """Test async version of the run method."""
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.0-flash" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    async def test_live_run_async_streaming(self):
        """Test async version with streaming."""
        responses = ""
        counter = 0

        async def async_callback(chunk: StreamingChunk) -> None:
            nonlocal counter, responses
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = GoogleGenAIChatGenerator()
        results = await component.run_async(
            [ChatMessage.from_user("What's the capital of France?")], streaming_callback=async_callback
        )

        assert len(results["replies"]) == 1
        assert counter > 0, "No streaming chunks received"
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert message.meta["finish_reason"] == "stop"

    async def test_live_run_async_with_tools(self, tools):
        """Test async version with tools."""
        component = GoogleGenAIChatGenerator(tools=tools)
        results = await component.run_async([ChatMessage.from_user("What's the weather like in Paris?")])

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert tool_message.tool_calls is not None, "Tool message has no tool calls"
        assert len(tool_message.tool_calls) == 1, "Tool message has multiple tool calls"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "stop"

    async def test_concurrent_async_calls(self):
        """Test multiple concurrent async calls."""
        component = GoogleGenAIChatGenerator()

        # Create multiple tasks
        tasks = []
        for i in range(3):
            messages = [ChatMessage.from_user(f"What's the capital of country number {i + 1}? Just say the city name.")]
            task = component.run_async(messages)
            tasks.append(task)

        # Run concurrently
        results = await asyncio.gather(*tasks)

        # Verify all calls completed successfully
        assert len(results) == 3
        for result in results:
            assert len(result["replies"]) == 1
            assert result["replies"][0].text
            assert result["replies"][0].meta["model"]
            assert result["replies"][0].meta["finish_reason"] == "stop"
