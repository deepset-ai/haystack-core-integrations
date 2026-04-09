# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FileContent,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    TextContent,
    ToolCall,
)
from haystack.tools import Tool, Toolset, create_tool_from_function
from haystack.utils.auth import Secret
from pydantic import BaseModel

from haystack_integrations.components.generators.google_genai.chat.chat_generator import (
    GoogleGenAIChatGenerator,
)


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


@pytest.fixture
def mock_response():
    mock_part = Mock()
    mock_part.text = "Hello"
    mock_part.function_call = None
    mock_part.thought = False
    mock_part.thought_signature = None

    mock_content = Mock()
    mock_content.parts = [mock_part]
    mock_candidate = Mock()
    mock_candidate.content = mock_content
    mock_candidate.finish_reason = "STOP"

    mock_usage = Mock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 5
    mock_usage.total_token_count = 15
    mock_usage.thoughts_token_count = None
    mock_usage.cached_content_token_count = None

    resp = Mock()
    resp.candidates = [mock_candidate]
    resp.usage_metadata = mock_usage
    return resp


@pytest.fixture
def mock_streaming_chunk():
    def _make(text="Hello", finish_reason=None):
        mock_part = Mock()
        mock_part.text = text
        mock_part.function_call = None
        mock_part.thought = False
        mock_part.thought_signature = None

        mock_content = Mock()
        mock_content.parts = [mock_part]
        mock_candidate = Mock()
        mock_candidate.finish_reason = finish_reason
        mock_candidate.content = mock_content

        chunk = Mock()
        chunk.candidates = [mock_candidate]
        chunk.usage_metadata = None
        return chunk

    return _make


def test_supported_models_is_non_empty_list_of_strings():
    assert isinstance(GoogleGenAIChatGenerator.SUPPORTED_MODELS, list)
    assert GoogleGenAIChatGenerator.SUPPORTED_MODELS
    assert all(isinstance(model, str) and model for model in GoogleGenAIChatGenerator.SUPPORTED_MODELS)


class TestGoogleGenAIChatGeneratorInitSerDe:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        assert component._model == "gemini-2.5-flash"
        assert component._generation_kwargs == {}
        assert component._safety_settings == []
        assert component._streaming_callback is None
        assert component._tools is None
        assert component._api_key is not None
        assert component._api_key.resolve_value() == "test-api-key"
        assert component._timeout is None
        assert component._max_retries is None

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
            model="gemini-2.5-flash",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"temperature": 0.5, "max_output_tokens": 100},
            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}],
            tools=[tool],
            timeout=30.0,
            max_retries=5,
        )
        assert component._model == "gemini-2.5-flash"
        assert component._streaming_callback is print_streaming_chunk
        assert component._generation_kwargs == {"temperature": 0.5, "max_output_tokens": 100}
        assert component._safety_settings == [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        ]
        assert component._tools == [tool]
        assert component._timeout == 30.0
        assert component._max_retries == 5

    def test_init_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(tools=toolset)
        assert generator._tools == toolset

    def test_init_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test initialization with a mixed list of Tools and Toolsets."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            function=weather,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
            function=weather,
        )
        tool3 = Tool(
            name="tool3",
            description="Third tool",
            parameters={"type": "object", "properties": {"z": {"type": "string"}}, "required": ["z"]},
            function=weather,
        )

        toolset1 = Toolset([tool2])

        # Initialize with mixed list: Tool, Toolset, Tool
        generator = GoogleGenAIChatGenerator(tools=[tool1, toolset1, tool3])

        assert generator._tools == [tool1, toolset1, tool3]
        assert isinstance(generator._tools, list)
        assert len(generator._tools) == 3
        assert isinstance(generator._tools[0], Tool)
        assert isinstance(generator._tools[1], Toolset)
        assert isinstance(generator._tools[2], Tool)

    def test_to_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        generator = GoogleGenAIChatGenerator(tools=toolset)
        data = generator.to_dict()

        assert data["init_parameters"]["tools"]["type"] == "haystack.tools.toolset.Toolset"
        assert "tools" in data["init_parameters"]["tools"]["data"]
        assert len(data["init_parameters"]["tools"]["data"]["tools"]) == len(tools)
        assert data["init_parameters"]["timeout"] is None
        assert data["init_parameters"]["max_retries"] is None

    def test_from_dict_with_toolset(self, tools, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        toolset = Toolset(tools)
        component = GoogleGenAIChatGenerator(tools=toolset)
        data = component.to_dict()

        deserialized_component = GoogleGenAIChatGenerator.from_dict(data)

        assert isinstance(deserialized_component._tools, Toolset)
        assert len(deserialized_component._tools) == len(tools)
        assert all(isinstance(tool, Tool) for tool in deserialized_component._tools)

    def test_serde_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test serialization/deserialization with mixed Tools and Toolsets."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        tool1 = Tool(
            name="tool1",
            description="First tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            function=weather,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            parameters={"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
            function=weather,
        )

        toolset1 = Toolset([tool2])

        generator = GoogleGenAIChatGenerator(tools=[tool1, toolset1])
        data = generator.to_dict()

        # Verify serialization preserves structure
        assert isinstance(data["init_parameters"]["tools"], list)
        assert len(data["init_parameters"]["tools"]) == 2
        assert data["init_parameters"]["tools"][0]["type"].endswith("Tool")
        assert data["init_parameters"]["tools"][1]["type"].endswith("Toolset")

        # Verify deserialization
        restored = GoogleGenAIChatGenerator.from_dict(data)
        assert isinstance(restored._tools, list)
        assert len(restored._tools) == 2
        assert isinstance(restored._tools[0], Tool)
        assert isinstance(restored._tools[1], Toolset)
        assert restored._tools[0].name == "tool1"
        assert len(restored._tools[1]) == 1

    def test_to_dict_with_response_format_pydantic(self, monkeypatch):
        """Test that to_dict serializes a Pydantic response_format to a JSON schema dict."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        class City(BaseModel):
            name: str
            country: str
            population: int

        generator = GoogleGenAIChatGenerator(generation_kwargs={"response_format": City})
        data = generator.to_dict()

        response_format = data["init_parameters"]["generation_kwargs"]["response_format"]
        assert response_format == {
            "properties": {
                "name": {"title": "Name", "type": "string"},
                "country": {"title": "Country", "type": "string"},
                "population": {"title": "Population", "type": "integer"},
            },
            "required": ["name", "country", "population"],
            "title": "City",
            "type": "object",
        }

    def test_to_dict_with_response_format_dict(self, monkeypatch):
        """Test that to_dict preserves a dict response_format as is."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        generator = GoogleGenAIChatGenerator(generation_kwargs={"response_format": schema})
        data = generator.to_dict()

        assert data["init_parameters"]["generation_kwargs"]["response_format"] == schema

    def test_serde_with_response_format(self, monkeypatch):
        """Test serialization/deserialization round-trip with response_format."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        generator = GoogleGenAIChatGenerator(generation_kwargs={"response_format": schema, "temperature": 0.5})
        data = generator.to_dict()

        restored = GoogleGenAIChatGenerator.from_dict(data)
        assert restored._generation_kwargs["response_format"] == schema
        assert restored._generation_kwargs["temperature"] == 0.5


class TestGoogleGenAIChatGeneratorRun:
    def test_run_non_streaming(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.models.generate_content = Mock(return_value=mock_response)

        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        assert results["replies"][0].text == "Hello"
        assert results["replies"][0].meta["model"] == "gemini-2.5-flash"
        assert results["replies"][0].meta["finish_reason"] == "stop"
        component._client.models.generate_content.assert_called_once()

    def test_run_streaming(self, monkeypatch, mock_streaming_chunk):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()

        chunks = [mock_streaming_chunk(text="Hello"), mock_streaming_chunk(text=" world", finish_reason="STOP")]
        component._client.models.generate_content_stream = Mock(return_value=iter(chunks))

        callback_chunks = []

        def callback(chunk):
            callback_chunks.append(chunk)

        results = component.run(
            [ChatMessage.from_user("Say hello")],
            streaming_callback=callback,
        )

        assert len(results["replies"]) == 1
        assert "Hello" in results["replies"][0].text
        assert " world" in results["replies"][0].text
        assert len(callback_chunks) == 2

    def test_run_extracts_system_message(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.models.generate_content = Mock(return_value=mock_response)

        messages = [
            ChatMessage.from_system("You are a helpful assistant"),
            ChatMessage.from_user("Hello"),
        ]
        component.run(messages)

        call_kwargs = component._client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.system_instruction == "You are a helpful assistant"
        # Only the user message should be in contents
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert len(contents) == 1

    def test_run_with_tools_passes_tools_to_config(self, monkeypatch, mock_response, tools):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.models.generate_content = Mock(return_value=mock_response)

        component.run([ChatMessage.from_user("Weather in Paris?")], tools=tools)

        call_kwargs = component._client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.tools is not None

    def test_run_with_safety_settings(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        safety = [{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}]
        component = GoogleGenAIChatGenerator()
        component._client.models.generate_content = Mock(return_value=mock_response)

        component.run([ChatMessage.from_user("Hello")], safety_settings=safety)

        call_kwargs = component._client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert len(config.safety_settings) == 1
        assert config.safety_settings[0].category == "HARM_CATEGORY_HARASSMENT"

    def test_run_with_generation_kwargs_override(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator(generation_kwargs={"temperature": 0.5})
        component._client.models.generate_content = Mock(return_value=mock_response)

        component.run([ChatMessage.from_user("Hello")], generation_kwargs={"temperature": 0.9})

        call_kwargs = component._client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.temperature == 0.9

    def test_run_thinking_error_raises_helpful_message(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})
        component._client.models.generate_content = Mock(side_effect=Exception("thinking_config is not supported"))

        with pytest.raises(RuntimeError, match="Thinking configuration error"):
            component.run([ChatMessage.from_user("Hello")])

    def test_run_generic_error_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.models.generate_content = Mock(side_effect=Exception("Connection timeout"))

        with pytest.raises(RuntimeError, match="Error in Google Gen AI chat generation"):
            component.run([ChatMessage.from_user("Hello")])

    def test_run_streaming_error_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()

        def failing_stream():
            msg = "Stream interrupted"
            raise Exception(msg)
            yield  # makes this a generator

        component._client.models.generate_content_stream = Mock(return_value=failing_stream())

        with pytest.raises(RuntimeError, match="Error in streaming response"):
            component.run(
                [ChatMessage.from_user("Hello")],
                streaming_callback=lambda chunk: None,
            )

    def test_from_dict_with_streaming_callback(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator(streaming_callback=print_streaming_chunk)
        data = component.to_dict()

        restored = GoogleGenAIChatGenerator.from_dict(data)
        assert restored._streaming_callback is print_streaming_chunk

    @pytest.mark.asyncio
    async def test_run_async_non_streaming(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        results = await component.run_async([ChatMessage.from_user("Capital of France?")])

        assert len(results["replies"]) == 1
        assert results["replies"][0].text == "Hello"
        assert results["replies"][0].meta["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_run_async_streaming(self, monkeypatch, mock_streaming_chunk):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()

        chunk = mock_streaming_chunk(text="Hello async", finish_reason="STOP")

        async def mock_stream():
            yield chunk

        component._client.aio.models.generate_content_stream = AsyncMock(return_value=mock_stream())

        callback_chunks = []

        async def callback(c):
            callback_chunks.append(c)

        results = await component.run_async(
            [ChatMessage.from_user("Hello")],
            streaming_callback=callback,
        )

        assert len(results["replies"]) == 1
        assert "Hello async" in results["replies"][0].text
        assert len(callback_chunks) == 1

    @pytest.mark.asyncio
    async def test_run_async_extracts_system_message(self, monkeypatch, mock_response):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        messages = [
            ChatMessage.from_system("Be helpful"),
            ChatMessage.from_user("Hi"),
        ]
        await component.run_async(messages)

        call_kwargs = component._client.aio.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config") or call_kwargs[1].get("config")
        assert config.system_instruction == "Be helpful"
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        assert len(contents) == 1

    @pytest.mark.asyncio
    async def test_run_async_thinking_error_raises_helpful_message(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})
        component._client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("thinking_config is not supported")
        )

        with pytest.raises(RuntimeError, match="Thinking configuration error"):
            await component.run_async([ChatMessage.from_user("Hello")])

    @pytest.mark.asyncio
    async def test_run_async_generic_error_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()
        component._client.aio.models.generate_content = AsyncMock(side_effect=Exception("Connection timeout"))

        with pytest.raises(RuntimeError, match="Error in async Google Gen AI chat generation"):
            await component.run_async([ChatMessage.from_user("Hello")])

    @pytest.mark.asyncio
    async def test_run_async_streaming_error_raises_runtime_error(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key")
        component = GoogleGenAIChatGenerator()

        async def failing_stream():
            msg = "Async stream interrupted"
            raise Exception(msg)
            yield  # unreachable, but makes this an async generator

        component._client.aio.models.generate_content_stream = AsyncMock(return_value=failing_stream())

        with pytest.raises(RuntimeError, match="Error in async streaming response"):
            await component.run_async(
                [ChatMessage.from_user("Hello")],
                streaming_callback=AsyncMock(),
            )


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
class TestGoogleGenAIChatGeneratorInference:
    def test_live_run(self) -> None:
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.5-flash" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    def test_run_with_multiple_images_mixed_content(self, test_files_path):
        """Test that multiple images with interleaved text maintain proper ordering."""
        client = GoogleGenAIChatGenerator()

        # Load both test images
        apple_path = test_files_path / "apple.jpg"
        banana_path = test_files_path / "banana.png"

        apple_content = ImageContent.from_file_path(apple_path, size=(100, 100))
        banana_content = ImageContent.from_file_path(banana_path, size=(100, 100))

        # Create message with interleaved text and images to test ordering preservation
        chat_message = ChatMessage.from_user(
            content_parts=[
                "Here are two fruits. First image:",
                apple_content,
                "Second image:",
                banana_content,
                "What fruits do you see? List them in order.",
            ]
        )

        response = client.run([chat_message])

        first_reply = response["replies"][0]
        assert isinstance(first_reply, ChatMessage)
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT)
        assert first_reply.text

        # Verify both fruits are mentioned in the response
        response_text = first_reply.text.lower()
        assert "apple" in response_text, "Apple should be mentioned in the response"
        assert "banana" in response_text, "Banana should be mentioned in the response"

        # Verify that apple is mentioned before banana (preserving our input order)
        apple_pos = response_text.find("apple")
        banana_pos = response_text.find("banana")
        assert apple_pos < banana_pos, (
            f"Apple should be mentioned before banana in the response. Got: {first_reply.text}"
        )

    def test_live_run_with_file_content(self, test_files_path):
        pdf_path = test_files_path / "sample_pdf_3.pdf"

        file_content = FileContent.from_file_path(file_path=pdf_path)

        chat_messages = [
            ChatMessage.from_user(
                content_parts=[file_content, "Is this document a paper about LLMs? Respond with 'yes' or 'no' only."]
            )
        ]

        generator = GoogleGenAIChatGenerator()
        results = generator.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        assert message.is_from(ChatRole.ASSISTANT)

        assert message.text
        indicates_no = any(
            phrase in message.text.lower()
            for phrase in (
                "no",
                "nope",
                "not about",
                "not a paper about",
                "it is not",
                "it's not",
                "the answer is no",
                "does not",
                "doesn't",
                "negative",
            )
        )

        assert indicates_no is True

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
        # Google Gen AI (gemini-2.5-flash and gemini-2.5-pro-preview-05-06) does not provide ids for tool calls although
        # it is in the response schema, revisit in future to see if there are changes and id is provided
        # assert tool_message.tool_calls[0].id is not None, "Tool call has no id"
        assert tool_message.tool_calls[0].tool_name == "weather"
        assert tool_message.tool_calls[0].arguments == {"city": "Paris"}

        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"
        assert tool_message.meta["finish_reason"] == "tool_calls"

        tool_call = tool_message.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

    def test_live_run_with_parallel_tools(self, tools):
        """
        Integration test that the GoogleGenAIChatGenerator component can run with parallel tools.
        """
        initial_messages = [
            ChatMessage.from_user(
                "What's the weather like in Paris and Berlin? Produce a separate tool call for each city."
            )
        ]
        component = GoogleGenAIChatGenerator(model="gemini-3-flash-preview", tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.meta["finish_reason"] == "tool_calls"

        # Google GenAI should make tool calls for both cities
        assert len(message.tool_calls) == 2
        paris_index = 0 if message.tool_calls[0].arguments["city"].lower() == "paris" else 1
        tool_call_paris = message.tool_calls[paris_index]
        assert isinstance(tool_call_paris, ToolCall)
        assert tool_call_paris.tool_name == "weather"
        assert tool_call_paris.arguments["city"].lower() == "paris"

        tool_call_berlin = message.tool_calls[1 - paris_index]
        assert isinstance(tool_call_berlin, ToolCall)
        assert tool_call_berlin.tool_name == "weather"
        assert tool_call_berlin.arguments["city"].lower() == "berlin"

        # Google GenAI expects results from both tools in separate messages
        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call_paris),
            ChatMessage.from_tool(tool_result="15°C, cloudy", origin=tool_call_berlin),
        ]

        # Response from the model contains results from both tools
        results = component.run(new_messages)
        final_message = results["replies"][0]

        assert not final_message.tool_calls, "Message has tool calls and it should not have any"
        assert len(final_message.text) > 0, "Message has no text"
        assert final_message.text and "paris" in final_message.text.lower() and "berlin" in final_message.text.lower()
        # Check that the response mentions both temperature readings
        assert "22" in final_message.text and "15" in final_message.text

    def test_live_run_with_thinking(self):
        """
        Integration test for the thinking feature with a model that supports it.
        """
        # We use a model that supports the thinking feature
        chat_messages = [ChatMessage.from_user("Why is the sky blue? Explain in one sentence.")]
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview", generation_kwargs={"thinking_level": "low"}
        )
        results = component.run(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text

        # Check for reasoning content, which should be present when thinking is enabled
        assert message.reasonings
        assert len(message.reasonings) > 0
        assert all(isinstance(r, ReasoningContent) for r in message.reasonings)

        # Check for thinking token usage
        assert "usage" in message.meta
        assert "thoughts_token_count" in message.meta["usage"]
        assert message.meta["usage"]["thoughts_token_count"] is not None
        assert message.meta["usage"]["thoughts_token_count"] > 0

    def test_live_run_with_thinking_and_tools_multi_turn(self, tools):
        """
        Integration test for thought signatures preservation in multi-turn conversations with tools.
        This verifies that thought context is maintained across turns when using tools with thinking.
        """
        # Use a model that supports thinking with tools
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview",
            tools=tools,
            generation_kwargs={"thinking_level": "low"},  # Dynamic allocation
        )

        # First turn: Ask about the weather
        messages = [ChatMessage.from_user("What's the weather in Paris?")]
        result = component.run(messages)

        assert len(result["replies"]) == 1
        first_response = result["replies"][0]

        # Should have tool calls
        assert first_response.tool_calls
        assert len(first_response.tool_calls) == 1
        assert first_response.tool_calls[0].tool_name == "weather"

        # Check for thought signatures in meta (only present with tools)
        assert "thought_signatures" in first_response.meta
        assert len(first_response.meta["thought_signatures"]) > 0

        # Second turn: Provide tool result and continue conversation
        tool_call = first_response.tool_calls[0]
        messages.extend(
            [
                first_response,  # Include the assistant's response with thought signatures
                ChatMessage.from_tool(tool_result="22°C, sunny", origin=tool_call),
                ChatMessage.from_user("Is that good weather for a picnic?"),
            ]
        )

        # The thought signatures from first_response should be preserved automatically
        result2 = component.run(messages)

        assert len(result2["replies"]) == 1
        second_response = result2["replies"][0]

        # check that the thought signatures are there
        assert "thought_signatures" in second_response.meta
        assert len(second_response.meta["thought_signatures"]) > 0

        # Should have a text response about picnic weather
        assert second_response.text
        assert "picnic" in second_response.text.lower() or "yes" in second_response.text.lower()

        # The model should maintain context from previous turns
        assert "22" in second_response.text or "sunny" in second_response.text.lower()

    def test_live_run_with_thinking_unsupported_model_fails_fast(self):
        """
        Integration test to verify that thinking configuration fails fast with unsupported models.
        """
        # gemini-2.0-flash does not support thinking
        chat_messages = [ChatMessage.from_user("Why is the sky blue?")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})

        # The call should raise a RuntimeError with a helpful message
        with pytest.raises(RuntimeError) as exc_info:
            component.run(chat_messages)

        # Verify the error message is helpful and mentions thinking configuration
        error_message = str(exc_info.value)
        assert "Thinking configuration error" in error_message
        assert "gemini-2.0" in error_message
        assert "thinking_budget" in error_message or "thinking features" in error_message
        assert "Try removing" in error_message or "use a different model" in error_message

    def test_live_run_with_structured_output_pydantic(self):
        """Test that response_format with a Pydantic model returns valid structured JSON output."""

        class City(BaseModel):
            name: str
            country: str
            population: int

        component = GoogleGenAIChatGenerator(generation_kwargs={"response_format": City})
        results = component.run([ChatMessage.from_user("Tell me about Paris. Respond in JSON.")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text

        parsed = json.loads(message.text)
        assert "name" in parsed
        assert "country" in parsed
        assert "population" in parsed

    def test_live_run_with_structured_output_dict_schema(self):
        """Test that response_format with a JSON schema dict returns valid structured JSON output."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["name", "country"],
        }

        component = GoogleGenAIChatGenerator(generation_kwargs={"response_format": schema})
        results = component.run([ChatMessage.from_user("Tell me about Paris. Respond in JSON.")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text

        parsed = json.loads(message.text)
        assert "name" in parsed
        assert "country" in parsed

    def test_live_run_agent_with_images_in_tool_result(self, test_files_path):
        def retrieve_image():
            return [
                TextContent("Here is the retrieved image."),
                ImageContent.from_file_path(test_files_path / "apple.jpg", size=(100, 100)),
            ]

        image_retriever_tool = create_tool_from_function(
            name="retrieve_image", description="Tool to retrieve an image", function=retrieve_image
        )
        image_retriever_tool.outputs_to_string = {"raw_result": True}

        agent = Agent(
            chat_generator=GoogleGenAIChatGenerator(model="gemini-3-flash-preview"),
            system_prompt="You are an Agent that can retrieve images and describe them.",
            tools=[image_retriever_tool],
        )

        user_message = ChatMessage.from_user("Retrieve the image and describe it in max 5 words.")
        result = agent.run(messages=[user_message])

        assert "apple" in result["last_message"].text.lower()


@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY", None),
    reason="Export an env var called GOOGLE_API_KEY containing the Google API key to run this test.",
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncGoogleGenAIChatGeneratorInference:
    """Test class for async functionality of GoogleGenAIChatGenerator."""

    async def test_live_run_async(self) -> None:
        """Test async version of the run method."""
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = GoogleGenAIChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text and "paris" in message.text.lower(), "Response does not contain Paris"
        assert "gemini-2.5-flash" in message.meta["model"]
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
        assert tool_message.meta["finish_reason"] == "tool_calls"

    async def test_live_run_async_with_thinking(self):
        """
        Async integration test for the thinking feature.
        """
        chat_messages = [ChatMessage.from_user("Why is the sky blue? Explain in one sentence.")]
        component = GoogleGenAIChatGenerator(
            model="gemini-3-flash-preview", generation_kwargs={"thinking_level": "low"}
        )
        results = await component.run_async(chat_messages)

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert message.text

        # Check for reasoning content
        assert message.reasonings
        assert len(message.reasonings) > 0
        assert all(isinstance(r, ReasoningContent) for r in message.reasonings)

        # Check for thinking token usage
        assert "usage" in message.meta
        assert "thoughts_token_count" in message.meta["usage"]
        assert message.meta["usage"]["thoughts_token_count"] is not None
        assert message.meta["usage"]["thoughts_token_count"] > 0

    async def test_live_run_async_with_thinking_unsupported_model_fails_fast(self):
        """
        Async integration test to verify that thinking configuration fails fast with unsupported models.
        This tests the fail-fast principle - no silent fallbacks.
        """
        # Use a model that does NOT support thinking features (gemini-2.0-flash)
        chat_messages = [ChatMessage.from_user("Why is the sky blue?")]
        component = GoogleGenAIChatGenerator(model="gemini-2.0-flash", generation_kwargs={"thinking_budget": 1024})

        # The call should raise a RuntimeError with a helpful message
        with pytest.raises(RuntimeError) as exc_info:
            await component.run_async(chat_messages)

        # Verify the error message is helpful and mentions thinking configuration
        error_message = str(exc_info.value)
        assert "Thinking configuration error" in error_message
        assert "gemini-2.0" in error_message
        assert "thinking_budget" in error_message or "thinking features" in error_message
        assert "Try removing" in error_message or "use a different model" in error_message

    async def test_live_run_async_with_structured_output(self):
        """Async integration test for structured output with a Pydantic model."""

        class City(BaseModel):
            name: str
            country: str
            population: int

        component = GoogleGenAIChatGenerator(generation_kwargs={"response_format": City})
        results = await component.run_async([ChatMessage.from_user("Tell me about Paris. Respond in JSON.")])

        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text

        parsed = json.loads(message.text)
        assert "name" in parsed
        assert "country" in parsed
        assert "population" in parsed

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
