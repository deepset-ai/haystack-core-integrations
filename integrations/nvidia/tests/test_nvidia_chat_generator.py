# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.nvidia.chat.chat_generator import NvidiaChatGenerator
from haystack_integrations.utils.nvidia.models import DEFAULT_API_URL


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


def echo_function(text: str) -> str:
    """Echo a text."""
    return text


@pytest.fixture
def tools():
    tool_parameters = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather,
    )

    return [tool]


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="meta/llama-3.1-8b-instruct",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


@pytest.fixture
def mock_async_chat_completion():
    """
    Mock the Async OpenAI API completion response and reuse it for async tests
    """
    with patch(
        "openai.resources.chat.completions.AsyncCompletions.create",
        new_callable=AsyncMock,
    ) as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="meta/llama-3.1-8b-instruct",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello world!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={
                "prompt_tokens": 57,
                "completion_tokens": 40,
                "total_tokens": 97,
            },
        )
        # For async mocks, the return value should be awaitable
        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestNvidiaChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        component = NvidiaChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "meta/llama-3.1-8b-instruct"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            NvidiaChatGenerator()

    def test_init_with_parameters(self):
        component = NvidiaChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="meta/llama-3.1-8b-instruct",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "meta/llama-3.1-8b-instruct"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        component = NvidiaChatGenerator()
        data = component.to_dict()

        assert (
            data["type"] == "haystack_integrations.components.generators.nvidia.chat.chat_generator.NvidiaChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["NVIDIA_API_KEY"], "strict": True, "type": "env_var"},
            "model": "meta/llama-3.1-8b-instruct",
            "streaming_callback": None,
            "api_base_url": DEFAULT_API_URL,
            "generation_kwargs": {},
            "tools": None,
            "timeout": None,
            "max_retries": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_extra_body(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        extra_body = {
            "guardrails": {"config_id": "demo-self-check-input-output"},
        }
        component = NvidiaChatGenerator(generation_kwargs={"extra_body": extra_body})
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["extra_body"] == extra_body
        assert kwargs["model"] == "meta/llama-3.1-8b-instruct"
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the capital of France"},
        ]

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = NvidiaChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "meta/llama-3.1-8b-instruct" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = NvidiaChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = NvidiaChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "meta/llama-3.1-8b-instruct" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_guided_json_schema(self):
        json_schema = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "rating": {"type": "number"}},
            "required": ["title", "rating"],
        }
        chat_messages = [
            ChatMessage.from_user(
                """
            Return the title and the rating based on the following movie review according
            to the provided json schema.
            Review: Inception is a really well made film. I rate it four stars out of five."""
            )
        ]

        component = NvidiaChatGenerator(
            model="meta/llama-3.1-70b-instruct",
            generation_kwargs={"extra_body": {"nvext": {"guided_json": json_schema}}},
        )

        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0].text
        output = json.loads(message)
        assert output["title"] == "Inception"
        assert "rating" in output

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_json_object(self):
        chat_messages = [
            ChatMessage.from_user(
                """
            Return the title and the rating based on the following movie review according
            to the provided json schema.
            Review: Inception is a really well made film. I rate it four stars out of five."""
            )
        ]

        component = NvidiaChatGenerator(
            model="meta/llama-3.1-70b-instruct",
            generation_kwargs={"response_format": {"type": "json_object"}},
        )

        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0].text
        output = json.loads(message)
        assert "title" in output
        assert "rating" in output
        assert isinstance(output["rating"], int)
        assert "Inception" in output["title"]

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    def test_integration_mixing_tools_and_toolset(self):
        """Test mixing Tool list and Toolset at runtime."""

        def weather_function(city: str) -> str:
            """Get weather information for a city."""
            return f"Weather in {city}: 22°C, sunny"

        def time_function(city: str) -> str:
            """Get current time in a city."""
            return f"Current time in {city}: 14:30"

        def echo_function(text: str) -> str:
            """Echo a text."""
            return text

        # Create tools
        weather_tool = Tool(
            name="weather",
            description="Get weather information for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather_function,
        )

        time_tool = Tool(
            name="time",
            description="Get current time in a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=time_function,
        )

        echo_tool = Tool(
            name="echo",
            description="Echo a text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo_function,
        )

        # Create Toolset with weather and time tools
        toolset = Toolset([weather_tool, time_tool])

        # Initialize without tools
        component = NvidiaChatGenerator()

        # Mix tools and toolset at runtime
        messages = [ChatMessage.from_user("What's the weather in Tokyo and echo 'test'")]
        results = component.run(messages, tools=[echo_tool, toolset])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Should have access to both echo tool and tools from toolset
        assert message.tool_calls is not None
        assert len(message.tool_calls) >= 1

        # Check that we can use tools from both the list and toolset
        tool_names = [call.tool_name for call in message.tool_calls]
        assert "echo" in tool_names or "weather" in tool_names

    def test_to_dict_with_mixed_tools_and_toolset(self, tools, monkeypatch):
        """Test serialization with a mixed list containing both Tool and Toolset objects."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")

        # Create additional tools for the toolset using module-level function
        echo_tool = Tool(
            name="echo",
            description="Echo a text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo_function,
        )

        # Create a mixed list: some individual tools + a toolset
        toolset = Toolset([echo_tool])
        mixed_tools = [*tools, toolset]  # List containing both Tool objects and a Toolset

        component = NvidiaChatGenerator(model="meta/llama-3.1-8b-instruct", tools=mixed_tools)
        data = component.to_dict()

        assert data["init_parameters"]["tools"] is not None
        assert isinstance(data["init_parameters"]["tools"], list)
        assert len(data["init_parameters"]["tools"]) == len(mixed_tools)

        # Check that we have both Tool and Toolset in the serialized data
        tool_types = [tool["type"] for tool in data["init_parameters"]["tools"]]
        assert "haystack.tools.tool.Tool" in tool_types
        assert "haystack.tools.toolset.Toolset" in tool_types

    def test_from_dict_with_mixed_tools_and_toolset(self, tools, monkeypatch):
        """Test deserialization with a mixed list containing both Tool and Toolset objects."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")

        # Create additional tools for the toolset using module-level function
        echo_tool = Tool(
            name="echo",
            description="Echo a text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo_function,
        )

        # Create a mixed list: some individual tools + a toolset
        toolset = Toolset([echo_tool])
        mixed_tools = [*tools, toolset]  # List containing both Tool objects and a Toolset

        component = NvidiaChatGenerator(model="meta/llama-3.1-8b-instruct", tools=mixed_tools)
        data = component.to_dict()

        deserialized_component = NvidiaChatGenerator.from_dict(data)

        assert isinstance(deserialized_component.tools, list)
        assert len(deserialized_component.tools) == len(mixed_tools)

        # Check that we have both Tool and Toolset objects in the deserialized list
        tool_types = [type(tool).__name__ for tool in deserialized_component.tools]
        assert "Tool" in tool_types
        assert "Toolset" in tool_types


class TestNvidiaChatGeneratorAsync:
    def test_init_default_async(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
        component = NvidiaChatGenerator()

        assert isinstance(component.async_client, AsyncOpenAI)
        assert component.async_client.api_key == "test-api-key"
        assert not component.generation_kwargs

    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_async_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaChatGenerator()
        response = await component.run_async(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_params(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        component = NvidiaChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = await component.run_async(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_extra_body(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "fake-api-key")
        extra_body = {
            "guardrails": {"config_id": "demo-self-check-input-output"},
        }
        component = NvidiaChatGenerator(generation_kwargs={"extra_body": extra_body})
        response = await component.run_async(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["extra_body"] == extra_body
        assert kwargs["model"] == "meta/llama-3.1-8b-instruct"
        assert kwargs["messages"] == [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What's the capital of France"},
        ]

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = NvidiaChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "meta/llama-3.1-8b-instruct" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_streaming_async(self):
        counter = 0
        responses = ""

        async def callback(chunk: StreamingChunk):
            nonlocal counter
            nonlocal responses
            counter += 1
            responses += chunk.content if chunk.content else ""

        component = NvidiaChatGenerator(streaming_callback=callback)
        results = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "meta/llama-3.1-8b-instruct" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert counter > 1
        assert "Paris" in responses

    @pytest.mark.skipif(
        not os.environ.get("NVIDIA_API_KEY", None),
        reason="Export an env var called NVIDIA_API_KEY containing the NVIDIA API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_integration_mixing_tools_and_toolset_async(self):
        """Test mixing Tool list and Toolset at runtime in async mode."""

        def weather_function(city: str) -> str:
            """Get weather information for a city."""
            return f"Weather in {city}: 22°C, sunny"

        def time_function(city: str) -> str:
            """Get current time in a city."""
            return f"Current time in {city}: 14:30"

        def echo_function(text: str) -> str:
            """Echo a text."""
            return text

        # Create tools
        weather_tool = Tool(
            name="weather",
            description="Get weather information for a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather_function,
        )

        time_tool = Tool(
            name="time",
            description="Get current time in a city",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=time_function,
        )

        echo_tool = Tool(
            name="echo",
            description="Echo a text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo_function,
        )

        # Create Toolset with weather and time tools
        toolset = Toolset([weather_tool, time_tool])

        # Initialize without tools
        component = NvidiaChatGenerator()

        # Mix tools and toolset at runtime
        messages = [ChatMessage.from_user("What's the weather in Tokyo and echo 'test'")]
        results = await component.run_async(messages, tools=[echo_tool, toolset])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Should have access to both echo tool and tools from toolset
        assert message.tool_calls is not None
        assert len(message.tool_calls) >= 1

        # Check that we can use tools from both the list and toolset
        tool_names = [call.tool_name for call in message.tool_calls]
        assert "echo" in tool_names or "weather" in tool_names
