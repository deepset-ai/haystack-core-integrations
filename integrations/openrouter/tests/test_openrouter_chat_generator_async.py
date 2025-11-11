import os
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest
import pytz
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    StreamingChunk,
)
from haystack.tools import Tool, Toolset
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.openrouter.chat.chat_generator import (
    OpenRouterChatGenerator,
)


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


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
            model="openai/gpt-4o-mini",
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


class TestOpenRouterChatGeneratorAsync:
    def test_init_default_async(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
        component = OpenRouterChatGenerator()

        assert isinstance(component.async_client, AsyncOpenAI)
        assert component.async_client.api_key == "test-api-key"
        assert component.async_client.base_url == "https://openrouter.ai/api/v1/"
        assert not component.generation_kwargs

    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_async_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator()
        response = await component.run_async(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_params(self, chat_messages, mock_async_chat_completion, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = await component.run_async(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        _, kwargs = mock_async_chat_completion.call_args
        assert kwargs["extra_body"]["max_tokens"] == 10
        assert kwargs["extra_body"]["temperature"] == 0.5

        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_async(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenRouterChatGenerator()
        results = await component.run_async(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "openai/gpt-4o-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
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

        component = OpenRouterChatGenerator(streaming_callback=callback)
        results = await component.run_async([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "openai/gpt-4o-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert counter > 1
        assert "Paris" in responses

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_with_tools_and_response_async(self, tools):
        """
        Integration test that the OpenRouterChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenRouterChatGenerator(tools=tools)
        results = await component.run_async(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_call:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # Pass the tool result to the model to get the final response
        results = await component.run_async(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_live_run_with_tools_streaming_async(self, tools):
        """
        Integration test that the OpenRouterChatGenerator component can run with tools and streaming.
        """

        counter = 0
        tool_calls = []

        async def callback(chunk: StreamingChunk):
            nonlocal counter
            nonlocal tool_calls
            counter += 1
            if chunk.meta.get("tool_calls"):
                tool_calls.extend(chunk.meta["tool_calls"])

        component = OpenRouterChatGenerator(tools=tools, streaming_callback=callback)
        results = await component.run_async(
            [ChatMessage.from_user("What's the weather like in Paris?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(results["replies"]) > 0, "No replies received"
        assert counter > 1, "Streaming callback was not called multiple times"
        assert tool_calls, "No tool calls received in streaming"

        # Find the message with tool calls
        tool_message = None
        for message in results["replies"]:
            if message.tool_call:
                tool_message = message
                break

        assert tool_message is not None, "No message with tool call found"
        assert isinstance(tool_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_call = tool_message.tool_call
        assert tool_call.id, "Tool call does not contain value for 'id' key"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenRouter API key to run this test.",
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

        # Initialize with no tools, we'll pass them at runtime
        component = OpenRouterChatGenerator()

        # Pass mixed list: echo_tool (individual) and toolset (weather + time) at runtime
        # This tests that both individual tools and toolsets can be combined
        messages = [ChatMessage.from_user("Echo this: Hello World")]
        results = await component.run_async(messages, tools=[echo_tool, toolset])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Should be able to use echo_tool from the runtime mixed list
        assert message.tool_calls is not None
        tool_call = message.tool_calls[0]
        assert tool_call.tool_name == "echo"
        assert tool_call.arguments == {"text": "Hello World"}
