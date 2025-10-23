from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.llama_stack.chat.chat_generator import LlamaStackChatGenerator


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's the capital of France"),
    ]


def weather(city: str):
    """Get weather for a given city."""
    return f"The weather in {city} is sunny and 32°C"


def population(city: str):
    """Get population for a given city."""
    return f"The population of {city} is 2.2 million"


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=weather,
    )

    return [tool]


@pytest.fixture
def mixed_tools():
    """Fixture that returns a mixed list of Tool and Toolset."""
    weather_tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=weather,
    )
    population_tool = Tool(
        name="population",
        description="useful to determine the population of a given location",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        function=population,
    )
    toolset = Toolset([population_tool])
    return [weather_tool, toolset]


@pytest.fixture
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="ollama/llama3.2:3b",
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


class TestLlamaStackChatGenerator:
    def test_init_default(self):
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
        assert component.model == "ollama/llama3.2:3b"
        assert component.api_base_url == "http://localhost:8321/v1/openai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_with_parameters(self):
        component = LlamaStackChatGenerator(
            model="ollama/llama3.2:3b",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.model == "ollama/llama3.2:3b"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(
        self,
    ):
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
        )

        expected_params = {
            "model": "ollama/llama3.2:3b",
            "streaming_callback": None,
            "api_base_url": "http://localhost:8321/v1/openai/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
            "tools_strict": False,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(
        self,
    ):
        component = LlamaStackChatGenerator(
            model="ollama/llama3.2:3b",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
            tools_strict=True,
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
        )

        expected_params = {
            "model": "ollama/llama3.2:3b",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
            "tools_strict": True,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(
        self,
    ):
        data = {
            "type": (
                "haystack_integrations.components.generators.llama_stack.chat.chat_generator.LlamaStackChatGenerator"
            ),
            "init_parameters": {
                "model": "ollama/llama3.2:3b",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
                "tools_strict": False,
            },
        }
        component = LlamaStackChatGenerator.from_dict(data)
        assert component.model == "ollama/llama3.2:3b"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10
        assert not component.tools_strict

    def test_init_with_mixed_tools(self):
        def tool_fn(city: str) -> str:
            return city

        weather_tool = Tool(
            name="weather",
            description="Weather lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        population_tool = Tool(
            name="population",
            description="Population lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=tool_fn,
        )
        toolset = Toolset([population_tool])

        generator = LlamaStackChatGenerator(
            model="ollama/llama3.2:3b",
            tools=[weather_tool, toolset],
        )

        assert generator.tools == [weather_tool, toolset]

    def test_run(self, chat_messages, mock_chat_completion):  # noqa: ARG002
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion):
        component = LlamaStackChatGenerator(
            model="ollama/llama3.2:3b",
            generation_kwargs={"max_tokens": 10, "temperature": 0.5},
        )
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        # for LlamaStack
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5
        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "llama3.2:3b" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

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
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b", streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "llama3.2:3b" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b", tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text == ""

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the LlamaStackChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b", tools=tools)
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

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
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.integration
    def test_live_run_with_mixed_tools(self, mixed_tools):
        """
        Integration test that verifies LlamaStackChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """
        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = LlamaStackChatGenerator(model="ollama/llama3.2:3b", tools=mixed_tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert isinstance(tool_call_message, ChatMessage), "Tool message is not a ChatMessage instance"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_calls = tool_call_message.tool_calls
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

        # Verify we got calls to both weather and population tools
        tool_names = {tc.tool_name for tc in tool_calls}
        assert "weather" in tool_names, "Expected 'weather' tool call"
        assert "population" in tool_names, "Expected 'population' tool call"

        # Verify tool call details
        for tool_call in tool_calls:
            assert tool_call.id, "Tool call does not contain value for 'id' key"
            assert tool_call.tool_name in ["weather", "population"]
            assert "city" in tool_call.arguments
            assert tool_call.arguments["city"] in ["Paris", "Berlin"]
            assert tool_call_message.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, tool_call_message, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_call
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()
