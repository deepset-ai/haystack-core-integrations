import json
import os
from datetime import datetime
from unittest.mock import ANY, patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ComponentInfo, StreamingChunk, ToolCall, ToolCallDelta
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from haystack_integrations.components.generators.mistral.chat.chat_generator import MistralChatGenerator


class CollectorCallback:
    """
    Callback to collect streaming chunks for testing purposes.
    """

    def __init__(self):
        self.chunks = []

    def __call__(self, chunk: StreamingChunk) -> None:
        self.chunks.append(chunk)


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
    return f"The population of {city} is 2.1 million"


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
            model="mistral-small-latest",
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


class TestMistralChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        component = MistralChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "mistral-small-latest"
        assert component.api_base_url == "https://api.mistral.ai/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            MistralChatGenerator()

    def test_init_with_parameters(self):
        component = MistralChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="mistral-small",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "mistral-small"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")
        component = MistralChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
            "model": "mistral-small-latest",
            "streaming_callback": None,
            "api_base_url": "https://api.mistral.ai/v1",
            "generation_kwargs": {},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")

        class NobelPrizeInfo(BaseModel):
            recipient_name: str
            award_year: int

        schema = {
            "json_schema": {
                "name": "NobelPrizeInfo",
                "schema": {
                    "additionalProperties": False,
                    "properties": {
                        "award_year": {
                            "title": "Award Year",
                            "type": "integer",
                        },
                        "recipient_name": {
                            "title": "Recipient Name",
                            "type": "string",
                        },
                    },
                    "required": [
                        "recipient_name",
                        "award_year",
                    ],
                    "title": "NobelPrizeInfo",
                    "type": "object",
                },
                "strict": True,
            },
            "type": "json_schema",
        }

        component = MistralChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="mistral-small",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params", "response_format": NobelPrizeInfo},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "mistral-small",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params", "response_format": schema},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-small",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        component = MistralChatGenerator.from_dict(data)
        assert component.model == "mistral-small"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("MISTRAL_API_KEY")

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["MISTRAL_API_KEY"], "strict": True, "type": "env_var"},
                "model": "mistral-small",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            MistralChatGenerator.from_dict(data)

    def test_init_with_mixed_tools(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-api-key")

        weather_tool = Tool(
            name="weather",
            description="Weather lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=weather,
        )
        population_tool = Tool(
            name="population",
            description="Population lookup",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            function=population,
        )
        toolset = Toolset([population_tool])

        component = MistralChatGenerator(tools=[weather_tool, toolset])

        assert component.tools == [weather_tool, toolset]

    def test_handle_stream_response(self):
        mistral_chunks = [
            ChatCompletionChunk(
                id="76535283139540de943bc2036121d4c5",
                choices=[ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0)],
                created=1750076261,
                model="mistral-small-latest",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="76535283139540de943bc2036121d4c5",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="FL1FFlqUG",
                                    function=ChoiceDeltaToolCallFunction(arguments='{"city": "Paris"}', name="weather"),
                                ),
                                ChoiceDeltaToolCall(
                                    index=1,
                                    id="xSuhp66iB",
                                    function=ChoiceDeltaToolCallFunction(
                                        arguments='{"city": "Berlin"}', name="weather"
                                    ),
                                ),
                            ],
                        ),
                        finish_reason="tool_calls",
                        index=0,
                    )
                ],
                created=1750076261,
                model="mistral-small-latest",
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=35,
                    prompt_tokens=77,
                    total_tokens=112,
                ),
            ),
        ]

        collector_callback = CollectorCallback()
        llm = MistralChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(mistral_chunks, callback=collector_callback)[0]  # type: ignore

        # Verify the callback collected the expected number of chunks
        # We expect 2 chunks: one for the initial empty content and one for the tool calls
        assert len(collector_callback.chunks) == 2
        assert collector_callback.chunks[0] == StreamingChunk(
            content="",
            meta={
                "model": "mistral-small-latest",
                "index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": ANY,
                "usage": None,
            },
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
                name=None,
            ),
        )
        assert collector_callback.chunks[1] == StreamingChunk(
            content="",
            meta={
                "model": "mistral-small-latest",
                "index": 0,
                "tool_calls": [
                    ChoiceDeltaToolCall(
                        index=0,
                        id="FL1FFlqUG",
                        function=ChoiceDeltaToolCallFunction(arguments='{"city": "Paris"}', name="weather"),
                    ),
                    ChoiceDeltaToolCall(
                        index=1,
                        id="xSuhp66iB",
                        function=ChoiceDeltaToolCallFunction(arguments='{"city": "Berlin"}', name="weather"),
                    ),
                ],
                "finish_reason": "tool_calls",
                "received_at": ANY,
                "usage": {
                    "completion_tokens": 35,
                    "prompt_tokens": 77,
                    "total_tokens": 112,
                    "completion_tokens_details": None,
                    "prompt_tokens_details": None,
                },
            },
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=[
                ToolCallDelta(index=0, tool_name="weather", arguments='{"city": "Paris"}', id="FL1FFlqUG"),
                ToolCallDelta(index=1, tool_name="weather", arguments='{"city": "Berlin"}', id="xSuhp66iB"),
            ],
            start=True,
            finish_reason="tool_calls",
        )

        # Assert text is empty
        assert result.text is None

        # Verify both tool calls were found and processed
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "FL1FFlqUG"
        assert result.tool_calls[0].tool_name == "weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.tool_calls[1].id == "xSuhp66iB"
        assert result.tool_calls[1].tool_name == "weather"
        assert result.tool_calls[1].arguments == {"city": "Berlin"}

        # Verify meta information
        assert result.meta["model"] == "mistral-small-latest"
        assert result.meta["finish_reason"] == "tool_calls"
        assert result.meta["index"] == 0
        assert result.meta["completion_start_time"] is not None
        assert result.meta["usage"] == {
            "completion_tokens": 35,
            "prompt_tokens": 77,
            "total_tokens": 112,
            "completion_tokens_details": None,
            "prompt_tokens_details": None,
        }

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-api-key")
        component = MistralChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "fake-api-key")
        component = MistralChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
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

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = MistralChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "mistral-small-latest" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = MistralChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
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
        component = MistralChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "mistral-small-latest" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_response_format(self):
        class NobelPrizeInfo(BaseModel):
            recipient_name: str
            award_year: int
            category: str
            achievement_description: str
            nationality: str

        chat_messages = [
            ChatMessage.from_user(
                "In 2021, American scientist David Julius received the Nobel Prize in"
                " Physiology or Medicine for his groundbreaking discoveries on how the human body"
                " senses temperature and touch."
            )
        ]
        component = MistralChatGenerator(generation_kwargs={"response_format": NobelPrizeInfo})
        results = component.run(chat_messages)
        assert isinstance(results, dict)
        assert "replies" in results
        assert isinstance(results["replies"], list)
        assert len(results["replies"]) == 1
        assert isinstance(results["replies"][0], ChatMessage)
        message = results["replies"][0]
        assert isinstance(message.text, str)
        msg = json.loads(message.text)
        assert msg["recipient_name"] == "David Julius"
        assert msg["award_year"] == 2021
        assert "category" in msg
        assert "achievement_description" in msg
        assert msg["nationality"] == "American"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_json_schema(self):
        response_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "CapitalCity",
                "strict": True,
                "schema": {
                    "title": "CapitalCity",
                    "type": "object",
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
            },
        }

        chat_messages = [ChatMessage.from_user("What's the capital of France?")]
        comp = MistralChatGenerator(generation_kwargs={"response_format": response_schema})
        results = comp.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Paris" in msg["city"]
        assert isinstance(msg["country"], str)
        assert "France" in msg["country"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = MistralChatGenerator(tools=tools)
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

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the MistralChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = MistralChatGenerator(tools=tools)
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "any"})

        assert len(results["replies"]) == 1

        # Find the message with tool calls
        tool_message = results["replies"][0]

        assert isinstance(tool_message, ChatMessage)
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)

        for tool_call in tool_calls:
            assert tool_call.id is not None
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]
        assert tool_message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            initial_messages[0],
            tool_message,
            ChatMessage.from_tool(tool_result="22° C and sunny", origin=tool_calls[0]),
            ChatMessage.from_tool(tool_result="16° C and windy", origin=tool_calls[1]),
        ]
        # Pass the tool result to the model to get the final response
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert final_message.is_from(ChatRole.ASSISTANT)
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the MistralChatGenerator component can run with tools and streaming.
        """
        component = MistralChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris and Berlin?")],
            generation_kwargs={"tool_choice": "any"},
        )

        assert len(results["replies"]) == 1

        # Find the message with tool calls
        tool_message = results["replies"][0]

        assert isinstance(tool_message, ChatMessage)
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) == 2
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)

        for tool_call in tool_calls:
            assert tool_call.id is not None
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

        arguments = [tool_call.arguments for tool_call in tool_calls]
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_pipeline_with_mistral_chat_generator(self, tools):
        """
        Test that the MistralChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", MistralChatGenerator(tools=tools))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={
                "generator": {
                    "messages": [ChatMessage.from_user("What's the weather like in Paris?")],
                    "generation_kwargs": {"tool_choice": "any"},
                }
            }
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of MistralChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock API key
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = MistralChatGenerator(
            model="mistral-small",
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        # Create and configure pipeline
        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        # Get pipeline dictionary and verify its structure
        pipeline_dict = pipeline.to_dict()
        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.mistral.chat.chat_generator.MistralChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["MISTRAL_API_KEY"], "strict": True},
                        "model": "mistral-small",
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.mistral.ai/v1",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_mistral_chat_generator.weather",
                                },
                            }
                        ],
                        "http_client_kwargs": None,
                        "timeout": None,
                        "max_retries": None,
                    },
                }
            },
            "connections": [],
        }

        if not hasattr(pipeline, "_connection_type_validation"):
            expected_dict.pop("connection_type_validation")

        # add outputs_to_string, inputs_from_state and outputs_to_state tool parameters for compatibility with
        # haystack-ai>=2.12.0
        if hasattr(tool, "outputs_to_string"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"]["outputs_to_string"] = (
                tool.outputs_to_string
            )
        if hasattr(tool, "inputs_from_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"]["inputs_from_state"] = (
                tool.inputs_from_state
            )
        if hasattr(tool, "outputs_to_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"]["outputs_to_state"] = (
                tool.outputs_to_state
            )

        assert pipeline_dict == expected_dict

        # Test YAML serialization/deserialization
        pipeline_yaml = pipeline.dumps()
        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

        # Verify the loaded pipeline's generator has the same configuration
        loaded_generator = new_pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback == generator.streaming_callback
        assert len(loaded_generator.tools) == len(generator.tools)
        assert loaded_generator.tools[0].name == generator.tools[0].name
        assert loaded_generator.tools[0].description == generator.tools[0].description
        assert loaded_generator.tools[0].parameters == generator.tools[0].parameters

    @pytest.mark.skipif(
        not os.environ.get("MISTRAL_API_KEY", None),
        reason="Export an env var called MISTRAL_API_KEY containing the Mistral API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_mixed_tools(self, mixed_tools):
        """
        Integration test that verifies MistralChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """
        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = MistralChatGenerator(tools=mixed_tools)
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
