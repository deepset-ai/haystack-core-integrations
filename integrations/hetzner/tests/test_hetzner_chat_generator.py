# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk

try:
    from haystack.components.tools import ToolInvoker
except ImportError:  # ToolInvoker was removed in Haystack 3.0
    ToolInvoker = None
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from pydantic import BaseModel

from haystack_integrations.components.generators.hetzner.chat.chat_generator import HetznerChatGenerator

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_API_BASE_URL = "https://inference.hetzner.com/api/v1"


class CalendarEvent(BaseModel):
    event_name: str
    event_date: str
    event_location: str


@pytest.fixture
def calendar_event_model():
    return CalendarEvent


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
            model=DEFAULT_MODEL,
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
            usage=CompletionUsage(prompt_tokens=57, completion_tokens=40, total_tokens=97),
        )

        mock_chat_completion_create.return_value = completion
        yield mock_chat_completion_create


class TestHetznerChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = HetznerChatGenerator(api_key=Secret.from_env_var("ENV_VAR"))
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == DEFAULT_MODEL
        assert component.api_base_url == DEFAULT_API_BASE_URL
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_supported_models(self) -> None:
        """SUPPORTED_MODELS lists the models documented by Hetzner."""
        assert HetznerChatGenerator.SUPPORTED_MODELS == ["Qwen/Qwen3.6-35B-A3B-FP8", "Qwen3.8-27B"]
        assert HetznerChatGenerator.SUPPORTED_MODELS[0] == DEFAULT_MODEL

    def test_init_with_unsupported_model_warns(self, caplog, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        component = HetznerChatGenerator(model="some-model-not-served-by-hetzner")
        # unknown models are passed on to the API, the user is only warned
        assert component.model == "some-model-not-served-by-hetzner"
        assert "not in the list of models known to be served by the Hetzner Inference API" in caplog.text

    def test_warm_up(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        component = HetznerChatGenerator()
        component.warm_up()  # with haystack-ai >= 3.0 the client is created during warm-up
        assert component.client.api_key == "test-api-key"

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("HETZNER_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            # haystack-ai 2.x raises at init; haystack-ai >= 3.0 raises when the client is created in warm_up
            component = HetznerChatGenerator()
            component.warm_up()

    def test_init_with_parameters(self):
        component = HetznerChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="Qwen3.8-27B",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.api_key.resolve_value() == "test-api-key"
        assert component.model == "Qwen3.8-27B"
        assert component.api_base_url == "test-base-url"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")
        component = HetznerChatGenerator(api_key=Secret.from_env_var("HETZNER_API_KEY"))
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.hetzner.chat.chat_generator.HetznerChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["HETZNER_API_KEY"], "strict": True, "type": "env_var"},
            "model": DEFAULT_MODEL,
            "streaming_callback": None,
            "api_base_url": DEFAULT_API_BASE_URL,
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch, calendar_event_model):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = HetznerChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="Qwen3.8-27B",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
                "response_format": calendar_event_model,
            },
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.hetzner.chat.chat_generator.HetznerChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "Qwen3.8-27B",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {
                "max_tokens": 10,
                "some_test_param": "test-params",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "CalendarEvent",
                        "strict": True,
                        "schema": {
                            "properties": {
                                "event_name": {"title": "Event Name", "type": "string"},
                                "event_date": {"title": "Event Date", "type": "string"},
                                "event_location": {"title": "Event Location", "type": "string"},
                            },
                            "required": ["event_name", "event_date", "event_location"],
                            "title": "CalendarEvent",
                            "type": "object",
                            "additionalProperties": False,
                        },
                    },
                },
            },
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.hetzner.chat.chat_generator.HetznerChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["HETZNER_API_KEY"], "strict": True, "type": "env_var"},
                "model": DEFAULT_MODEL,
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = HetznerChatGenerator.from_dict(data)
        assert component.model == DEFAULT_MODEL
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("HETZNER_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        component = HetznerChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("HETZNER_API_KEY", "fake-api-key")
        component = HetznerChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # check that the component calls the Hetzner API with the correct parameters
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5
        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_init_with_mixed_tools(self, monkeypatch, mixed_tools):
        """Test that HetznerChatGenerator can be initialized with mixed Tool and Toolset."""
        monkeypatch.setenv("HETZNER_API_KEY", "test-api-key")

        component = HetznerChatGenerator(tools=mixed_tools)

        assert component.tools == mixed_tools

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of HetznerChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock API key
        monkeypatch.setenv("ENV_VAR", "test-key")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = HetznerChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        # Create and configure pipeline
        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        # Get pipeline dictionary and verify its structure
        pipeline_dict = pipeline.to_dict()

        # the Tool serialization format is owned by haystack-ai and varies across its versions; the
        # dumps/loads round-trip below covers the tools, so exclude them from the pinned-dict comparison
        tools_entries = pipeline_dict["components"]["generator"]["init_parameters"].pop("tools")
        assert len(tools_entries) == 1
        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.hetzner.chat.chat_generator.HetznerChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": True},
                        "model": DEFAULT_MODEL,
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": DEFAULT_API_BASE_URL,
                        "generation_kwargs": {"temperature": 0.7},
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
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = HetznerChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert DEFAULT_MODEL in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = HetznerChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
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
        component = HetznerChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert DEFAULT_MODEL in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = HetznerChatGenerator(tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the HetznerChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = HetznerChatGenerator(tools=tools)
        results = component.run(messages=initial_messages, generation_kwargs={"tool_choice": "auto"})

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
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
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
        component = HetznerChatGenerator(generation_kwargs={"response_format": response_schema})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "Paris" in msg["city"]
        assert isinstance(msg["country"], str)
        assert "France" in msg["country"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_pydantic_model(self, calendar_event_model):
        chat_messages = [
            ChatMessage.from_user("The marketing summit takes place on October 12th at the Hilton Hotel downtown.")
        ]
        component = HetznerChatGenerator(generation_kwargs={"response_format": calendar_event_model})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        msg = json.loads(message.text)
        assert "marketing summit" in msg["event_name"].lower()
        assert isinstance(msg["event_date"], str)
        assert isinstance(msg["event_location"], str)

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_mixed_tools(self, mixed_tools):
        """
        Integration test that verifies HetznerChatGenerator works with mixed Tool and Toolset.
        """
        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = HetznerChatGenerator(tools=mixed_tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        # Find the message with tool calls
        tool_call_message = None
        for message in results["replies"]:
            if message.tool_calls:
                tool_call_message = message
                break

        assert tool_call_message is not None, "No message with tool call found"
        assert ChatMessage.is_from(tool_call_message, ChatRole.ASSISTANT), "Tool message is not from the assistant"

        tool_names = {tool_call.tool_name for tool_call in tool_call_message.tool_calls}
        assert "weather" in tool_names, "Expected 'weather' tool call"
        assert "population" in tool_names, "Expected 'population' tool call"

    @pytest.mark.skipif(
        not os.environ.get("HETZNER_API_KEY", None),
        reason="Export an env var called HETZNER_API_KEY containing the Hetzner API token to run this test.",
    )
    @pytest.mark.integration
    @pytest.mark.skipif(ToolInvoker is None, reason="ToolInvoker is not available in the installed haystack-ai version")
    def test_pipeline_with_hetzner_chat_generator(self, tools):
        """
        Test that the HetznerChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", HetznerChatGenerator(tools=tools))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={
                "generator": {
                    "messages": [ChatMessage.from_user("What's the weather like in Paris?")],
                    "generation_kwargs": {"tool_choice": "auto"},
                }
            }
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )


class TestChatCompletionChunkConversion:
    def test_handle_stream_response(self):
        hetzner_chunks = [
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0)],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    id="call_zznlVyVfK0GJwY28SShJpDCh",
                                    function=ChoiceDeltaToolCallFunction(arguments="", name="weather"),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                    )
                ],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='{"ci'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                    )
                ],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='ty": '),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                    )
                ],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='"Paris"}'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                    )
                ],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        finish_reason="tool_calls",
                        index=0,
                    )
                ],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="chatcmpl-1750162525",
                choices=[ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0)],
                created=1750162525,
                model=DEFAULT_MODEL,
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=42,
                    prompt_tokens=55,
                    total_tokens=97,
                    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
                ),
            ),
        ]

        collector_callback = CollectorCallback()
        llm = HetznerChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(hetzner_chunks, callback=collector_callback)[0]  # type: ignore

        # Assert text is empty
        assert result.text is None

        # Verify the tool call was found and processed
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_zznlVyVfK0GJwY28SShJpDCh"
        assert result.tool_calls[0].tool_name == "weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}

        # Verify meta information
        assert result.meta["model"] == DEFAULT_MODEL
        assert result.meta["finish_reason"] == "tool_calls"
        assert result.meta["index"] == 0
        assert result.meta["completion_start_time"] is not None

        usage = result.meta["usage"]
        assert usage["completion_tokens"] == 42
        assert usage["prompt_tokens"] == 55
        assert usage["total_tokens"] == 97
        assert usage["completion_tokens_details"]["reasoning_tokens"] == 0
        assert usage["prompt_tokens_details"]["cached_tokens"] == 0
