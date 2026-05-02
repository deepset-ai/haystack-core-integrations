import json
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ReasoningContent, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails
from pydantic import BaseModel

from haystack_integrations.components.generators.openrouter.chat.chat_generator import (
    OpenRouterChatGenerator,
    _convert_openrouter_completion_to_chat_message,
    _extract_reasoning,
)


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
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="openai/gpt-5-mini",
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


class TestOpenRouterChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
        component = OpenRouterChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "openai/gpt-5-mini"
        assert component.api_base_url == "https://openrouter.ai/api/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            OpenRouterChatGenerator()

    def test_init_with_parameters(self):
        component = OpenRouterChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="openai/gpt-5",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "openai/gpt-5"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key")
        component = OpenRouterChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["OPENROUTER_API_KEY"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-5-mini",
            "streaming_callback": None,
            "api_base_url": "https://openrouter.ai/api/v1",
            "generation_kwargs": {},
            "extra_headers": None,
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = OpenRouterChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="openai/gpt-5",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            extra_headers={"test-header": "test-value"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-5",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "extra_headers": {"test-header": "test-value"},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        data = {
            "type": (
                "haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["OPENROUTER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "openai/gpt-5-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "extra_headers": {"test-header": "test-value"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = OpenRouterChatGenerator.from_dict(data)
        assert component.model == "openai/gpt-5-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("OPENROUTER_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.extra_headers == {"test-header": "test-value"}
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        data = {
            "type": (
                "haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["OPENROUTER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "openai/gpt-5-mini",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "extra_headers": {"test-header": "test-value"},
                "timeout": 10,
                "max_retries": 10,
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            OpenRouterChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        # for OpenRouter, these are passed in the extra_body parameter
        _, kwargs = mock_chat_completion.call_args
        assert kwargs["extra_body"]["max_tokens"] == 10
        assert kwargs["extra_body"]["temperature"] == 0.5
        # check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_prepare_api_call_with_tools_strict(self, chat_messages, tools, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator(tools=tools)
        api_args = component._prepare_api_call(messages=chat_messages, tools_strict=True)

        assert api_args["tools"][0]["type"] == "function"
        function_spec = api_args["tools"][0]["function"]
        assert function_spec["name"] == "weather"
        assert function_spec["strict"] is True
        assert function_spec["parameters"]["additionalProperties"] is False

    def test_prepare_api_call_raises_when_streaming_with_multiple_responses(self, chat_messages, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator(generation_kwargs={"n": 2})
        with pytest.raises(ValueError, match="Cannot stream multiple responses"):
            component._prepare_api_call(messages=chat_messages, streaming_callback=print_streaming_chunk)

    def test_prepare_api_call_with_response_format_and_streaming(self, chat_messages, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        response_format = {"type": "json_schema", "json_schema": {"name": "Foo", "schema": {"type": "object"}}}
        component = OpenRouterChatGenerator(generation_kwargs={"response_format": response_format})
        api_args = component._prepare_api_call(messages=chat_messages, streaming_callback=print_streaming_chunk)

        assert api_args["stream"] is True
        assert api_args["openai_endpoint"] == "create"
        assert api_args["response_format"] == response_format

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenRouter API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = OpenRouterChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "openai/gpt-5-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = OpenRouterChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
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
        component = OpenRouterChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "openai/gpt-5-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = OpenRouterChatGenerator(tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert not message.text

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert "paris" in tool_call.arguments["city"].lower(), f"Expected 'paris' in city: {tool_call.arguments}"
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the OpenRouterChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = OpenRouterChatGenerator(tools=tools)
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
        # Extract city names and check they contain the expected cities
        # (LLM may return "Paris, France" or "Berlin, Germany" instead of just city names)
        cities = [arg["city"].lower() for arg in arguments]
        assert len(cities) == 2
        assert any("berlin" in city for city in cities), f"Expected 'berlin' in one of {cities}"
        assert any("paris" in city for city in cities), f"Expected 'paris' in one of {cities}"
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
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the OpenRouterChatGenerator component can run with tools and streaming.
        """
        component = OpenRouterChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris and Berlin?")],
            generation_kwargs={"tool_choice": "auto"},
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
        # Extract city names and check they contain the expected cities
        # (LLM may return "Paris, France" or "Berlin, Germany" instead of just city names)
        cities = [arg["city"].lower() for arg in arguments]
        assert len(cities) == 2
        assert any("berlin" in city for city in cities), f"Expected 'berlin' in one of {cities}"
        assert any("paris" in city for city in cities), f"Expected 'paris' in one of {cities}"
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_pipeline_with_openrouter_chat_generator(self, tools):
        """
        Test that the OpenRouterChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", OpenRouterChatGenerator(tools=tools))
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

        result = results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        assert "paris" in result.lower(), f"Expected 'paris' in result: {result}"
        assert "sunny and 32°c" in result.lower(), f"Expected 'sunny and 32°c' in result: {result}"

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenRouter API key to run this test.",
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

        # Use a more explicit prompt that emphasizes JSON structure requirement
        # gpt-5-mini is very rarely flaky but to harden CI we'll be more explicit
        chat_messages = [
            ChatMessage.from_user(
                "What's the capital of France? "
                "You must respond with a JSON object containing 'city' and 'country' fields."
            )
        ]
        comp = OpenRouterChatGenerator(
            model="openai/gpt-5-mini", generation_kwargs={"response_format": response_schema}
        )
        results = comp.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        try:
            msg = json.loads(message.text)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Failed to parse response as JSON. "
                f"Expected JSON with 'city' and 'country' fields, but got: {message.text!r}. "
                f"Error: {e}"
            )

        # Validate JSON structure with descriptive error messages
        assert "city" in msg, f"Response JSON missing 'city' field. Got: {msg}"
        assert "country" in msg, f"Response JSON missing 'country' field. Got: {msg}"
        assert "paris" in msg["city"].lower(), f"Expected 'Paris' in city field, got: {msg['city']}"
        assert isinstance(msg["country"], str), f"Expected country to be string, got: {type(msg['country'])}"
        assert "france" in msg["country"].lower(), f"Expected 'France' in country field, got: {msg['country']}"
        assert message.meta["finish_reason"] == "stop"

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of OpenRouterChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock API key
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = OpenRouterChatGenerator(
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
                    "type": "haystack_integrations.components.generators.openrouter.chat.chat_generator.OpenRouterChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["OPENROUTER_API_KEY"], "strict": True},
                        "model": "openai/gpt-5-mini",
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://openrouter.ai/api/v1",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_openrouter_chat_generator.weather",
                                },
                            }
                        ],
                        "http_client_kwargs": None,
                        "extra_headers": None,
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
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenRouter API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_response_format_pydantic_model(self, calendar_event_model):
        # Use a more explicit prompt that emphasizes JSON structure requirement
        chat_messages = [
            ChatMessage.from_user(
                "The marketing summit takes place on October12th at the Hilton Hotel downtown. "
                "Extract the event information and respond with a JSON object containing "
                "'event_name', 'event_date', and 'event_location' fields."
            )
        ]
        component = OpenRouterChatGenerator(
            model="openai/gpt-5-mini", generation_kwargs={"response_format": calendar_event_model}
        )
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]

        # Add better error message if JSON parsing fails
        try:
            msg = json.loads(message.text)
        except json.JSONDecodeError as e:
            pytest.fail(
                f"Failed to parse response as JSON. "
                f"Expected JSON with 'event_name', 'event_date', and 'event_location' fields, "
                f"but got: {message.text!r}. Error: {e}"
            )

        # Validate JSON structure with descriptive error messages
        assert "event_name" in msg, f"Response JSON missing 'event_name' field. Got: {msg}"
        assert "event_date" in msg, f"Response JSON missing 'event_date' field. Got: {msg}"
        assert "event_location" in msg, f"Response JSON missing 'event_location' field. Got: {msg}"
        assert "marketing summit" in msg["event_name"].lower(), (
            f"Expected 'Marketing Summit' in event_name, got: {msg['event_name']}"
        )
        assert isinstance(msg["event_date"], str), f"Expected event_date to be string, got: {type(msg['event_date'])}"
        assert isinstance(msg["event_location"], str), (
            f"Expected event_location to be string, got: {type(msg['event_location'])}"
        )

    @pytest.mark.skipif(
        not os.environ.get("OPENROUTER_API_KEY", None),
        reason="Export an env var called OPENROUTER_API_KEY containing the OpenRouter API key to run this test.",
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

        # Initialize with no tools, we'll pass them at runtime
        component = OpenRouterChatGenerator()

        # Pass mixed list: echo_tool (individual) and toolset (weather + time) at runtime
        # This tests that both individual tools and toolsets can be combined
        messages = [ChatMessage.from_user("Echo this via tool: Hello World")]
        results = component.run(messages, tools=[echo_tool, toolset])

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # Should be able to use echo_tool from the runtime mixed list
        assert message.tool_calls is not None
        tool_call = message.tool_calls[0]
        assert tool_call.tool_name == "echo"
        assert tool_call.arguments == {"text": "Hello World"}


class TestChatCompletionChunkConversion:
    def test_handle_stream_response(self):
        openrouter_chunks = [
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0, native_finish_reason=None)
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
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
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
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
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
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
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='"Paris'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=0,
                                    function=ChoiceDeltaToolCallFunction(arguments='"}'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=1,
                                    id="call_Mh1uOyW3Ys4gwydHjNHILHGX",
                                    function=ChoiceDeltaToolCallFunction(arguments="", name="weather"),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                service_tier=None,
                system_fingerprint="fp_34a54ae93c",
                usage=None,
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=1,
                                    id=None,
                                    function=ChoiceDeltaToolCallFunction(arguments='{"ci'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=1,
                                    function=ChoiceDeltaToolCallFunction(arguments='ty": '),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=1,
                                    function=ChoiceDeltaToolCallFunction(arguments='"Berli'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                ChoiceDeltaToolCall(
                                    index=1,
                                    function=ChoiceDeltaToolCallFunction(arguments='n"}'),
                                    type="function",
                                )
                            ],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        finish_reason="tool_calls",
                        index=0,
                        native_finish_reason="tool_calls",
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                system_fingerprint="fp_34a54ae93c",
                provider="OpenAI",
            ),
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="openai/gpt-5-mini",
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=42,
                    prompt_tokens=55,
                    total_tokens=97,
                    completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
                    prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
                ),
                provider="OpenAI",
            ),
        ]

        collector_callback = CollectorCallback()
        llm = OpenRouterChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(openrouter_chunks, callback=collector_callback)[0]  # type: ignore

        # Assert text is empty
        assert result.text is None

        # Verify both tool calls were found and processed
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].id == "call_zznlVyVfK0GJwY28SShJpDCh"
        assert result.tool_calls[0].tool_name == "weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.tool_calls[1].id == "call_Mh1uOyW3Ys4gwydHjNHILHGX"
        assert result.tool_calls[1].tool_name == "weather"
        assert result.tool_calls[1].arguments == {"city": "Berlin"}

        # Verify meta information
        assert result.meta["model"] == "openai/gpt-5-mini"
        assert result.meta["finish_reason"] == "tool_calls"
        assert result.meta["index"] == 0
        assert result.meta["completion_start_time"] is not None
        assert result.meta["usage"] == {
            "completion_tokens": 42,
            "prompt_tokens": 55,
            "total_tokens": 97,
            "completion_tokens_details": {
                "accepted_prediction_tokens": None,
                "audio_tokens": None,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": None,
            },
            "prompt_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 0,
            },
        }


class TestReasoningSupport:
    def test_extract_reasoning_with_text_and_details(self):
        msg = SimpleNamespace(
            reasoning="Let me think step by step...",
            reasoning_details=[{"type": "reasoning.text", "text": "Let me think step by step..."}],
        )
        result = _extract_reasoning(msg)
        assert result is not None
        assert result.reasoning_text == "Let me think step by step..."
        assert result.extra["reasoning_details"] == [{"type": "reasoning.text", "text": "Let me think step by step..."}]

    def test_extract_reasoning_returns_none_without_reasoning(self):
        msg = SimpleNamespace()
        result = _extract_reasoning(msg)
        assert result is None

    def test_extract_reasoning_from_details_only(self):
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_details=[
                {"type": "reasoning.text", "text": "Step 1. "},
                {"type": "reasoning.summary", "summary": "Conclusion."},
            ],
        )
        result = _extract_reasoning(msg)
        assert result is not None
        assert result.reasoning_text == "Step 1. Conclusion."
        assert len(result.extra["reasoning_details"]) == 2

    def test_extract_reasoning_handles_model_dump_objects(self):
        detail = Mock()
        detail.model_dump.return_value = {"type": "reasoning.text", "text": "Thinking..."}
        msg = SimpleNamespace(
            reasoning="Thinking...",
            reasoning_details=[detail],
        )
        result = _extract_reasoning(msg)
        assert result is not None
        assert result.extra["reasoning_details"] == [{"type": "reasoning.text", "text": "Thinking..."}]

    def test_extract_reasoning_vars_fallback(self):
        detail = SimpleNamespace(type="reasoning.text", text="Fallback path")
        msg = SimpleNamespace(
            reasoning="Fallback path",
            reasoning_details=[detail],
        )
        result = _extract_reasoning(msg)
        assert result is not None
        assert result.extra["reasoning_details"] == [{"type": "reasoning.text", "text": "Fallback path"}]

    def test_extract_reasoning_unknown_detail_type(self):
        msg = SimpleNamespace(
            reasoning=None,
            reasoning_details=[
                {"type": "reasoning.internal_monologue", "content": "hidden"},
                {"type": "reasoning.text", "text": "Visible."},
            ],
        )
        result = _extract_reasoning(msg)
        assert result is not None
        assert result.reasoning_text == "Visible."
        assert len(result.extra["reasoning_details"]) == 2

    def test_convert_completion_with_reasoning_and_tool_calls(self):
        completion = ChatCompletion(
            id="test-reasoning-tools",
            model="deepseek/deepseek-r1",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="tool_calls",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(
                        content=None,
                        role="assistant",
                        reasoning="I need to check the weather.",
                        reasoning_details=[{"type": "reasoning.text", "text": "I need to check the weather."}],
                        tool_calls=[
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                            }
                        ],
                    ),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
        )

        result = _convert_openrouter_completion_to_chat_message(completion, completion.choices[0])
        assert result.text is None
        assert result.reasoning is not None
        assert result.reasoning.reasoning_text == "I need to check the weather."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "weather"
        assert result.tool_calls[0].arguments == {"city": "Paris"}
        assert result.meta["finish_reason"] == "tool_calls"

    def test_convert_completion_with_reasoning(self):
        completion = ChatCompletion(
            id="test-reasoning",
            model="deepseek/deepseek-r1",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(
                        content="The answer is 42.",
                        role="assistant",
                        reasoning="Let me think about this...",
                        reasoning_details=[{"type": "reasoning.text", "text": "Let me think about this..."}],
                    ),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
        )

        result = _convert_openrouter_completion_to_chat_message(completion, completion.choices[0])
        assert result.text == "The answer is 42."
        assert result.reasoning is not None
        assert result.reasoning.reasoning_text == "Let me think about this..."
        assert result.reasoning.extra["reasoning_details"] == [
            {"type": "reasoning.text", "text": "Let me think about this..."}
        ]
        assert result.meta["model"] == "deepseek/deepseek-r1"
        assert result.meta["finish_reason"] == "stop"

    def test_convert_completion_without_reasoning(self):
        completion = ChatCompletion(
            id="test-no-reasoning",
            model="openai/gpt-5-mini",
            object="chat.completion",
            choices=[
                Choice(
                    finish_reason="stop",
                    logprobs=None,
                    index=0,
                    message=ChatCompletionMessage(content="Hello!", role="assistant"),
                )
            ],
            created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        result = _convert_openrouter_completion_to_chat_message(completion, completion.choices[0])
        assert result.text == "Hello!"
        assert result.reasoning is None

    def test_run_with_reasoning(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")

        with patch("openai.resources.chat.completions.Completions.create") as mock_create:
            completion = ChatCompletion(
                id="test-run-reasoning",
                model="deepseek/deepseek-r1",
                object="chat.completion",
                choices=[
                    Choice(
                        finish_reason="stop",
                        logprobs=None,
                        index=0,
                        message=ChatCompletionMessage(
                            content="Paris is the capital of France.",
                            role="assistant",
                            reasoning="The user asked about capitals. France's capital is Paris.",
                            reasoning_details=[
                                {
                                    "type": "reasoning.text",
                                    "text": "The user asked about capitals. France's capital is Paris.",
                                }
                            ],
                        ),
                    )
                ],
                created=int(datetime.now(tz=pytz.timezone("UTC")).timestamp()),
                usage={"prompt_tokens": 57, "completion_tokens": 40, "total_tokens": 97},
            )
            mock_create.return_value = completion

            component = OpenRouterChatGenerator(
                model="deepseek/deepseek-r1",
                generation_kwargs={"reasoning": {"effort": "high"}},
            )
            response = component.run([ChatMessage.from_user("What's the capital of France?")])

            assert len(response["replies"]) == 1
            reply = response["replies"][0]
            assert reply.text == "Paris is the capital of France."
            assert reply.reasoning is not None
            assert "capitals" in reply.reasoning.reasoning_text
            assert reply.reasoning.extra["reasoning_details"][0]["type"] == "reasoning.text"

    def test_handle_stream_response_with_reasoning(self):
        chunks = [
            ChatCompletionChunk(
                id="gen-reasoning",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="deepseek/deepseek-r1",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="gen-reasoning",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(
                            content=None,
                            role="assistant",
                            reasoning_details=[{"type": "reasoning.text", "text": "Thinking about capitals..."}],
                        ),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="deepseek/deepseek-r1",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="gen-reasoning",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="Paris.", role="assistant"),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="deepseek/deepseek-r1",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="gen-reasoning",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        finish_reason="stop",
                        index=0,
                        native_finish_reason="stop",
                    )
                ],
                created=1750162525,
                model="deepseek/deepseek-r1",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="gen-reasoning",
                choices=[
                    ChoiceChunk(
                        delta=ChoiceDelta(content="", role="assistant"),
                        index=0,
                        native_finish_reason=None,
                    )
                ],
                created=1750162525,
                model="deepseek/deepseek-r1",
                object="chat.completion.chunk",
                usage=CompletionUsage(
                    completion_tokens=50,
                    prompt_tokens=30,
                    total_tokens=80,
                ),
            ),
        ]

        collector = CollectorCallback()
        llm = OpenRouterChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(chunks, callback=collector)[0]

        assert result.text == "Paris."
        assert result.reasoning is not None
        assert "capitals" in result.reasoning.reasoning_text

    def test_prepare_api_call_preserves_reasoning(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-api-key")
        component = OpenRouterChatGenerator()

        reasoning = ReasoningContent(
            reasoning_text="Step by step analysis...",
            extra={"reasoning_details": [{"type": "reasoning.text", "text": "Step by step analysis..."}]},
        )
        messages = [
            ChatMessage.from_user("Explain quantum computing."),
            ChatMessage.from_assistant(text="Quantum computing uses qubits.", reasoning=reasoning),
            ChatMessage.from_user("Tell me more."),
        ]

        api_args = component._prepare_api_call(messages=messages)
        formatted = api_args["messages"]

        assert "reasoning_details" in formatted[1]
        assert formatted[1]["reasoning_details"] == [{"type": "reasoning.text", "text": "Step by step analysis..."}]
        assert "reasoning_details" not in formatted[0]
        assert "reasoning_details" not in formatted[2]
