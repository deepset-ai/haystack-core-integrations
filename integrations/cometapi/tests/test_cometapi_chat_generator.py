import os
from dataclasses import asdict
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

from haystack_integrations.components.generators.cometapi.chat.chat_generator import CometAPIChatGenerator


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
            model="gpt-4o-mini",
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


class TestCometAPIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COMET_API_KEY", "test-api-key")
        component = CometAPIChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.api_base_url == "https://api.cometapi.com/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("COMET_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            CometAPIChatGenerator()

    def test_init_with_parameters(self):
        component = CometAPIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("COMET_API_KEY", "test-api-key")
        component = CometAPIChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["COMET_API_KEY"], "strict": True, "type": "env_var"},
            "model": "gpt-4o-mini",
            "streaming_callback": None,
            "api_base_url": "https://api.cometapi.com/v1",
            "generation_kwargs": {},
            "timeout": None,
            "max_retries": None,
            "tools": None,
            "http_client_kwargs": None,
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = CometAPIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="gpt-4o-mini",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "gpt-4o-mini",
            "api_base_url": "https://api.cometapi.com/v1",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            "timeout": 10,
            "max_retries": 10,
            "tools": None,
            "http_client_kwargs": {"proxy": "http://localhost:8080"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COMET_API_KEY", "fake-api-key")
        data = {
            "type": ("haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"),
            "init_parameters": {
                "api_key": {"env_vars": ["COMET_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = CometAPIChatGenerator.from_dict(data)
        assert component.model == "gpt-4o-mini"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "https://api.cometapi.com/v1"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("COMET_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COMET_API_KEY", raising=False)
        data = {
            "type": ("haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"),
            "init_parameters": {
                "api_key": {"env_vars": ["COMET_API_KEY"], "strict": True, "type": "env_var"},
                "model": "gpt-4o-mini",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            CometAPIChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("COMET_API_KEY", "fake-api-key")
        component = CometAPIChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("COMET_API_KEY", "fake-api-key")
        component = CometAPIChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
        response = component.run(chat_messages)

        # check that the component calls the OpenAI API with the correct parameters
        # for cometapi, these are passed in the extra_body parameter
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
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the cometapi API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = CometAPIChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "gpt-4o-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = CometAPIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
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
        component = CometAPIChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "gpt-4o-mini" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = CometAPIChatGenerator(tools=tools)
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert message.text is None

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the CometAPIChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = CometAPIChatGenerator(tools=tools)
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
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the CometAPIChatGenerator component can run with tools and streaming.
        """
        component = CometAPIChatGenerator(tools=tools, streaming_callback=print_streaming_chunk)
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
        assert sorted(arguments, key=lambda x: x["city"]) == [{"city": "Berlin"}, {"city": "Paris"}]
        assert tool_message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("COMET_API_KEY", None),
        reason="Export an env var called COMET_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_pipeline_with_cometapi_chat_generator(self, tools):
        """
        Test that the CometAPIChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", CometAPIChatGenerator(tools=tools))
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

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of CometAPIChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock API key
        monkeypatch.setenv("COMET_API_KEY", "test-key")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = CometAPIChatGenerator(
            model="gpt-4o-mini",
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
            "components": {
                "generator": {
                    "type": (
                        "haystack_integrations.components.generators.cometapi.chat.chat_generator.CometAPIChatGenerator"
                    ),
                    "init_parameters": {
                        "model": "gpt-4o-mini",
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.cometapi.com/v1",
                        "organization": None,
                        "generation_kwargs": {"temperature": 0.7},
                        "api_key": {"type": "env_var", "env_vars": ["COMET_API_KEY"], "strict": True},
                        "timeout": None,
                        "max_retries": None,
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "test_cometapi_chat_generator.weather",
                                },
                            }
                        ],
                        "tools_strict": False,
                        "http_client_kwargs": None,
                    },
                }
            },
            "connections": [],
            "connection_type_validation": True,
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

        # Verify the loaded pipeline's generator has the same configuration
        loaded_generator = pipeline.get_component("generator")
        assert loaded_generator.model == generator.model
        assert loaded_generator.generation_kwargs == generator.generation_kwargs
        assert loaded_generator.streaming_callback == generator.streaming_callback
        assert len(loaded_generator.tools) == len(generator.tools)
        assert loaded_generator.tools[0].name == generator.tools[0].name
        assert loaded_generator.tools[0].description == generator.tools[0].description
        assert loaded_generator.tools[0].parameters == generator.tools[0].parameters


class TestChatCompletionChunkConversion:
    def test_handle_stream_response(self):
        cometapi_chunks = [
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0, native_finish_reason=None)
                ],
                created=1750162525,
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
                model="gpt-4o-mini",
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
        llm = CometAPIChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(cometapi_chunks, callback=collector_callback)[0]  # type: ignore

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
        assert result.meta["model"] == "gpt-4o-mini"
        assert result.meta["finish_reason"] == "tool_calls"
        assert result.meta["index"] == 0
        assert result.meta["completion_start_time"] is not None

        # Normalize usage details before asserting
        usage = result.meta["usage"]

        if hasattr(usage["completion_tokens_details"], "model_dump"):
            usage["completion_tokens_details"] = usage["completion_tokens_details"].model_dump()
        if hasattr(usage["prompt_tokens_details"], "model_dump"):
            usage["prompt_tokens_details"] = usage["prompt_tokens_details"].model_dump()

        # For dataclass fallback
        if not isinstance(usage["completion_tokens_details"], dict):
            usage["completion_tokens_details"] = asdict(usage["completion_tokens_details"])
        if not isinstance(usage["prompt_tokens_details"], dict):
            usage["prompt_tokens_details"] = asdict(usage["prompt_tokens_details"])

        assert usage == {
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
