import os
from datetime import datetime
from unittest.mock import patch

import pytest
import pytz
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, StreamingChunk, ToolCall
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret
from openai import OpenAIError
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChoiceChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.completion_usage import CompletionTokensDetails, CompletionUsage, PromptTokensDetails

from haystack_integrations.components.generators.togetherai.chat.chat_generator import TogetherAIChatGenerator


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
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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


class TestTogetherAIChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = TogetherAIChatGenerator(api_key=Secret.from_env_var("ENV_VAR"))
        assert component.client.api_key == "test-api-key"
        assert component.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert component.api_base_url == "https://api.together.xyz/v1"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            TogetherAIChatGenerator()

    def test_init_with_parameters(self):
        component = TogetherAIChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="openai/gpt-oss-20b",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "openai/gpt-oss-20b"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "test-api-key")
        component = TogetherAIChatGenerator(api_key=Secret.from_env_var("TOGETHER_API_KEY"))
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.togetherai.chat.chat_generator.TogetherAIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "streaming_callback": None,
            "api_base_url": "https://api.together.xyz/v1",
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
        component = TogetherAIChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="openai/gpt-oss-20b",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            timeout=10,
            max_retries=10,
            tools=None,
            http_client_kwargs={"proxy": "http://localhost:8080"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.togetherai.chat.chat_generator.TogetherAIChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "openai/gpt-oss-20b",
            "api_base_url": "test-base-url",
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
        monkeypatch.setenv("TOGETHER_API_KEY", "fake-api-key")
        data = {
            "type": (
                "haystack_integrations.components.generators.togetherai.chat.chat_generator.TogetherAIChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
                "tools": None,
                "http_client_kwargs": {"proxy": "http://localhost:8080"},
            },
        }
        component = TogetherAIChatGenerator.from_dict(data)
        assert component.model == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("TOGETHER_API_KEY")
        assert component.http_client_kwargs == {"proxy": "http://localhost:8080"}
        assert component.tools is None
        assert component.timeout == 10
        assert component.max_retries == 10

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        data = {
            "type": (
                "haystack_integrations.components.generators.togetherai.chat.chat_generator.TogetherAIChatGenerator"
            ),
            "init_parameters": {
                "api_key": {"env_vars": ["TOGETHER_API_KEY"], "strict": True, "type": "env_var"},
                "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo ",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "timeout": 10,
                "max_retries": 10,
            },
        }
        with pytest.raises(ValueError, match=r"None of the .* environment variables are set"):
            TogetherAIChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("TOGETHER_API_KEY", "fake-api-key")
        component = TogetherAIChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "fake-api-key")
        component = TogetherAIChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
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
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = TogetherAIChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = TogetherAIChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
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
        component = TogetherAIChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "meta-llama/Llama-3.3-70B-Instruct-Turbo" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = TogetherAIChatGenerator(tools=tools)
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
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the TogetherAIChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = TogetherAIChatGenerator(tools=tools)
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
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the TogetherAIChatGenerator component can run with tools and streaming.
        """
        component = TogetherAIChatGenerator(
            model="openai/gpt-oss-120b", tools=tools, streaming_callback=print_streaming_chunk
        )
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris and Berlin?")],
            generation_kwargs={"tool_choice": "auto"},
        )

        assert len(results["replies"]) == 1

        # Find the message with tool calls
        tool_message = results["replies"][0]

        assert isinstance(tool_message, ChatMessage)
        tool_calls = tool_message.tool_calls
        assert len(tool_calls) > 0
        assert ChatMessage.is_from(tool_message, ChatRole.ASSISTANT)

        for tool_call in tool_calls:
            assert tool_call.id is not None
            assert isinstance(tool_call, ToolCall)
            assert tool_call.tool_name == "weather"

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_pipeline_with_togetherai_chat_generator(self, tools):
        """
        Test that the TogetherAIChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", TogetherAIChatGenerator(tools=tools))
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
        Test serialization/deserialization of TogetherAIChatGenerator in a Pipeline,
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
        generator = TogetherAIChatGenerator(
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
        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.togetherai.chat.chat_generator.TogetherAIChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["ENV_VAR"], "strict": True},
                        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.together.xyz/v1",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_togetherai_chat_generator.weather",
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

    def test_init_with_mixed_tools(self, monkeypatch):
        """Test that TogetherAIChatGenerator can be initialized with mixed Tool and Toolset."""
        monkeypatch.setenv("TOGETHER_API_KEY", "test-api-key")

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

        generator = TogetherAIChatGenerator(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            tools=[weather_tool, toolset],
        )

        assert generator.tools == [weather_tool, toolset]

    @pytest.mark.skipif(
        not os.environ.get("TOGETHER_API_KEY", None),
        reason="Export an env var called TOGETHER_API_KEY containing the Together AI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_mixed_tools(self, mixed_tools):
        """
        Integration test that verifies TogetherAIChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """
        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = TogetherAIChatGenerator(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", tools=mixed_tools)
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


class TestChatCompletionChunkConversion:
    def test_handle_stream_response(self):
        togetherai_chunks = [
            ChatCompletionChunk(
                id="gen-1750162525-tc7ParBHvsqd6rYhCDtK",
                choices=[
                    ChoiceChunk(delta=ChoiceDelta(content="", role="assistant"), index=0, native_finish_reason=None)
                ],
                created=1750162525,
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
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
        llm = TogetherAIChatGenerator(api_key=Secret.from_token("test-api-key"))
        result = llm._handle_stream_response(togetherai_chunks, callback=collector_callback)[0]  # type: ignore

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
        assert result.meta["model"] == "meta-llama/Llama-3.3-70B-Instruct-Turbo"
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
