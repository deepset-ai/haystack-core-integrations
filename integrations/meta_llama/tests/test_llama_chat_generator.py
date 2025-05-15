# Copyright (c) Meta Platforms, Inc. and affiliates

import os
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
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from haystack_integrations.components.generators.meta_llama.chat.chat_generator import (
    MetaLlamaChatGenerator,
)


@pytest.fixture
def chat_messages():
    return [
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
def mock_chat_completion():
    """
    Mock the OpenAI API completion response and reuse it for tests
    """
    with patch("openai.resources.chat.completions.Completions.create") as mock_chat_completion_create:
        completion = ChatCompletion(
            id="foo",
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
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


class TestLlamaChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("LLAMA_API_KEY", "test-api-key")
        component = MetaLlamaChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "Llama-4-Scout-17B-16E-Instruct-FP8"
        assert component.api_base_url == "https://api.llama.com/compat/v1/"
        assert component.streaming_callback is None
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("LLAMA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            MetaLlamaChatGenerator()

    def test_init_with_parameters(self):
        component = MetaLlamaChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "Llama-4-Scout-17B-16E-Instruct-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("LLAMA_API_KEY", "test-api-key")
        component = MetaLlamaChatGenerator()
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator"
        )

        expected_params = {
            "api_key": {
                "env_vars": ["LLAMA_API_KEY"],
                "strict": True,
                "type": "env_var",
            },
            "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
            "streaming_callback": None,
            "api_base_url": "https://api.llama.com/compat/v1/",
            "generation_kwargs": {},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = MetaLlamaChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        data = component.to_dict()

        assert (
            data["type"]
            == "haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator"
        )

        expected_params = {
            "api_key": {"env_vars": ["ENV_VAR"], "strict": True, "type": "env_var"},
            "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
            "api_base_url": "test-base-url",
            "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
            "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
        }

        for key, value in expected_params.items():
            assert data["init_parameters"][key] == value

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("LLAMA_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["LLAMA_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        component = MetaLlamaChatGenerator.from_dict(data)
        assert component.model == "Llama-4-Scout-17B-16E-Instruct-FP8"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }
        assert component.api_key == Secret.from_env_var("LLAMA_API_KEY")

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("LLAMA_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.meta_llama.chat.chat_generator.MetaLlamaChatGenerator",
            "init_parameters": {
                "api_key": {
                    "env_vars": ["LLAMA_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
                "api_base_url": "test-base-url",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            MetaLlamaChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion, monkeypatch):  # noqa: ARG002
        monkeypatch.setenv("LLAMA_API_KEY", "fake-api-key")
        component = MetaLlamaChatGenerator()
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_chat_completion, monkeypatch):
        monkeypatch.setenv("LLAMA_API_KEY", "fake-api-key")
        component = MetaLlamaChatGenerator(generation_kwargs={"max_tokens": 10, "temperature": 0.5})
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

    def test_check_abnormal_completions(self, caplog):
        component = MetaLlamaChatGenerator(api_key=Secret.from_token("test-api-key"))
        messages = [
            ChatMessage.from_assistant(
                "",
                meta={
                    "finish_reason": "content_filter" if i % 2 == 0 else "length",
                    "index": i,
                },
            )
            for i, _ in enumerate(range(4))
        ]

        for m in messages:
            component._check_finish_reason(m.meta)

        # check truncation warning
        message_template = (
            "The completion for index {index} has been truncated before reaching a natural stopping point. "
            "Increase the max_tokens parameter to allow for longer completions."
        )

        for index in [1, 3]:
            assert caplog.records[index].message == message_template.format(index=index)

        # check content filter warning
        message_template = "The completion for index {index} has been truncated due to the content filter."
        for index in [0, 2]:
            assert caplog.records[index].message == message_template.format(index=index)

    @pytest.mark.skipif(
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = MetaLlamaChatGenerator()
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "Llama-4-Scout-17B-16E-Instruct-FP8" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = MetaLlamaChatGenerator(model="something-obviously-wrong")
        with pytest.raises(OpenAIError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
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
        component = MetaLlamaChatGenerator(streaming_callback=callback)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "Llama-4-Scout-17B-16E-Instruct-FP8" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

        assert callback.counter > 1
        assert "Paris" in callback.responses

    @pytest.mark.skipif(
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        chat_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = MetaLlamaChatGenerator(tools=tools)
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
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_and_response(self, tools):
        """
        Integration test that the MetaLlamaChatGenerator component can run with tools and get a response.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = MetaLlamaChatGenerator(tools=tools)
        results = component.run(messages=initial_messages)

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

    @pytest.mark.skipif(
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the MetaLlamaChatGenerator component can run with tools and streaming.
        """

        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0
                self.tool_calls = []

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                if chunk.content:
                    self.responses += chunk.content
                if chunk.meta.get("tool_calls"):
                    self.tool_calls.extend(chunk.meta["tool_calls"])

        callback = Callback()
        component = MetaLlamaChatGenerator(tools=tools, streaming_callback=callback)
        results = component.run(
            [ChatMessage.from_user("What's the weather like in Paris?")],
        )

        assert len(results["replies"]) > 0, "No replies received"
        assert callback.counter > 1, "Streaming callback was not called multiple times"
        assert callback.tool_calls, "No tool calls received in streaming"

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
        not os.environ.get("LLAMA_API_KEY", None),
        reason="Export an env var called LLAMA_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_pipeline_with_llama_chat_generator(self, tools):
        """
        Test that the MetaLlamaChatGenerator component can be used in a pipeline
        """
        pipeline = Pipeline()
        pipeline.add_component("generator", MetaLlamaChatGenerator(tools=tools))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=tools))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={
                "generator": {
                    "messages": [ChatMessage.from_user("What's the weather like in Paris?")],
                }
            }
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of MetaLlamaChatGenerator in a Pipeline,
        including YAML conversion and detailed dictionary validation
        """
        # Set mock API key
        monkeypatch.setenv("LLAMA_API_KEY", "test-key")

        # Create a test tool
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        # Create generator with specific configuration
        generator = MetaLlamaChatGenerator(
            model="Llama-4-Scout-17B-16E-Instruct-FP8",
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
                    "type": (
                        "haystack_integrations.components.generators.meta_llama.chat."
                        "chat_generator.MetaLlamaChatGenerator"
                    ),
                    "init_parameters": {
                        "api_key": {
                            "type": "env_var",
                            "env_vars": ["LLAMA_API_KEY"],
                            "strict": True,
                        },
                        "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.llama.com/compat/v1/",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_llama_chat_generator.weather",
                                },
                            }
                        ],
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
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "outputs_to_string"
            ] = tool.outputs_to_string
        if hasattr(tool, "inputs_from_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "inputs_from_state"
            ] = tool.inputs_from_state
        if hasattr(tool, "outputs_to_state"):
            expected_dict["components"]["generator"]["init_parameters"]["tools"][0]["data"][
                "outputs_to_state"
            ] = tool.outputs_to_state

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
