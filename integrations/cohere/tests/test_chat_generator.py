import os
from unittest.mock import MagicMock, Mock

import pytest
from cohere.core import ApiError
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ComponentInfo, StreamingChunk, ToolCall
from haystack.tools import Tool
from haystack.utils import Secret

from haystack_integrations.components.generators.cohere import (
    CohereChatGenerator,
)
from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _finalize_streaming_message,
    _format_message,
    _initialize_streaming_state,
    _parse_streaming_response,
    _process_cohere_chunk,
)


def weather(city: str) -> str:
    return f"The weather in {city} is sunny and 32°C"


def stock_price(ticker: str):
    return f"The current price of {ticker} is $100"


class TestUtils:
    def test_format_message_empty_message_raises_error(self):
        message = ChatMessage.from_user("")

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_tool_call_result_with_none_id_raises_error(self):
        tool_call = ToolCall(id=None, tool_name="test_tool", arguments={})

        message = ChatMessage.from_tool(tool_result="test result", origin=tool_call, error="no error")

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_tool_call_with_none_id_raises_error(self):
        tool_call = ToolCall(id=None, tool_name="test_tool", arguments={})

        message = ChatMessage.from_assistant("", tool_calls=[tool_call])

        with pytest.raises(ValueError):
            _format_message(message)

    def test_process_cohere_chunk_none_chunk_returns_none(self):
        state = _initialize_streaming_state()
        result = _process_cohere_chunk(None, state, "test-model")
        assert result is None

    def test_process_cohere_chunk_unknown_type_returns_none(self):
        chunk = Mock()
        chunk.type = "unknown-type"
        state = _initialize_streaming_state()

        result = _process_cohere_chunk(chunk, state, "test-model")
        assert result is None

    def test_process_cohere_chunk_tool_call_end_without_current_tool_call(self):
        chunk = Mock()
        chunk.type = "tool-call-end"

        state = _initialize_streaming_state()
        state["current_tool_call"] = None  # Explicitly set to None

        result = _process_cohere_chunk(chunk, state, "test-model")

        assert result is None
        assert len(state["tool_calls"]) == 0

    def test_process_cohere_chunk_tool_call_complete_flow(self):
        state = _initialize_streaming_state()
        model = "test-model"

        # Tool call start
        start_chunk = Mock()
        start_chunk.type = "tool-call-start"
        start_chunk.delta.message.tool_calls.id = "test-id"
        start_chunk.delta.message.tool_calls.function.name = "test_function"

        _process_cohere_chunk(start_chunk, state, model)
        assert state["current_tool_call"] is not None
        assert state["current_tool_call"].id == "test-id"

        # Tool call delta
        delta_chunk = Mock()
        delta_chunk.type = "tool-call-delta"
        delta_chunk.delta.message.tool_calls.function.arguments = '{"key": "value"'

        _process_cohere_chunk(delta_chunk, state, model)
        assert state["current_tool_arguments"] == '{"key": "value"'

        # Another delta to complete the JSON
        delta_chunk2 = Mock()
        delta_chunk2.type = "tool-call-delta"
        delta_chunk2.delta.message.tool_calls.function.arguments = "}"

        _process_cohere_chunk(delta_chunk2, state, model)
        assert state["current_tool_arguments"] == '{"key": "value"}'

        # Tool call end
        end_chunk = Mock()
        end_chunk.type = "tool-call-end"

        _process_cohere_chunk(end_chunk, state, model)

        assert len(state["tool_calls"]) == 1
        assert state["tool_calls"][0].arguments == {"key": "value"}
        assert state["current_tool_call"] is None
        assert state["current_tool_arguments"] == ""

    def test_finalize_streaming_message_with_tool_calls(self):
        state = {
            "response_text": "",
            "tool_plan": "I need to check the weather",
            "tool_calls": [ToolCall(id="test-id", tool_name="weather", arguments={"city": "Paris"})],
            "current_tool_call": None,
            "current_tool_arguments": "",
            "captured_meta": {
                "model": "test-model",
                "finish_reason": "COMPLETE",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }

        message = _finalize_streaming_message(state)

        assert message.text == "I need to check the weather"
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].tool_name == "weather"
        assert message.meta["model"] == "test-model"

    def test_finalize_streaming_message_without_tool_calls(self):
        state = {
            "response_text": "Simple response text",
            "tool_plan": "",
            "tool_calls": [],
            "current_tool_call": None,
            "current_tool_arguments": "",
            "captured_meta": {
                "model": "test-model",
                "finish_reason": "COMPLETE",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        }

        message = _finalize_streaming_message(state)

        assert message.text == "Simple response text"
        assert len(message.tool_calls) == 0
        assert message.meta["model"] == "test-model"

    def test_parse_streaming_response_uses_component_info(self):
        mock_cohere_chunk = MagicMock()
        mock_cohere_chunk.type = "content-delta"
        mock_cohere_chunk.delta.message.content.text = "Hello"

        mock_response = [mock_cohere_chunk]

        captured_chunks = []

        def callback(chunk: StreamingChunk):
            captured_chunks.append(chunk)

        component_info = ComponentInfo(name="test_component", type="test_type")

        message = _parse_streaming_response(
            response=mock_response,
            model="test-model",
            streaming_callback=callback,
            component_info=component_info,
        )

        assert len(captured_chunks) == 1
        assert captured_chunks[0].component_info == component_info
        assert captured_chunks[0].content == "Hello"

        assert message.text == "Hello"


class TestCohereChatGenerator:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

        component = CohereChatGenerator()
        assert component.api_key == Secret.from_env_var(["COHERE_API_KEY", "CO_API_KEY"])
        assert component.model == "command-r-08-2024"
        assert component.streaming_callback is None
        assert component.api_base_url == "https://api.cohere.com"
        assert not component.generation_kwargs

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        with pytest.raises(ValueError):
            CohereChatGenerator()

    def test_init_with_parameters(self):
        component = CohereChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
            },
        )
        assert component.api_key == Secret.from_token("test-api-key")
        assert component.model == "command-nightly"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }

    def test_to_dict_default(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        component = CohereChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "streaming_callback": None,
                "api_key": {
                    "env_vars": ["COHERE_API_KEY", "CO_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "api_base_url": "https://api.cohere.com",
                "generation_kwargs": {},
                "tools": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        component = CohereChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="command-nightly",
            streaming_callback=print_streaming_chunk,
            api_base_url="test-base-url",
            generation_kwargs={
                "max_tokens": 10,
                "some_test_param": "test-params",
            },
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-nightly",
                "api_key": {
                    "env_vars": ["ENV_VAR"],
                    "strict": False,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "api_base_url": "test-base-url",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
                "tools": None,
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("COHERE_API_KEY", "fake-api-key")
        monkeypatch.setenv("CO_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "api_base_url": "test-base-url",
                "api_key": {
                    "env_vars": ["ENV_VAR"],
                    "strict": False,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        component = CohereChatGenerator.from_dict(data)
        assert component.model == "command-r-08-2024"
        assert component.streaming_callback is print_streaming_chunk
        assert component.api_base_url == "test-base-url"
        assert component.generation_kwargs == {
            "max_tokens": 10,
            "some_test_param": "test-params",
        }

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
            "init_parameters": {
                "model": "command-r-08-2024",
                "api_base_url": "test-base-url",
                "api_key": {
                    "env_vars": ["COHERE_API_KEY", "CO_API_KEY"],
                    "strict": True,
                    "type": "env_var",
                },
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {
                    "max_tokens": 10,
                    "some_test_param": "test-params",
                },
            },
        }
        with pytest.raises(ValueError):
            CohereChatGenerator.from_dict(data)

    def test_serde_in_pipeline(self, monkeypatch):
        """
        Test serialization/deserialization of CohereChatGenerator in a Pipeline,
        including detailed dictionary validation
        """
        monkeypatch.setenv("COHERE_API_KEY", "test-api-key")

        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=weather,
        )

        generator = CohereChatGenerator(
            model="command-r-08-2024",
            generation_kwargs={"temperature": 0.7},
            streaming_callback=print_streaming_chunk,
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()

        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": "haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",  # noqa: E501
                    "init_parameters": {
                        "model": "command-r-08-2024",
                        "api_key": {"type": "env_var", "env_vars": ["COHERE_API_KEY", "CO_API_KEY"], "strict": True},
                        "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                        "api_base_url": "https://api.cohere.com",
                        "generation_kwargs": {"temperature": 0.7},
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "weather",
                                    "description": "useful to determine the weather in a given location",
                                    "parameters": {"city": {"type": "string"}},
                                    "function": "tests.test_chat_generator.weather",
                                    "outputs_to_string": tool.outputs_to_string,
                                    "inputs_from_state": tool.inputs_from_state,
                                    "outputs_to_state": tool.outputs_to_state,
                                },
                            }
                        ],
                    },
                }
            },
            "connections": [],
        }

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
    not os.environ.get("COHERE_API_KEY", None) and not os.environ.get("CO_API_KEY", None),
    reason="Export an env var called COHERE_API_KEY/CO_API_KEY containing the Cohere API key to run this test.",
)
@pytest.mark.integration
class TestCohereChatGeneratorInference:
    def test_live_run(self):
        chat_messages = [ChatMessage.from_user("What's the capital of France")]
        component = CohereChatGenerator(generation_kwargs={"temperature": 0.8})
        results = component.run(chat_messages)
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    def test_live_run_wrong_model(self):
        component = CohereChatGenerator(model="something-obviously-wrong")
        with pytest.raises(ApiError):
            component.run([ChatMessage.from_assistant("What's the capital of France")])

    def test_live_run_streaming(self):
        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                assert chunk.component_info is not None
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""

        callback = Callback()
        component = CohereChatGenerator(streaming_callback=callback, stream=True)
        results = component.run([ChatMessage.from_user("What's the capital of France? answer in a word")])

        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert message.meta["finish_reason"] == "COMPLETE"
        assert callback.counter > 1
        assert "Paris" in callback.responses
        assert "usage" in message.meta
        assert "prompt_tokens" in message.meta["usage"]
        assert "completion_tokens" in message.meta["usage"]

    def test_tools_use_old_way(self):
        # See https://docs.cohere.com/docs/structured-outputs-json for more information
        tools_schema = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Retrieves the current stock price for a given ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                            }
                        },
                        "required": ["ticker"],
                    },
                },
            }
        ]
        client = CohereChatGenerator(model="command-r-08-2024")
        response = client.run(
            messages=[ChatMessage.from_user("What is the current price of AAPL?")],
            generation_kwargs={"tools": tools_schema},
        )
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply text should be a tool plan"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"

        assert first_reply.tool_calls, "First reply has no tool calls"
        assert len(first_reply.tool_calls) == 1, "First reply has more than one tool call"
        assert first_reply.tool_calls[0].tool_name == "get_stock_price", "First tool call is not get_stock_price"
        assert first_reply.tool_calls[0].arguments == {"ticker": "AAPL"}, "First tool call arguments are not correct"

    def test_tools_use_with_tools(self):
        stock_price_tool = Tool(
            name="get_stock_price",
            description="Retrieves the current stock price for a given ticker symbol.",
            parameters={
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple Inc.",
                    }
                },
                "required": ["ticker"],
            },
            function=stock_price,
        )
        initial_messages = [ChatMessage.from_user("What is the current price of AAPL?")]
        client = CohereChatGenerator(model="command-r-08-2024")
        response = client.run(
            messages=initial_messages,
            tools=[stock_price_tool],
        )
        replies = response["replies"]
        assert isinstance(replies, list), "Replies is not a list"
        assert len(replies) > 0, "No replies received"

        first_reply = replies[0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert first_reply.text, "First reply text should be a tool plan"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"

        assert first_reply.tool_calls, "First reply has no tool calls"
        assert len(first_reply.tool_calls) == 1, "First reply has more than one tool call"
        assert first_reply.tool_calls[0].tool_name == "get_stock_price", "First tool call is not get_stock_price"
        assert first_reply.tool_calls[0].arguments == {"ticker": "AAPL"}, "First tool call arguments are not correct"

        # Test with tool result
        new_messages = [
            initial_messages[0],
            first_reply,
            ChatMessage.from_tool(tool_result="150.23", origin=first_reply.tool_calls[0]),
        ]
        results = client.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "150.23" in final_message.text

    def test_live_run_with_tools_streaming(self):
        """
        Test that the CohereChatGenerator can run with tools and streaming callback.
        """
        weather_tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for, e.g. Paris, London",
                    }
                },
                "required": ["city"],
            },
            function=weather,
        )

        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = CohereChatGenerator(
            model="command-r-08-2024",  # Cohere's model that supports tools
            tools=[weather_tool],
            streaming_callback=print_streaming_chunk,
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"
        first_reply = results["replies"][0]

        assert isinstance(first_reply, ChatMessage), "Reply is not a ChatMessage instance"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "Reply is not from the assistant"
        assert first_reply.tool_calls, "No tool calls in the reply"

        tool_call = first_reply.tool_calls[0]
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}

        # Test with tool result
        new_messages = [
            initial_messages[0],
            first_reply,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    def test_pipeline_with_cohere_chat_generator(self):
        """
        Test that the CohereChatGenerator component can be used in a pipeline
        """
        weather_tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get weather for, e.g. Paris, London",
                    }
                },
                "required": ["city"],
            },
            function=weather,
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", CohereChatGenerator(model="command-r-08-2024", tools=[weather_tool]))
        pipeline.add_component("tool_invoker", ToolInvoker(tools=[weather_tool]))

        pipeline.connect("generator", "tool_invoker")

        results = pipeline.run(
            data={"generator": {"messages": [ChatMessage.from_user("What's the weather like in Paris?")]}}
        )

        assert (
            "The weather in Paris is sunny and 32°C"
            == results["tool_invoker"]["tool_messages"][0].tool_call_result.result
        )
