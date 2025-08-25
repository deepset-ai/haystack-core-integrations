import os
from typing import Optional
from unittest.mock import MagicMock

import pytest
from cohere.core import ApiError
from haystack import Pipeline
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage, ChatRole, ComponentInfo, StreamingChunk, ToolCall
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from haystack.tools import Tool
from haystack.utils import Secret

from haystack_integrations.components.generators.cohere import CohereChatGenerator
from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _convert_cohere_chunk_to_streaming_chunk,
    _format_message,
    _parse_streaming_response,
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

        message = ChatMessage.from_tool(tool_result="test result", origin=tool_call, error=False)

        with pytest.raises(ValueError):
            _format_message(message)

    def test_format_message_tool_call_with_none_id_raises_error(self):
        tool_call = ToolCall(id=None, tool_name="test_tool", arguments={})

        message = ChatMessage.from_assistant("", tool_calls=[tool_call])

        with pytest.raises(ValueError):
            _format_message(message)

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
        assert component.model == "command-r-plus"
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
                "model": "command-r-plus",
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
                "model": "command-r-plus",
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
        assert component.model == "command-r-plus"
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
                "model": "command-r-plus",
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
            model="command-r-plus",
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
                        "model": "command-r-plus",
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
        assert message.meta["finish_reason"] == "stop"
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


def create_mock_cohere_chunk(chunk_type: str, index: Optional[int] = None, **kwargs):
    """aux function to create properly configured mock Cohere chunks"""
    chunk = MagicMock()
    chunk.type = chunk_type
    chunk.index = index

    # Create delta mock
    delta = MagicMock()
    chunk.delta = delta

    # Configure based on chunk type
    if chunk_type == "content-delta":
        message = MagicMock()
        content = MagicMock()
        content.text = kwargs.get("text", "")
        message.content = content
        delta.message = message

    elif chunk_type == "tool-plan-delta":
        message = MagicMock()
        message.tool_plan = kwargs.get("tool_plan", "")
        delta.message = message

    elif chunk_type == "tool-call-start":
        message = MagicMock()
        tool_calls = MagicMock()
        tool_calls.id = kwargs.get("tool_call_id", "")
        function = MagicMock()
        function.name = kwargs.get("tool_name", "")
        function.arguments = kwargs.get("arguments", None)
        tool_calls.function = function
        message.tool_calls = tool_calls
        delta.message = message

    elif chunk_type == "tool-call-delta":
        message = MagicMock()
        tool_calls = MagicMock()
        function = MagicMock()
        function.arguments = kwargs.get("arguments", "")
        tool_calls.function = function
        message.tool_calls = tool_calls
        delta.message = message

    elif chunk_type == "tool-call-end":
        # No specific configuration needed
        pass

    elif chunk_type == "message-end":
        delta.finish_reason = kwargs.get("finish_reason", None)
        if "usage" in kwargs:
            if isinstance(kwargs["usage"], dict):
                delta.usage = kwargs["usage"]
            else:
                usage = MagicMock()
                billed_units = MagicMock()
                billed_units.input_tokens = kwargs["usage"]["input_tokens"]
                billed_units.output_tokens = kwargs["usage"]["output_tokens"]
                usage.billed_units = billed_units
                delta.usage = usage
        else:
            delta.usage = None

    return chunk


@pytest.fixture
def cohere_chunks():
    """Mocked Cohere streaming response chunks to test conversion function"""
    return [
        # Chunk 1: Initial content-delta with text
        create_mock_cohere_chunk("content-delta", text="I'll help you get the weather", index=0),
        # Chunk 2: Tool plan delta
        create_mock_cohere_chunk("tool-plan-delta", tool_plan="I need to call the weather tool"),
        # Chunk 3: Tool call start - first tool call
        create_mock_cohere_chunk(
            "tool-call-start", tool_call_id="call_weather_paris_123", tool_name="weather", index=1, arguments=None
        ),
        # Chunk 4: Tool call delta - arguments streaming
        create_mock_cohere_chunk("tool-call-delta", index=1, arguments='{"ci'),
        # Chunk 5: Tool call delta - more arguments
        create_mock_cohere_chunk("tool-call-delta", index=1, arguments='ty": "'),
        # Chunk 6: Tool call delta - city name
        create_mock_cohere_chunk("tool-call-delta", index=1, arguments='Paris"'),
        # Chunk 7: Tool call delta - closing brace
        create_mock_cohere_chunk("tool-call-delta", index=1, arguments="}"),
        # Chunk 8: Tool call end - first tool call complete
        create_mock_cohere_chunk("tool-call-end", index=1),
        # Chunk 9: Tool call start - second tool call
        create_mock_cohere_chunk(
            "tool-call-start", tool_call_id="call_weather_berlin_456", tool_name="weather", index=2, arguments=None
        ),
        # Chunk 10: Tool call delta - second tool arguments
        create_mock_cohere_chunk("tool-call-delta", index=2, arguments='{"ci'),
        # Chunk 11: Tool call delta - more second tool arguments
        create_mock_cohere_chunk("tool-call-delta", index=2, arguments='ty": "'),
        # Chunk 12: Tool call delta - second city name
        create_mock_cohere_chunk("tool-call-delta", index=2, arguments='Berlin"'),
        # Chunk 13: Tool call delta - closing brace for second tool
        create_mock_cohere_chunk("tool-call-delta", index=2, arguments="}"),
        # Chunk 14: Tool call end - second tool call complete
        create_mock_cohere_chunk("tool-call-end", index=2),
        # Chunk 15: Message end with finish reason and usage
        create_mock_cohere_chunk(
            "message-end",
            finish_reason="TOOL_CALLS",
            usage={
                "billed_units": {"input_tokens": 9, "output_tokens": 75},
                "tokens": {"input_tokens": 150, "output_tokens": 75},
            },
        ),
    ]


@pytest.fixture
def expected_streaming_chunks():
    """Fixture providing expected Haystack StreamingChunk objects for conversion testing."""

    return [
        # Chunk 1: Content delta
        StreamingChunk(
            content="I'll help you get the weather",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=None,
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 2: Tool plan delta
        StreamingChunk(
            content="I need to call the weather tool",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=None,
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 3: Tool call start
        StreamingChunk(
            content="",
            index=0,
            start=True,  # Tool call start
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    id="call_weather_paris_123",
                    tool_name="weather",
                    arguments=None,
                )
            ],
            meta={
                "model": "command-r-08-2024",
                "tool_call_id": "call_weather_paris_123",
            },
        ),
        # Chunk 4: Tool call delta - arguments
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,  # Name was set in start chunk
                    arguments='{"ci',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 5: Tool call delta - more arguments
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments='ty": "',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 6: Tool call delta - city name
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments='Paris"',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 7: Tool call delta - closing brace
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments="}",
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 8: Tool call end
        StreamingChunk(
            content="",
            index=0,
            start=True,
            finish_reason=None,
            tool_calls=None,
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 9: Tool call start - second tool
        StreamingChunk(
            content="",
            index=0,
            start=True,  # Tool call start
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    id="call_weather_berlin_456",
                    tool_name="weather",
                    arguments=None,
                )
            ],
            meta={
                "model": "command-r-08-2024",
                "tool_call_id": "call_weather_berlin_456",
            },
        ),
        # Chunk 10: Tool call delta - second tool arguments
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments='{"ci',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 11: Tool call delta - more second tool arguments
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments='ty": "',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 12: Tool call delta - second city name
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments='Berlin"',
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 13: Tool call delta - closing brace for second tool
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason=None,
            tool_calls=[
                ToolCallDelta(
                    index=0,
                    tool_name=None,
                    arguments="}",
                )
            ],
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 14: Tool call end - second tool
        StreamingChunk(
            content="",
            index=0,
            start=True,
            finish_reason=None,
            tool_calls=None,
            meta={
                "model": "command-r-08-2024",
            },
        ),
        # Chunk 15: Message end with finish reason and usage
        StreamingChunk(
            content="",
            index=0,
            start=False,
            finish_reason="tool_calls",
            tool_calls=None,
            meta={
                "model": "command-r-08-2024",
                "finish_reason": "TOOL_CALLS",
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 75,
                },
            },
        ),
    ]


class TestCohereChunkConversion:
    def test_convert_cohere_chunk_to_streaming_chunk_complete_sequence(self, cohere_chunks, expected_streaming_chunks):
        # TODO: the indexes are not correctly checked because of missing global_index
        # all the streaming chunks have index 0, but the expected indexes increase with tool calls.

        for cohere_chunk, haystack_chunk in zip(cohere_chunks, expected_streaming_chunks):
            stream_chunk = _convert_cohere_chunk_to_streaming_chunk(
                chunk=cohere_chunk,
                model="command-r-08-2024",
            )
            assert stream_chunk == haystack_chunk

    def test_convert_cohere_chunk_with_empty_tool_calls(self):
        chunk = create_mock_cohere_chunk(
            "tool-call-start",
            tool_call_id="call_empty_123",
            tool_name=None,  # missing tool name
            arguments=None,
        )
        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")

        assert result.content == ""
        assert result.start is False
        assert result.tool_calls is None
        assert result.index == 0
        assert result.meta["model"] == "command-r-08-2024"

    def test_convert_tool_plan_delta_chunk(self):
        chunk = create_mock_cohere_chunk("tool-plan-delta", tool_plan="I will call the weather tool")

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == "I will call the weather tool"
        assert result.index == 0
        assert result.start is False
        assert result.tool_calls is None

    def test_convert_tool_call_start_chunk(self):
        chunk = create_mock_cohere_chunk(
            "tool-call-start", tool_call_id="call_123", tool_name="weather", arguments=None
        )

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.index == 0
        assert result.start is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].tool_name == "weather"
        assert result.tool_calls[0].arguments is None
        assert result.meta["tool_call_id"] == "call_123"

    def test_convert_tool_call_delta_chunk(self):
        chunk = create_mock_cohere_chunk("tool-call-delta", arguments='{"city": "Paris"}')

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.index == 0
        assert result.start is False
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name is None  # name was set in start chunk
        assert result.tool_calls[0].arguments == '{"city": "Paris"}'

    def test_convert_tool_call_end_chunk(self):
        chunk = create_mock_cohere_chunk("tool-call-end")

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.index == 0
        assert result.start
        assert result.tool_calls is None

    def test_convert_message_end_chunk(self):
        chunk = create_mock_cohere_chunk(
            "message-end",
            finish_reason="COMPLETE",
            usage={
                "billed_units": {"input_tokens": 9, "output_tokens": 75},
                "tokens": {"input_tokens": 150, "output_tokens": 50},
            },
        )

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.index == 0
        assert result.start is False
        assert result.finish_reason == "stop"  # Mapped from "COMPLETE"
        assert result.tool_calls is None
        assert result.meta["finish_reason"] == "COMPLETE"
        assert result.meta["usage"] == {
            "prompt_tokens": 9,
            "completion_tokens": 75,
        }

    def test_convert_message_end_chunk_max_tokens(self):
        chunk = create_mock_cohere_chunk(
            "message-end",
            finish_reason="MAX_TOKENS",
            usage={
                "billed_units": {"input_tokens": 9, "output_tokens": 75},
                "tokens": {"input_tokens": 200, "output_tokens": 100},
            },
        )

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.finish_reason == "length"  # mapped from "MAX_TOKENS"
        assert result.meta["finish_reason"] == "MAX_TOKENS"
        assert result.meta["usage"] == {
            "prompt_tokens": 9,
            "completion_tokens": 75,
        }

    def test_convert_unknown_chunk_type(self):
        chunk = create_mock_cohere_chunk("unknown-chunk-type")
        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.start is False
        assert result.tool_calls is None
        assert result.finish_reason is None

    def test_convert_with_component_info(self):
        component_info = ComponentInfo(name="test_component", type="test_type")
        chunk = create_mock_cohere_chunk("content-delta", text="Test content")

        result = _convert_cohere_chunk_to_streaming_chunk(
            chunk=chunk, component_info=component_info, model="command-r-08-2024"
        )
        assert result.component_info == component_info
        assert result.content == "Test content"

    def test_finish_reason_mapping(self):
        finish_reasons = [
            ("COMPLETE", "stop"),
            ("MAX_TOKENS", "length"),
            ("TOOL_CALLS", "tool_calls"),
        ]

        for cohere_reason, haystack_reason in finish_reasons:
            chunk = create_mock_cohere_chunk("message-end", finish_reason=cohere_reason)

            result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
            assert result.finish_reason == haystack_reason
            assert result.meta["finish_reason"] == cohere_reason

    def test_usage_extraction_other_cases(self):
        # missing usage data
        chunk = create_mock_cohere_chunk("message-end", finish_reason="COMPLETE")
        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")

        assert result.meta["usage"] == {"completion_tokens": 0.0, "prompt_tokens": 0.0}

        # malformed usage data
        chunk = create_mock_cohere_chunk(
            "message-end", finish_reason="COMPLETE", usage={"billed_units": {"invalid_key": 100}}
        )

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=chunk, model="command-r-08-2024")
        assert result.meta["usage"] == {"completion_tokens": 0.0, "prompt_tokens": 0.0}
