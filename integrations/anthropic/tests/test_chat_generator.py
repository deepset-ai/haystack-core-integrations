# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from unittest.mock import AsyncMock, Mock, patch

import anthropic
import pytest
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    SignatureDelta,
    TextBlock,
    TextBlockParam,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta
from haystack import Pipeline
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, print_streaming_chunk
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    ImageContent,
    StreamingChunk,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import Tool, Toolset
from haystack.utils.auth import Secret

from haystack_integrations.components.generators.anthropic.chat.chat_generator import (
    AnthropicChatGenerator,
)
from haystack_integrations.components.generators.anthropic.chat.utils import (
    FINISH_REASON_MAPPING,
    _convert_anthropic_chunk_to_streaming_chunk,
    _convert_chat_completion_to_chat_message,
    _convert_messages_to_anthropic_format,
)


def hello_world():
    return "Hello, World!"


def population(city: str) -> str:
    return f"The population of {city} is 2.2 million"


@pytest.fixture
def tool_with_no_parameters():
    tool = Tool(
        name="hello_world",
        description="This prints hello world",
        parameters={"properties": {}, "type": "object"},
        function=hello_world,
    )
    return tool


@pytest.fixture
def tools():
    tool_parameters = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = Tool(
        name="weather",
        description="useful to determine the weather in a given location",
        parameters=tool_parameters,
        function=lambda x: x,
    )
    return [tool]


@pytest.fixture
def chat_messages():
    return [
        ChatMessage.from_user("What's the capital of France"),
    ]


@pytest.fixture
def mock_anthropic_completion():
    with patch("anthropic.resources.messages.Messages.create") as mock_anthropic:
        completion = Message(
            id="foo",
            type="message",
            model="claude-sonnet-4-20250514",
            role="assistant",
            content=[TextBlockParam(type="text", text="Hello! I'm Claude.")],
            stop_reason="end_turn",
            usage={"input_tokens": 10, "output_tokens": 20},
        )
        mock_anthropic.return_value = completion
        yield mock_anthropic


class TestAnthropicChatGenerator:
    def test_init_default(self, monkeypatch):
        """
        Test the default initialization of the AnthropicChatGenerator component.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-sonnet-4-20250514"
        assert component.streaming_callback is None
        assert not component.generation_kwargs
        assert component.tools is None

    def test_init_fail_wo_api_key(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to initialize without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError):
            AnthropicChatGenerator()

    def test_init_fail_with_duplicate_tool_names(self, monkeypatch, tools):
        """
        Test that the AnthropicChatGenerator component fails to initialize with duplicate tool names.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        duplicate_tools = [tools[0], tools[0]]
        with pytest.raises(ValueError):
            AnthropicChatGenerator(tools=duplicate_tools)

    def test_init_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component initializes with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=lambda x: x)

        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"),
            model="claude-sonnet-4-20250514",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-sonnet-4-20250514"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.tools == [tool]

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component initializes with parameters and env vars.
        """
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        component = AnthropicChatGenerator(
            model="claude-sonnet-4-20250514",
            api_key=Secret.from_token("test-api-key"),
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
        )
        assert component.client.api_key == "test-api-key"
        assert component.model == "claude-sonnet-4-20250514"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}

    def test_to_dict_default(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        data = component.to_dict()
        assert data == {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-20250514",
                "streaming_callback": None,
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {},
                "tools": None,
                "timeout": None,
                "max_retries": None,
            },
        }

    def test_to_dict_with_parameters(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be serialized to a dictionary with parameters.
        """
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        monkeypatch.setenv("ENV_VAR", "test-api-key")
        component = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ENV_VAR"),
            model="claude-sonnet-4-20250514",
            streaming_callback=print_streaming_chunk,
            generation_kwargs={"max_tokens": 10, "some_test_param": "test-params"},
            tools=[tool],
            timeout=10.0,
            max_retries=1,
        )
        data = component.to_dict()

        expected_dict = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-20250514",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "ignore_tools_thinking_messages": True,
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {
                                "x": {
                                    "type": "string",
                                },
                            },
                        },
                        "type": "haystack.tools.tool.Tool",
                    }
                ],
                "timeout": 10.0,
                "max_retries": 1,
            },
        }

        # add outputs_to_string, inputs_from_state and outputs_to_state tool parameters for compatibility with
        # haystack-ai>=2.12.0
        if hasattr(tool, "outputs_to_string"):
            expected_dict["init_parameters"]["tools"][0]["data"]["outputs_to_string"] = tool.outputs_to_string
        if hasattr(tool, "inputs_from_state"):
            expected_dict["init_parameters"]["tools"][0]["data"]["inputs_from_state"] = tool.inputs_from_state
        if hasattr(tool, "outputs_to_state"):
            expected_dict["init_parameters"]["tools"][0]["data"]["outputs_to_state"] = tool.outputs_to_state

        assert data == expected_dict

    def test_from_dict(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component can be deserialized from a dictionary.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-20250514",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
                "tools": [
                    {
                        "type": "haystack.tools.tool.Tool",
                        "data": {
                            "description": "description",
                            "function": "builtins.print",
                            "name": "name",
                            "parameters": {
                                "x": {
                                    "type": "string",
                                },
                            },
                        },
                    },
                ],
            },
        }
        component = AnthropicChatGenerator.from_dict(data)

        assert isinstance(component, AnthropicChatGenerator)
        assert component.model == "claude-sonnet-4-20250514"
        assert component.streaming_callback is print_streaming_chunk
        assert component.generation_kwargs == {"max_tokens": 10, "some_test_param": "test-params"}
        assert component.api_key == Secret.from_env_var("ANTHROPIC_API_KEY")
        assert component.tools == [
            Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)
        ]

    def test_from_dict_fail_wo_env_var(self, monkeypatch):
        """
        Test that the AnthropicChatGenerator component fails to deserialize from a dictionary without an API key.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "type": "env_var", "strict": True},
                "model": "claude-sonnet-4-20250514",
                "streaming_callback": "haystack.components.generators.utils.print_streaming_chunk",
                "generation_kwargs": {"max_tokens": 10, "some_test_param": "test-params"},
            },
        }
        with pytest.raises(ValueError):
            AnthropicChatGenerator.from_dict(data)

    def test_run(self, chat_messages, mock_chat_completion):
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        response = component.run(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    def test_run_with_params(self, chat_messages, mock_anthropic_completion):
        """
        Test that the AnthropicChatGenerator component can run with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = component.run(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        _, kwargs = mock_anthropic_completion.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert "Hello! I'm Claude." in response["replies"][0].text
        assert response["replies"][0].meta["model"] == "claude-sonnet-4-20250514"
        assert response["replies"][0].meta["finish_reason"] == "stop"

    def test_check_duplicate_tool_names(self, tools):
        """Test that the AnthropicChatGenerator component fails to initialize with duplicate tool names."""
        with pytest.raises(ValueError):
            AnthropicChatGenerator(tools=tools + tools)

    def test_convert_chat_completion_to_chat_message(self, mock_chat_completion):
        """
        Test converting Anthropic chat completion to ChatMessage
        """
        chat_completion = mock_chat_completion.return_value

        chat_message = _convert_chat_completion_to_chat_message(chat_completion, ignore_tools_thinking_messages=True)
        assert chat_message.text == "Hello, world!"
        assert chat_message.role == "assistant"
        assert chat_message.meta["model"] == "claude-sonnet-4-20250514"
        assert "usage" in chat_message.meta
        assert chat_message.meta["usage"]["prompt_tokens"] == 57
        assert chat_message.meta["usage"]["completion_tokens"] == 40

    def test_convert_chat_completion_to_chat_message_with_reasoning_and_tool_call(self):
        """
        Test converting Anthropic chat completion to ChatMessage
        """
        chat_completion = Message(
            id="msg_01MZF",
            content=[
                ThinkingBlock(signature="sign1", thinking="User has asked 2 questions", type="thinking"),
                TextBlock(citations=None, text="I'll provide the answers!", type="text"),
                ToolUseBlock(
                    id="toolu_01XEkx", input={"expression": "7 * (4 + 2)"}, name="calculator", type="tool_use"
                ),
            ],
            model="claude-sonnet-4-20250514",
            role="assistant",
            stop_reason="tool_use",
            stop_sequence=None,
            type="message",
            usage=Usage(input_tokens=507, output_tokens=219),
        )
        chat_message = _convert_chat_completion_to_chat_message(chat_completion, ignore_tools_thinking_messages=False)
        assert chat_message.text == "I'll provide the answers!"
        assert chat_message.reasoning.reasoning_text == "User has asked 2 questions"
        assert chat_message.reasoning.extra == {
            "reasoning_contents": [
                {"reasoning_content": {"reasoning_text": {"text": "User has asked 2 questions", "signature": "sign1"}}}
            ]
        }
        assert chat_message.meta["model"] == "claude-sonnet-4-20250514"
        assert chat_message.meta["finish_reason"] == "tool_calls"
        assert "usage" in chat_message.meta
        assert chat_message.meta["usage"]["prompt_tokens"] == 507
        assert chat_message.meta["usage"]["completion_tokens"] == 219

    def test_convert_anthropic_completion_chunks_with_multiple_tool_calls_and_reasoning_to_streaming_chunks(self):
        """
        Test converting Anthropic stream events with tools to Haystack StreamingChunks
        """
        component = AnthropicChatGenerator(api_key=Secret.from_token("test-api-key"))
        component_info = ComponentInfo.from_component(component)

        raw_chunks = []

        # Test message_start chunk
        message_start_chunk = RawMessageStartEvent(
            message=Message(
                id="msg_01ApGaijiGeLtxWLCKUKELfT",
                content=[],
                model="claude-sonnet-4-20250514",
                role="assistant",
                stop_reason=None,
                stop_sequence=None,
                type="message",
                usage=Usage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                    input_tokens=393,
                    output_tokens=3,
                    server_tool_use=None,
                    service_tier="standard",
                ),
            ),
            type="message_start",
        )
        raw_chunks.append(message_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta["message"]["model"] == message_start_chunk.message.model
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""

        # Test content_block_start for reasoning
        reasoning_block_start_chunk = RawContentBlockStartEvent(
            content_block=ThinkingBlock(type="thinking", signature="", thinking=""), index=0, type="content_block_start"
        )
        raw_chunks.append(reasoning_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 0
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""

        # Test content_block_delta for reasoning
        reasoning_delta_chunk = RawContentBlockDeltaEvent(
            delta=ThinkingDelta(thinking="The user is asking 2 questions.", type="thinking_delta"),
            index=0,
            type="content_block_delta",
        )
        raw_chunks.append(reasoning_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_delta_chunk.model_dump()
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 0
        assert streaming_chunk.tool_calls is None
        assert not streaming_chunk.start

        # Test content_block_delta for reasoning signature
        reasoning_signature_delta_chunk = RawContentBlockDeltaEvent(
            delta=SignatureDelta(signature="1234567890", type="signature_delta"),
            index=0,
            type="content_block_delta",
        )
        raw_chunks.append(reasoning_signature_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            reasoning_signature_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == reasoning_signature_delta_chunk.model_dump()
        assert streaming_chunk.content == ""

        # Test content_block_start for text
        text_block_start_chunk = RawContentBlockStartEvent(
            content_block=TextBlock(citations=None, text="", type="text"), index=1, type="content_block_start"
        )
        raw_chunks.append(text_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            text_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == text_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 1
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""

        # Test content_block_delta with text_delta
        text_delta_chunk = RawContentBlockDeltaEvent(
            delta=TextDelta(text="I'll calculate the factorial of 5", type="text_delta"),
            index=1,
            type="content_block_delta",
        )
        raw_chunks.append(text_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            text_delta_chunk, component_info=component_info, tool_call_index=0
        )

        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == text_delta_chunk.model_dump()
        assert streaming_chunk.content == text_delta_chunk.delta.text
        assert not streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 1
        assert streaming_chunk.tool_calls is None

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Test content_block_start for tool_use
        tool_block_start_chunk = RawContentBlockStartEvent(
            content_block=ToolUseBlock(
                id="toolu_011dE5KDKxSh6hi85EnRKZT3", input={}, name="calculator", type="tool_use"
            ),
            index=2,
            type="content_block_start",
        )
        raw_chunks.append(tool_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            tool_block_start_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == tool_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=0, id="toolu_011dE5KDKxSh6hi85EnRKZT3", tool_name="calculator", arguments=None)
        ]
        assert streaming_chunk.content == ""

        # Test content_block_delta with input_json_delta (empty)
        empty_json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json="", type="input_json_delta"), index=2, type="content_block_delta"
        )
        raw_chunks.append(empty_json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            empty_json_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == empty_json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=0, arguments=empty_json_delta_chunk.delta.partial_json)
        ]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert not streaming_chunk.start

        # Test content_block_delta with input_json_delta (with content)
        json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json='{"expression": 5 ', type="input_json_delta"),
            index=2,
            type="content_block_delta",
        )
        raw_chunks.append(json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            json_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [ToolCallDelta(index=0, arguments=json_delta_chunk.delta.partial_json)]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 2
        assert not streaming_chunk.start

        # Test message_delta chunk
        message_delta_chunk = RawMessageDeltaEvent(
            delta=Delta(stop_reason="tool_use", stop_sequence=None),
            type="message_delta",
            usage=MessageDeltaUsage(
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                input_tokens=None,
                output_tokens=77,
                server_tool_use=None,
            ),
        )
        raw_chunks.append(message_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == message_delta_chunk.model_dump()
        assert streaming_chunk.finish_reason == FINISH_REASON_MAPPING.get(message_delta_chunk.delta.stop_reason)
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""
        assert not streaming_chunk.start

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Test content_block_start for second tool_call
        tool_block_start_chunk = RawContentBlockStartEvent(
            content_block=ToolUseBlock(
                id="toolu_011dE5KDKxSh6hi85EnRKZT4", input={}, name="factorial", type="tool_use"
            ),
            index=3,
            type="content_block_start",
        )
        raw_chunks.append(tool_block_start_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            tool_block_start_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == tool_block_start_chunk.model_dump()
        assert streaming_chunk.start
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=1, id="toolu_011dE5KDKxSh6hi85EnRKZT4", tool_name="factorial", arguments=None)
        ]
        assert streaming_chunk.content == ""

        # Test content_block_delta with input_json_delta (empty)
        empty_json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json="", type="input_json_delta"), index=3, type="content_block_delta"
        )
        raw_chunks.append(empty_json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            empty_json_delta_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == empty_json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [
            ToolCallDelta(index=1, arguments=empty_json_delta_chunk.delta.partial_json)
        ]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert not streaming_chunk.start

        # Test content_block_delta with input_json_delta (with content)
        json_delta_chunk = RawContentBlockDeltaEvent(
            delta=InputJSONDelta(partial_json='{"expression": 5 ', type="input_json_delta"),
            index=3,
            type="content_block_delta",
        )
        raw_chunks.append(json_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            json_delta_chunk, component_info=component_info, tool_call_index=1
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == json_delta_chunk.model_dump()
        assert streaming_chunk.tool_calls == [ToolCallDelta(index=1, arguments=json_delta_chunk.delta.partial_json)]
        assert streaming_chunk.content == ""
        assert streaming_chunk.finish_reason is None
        assert streaming_chunk.index == 3
        assert not streaming_chunk.start

        # Test message_delta chunk
        message_delta_chunk = RawMessageDeltaEvent(
            delta=Delta(stop_reason="tool_use", stop_sequence=None),
            type="message_delta",
            usage=MessageDeltaUsage(
                cache_creation_input_tokens=None,
                cache_read_input_tokens=None,
                input_tokens=None,
                output_tokens=77,
                server_tool_use=None,
            ),
        )
        raw_chunks.append(message_delta_chunk)
        streaming_chunk = _convert_anthropic_chunk_to_streaming_chunk(
            message_delta_chunk, component_info=component_info, tool_call_index=0
        )
        assert streaming_chunk.component_info == component_info
        assert streaming_chunk.meta == message_delta_chunk.model_dump()
        assert streaming_chunk.finish_reason == FINISH_REASON_MAPPING.get(message_delta_chunk.delta.stop_reason)
        assert streaming_chunk.index is None
        assert streaming_chunk.tool_calls is None
        assert streaming_chunk.content == ""
        assert not streaming_chunk.start

        # In response flow, here will be another content_block_stop chunk
        # content_block_stop_chunk = RawContentBlockStopEvent(index=0, type="content_block_stop")
        # but we don't stream it

        # Then a message_stop chunk
        # message_stop_chunk = RawMessageStopEvent(type="message_stop")
        # but we don't stream it

        generator = AnthropicChatGenerator(Secret.from_token("test-api-key"))
        message = generator._process_response(raw_chunks)
        assert message["replies"][0].meta["usage"] == {
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": None,
            "input_tokens": 393,
            "output_tokens": 77,
            "server_tool_use": None,
        }

    def test_convert_streaming_chunks_to_chat_message_with_multiple_tool_calls(self):
        """
        Test converting streaming chunks to a chat message with tool calls
        """
        # Create a sequence of streaming chunks that simulate Anthropic's response
        chunks = [
            # Message start with input tokens
            StreamingChunk(
                content="",
                meta={
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-20250514",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 0},
                    },
                },
                component_info=ComponentInfo.from_component(self),
                start=True,
            ),
            # Initial text content
            StreamingChunk(
                content="",
                meta={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                component_info=ComponentInfo.from_component(self),
                index=0,
                start=True,
            ),
            StreamingChunk(
                content="Let me check",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Let me check"},
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
            ),
            StreamingChunk(
                content=" the weather",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " the weather"},
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id="toolu_123", tool_name="weather", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments='{"city":')],
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments='"Paris"}')],
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "tool_use", "id": "toolu_224", "name": "factorial", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id="toolu_224", tool_name="factorial", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"expression":'},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id=None, tool_name=None, arguments='{"expression":')],
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": " 5 }"},
                },
                component_info=ComponentInfo.from_component(self),
                index=2,
                tool_calls=[ToolCallDelta(index=1, id=None, tool_name=None, arguments=" 5 }")],
            ),
            # Final message delta
            StreamingChunk(
                content="",
                meta={
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_calls", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
                component_info=ComponentInfo.from_component(self),
                finish_reason="tool_calls",
            ),
        ]

        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Cannot test creating ReasoningContent from StreamingChunk objects
        # because reasoning is added outside _convert_streaming_chunks_to_chat_message

        # Verify the message content
        assert message.text == "Let me check the weather"
        # Verify tool calls
        assert len(message.tool_calls) == 2
        tool_call = message.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        tool_call = message.tool_calls[1]
        assert tool_call.id == "toolu_224"
        assert tool_call.tool_name == "factorial"
        assert tool_call.arguments == {"expression": 5}

        # Verify meta information
        assert message._meta["index"] == 0
        assert message._meta["finish_reason"] == "tool_calls"
        assert message._meta["usage"] == {"output_tokens": 40}

    def test_convert_streaming_chunks_to_chat_message_tool_call_with_empty_arguments(self):
        """
        Test converting streaming chunks with an empty tool call arguments
        """

        # Create a sequence of streaming chunks that simulate Anthropic's response
        chunks = [
            # Message start with input tokens
            StreamingChunk(
                content="",
                meta={
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-sonnet-4-20250514",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 25, "output_tokens": 0},
                    },
                },
                component_info=ComponentInfo.from_component(self),
                index=0,
                start=True,
            ),
            # Initial text content
            StreamingChunk(
                content="",
                meta={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                component_info=ComponentInfo.from_component(self),
                index=1,
                start=True,
            ),
            StreamingChunk(
                content="Let me check",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Let me check"},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
            ),
            StreamingChunk(
                content=" the weather",
                meta={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": " the weather"},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
            ),
            # Tool use content
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "tool_use", "id": "toolu_123", "name": "weather", "input": {}},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id="toolu_123", tool_name="weather", arguments=None)],
                start=True,
            ),
            StreamingChunk(
                content="",
                meta={
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                tool_calls=[ToolCallDelta(index=0, id=None, tool_name=None, arguments="")],
            ),
            # Final message delta
            StreamingChunk(
                content="",
                meta={
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_calls", "stop_sequence": None},
                    "usage": {"output_tokens": 40},
                },
                component_info=ComponentInfo.from_component(self),
                index=1,
                finish_reason="tool_calls",
            ),
        ]

        message = _convert_streaming_chunks_to_chat_message(chunks)

        # Verify the message content
        assert message.text == "Let me check the weather"

        # Verify tool calls
        assert len(message.tool_calls) == 1
        tool_call = message.tool_calls[0]
        assert tool_call.id == "toolu_123"
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {}

        # Verify meta information
        assert message._meta["index"] == 0
        assert message._meta["finish_reason"] == "tool_calls"
        assert message._meta["usage"] == {"output_tokens": 40}

    def test_serde_in_pipeline(self):
        tool = Tool(name="name", description="description", parameters={"x": {"type": "string"}}, function=print)

        generator = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY", strict=False),
            model="claude-sonnet-4-20250514",
            generation_kwargs={"temperature": 0.6},
            tools=[tool],
        )

        pipeline = Pipeline()
        pipeline.add_component("generator", generator)

        pipeline_dict = pipeline.to_dict()
        type_ = "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator"

        expected_dict = {
            "metadata": {},
            "max_runs_per_component": 100,
            "connection_type_validation": True,
            "components": {
                "generator": {
                    "type": type_,
                    "init_parameters": {
                        "api_key": {"type": "env_var", "env_vars": ["ANTHROPIC_API_KEY"], "strict": False},
                        "model": "claude-sonnet-4-20250514",
                        "generation_kwargs": {"temperature": 0.6},
                        "ignore_tools_thinking_messages": True,
                        "streaming_callback": None,
                        "tools": [
                            {
                                "type": "haystack.tools.tool.Tool",
                                "data": {
                                    "name": "name",
                                    "description": "description",
                                    "parameters": {"x": {"type": "string"}},
                                    "function": "builtins.print",
                                },
                            }
                        ],
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

        pipeline_yaml = pipeline.dumps()

        new_pipeline = Pipeline.loads(pipeline_yaml)
        assert new_pipeline == pipeline

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run(self):
        """
        Integration test that the AnthropicChatGenerator component can run with default parameters.
        """
        component = AnthropicChatGenerator()
        results = component.run(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-sonnet-4-20250514" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_wrong_model(self, chat_messages):
        component = AnthropicChatGenerator(model="something-obviously-wrong")
        with pytest.raises(anthropic.NotFoundError):
            component.run(chat_messages)

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the OpenAI API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_streaming(self):
        """
        Integration test that the AnthropicChatGenerator component can run with streaming.
        """

        class Callback:
            def __init__(self):
                self.responses = ""
                self.counter = 0

            def __call__(self, chunk: StreamingChunk) -> None:
                self.counter += 1
                self.responses += chunk.content if chunk.content else ""
                assert chunk.component_info is not None
                assert chunk.component_info.type.endswith("chat_generator.AnthropicChatGenerator")

        callback = Callback()

        component = AnthropicChatGenerator(streaming_callback=callback, timeout=30.0, max_retries=1)
        results = component.run([ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text

        assert "claude-sonnet-4-20250514" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert callback.counter > 1
        assert "Paris" in callback.responses
        assert "input_tokens" in message.meta["usage"]
        assert "output_tokens" in message.meta["usage"]

    def test_convert_message_to_anthropic_format(self):
        """
        Test that the AnthropicChatGenerator component can convert a ChatMessage to Anthropic format.
        """
        messages = [ChatMessage.from_system("You are good assistant")]
        assert _convert_messages_to_anthropic_format(messages) == (
            [{"type": "text", "text": "You are good assistant"}],
            [],
        )

        messages = [ChatMessage.from_user("I have a question")]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [{"role": "user", "content": [{"type": "text", "text": "I have a question"}]}],
        )

        messages = [ChatMessage.from_assistant(text="I have an answer", meta={"finish_reason": "stop"})]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [{"role": "assistant", "content": [{"type": "text", "text": "I have an answer"}]}],
        )

        messages = [
            ChatMessage.from_assistant(
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})]
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}}],
                }
            ],
        )

        messages = [
            ChatMessage.from_assistant(
                text="",  # this should not happen, but we should handle it without errors
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})],
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}}],
                }
            ],
        )

        tool_result = json.dumps({"weather": "sunny", "temperature": "25"})
        messages = [
            ChatMessage.from_tool(
                tool_result=tool_result, origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            )
        ]
        assert _convert_messages_to_anthropic_format(messages) == (
            [],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "123",
                            "content": [{"type": "text", "text": '{"weather": "sunny", "temperature": "25"}'}],
                            "is_error": False,
                        }
                    ],
                }
            ],
        )

        messages = [
            ChatMessage.from_assistant(
                text="For that I'll need to check the weather",
                tool_calls=[ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})],
            )
        ]
        result = _convert_messages_to_anthropic_format(messages)
        assert result == (
            [],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "For that I'll need to check the weather"},
                        {"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}},
                    ],
                }
            ],
        )

    def test_convert_message_to_anthropic_format_complex(self):
        """
        Test that the AnthropicChatGenerator can convert a complex sequence of ChatMessages to Anthropic format.
        In particular, we check that different tool results are packed in a single dictionary with role=user.
        """

        messages = [
            ChatMessage.from_system("You are good assistant"),
            ChatMessage.from_user("What's the weather like in Paris? And how much is 2+2?"),
            ChatMessage.from_assistant(
                text="",
                tool_calls=[
                    ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"}),
                    ToolCall(id="456", tool_name="math", arguments={"expression": "2+2"}),
                ],
            ),
            ChatMessage.from_tool(
                tool_result="22° C", origin=ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
            ),
            ChatMessage.from_tool(
                tool_result="4", origin=ToolCall(id="456", tool_name="math", arguments={"expression": "2+2"})
            ),
        ]

        system_messages, non_system_messages = _convert_messages_to_anthropic_format(messages)

        assert system_messages == [{"type": "text", "text": "You are good assistant"}]
        assert non_system_messages == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather like in Paris? And how much is 2+2?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "123", "name": "weather", "input": {"city": "Paris"}},
                    {"type": "tool_use", "id": "456", "name": "math", "input": {"expression": "2+2"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "123",
                        "content": [{"type": "text", "text": "22° C"}],
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "456",
                        "content": [{"type": "text", "text": "4"}],
                        "is_error": False,
                    },
                ],
            },
        ]

    def test_convert_message_to_anthropic_format_with_image(self):
        """Test that a ChatMessage with ImageContent is converted to Anthropic format correctly."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        _, non_system_messages = _convert_messages_to_anthropic_format([message])

        assert len(non_system_messages) == 1
        anthropic_message = non_system_messages[0]
        assert anthropic_message["role"] == "user"
        assert len(anthropic_message["content"]) == 2

        # Check text and image blocks
        assert anthropic_message["content"][0]["type"] == "text"
        assert anthropic_message["content"][0]["text"] == "What's in this image?"
        assert anthropic_message["content"][1]["type"] == "image"
        assert anthropic_message["content"][1]["source"]["type"] == "base64"
        assert anthropic_message["content"][1]["source"]["media_type"] == "image/png"
        assert anthropic_message["content"][1]["source"]["data"] == base64_image

    def test_convert_message_to_anthropic_format_with_unsupported_mime_type(self):
        """Test that a ChatMessage with unsupported mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/bmp")  # Unsupported format
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        with pytest.raises(ValueError, match="Unsupported image format: image/bmp"):
            _convert_messages_to_anthropic_format([message])

    def test_convert_message_to_anthropic_format_with_none_mime_type(self):
        """Test that a ChatMessage with None mime type raises ValueError."""
        base64_image = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        image_content = ImageContent(base64_image=base64_image, mime_type="image/png")
        # Manually set mime_type to None to test the validation
        image_content.mime_type = None
        message = ChatMessage.from_user(content_parts=["What's in this image?", image_content])

        with pytest.raises(ValueError, match="Unsupported image format: None"):
            _convert_messages_to_anthropic_format([message])

    def test_convert_message_to_anthropic_invalid(self):
        """
        Test that the AnthropicChatGenerator component fails to convert an invalid ChatMessage to Anthropic format.
        """
        message = ChatMessage(_role=ChatRole.ASSISTANT, _content=[])
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

        tool_call_null_id = ToolCall(id=None, tool_name="weather", arguments={"city": "Paris"})
        message = ChatMessage.from_assistant(tool_calls=[tool_call_null_id])
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

        message = ChatMessage.from_tool(tool_result="result", origin=tool_call_null_id)
        with pytest.raises(ValueError):
            _convert_messages_to_anthropic_format([message])

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert "completion_tokens" in message.meta["usage"]

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        # the model tends to make tool calls if provided with tools, so we don't pass them here
        results = component.run(new_messages, generation_kwargs={"max_tokens": 50})

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    def test_init_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test initialization with a mixed list of Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=lambda x: x)
        tool2 = Tool(
            name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=lambda y: y
        )
        tool3 = Tool(name="tool3", description="Third tool", parameters={"z": {"type": "string"}}, function=lambda z: z)
        toolset1 = Toolset([tool2])

        generator = AnthropicChatGenerator(tools=[tool1, toolset1, tool3])

        assert generator.tools == [tool1, toolset1, tool3]
        assert isinstance(generator.tools, list)
        assert len(generator.tools) == 3

    def test_serde_with_mixed_tools_and_toolsets(self, monkeypatch):
        """Test serialization/deserialization with mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=print)
        tool2 = Tool(name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=print)
        toolset1 = Toolset([tool2])

        generator = AnthropicChatGenerator(tools=[tool1, toolset1])
        data = generator.to_dict()

        # Verify serialization preserves structure
        assert isinstance(data["init_parameters"]["tools"], list)
        assert len(data["init_parameters"]["tools"]) == 2
        assert data["init_parameters"]["tools"][0]["type"] == "haystack.tools.tool.Tool"
        assert data["init_parameters"]["tools"][1]["type"] == "haystack.tools.toolset.Toolset"

        # Verify deserialization
        restored = AnthropicChatGenerator.from_dict(data)
        assert isinstance(restored.tools, list)
        assert len(restored.tools) == 2
        assert isinstance(restored.tools[0], Tool)
        assert isinstance(restored.tools[1], Toolset)
        assert restored.tools[0].name == "tool1"
        assert next(iter(restored.tools[1])).name == "tool2"

    def test_run_with_mixed_tools_and_toolsets(self, chat_messages, mock_anthropic_completion, monkeypatch):
        """Test that the run method works with mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="tool1", description="First tool", parameters={"x": {"type": "string"}}, function=lambda x: x)
        tool2 = Tool(
            name="tool2", description="Second tool", parameters={"y": {"type": "string"}}, function=lambda y: y
        )
        toolset1 = Toolset([tool2])

        component = AnthropicChatGenerator(tools=[tool1, toolset1])
        response = component.run(chat_messages)

        # Check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1

        # Check that the component called the Anthropic API with the correct tools
        _, kwargs = mock_anthropic_completion.call_args
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 2
        assert kwargs["tools"][0]["name"] == "tool1"
        assert kwargs["tools"][1]["name"] == "tool2"

    def test_init_with_duplicate_tools_in_mixed_list(self, monkeypatch):
        """Test that initialization fails with duplicate tool names in mixed Tools and Toolsets."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        tool1 = Tool(name="duplicate", description="First", parameters={}, function=lambda: None)
        tool2 = Tool(name="duplicate", description="Second", parameters={}, function=lambda: None)
        toolset1 = Toolset([tool2])

        with pytest.raises(ValueError, match="duplicate"):
            AnthropicChatGenerator(tools=[tool1, toolset1])

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_toolset(self):
        """
        Integration test that the AnthropicChatGenerator component can run with a Toolset.
        """

        def weather_function(city: str) -> str:
            """Get weather information for a city."""
            weather_data = {"Paris": "22°C, sunny", "London": "15°C, rainy", "Tokyo": "18°C, cloudy"}
            return weather_data.get(city, "Weather data not available")

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

        echo_tool = Tool(
            name="echo",
            description="Echo a text",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=echo_function,
        )

        # Create Toolset
        toolset = Toolset([weather_tool, echo_tool])

        # Test with weather query
        initial_messages = [ChatMessage.from_user("What's the weather like in Tokyo?")]
        component = AnthropicChatGenerator(tools=toolset)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Tokyo"}
        assert message.meta["finish_reason"] == "tool_calls"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools and streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(
            tools=tools, streaming_callback=print_streaming_chunk, generation_kwargs={"max_tokens": 11000}
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # this is Anthropic message prior to tool call
        assert message.text is not None
        assert "weather" in message.text.lower()
        assert "paris" in message.text.lower()

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert "output_tokens" in message.meta["usage"]
        assert "input_tokens" in message.meta["usage"]

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tools_streaming_and_reasoning(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with tools and streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(
            tools=tools,
            streaming_callback=print_streaming_chunk,
            generation_kwargs={
                "thinking": {"type": "enabled", "budget_tokens": 10000},
                "max_tokens": 11000,
            },
        )
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # this is Anthropic thinking message prior to tool call
        assert message.reasoning.reasoning_text

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"
        assert "output_tokens" in message.meta["usage"]
        assert "input_tokens" in message.meta["usage"]

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_tool_with_no_args_streaming(self, tool_with_no_parameters):
        """
        Integration test that the AnthropicChatGenerator component can run with a tool that has no arguments and
        streaming.
        """
        initial_messages = [ChatMessage.from_user("Print Hello World using the print hello world tool.")]
        component = AnthropicChatGenerator(tools=[tool_with_no_parameters], streaming_callback=print_streaming_chunk)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # this is Anthropic thinking message prior to tool call
        assert message.text is not None

        # now we have the tool call
        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "hello_world"
        assert tool_call.arguments == {}
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="Hello World!", origin=tool_call),
        ]
        results = component.run(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "hello" in final_message.text.lower()

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_parallel_tools(self, tools):
        """
        Integration test that the AnthropicChatGenerator component can run with parallel tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris and Berlin?")]
        component = AnthropicChatGenerator(tools=tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        # now we have the tool call
        assert len(message.tool_calls) == 2
        tool_call_paris = message.tool_calls[0]
        assert isinstance(tool_call_paris, ToolCall)
        assert tool_call_paris.id is not None
        assert tool_call_paris.tool_name == "weather"
        assert tool_call_paris.arguments["city"] in {"Paris", "Berlin"}
        assert message.meta["finish_reason"] == "tool_calls"

        tool_call_berlin = message.tool_calls[1]
        assert isinstance(tool_call_berlin, ToolCall)
        assert tool_call_berlin.id is not None
        assert tool_call_berlin.tool_name == "weather"
        assert tool_call_berlin.arguments["city"] in {"Berlin", "Paris"}

        # Anthropic expects results from both tools in the same message
        # https://docs.anthropic.com/en/docs/build-with-claude/tool-use#handling-tool-use-and-tool-result-content-blocks
        # [optional] Continue the conversation by sending a new message with the role of user, and a content block
        # containing the tool_result type and the following information:
        # tool_use_id: The id of the tool use request this is a result for.
        # content: The result of the tool, as a string (e.g. "content": "15 degrees") or list of
        # nested content blocks (e.g. "content": [{"type": "text", "text": "15 degrees"}]).
        # These content blocks can use the text or image types.
        # is_error (optional): Set to true if the tool execution resulted in an error.
        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call_paris, error=False),
            ChatMessage.from_tool(tool_result="12° C", origin=tool_call_berlin, error=False),
        ]

        # Response from the model contains results from both tools
        results = component.run(new_messages)
        message = results["replies"][0]
        assert not message.tool_calls
        assert len(message.text) > 0
        assert "paris" in message.text.lower()
        assert "berlin" in message.text.lower()
        assert "22°" in message.text
        assert "12°" in message.text
        assert message.meta["finish_reason"] == "stop"

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_mixed_tools(self):
        """
        Integration test that verifies AnthropicChatGenerator works with mixed Tool and Toolset.
        This tests that the LLM can correctly invoke tools from both a standalone Tool and a Toolset.
        """

        def weather(city: str):
            return f"The weather in {city} is sunny and 32°C"

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

        population_tool = Tool(
            name="population",
            description="useful to determine the population of a given city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city to get population for, e.g. Paris, Berlin",
                    }
                },
                "required": ["city"],
            },
            function=population,
        )

        # Create a toolset with the population tool
        population_toolset = Toolset([population_tool])

        # Mix standalone tool with toolset
        mixed_tools = [weather_tool, population_toolset]

        initial_messages = [
            ChatMessage.from_user("What's the weather like in Paris and what is the population of Berlin?")
        ]
        component = AnthropicChatGenerator(tools=mixed_tools)
        results = component.run(messages=initial_messages)

        assert len(results["replies"]) > 0, "No replies received"

        first_reply = results["replies"][0]
        assert isinstance(first_reply, ChatMessage), "First reply is not a ChatMessage instance"
        assert ChatMessage.is_from(first_reply, ChatRole.ASSISTANT), "First reply is not from the assistant"
        assert first_reply.tool_calls, "First reply has no tool calls"

        tool_calls = first_reply.tool_calls
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
            assert first_reply.meta["finish_reason"] == "tool_calls"

        # Mock the response we'd get from ToolInvoker
        tool_result_messages = []
        for tool_call in tool_calls:
            if tool_call.tool_name == "weather":
                result = "The weather in Paris is sunny and 32°C"
            else:  # population
                result = "The population of Berlin is 2.2 million"
            tool_result_messages.append(ChatMessage.from_tool(tool_result=result, origin=tool_call))

        new_messages = [*initial_messages, first_reply, *tool_result_messages]
        results = component.run(new_messages)

        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "berlin" in final_message.text.lower()

    def test_prompt_caching_enabled(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API when prompt
        caching is enabled
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        assert component.generation_kwargs.get("extra_headers", {}).get("anthropic-beta") == "prompt-caching-2024-07-31"

    def test_prompt_caching_cache_control_without_extra_headers(self, monkeypatch, mock_chat_completion, caplog):
        """
        Test that the cache_control is removed from the messages when prompt caching is not enabled via extra_headers
        This is to avoid Anthropic errors when prompt caching is not enabled
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()

        messages = [ChatMessage.from_system("System message"), ChatMessage.from_user("User message")]

        # Add cache_control to messages
        for msg in messages:
            msg._meta["cache_control"] = {"type": "ephemeral"}

        # Invoke run with messages
        component.run(messages)

        # Check caplog for the warning message that should have been logged
        assert not any("Prompt caching is not enabled" in record.message for record in caplog.records)
        # Check that the Anthropic API was called without cache_control in messages so that it does not raise an error
        _, kwargs = mock_chat_completion.call_args
        for msg in kwargs["messages"]:
            assert "cache_control" not in msg

    @pytest.mark.parametrize("enable_caching", [True, False])
    def test_run_with_prompt_caching(self, monkeypatch, mock_chat_completion, enable_caching):
        """
        Test that the generation_kwargs extra_headers are correctly passed to the Anthropic API in both cases of
        prompt caching being enabled or not
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if enable_caching else {}
        component = AnthropicChatGenerator(generation_kwargs=generation_kwargs)

        messages = [ChatMessage.from_system("System message"), ChatMessage.from_user("User message")]

        component.run(messages)

        # Check that the Anthropic API was called with the correct headers
        _, kwargs = mock_chat_completion.call_args
        headers = kwargs.get("extra_headers", {})
        if enable_caching:
            assert "anthropic-beta" in headers
        else:
            assert "anthropic-beta" not in headers

    def test_to_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly serialized to a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator(
            generation_kwargs={"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}}
        )
        data = component.to_dict()
        assert (
            data["init_parameters"]["generation_kwargs"]["extra_headers"]["anthropic-beta"]
            == "prompt-caching-2024-07-31"
        )

    def test_from_dict_with_prompt_caching(self, monkeypatch):
        """
        Test that the generation_kwargs extra_headers are correctly deserialized from a dictionary
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        data = {
            "type": "haystack_integrations.components.generators.anthropic.chat.chat_generator.AnthropicChatGenerator",
            "init_parameters": {
                "api_key": {"env_vars": ["ANTHROPIC_API_KEY"], "strict": True, "type": "env_var"},
                "model": "claude-sonnet-4-20250514",
                "generation_kwargs": {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}},
            },
        }
        component = AnthropicChatGenerator.from_dict(data)
        assert component.generation_kwargs["extra_headers"]["anthropic-beta"] == "prompt-caching-2024-07-31"

    def test_cache_control_forwarded_for_all_block_types(self, monkeypatch, mock_chat_completion):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()

        sys_msg = ChatMessage.from_system("sys")
        sys_msg._meta["cache_control"] = {"type": "ephemeral"}

        usr_msg = ChatMessage.from_user(
            "doc chunk",
            meta={"cache_control": {"type": "ephemeral"}},
        )

        tool_call = ToolCall(id="123", tool_name="weather", arguments={"city": "Paris"})
        asst_msg = ChatMessage.from_assistant(
            tool_calls=[tool_call],
            meta={"cache_control": {"type": "ephemeral"}},
        )

        tool_res = ChatMessage.from_tool(
            origin=tool_call,
            tool_result="sunny",
            meta={"cache_control": {"type": "ephemeral"}},
        )

        component.run([sys_msg, usr_msg, asst_msg, tool_res])

        _, kwargs = mock_chat_completion.call_args

        for blk in kwargs["system"]:
            assert blk.get("cache_control") == {"type": "ephemeral"}

        assert all("cache_control" not in msg for msg in kwargs["messages"])

        for msg in kwargs["messages"]:
            for cblk in msg["content"]:
                assert cblk.get("cache_control") == {"type": "ephemeral"}

    @pytest.mark.parametrize(
        "beta_header",
        [
            "featureA,extended-cache-ttl-2025-04-11",
            "featureA , featureB , new-fancy-stuff",
        ],
    )
    def test_extra_headers_pass_through(self, monkeypatch, mock_chat_completion, beta_header):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")

        component = AnthropicChatGenerator(generation_kwargs={"extra_headers": {"anthropic-beta": beta_header}})
        component.run([ChatMessage.from_user("ping")])

        _, kwargs = mock_chat_completion.call_args
        assert kwargs["extra_headers"]["anthropic-beta"] == beta_header

    def test_convert_messages_attaches_cache_control(self):
        user = ChatMessage.from_user(
            "hello",
            meta={
                "cache_control": {
                    "type": "ephemeral",
                }
            },
        )
        sys = ChatMessage.from_system("hi", meta={"cache_control": {"type": "ephemeral", "example_key": "example_val"}})
        sys_blocks, non_sys = _convert_messages_to_anthropic_format([sys, user])

        assert sys_blocks[0]["cache_control"] == {"type": "ephemeral", "example_key": "example_val"}
        assert non_sys[0]["content"][0]["cache_control"]["type"] == "ephemeral"

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY", None), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.parametrize("cache_enabled", [True, False])
    def test_prompt_caching_live_run(self, cache_enabled):
        generation_kwargs = {"extra_headers": {"anthropic-beta": "prompt-caching-2024-07-31"}} if cache_enabled else {}

        claude_llm = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"), generation_kwargs=generation_kwargs
        )

        # see https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#cache-limitations
        system_message = ChatMessage.from_system("This is the cached, here we make it at least 1024 tokens long." * 70)
        if cache_enabled:
            system_message._meta["cache_control"] = {"type": "ephemeral"}

        messages = [system_message, ChatMessage.from_user("What's in cached content?")]
        result = claude_llm.run(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        token_usage = result["replies"][0].meta.get("usage")

        if cache_enabled:
            # either we created cache or we read it (depends on how you execute this integration test)
            assert (
                token_usage.get("cache_creation_input_tokens") > 1024
                or token_usage.get("cache_read_input_tokens") > 1024
            )
        else:
            assert token_usage["cache_creation_input_tokens"] == 0
            assert token_usage["cache_read_input_tokens"] == 0

    @pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
    @pytest.mark.integration
    @pytest.mark.parametrize("cache_enabled", [True, False])
    def test_prompt_caching_live_run_with_user_message(self, cache_enabled):
        claude_llm = AnthropicChatGenerator(
            api_key=Secret.from_env_var("ANTHROPIC_API_KEY"),
        )

        system_message = ChatMessage.from_system("Hello from system. Just a generic instruction.")

        user_message = ChatMessage.from_user("This is a user message that should be long enough to cache. " * 100)
        if cache_enabled:
            user_message._meta["cache_control"] = {"type": "ephemeral"}

        messages = [system_message, user_message]
        result = claude_llm.run(messages)

        assert "replies" in result
        assert len(result["replies"]) == 1
        token_usage = result["replies"][0].meta.get("usage")

        if cache_enabled:
            assert (
                token_usage.get("cache_creation_input_tokens", 0) > 1024
                or token_usage.get("cache_read_input_tokens", 0) > 1024
            ), f"Unexpected usage stats: {token_usage}"
        else:
            assert token_usage.get("cache_creation_input_tokens", 0) == 0
            assert token_usage.get("cache_read_input_tokens", 0) == 0

    @pytest.mark.parametrize("streaming_callback", [None, Mock()])
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_reasoning(self, streaming_callback):
        chat_generator = AnthropicChatGenerator(
            model="claude-sonnet-4-20250514",
            generation_kwargs={"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 11000},
            streaming_callback=streaming_callback,
        )

        message = ChatMessage.from_user("2+3?")
        response = chat_generator.run([message])["replies"][0]

        assert isinstance(response, ChatMessage)
        assert response.text and len(response.text) > 0
        assert response.reasoning is not None
        assert len(response.reasoning.reasoning_text) > 0

        new_message = ChatMessage.from_user("Now multiply the result by 10.")
        new_response = chat_generator.run([message, response, new_message])["replies"][0]
        assert isinstance(new_response, ChatMessage)
        assert new_response.text and len(new_response.text) > 0
        assert new_response.reasoning is not None
        assert len(new_response.reasoning.reasoning_text) > 0
        assert "reasoning_contents" in new_response.reasoning.extra

        if streaming_callback:
            streaming_callback.assert_called()

    @pytest.mark.parametrize("streaming_callback", [None, Mock()])
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    def test_live_run_with_redacted_thinking(self, streaming_callback):
        """
        This test uses a magic string to trigger the redacted thinking
        feature and works with claude-3-7-sonnet-20250219.
        """
        chat_generator = AnthropicChatGenerator(
            model="claude-3-7-sonnet-20250219",
            generation_kwargs={"thinking": {"type": "enabled", "budget_tokens": 10000}, "max_tokens": 11000},
            streaming_callback=streaming_callback,
        )

        message = ChatMessage.from_user(
            "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB"
        )
        response = chat_generator.run([message])["replies"][0]

        assert isinstance(response, ChatMessage)
        assert response.text and len(response.text) > 0
        assert response.reasoning is not None
        # we cannot be sure how many redacted blocks will be returned with streaming
        assert "[REDACTED]" in response.reasoning.reasoning_text
        assert "reasoning_contents" in response.reasoning.extra
        assert (
            len(response.reasoning.extra.get("reasoning_contents")[0].get("reasoning_content").get("redacted_thinking"))
            > 0
        )

        new_message = ChatMessage.from_user("Now tell me whats the capital of France.")
        new_response = chat_generator.run([message, response, new_message])["replies"][0]

        assert isinstance(new_response, ChatMessage)
        assert new_response.text and len(new_response.text) > 0
        assert new_response.reasoning is not None
        assert "[REDACTED]" in new_response.reasoning.reasoning_text
        assert "reasoning_contents" in new_response.reasoning.extra
        assert (
            len(
                new_response.reasoning.extra.get("reasoning_contents")[0]
                .get("reasoning_content")
                .get("redacted_thinking")
            )
            > 0
        )

        if streaming_callback:
            streaming_callback.assert_called()


class TestAnthropicChatGeneratorAsync:
    @pytest.fixture
    async def mock_anthropic_completion_async(self):
        with patch("anthropic.resources.messages.AsyncMessages.create") as mock_anthropic:
            completion = Message(
                id="foo",
                type="message",
                model="claude-sonnet-4-20250514",
                role="assistant",
                content=[TextBlockParam(type="text", text="Hello! I'm Claude.")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=10, output_tokens=20),
            )
            # Make the mock return an awaitable
            mock_anthropic.return_value = AsyncMock(return_value=completion)()
            yield mock_anthropic

    @pytest.fixture
    async def mock_anthropic_completion_async_with_tool(self):
        with patch("anthropic.resources.messages.AsyncMessages.create") as mock_anthropic:
            completion = Message(
                id="foo",
                type="message",
                model="claude-sonnet-4-20250514",
                role="assistant",
                content=[
                    TextBlockParam(type="text", text="Let me check the weather for you."),
                    {"type": "tool_use", "id": "tool_123", "name": "weather", "input": {"city": "Paris"}},
                ],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=20),
            )
            # Make the mock return an awaitable
            mock_anthropic.return_value = AsyncMock(return_value=completion)()
            yield mock_anthropic

    @pytest.mark.asyncio
    async def test_run_async(self, chat_messages, mock_anthropic_completion_async, monkeypatch):
        """
        Test that the async run method of AnthropicChatGenerator works correctly.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
        component = AnthropicChatGenerator()
        response = await component.run_async(chat_messages)

        # check that the component returns the correct ChatMessage response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert [isinstance(reply, ChatMessage) for reply in response["replies"]]

    @pytest.mark.asyncio
    async def test_run_async_with_params(self, chat_messages, mock_anthropic_completion_async):
        """
        Test that the async run method of AnthropicChatGenerator works with parameters.
        """
        component = AnthropicChatGenerator(
            api_key=Secret.from_token("test-api-key"), generation_kwargs={"max_tokens": 10, "temperature": 0.5}
        )
        response = await component.run_async(chat_messages)

        # Check that the component calls the Anthropic API with the correct parameters
        _, kwargs = mock_anthropic_completion_async.call_args
        assert kwargs["max_tokens"] == 10
        assert kwargs["temperature"] == 0.5

        # Check that the component returns the correct response
        assert isinstance(response, dict)
        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) == 1
        assert isinstance(response["replies"][0], ChatMessage)
        assert "Hello! I'm Claude." in response["replies"][0].text
        assert response["replies"][0].meta["model"] == "claude-sonnet-4-20250514"
        assert response["replies"][0].meta["finish_reason"] == "stop"
        assert "completion_tokens" in response["replies"][0].meta["usage"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async(self):
        """
        Integration test that the async run method of AnthropicChatGenerator works with default parameters.
        """
        component = AnthropicChatGenerator()
        results = await component.run_async(messages=[ChatMessage.from_user("What's the capital of France?")])
        assert len(results["replies"]) == 1
        message: ChatMessage = results["replies"][0]
        assert "Paris" in message.text
        assert "claude-sonnet-4-20250514" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert "completion_tokens" in message.meta["usage"]

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async_with_streaming(self):
        """
        Test that the async run method of AnthropicChatGenerator works with streaming.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator()  # No streaming callback during initialization

        counter = 0
        responses = ""

        # Create a callback that's compatible with async operations
        async def callback(chunk: StreamingChunk) -> None:
            nonlocal counter
            nonlocal responses
            counter += 1
            responses += chunk.content if chunk.content else ""
            assert chunk.component_info is not None
            assert chunk.component_info.type.endswith("chat_generator.AnthropicChatGenerator")

        # Run the async streaming test
        results = await component.run_async(messages=initial_messages, streaming_callback=callback)

        # Verify the results
        assert len(results["replies"]) == 1
        message = results["replies"][0]
        assert "paris" in message.text.lower()
        assert "claude-sonnet-4-20250514" in message.meta["model"]
        assert message.meta["finish_reason"] == "stop"
        assert "input_tokens" in message.meta["usage"]
        assert "output_tokens" in message.meta["usage"]

        # Verify streaming behavior
        assert counter > 1  # Should have received multiple chunks
        assert "paris" in responses.lower()  # Should have received the response in chunks

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic API key to run this test.",
    )
    @pytest.mark.integration
    async def test_live_run_async_with_tools(self, tools):
        """
        Integration test that the async run method works with tools.
        """
        initial_messages = [ChatMessage.from_user("What's the weather like in Paris?")]
        component = AnthropicChatGenerator(tools=tools)
        results = await component.run_async(messages=initial_messages)

        assert len(results["replies"]) == 1
        message = results["replies"][0]

        assert message.tool_calls
        tool_call = message.tool_call
        assert isinstance(tool_call, ToolCall)
        assert tool_call.id is not None
        assert tool_call.tool_name == "weather"
        assert tool_call.arguments == {"city": "Paris"}
        assert message.meta["finish_reason"] == "tool_calls"

        new_messages = [
            *initial_messages,
            message,
            ChatMessage.from_tool(tool_result="22° C", origin=tool_call),
        ]
        results = await component.run_async(new_messages)
        assert len(results["replies"]) == 1
        final_message = results["replies"][0]
        assert not final_message.tool_calls
        assert len(final_message.text) > 0
        assert "paris" in final_message.text.lower()
        assert "completion_tokens" in final_message.meta["usage"]

    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY", None),
        reason="Export an env var called ANTHROPIC_API_KEY containing the Anthropic token to run this test.",
    )
    def test_live_run_multimodal(self, test_files_path):
        """Integration test for multimodal functionality with real API."""
        image_path = test_files_path / "apple.jpg"
        # Resize the image to keep this test fast
        image_content = ImageContent.from_file_path(file_path=image_path, size=(100, 100))
        messages = [ChatMessage.from_user(content_parts=["What does this image show? Max 5 words", image_content])]

        generator = AnthropicChatGenerator(generation_kwargs={"max_tokens": 20})
        response = generator.run(messages=messages)

        assert "replies" in response
        assert isinstance(response["replies"], list)
        assert len(response["replies"]) > 0
        message = response["replies"][0]
        assert message.text
        assert len(message.text) > 0
        assert any(word in message.text.lower() for word in ["apple", "fruit", "red"])
