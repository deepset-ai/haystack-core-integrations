# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY

import pytest
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import Tool
from huggingface_hub import (
    ChatCompletionOutputFunctionDefinition,
    ChatCompletionOutputToolCall,
    ChatCompletionStreamOutput,
)

from haystack_integrations.components.generators.huggingface_api.chat.chat_generator import (
    _convert_chat_completion_stream_output_to_streaming_chunk,
    _convert_hfapi_tool_calls,
    _convert_message_to_hf_format_with_json_tool_arguments,
    _convert_tools_to_hfapi_tools,
    _resolve_schema_refs,
)


class TestConvertChatCompletionStreamOutputToStreamingChunk:
    @pytest.fixture
    def hf_stream_chunk(self):
        def create(delta=None, finish_reason=None, usage=None):
            return ChatCompletionStreamOutput.parse_obj_as_instance(
                {
                    "id": "completion_id",
                    "created": 1710498504,
                    "model": "some_model",
                    "system_fingerprint": "some_fingerprint",
                    "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    "usage": usage,
                }
            )

        return create

    def test_tool_call_without_index_raises(self, hf_stream_chunk):
        raw = hf_stream_chunk({"tool_calls": [{"id": "call_0", "function": {"name": "weather", "arguments": "{}"}}]})
        with pytest.raises(ValueError, match="without an index"):
            _convert_chat_completion_stream_output_to_streaming_chunk(raw)

    @pytest.mark.parametrize("reason", ["stop", "eos_token", "stop_sequence"])
    def test_stop_aliases_map_to_stop(self, hf_stream_chunk, reason):
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(hf_stream_chunk({}, finish_reason=reason))
        assert chunk.finish_reason == "stop"
        assert chunk.meta["finish_reason"] == reason

    @pytest.mark.parametrize("reason", ["stop", "eos_token", "stop_sequence"])
    def test_stop_after_tool_call_maps_to_tool_calls(self, hf_stream_chunk, reason):
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(
            hf_stream_chunk({}, finish_reason=reason), tool_calls_seen=True
        )
        assert chunk.finish_reason == "tool_calls"
        assert chunk.meta["finish_reason"] == reason

    @pytest.mark.parametrize("reason", ["length", "content_filter"])
    def test_length_and_content_filter_are_not_replaced_by_tool_calls(self, hf_stream_chunk, reason):
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(
            hf_stream_chunk({}, finish_reason=reason), tool_calls_seen=True
        )
        assert chunk.finish_reason == reason

    def test_initial_whitespace_has_no_chunk_index(self, hf_stream_chunk):
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(hf_stream_chunk({"content": " "}))
        assert chunk.content == " "
        assert chunk.index is None
        assert chunk.start is False

    @pytest.mark.parametrize("text", [" ", " world"])
    def test_more_text_keeps_same_chunk_index(self, hf_stream_chunk, text):
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(
            hf_stream_chunk({"content": text}), last_content_chunk=StreamingChunk(content="Hello", index=2)
        )
        assert chunk.content == text
        assert chunk.index == 2
        assert chunk.start is False

    def test_converts_text_chunk(self, hf_stream_chunk):
        raw = hf_stream_chunk({"role": "assistant", "content": " Paris"})
        chunk = _convert_chat_completion_stream_output_to_streaming_chunk(raw)
        assert chunk == StreamingChunk(
            content=" Paris",
            index=0,
            start=True,
            meta={"model": "some_model", "received_at": ANY, "finish_reason": None},
        )
        assert chunk.reasoning is None


class TestConvertHfapiToolCalls:
    @pytest.mark.parametrize("tool_calls", [None, []], ids=["none", "empty_list"])
    def test_returns_empty_list_when_no_tool_calls(self, tool_calls):
        assert _convert_hfapi_tool_calls(tool_calls) == []

    @pytest.mark.parametrize("arguments", [{"city": "Paris"}, '{"city": "Paris"}'], ids=["dict", "json"])
    def test_tool_call_arguments_as_dict_or_json(self, arguments):
        raw = [
            ChatCompletionOutputToolCall(
                id="paris",
                type="function",
                function=ChatCompletionOutputFunctionDefinition(arguments=arguments, name="weather", description=None),
            )
        ]
        assert _convert_hfapi_tool_calls(raw) == [
            ToolCall(id="paris", tool_name="weather", arguments={"city": "Paris"}),
        ]

    @pytest.mark.parametrize(
        "arguments, warning",
        [
            pytest.param("invalid JSON", "malformed JSON", id="invalid_json"),
            pytest.param(["invalid", "shape"], "Valid types are dict and str", id="unsupported_type"),
        ],
    )
    def test_invalid_tool_call_is_skipped_with_warning(self, arguments, warning, caplog):
        raw = [
            ChatCompletionOutputToolCall(
                id="invalid",
                type="function",
                function=ChatCompletionOutputFunctionDefinition(arguments=arguments, name="weather", description=None),
            ),
            ChatCompletionOutputToolCall(
                id="paris",
                type="function",
                function=ChatCompletionOutputFunctionDefinition(
                    arguments={"city": "Paris"},
                    name="weather",
                    description=None,
                ),
            ),
        ]
        assert _convert_hfapi_tool_calls(raw) == [
            ToolCall(id="paris", tool_name="weather", arguments={"city": "Paris"}),
        ]
        assert warning in caplog.text


class TestConvertMessageToHfFormatWithJsonToolArguments:
    def test_tool_arguments_are_json_encoded_without_changing_original_message(self):
        calls = [
            ToolCall(id="paris", tool_name="weather", arguments={"city": "Paris"}),
            ToolCall(id="clock", tool_name="get_time", arguments={}),
        ]
        message = ChatMessage.from_assistant(tool_calls=calls)
        converted = _convert_message_to_hf_format_with_json_tool_arguments(message)
        assert converted["tool_calls"] == [
            {"id": "paris", "type": "function", "function": {"name": "weather", "arguments": '{"city": "Paris"}'}},
            {"id": "clock", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
        ]
        assert message.tool_calls[0].arguments == {"city": "Paris"}
        assert message.tool_calls[1].arguments == {}


class TestConvertToolsToHfapiTools:
    @pytest.mark.parametrize("tools", [None, []], ids=["none", "empty_list"])
    def test_returns_none_when_no_tools(self, tools):
        assert _convert_tools_to_hfapi_tools(tools) is None

    def test_preserves_tool_name_description_and_parameters(self):
        tool = Tool(
            name="weather",
            description="useful to determine the weather in a given location",
            parameters={"city": {"type": "string"}},
            function=lambda city: city,
        )
        hf_tools = _convert_tools_to_hfapi_tools([tool])
        assert len(hf_tools) == 1
        assert hf_tools[0].type == "function"
        assert hf_tools[0].function.name == "weather"
        assert hf_tools[0].function.description == "useful to determine the weather in a given location"
        assert hf_tools[0].function.parameters == {"city": {"type": "string"}}

    def test_resolves_refs_in_tool_parameters(self):
        """Tool schemas with $defs are resolved before passing to HF API."""
        tool = Tool(
            name="get_user",
            description="Get user info",
            parameters={
                "$defs": {"User": {"type": "object", "properties": {"name": {"type": "string"}}}},
                "type": "object",
                "properties": {"user": {"$ref": "#/$defs/User"}},
            },
            function=lambda user: user,
        )
        hf_tools = _convert_tools_to_hfapi_tools([tool])
        assert hf_tools is not None
        assert len(hf_tools) == 1
        params = hf_tools[0].function.parameters
        assert "$defs" not in params
        assert params["properties"]["user"]["type"] == "object"


class TestResolveSchemaRefs:
    def test_preserves_schema_without_defs(self):
        """Schema without $defs is returned as-is."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assert _resolve_schema_refs(schema) == schema

    def test_inlines_references_and_removes_defs(self):
        """Schema with $defs and $ref is expanded correctly."""
        schema = {
            "$defs": {
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    "required": ["name"],
                }
            },
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
            "required": ["user"],
        }
        resolved = _resolve_schema_refs(schema)
        assert "$defs" not in resolved
        assert "$ref" not in resolved["properties"]["user"]
        assert resolved["properties"]["user"]["type"] == "object"
        assert resolved["properties"]["user"]["properties"]["name"] == {"type": "string"}

    def test_resolves_nested_references(self):
        """Schema with nested $ref references is expanded correctly."""
        schema = {
            "$defs": {
                "Address": {"type": "object", "properties": {"street": {"type": "string"}}},
                "User": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}, "address": {"$ref": "#/$defs/Address"}},
                },
            },
            "type": "object",
            "properties": {"user": {"$ref": "#/$defs/User"}},
        }
        resolved = _resolve_schema_refs(schema)
        assert "$defs" not in resolved
        user = resolved["properties"]["user"]
        assert user["properties"]["address"]["type"] == "object"
        assert user["properties"]["address"]["properties"]["street"] == {"type": "string"}
