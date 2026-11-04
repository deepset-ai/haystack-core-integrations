# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64

import pytest
from google.genai import interactions as gi
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    FileContent,
    ImageContent,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import Tool
from pydantic import BaseModel

from haystack_integrations.components.generators.google_genai.chat.interactions_utils import (
    _aggregate_streaming_chunks,
    _convert_interaction_event_to_streaming_chunk,
    _convert_interaction_to_chat_message,
    _convert_message_to_steps,
    _convert_messages_to_interaction_input,
    _convert_steps_to_chat_message,
    _convert_tools_to_interactions_format,
    _convert_usage,
    _map_status_to_finish_reason,
    _process_response_format,
)

MODEL = "gemini-3.8-flash"
# Older SDK versions require the timestamps on every Interaction
TIMESTAMPS = {"created": "2026-11-03T10:00:00Z", "updated": "2026-11-03T10:00:01Z"}
USAGE = {
    "total_input_tokens": 12,
    "total_output_tokens": 8,
    "total_thought_tokens": 5,
    "total_cached_tokens": 2,
    "total_tokens": 25,
}
EXPECTED_USAGE = {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 25,
    "thoughts_token_count": 5,
    "cached_content_token_count": 2,
}
THOUGHT_STEP = {"type": "thought", "signature": "c2lnLTE=", "summary": [{"type": "text", "text": "Check the weather."}]}
PNG_BASE64 = base64.b64encode(b"fake-png").decode()
DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def _interaction(steps, status="completed", usage=USAGE):
    return gi.Interaction.model_validate(
        {"id": "int_1", "status": status, "model": MODEL, "steps": steps, "usage": usage, **TIMESTAMPS}
    )


def _interaction_event(event_type, status, usage=None):
    interaction = {"id": "int_1", "status": status, "steps": [], **TIMESTAMPS}
    if usage:
        interaction["usage"] = usage
    event_class = gi.InteractionCreatedEvent if event_type == "interaction.created" else gi.InteractionCompletedEvent
    return event_class.model_validate({"event_type": event_type, "interaction": interaction})


def _step_start(index, step):
    return gi.StepStart.model_validate({"event_type": "step.start", "index": index, "step": step})


def _step_delta(index, delta):
    return gi.StepDelta.model_validate({"event_type": "step.delta", "index": index, "delta": delta})


def _step_stop(index):
    return gi.StepStop.model_validate({"event_type": "step.stop", "index": index})


@pytest.fixture
def component_info():
    return ComponentInfo(type="GoogleGenAIInteractionsChatGenerator")


class TestMessageConversion:
    def test_user_text(self):
        assert _convert_message_to_steps(ChatMessage.from_user("Hi")) == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]}
        ]

    def test_user_multimodal(self):
        message = ChatMessage.from_user(
            content_parts=[
                "Describe these",
                ImageContent(base64_image=PNG_BASE64, mime_type="image/png", validation=False),
                FileContent(base64_data=PNG_BASE64, mime_type="application/pdf", validation=False),
                FileContent(base64_data=PNG_BASE64, mime_type=DOCX_MIME_TYPE, validation=False),
                FileContent(base64_data=PNG_BASE64, mime_type="audio/mp3", validation=False),
                FileContent(base64_data=PNG_BASE64, mime_type="video/mp4", validation=False),
            ]
        )
        assert _convert_message_to_steps(message) == [
            {
                "type": "user_input",
                "content": [
                    {"type": "text", "text": "Describe these"},
                    {"type": "image", "data": PNG_BASE64, "mime_type": "image/png"},
                    {"type": "document", "data": PNG_BASE64, "mime_type": "application/pdf"},
                    {"type": "document", "data": PNG_BASE64, "mime_type": DOCX_MIME_TYPE},
                    {"type": "audio", "data": PNG_BASE64, "mime_type": "audio/mp3"},
                    {"type": "video", "data": PNG_BASE64, "mime_type": "video/mp4"},
                ],
            }
        ]

    def test_user_file_without_mime_type_raises(self):
        file = FileContent(base64_data=PNG_BASE64, validation=False)
        with pytest.raises(ValueError, match="MIME type is required"):
            _convert_message_to_steps(ChatMessage.from_user(content_parts=[file]))

    def test_user_image_without_mime_type_raises(self):
        image = ImageContent(base64_image=PNG_BASE64, validation=False)
        with pytest.raises(ValueError, match="MIME type is required"):
            _convert_message_to_steps(ChatMessage.from_user(content_parts=[image]))

    def test_empty_user_message_raises(self):
        with pytest.raises(ValueError, match="must contain at least one content part"):
            _convert_message_to_steps(ChatMessage.from_user(content_parts=[]))

    def test_assistant_with_thoughts_text_and_tool_calls(self):
        message = ChatMessage.from_assistant(
            text="Checking.",
            reasoning=ReasoningContent(reasoning_text="Check the weather.", extra={"thought_steps": [THOUGHT_STEP]}),
            tool_calls=[ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="fc_1")],
        )
        assert _convert_message_to_steps(message) == [
            THOUGHT_STEP,
            {"type": "model_output", "content": [{"type": "text", "text": "Checking."}]},
            {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}},
        ]

    def test_assistant_reasoning_from_other_provider_is_skipped(self):
        message = ChatMessage.from_assistant(text="Hi", reasoning=ReasoningContent(reasoning_text="other provider"))
        assert _convert_message_to_steps(message) == [
            {"type": "model_output", "content": [{"type": "text", "text": "Hi"}]}
        ]

    def test_assistant_tool_call_without_id_raises(self):
        message = ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="weather", arguments={})])
        with pytest.raises(ValueError, match="requires an ID on every ToolCall"):
            _convert_message_to_steps(message)

    def test_tool_results(self):
        origin = ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="fc_1")
        text_result = ChatMessage.from_tool(tool_result="22°C", origin=origin, error=True)
        multimodal_result = ChatMessage.from_tool(
            tool_result=[TextContent("A chart"), ImageContent(base64_image=PNG_BASE64, mime_type="image/png")],
            origin=origin,
        )
        assert _convert_message_to_steps(text_result) == [
            {"type": "function_result", "call_id": "fc_1", "name": "weather", "result": "22°C", "is_error": True}
        ]
        assert _convert_message_to_steps(multimodal_result) == [
            {
                "type": "function_result",
                "call_id": "fc_1",
                "name": "weather",
                "result": [
                    {"type": "text", "text": "A chart"},
                    {"type": "image", "data": PNG_BASE64, "mime_type": "image/png"},
                ],
                "is_error": False,
            }
        ]

    def test_tool_result_with_unsupported_content_raises(self):
        origin = ToolCall(tool_name="reader", arguments={}, id="fc_1")
        file = FileContent(base64_data=PNG_BASE64, mime_type="application/pdf", validation=False)
        with pytest.raises(ValueError, match="Only TextContent and ImageContent"):
            _convert_message_to_steps(ChatMessage.from_tool(tool_result=[file], origin=origin))

    def test_tool_result_without_call_id_raises(self):
        message = ChatMessage.from_tool(tool_result="ok", origin=ToolCall(tool_name="weather", arguments={}))
        with pytest.raises(ValueError, match="requires an ID on the ToolCall"):
            _convert_message_to_steps(message)

    def test_only_leading_system_message_becomes_system_instruction(self):
        messages = [
            ChatMessage.from_system("Be brief."),
            ChatMessage.from_user("Hi"),
            ChatMessage.from_system("Answer in French."),
        ]
        system_instruction, steps = _convert_messages_to_interaction_input(messages)
        assert system_instruction == "Be brief."
        assert steps == [
            {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "Answer in French."}]},
        ]

    def test_steps_round_trip_through_chat_message(self):
        steps = [
            THOUGHT_STEP,
            {"type": "model_output", "content": [{"type": "text", "text": "Checking."}]},
            {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}},
            {"type": "function_call", "id": "fc_2", "name": "weather", "arguments": {"city": "Rome"}},
        ]
        message = _convert_steps_to_chat_message(
            steps, model=MODEL, interaction_id="int_1", status="requires_action", usage={}
        )
        assert _convert_message_to_steps(message) == steps


def test_convert_tools_sanitizes_schema():
    parameters = {
        "type": "object",
        "$defs": {"Unit": {"type": "string", "enum": ["C", "F"]}},
        "properties": {"city": {"type": "string"}, "unit": {"$ref": "#/$defs/Unit"}},
        "additionalProperties": False,
    }
    tool = Tool(name="weather", description="Get the weather", parameters=parameters, function=lambda city: city)
    assert _convert_tools_to_interactions_format([tool]) == [
        {
            "type": "function",
            "name": "weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["C", "F"]}},
            },
        }
    ]


class City(BaseModel):
    name: str


class TestResponseFormat:
    @pytest.mark.parametrize(
        ("response_format", "expected_schema"),
        [(City, City.model_json_schema()), ({"type": "object"}, {"type": "object"})],
    )
    def test_converts_pydantic_model_and_json_schema(self, response_format, expected_schema):
        kwargs = {"response_format": response_format, "store": False}
        result = _process_response_format(kwargs)
        assert result == {
            "response_format": {"type": "text", "mime_type": "application/json", "schema": expected_schema},
            "store": False,
        }
        assert kwargs["response_format"] is response_format

    @pytest.mark.parametrize(
        "response_format",
        [None, {"type": "text", "mime_type": "application/json"}, [{"type": "text", "mime_type": "text/plain"}]],
    )
    def test_leaves_native_formats_unchanged(self, response_format):
        assert _process_response_format({"response_format": response_format}) == {"response_format": response_format}

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Unsupported response_format type"):
            _process_response_format({"response_format": "json"})


def test_convert_usage():
    assert _convert_usage(gi.Usage.model_validate(USAGE)) == EXPECTED_USAGE
    assert _convert_usage(None) == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


@pytest.mark.parametrize(
    ("status", "has_tool_calls", "expected"),
    [
        ("completed", False, "stop"),
        ("completed", True, "tool_calls"),
        ("requires_action", True, "tool_calls"),
        ("incomplete", False, "length"),
        ("budget_exceeded", False, "length"),
        ("cancelled", False, None),
        (None, False, None),
    ],
)
def test_map_status_to_finish_reason(status, has_tool_calls, expected):
    assert _map_status_to_finish_reason(status, has_tool_calls=has_tool_calls) == expected


class TestInteractionToChatMessage:
    def test_text_and_thought(self):
        interaction = _interaction(
            [THOUGHT_STEP, {"type": "model_output", "content": [{"type": "text", "text": "Sunny."}]}]
        )
        message = _convert_interaction_to_chat_message(interaction, "fallback-model")
        assert message.text == "Sunny."
        assert message.reasoning == ReasoningContent(
            reasoning_text="Check the weather.", extra={"thought_steps": [THOUGHT_STEP]}
        )
        assert message.tool_calls == []
        assert message.meta == {
            "model": MODEL,
            "interaction_id": "int_1",
            "status": "completed",
            "finish_reason": "stop",
            "usage": EXPECTED_USAGE,
        }

    def test_parallel_function_calls(self):
        interaction = _interaction(
            [
                THOUGHT_STEP,
                {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}},
                {"type": "function_call", "id": "fc_2", "name": "weather", "arguments": {"city": "Rome"}},
            ],
            status="requires_action",
        )
        message = _convert_interaction_to_chat_message(interaction, MODEL)
        assert message.text is None
        assert message.tool_calls == [
            ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="fc_1"),
            ToolCall(tool_name="weather", arguments={"city": "Rome"}, id="fc_2"),
        ]
        assert message.meta["finish_reason"] == "tool_calls"

    def test_multiple_text_items_and_unsupported_steps(self):
        interaction = _interaction(
            [
                {"type": "google_search_call", "id": "gs_1", "arguments": {"queries": ["weather"]}},
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Sunny "}, {"type": "text", "text": "today."}],
                },
            ]
        )
        message = _convert_interaction_to_chat_message(interaction, MODEL)
        assert message.texts == ["Sunny today."]
        assert message.reasoning is None

    def test_failed_interaction_raises(self):
        with pytest.raises(RuntimeError, match="int_1 failed"):
            _convert_interaction_to_chat_message(_interaction([], status="failed"), MODEL)


class TestStreaming:
    def test_text_stream(self, component_info):
        events = [
            _interaction_event("interaction.created", "in_progress"),
            _step_start(0, {"type": "model_output"}),
            _step_delta(0, {"type": "text", "text": "Hello"}),
            _step_delta(0, {"type": "text", "text": " world"}),
            _step_stop(0),
            _interaction_event("interaction.completed", "completed", usage=USAGE),
        ]
        chunks = [_convert_interaction_event_to_streaming_chunk(e, component_info, MODEL) for e in events]

        assert [(c.content, c.index, c.start, c.finish_reason) for c in chunks] == [
            ("", None, False, None),
            ("", 0, True, None),
            ("Hello", 0, False, None),
            (" world", 0, False, None),
            ("", None, False, None),
            ("", None, False, "stop"),
        ]
        assert all(c.component_info == component_info and c.meta["model"] == MODEL for c in chunks)
        assert chunks[0].meta["interaction_id"] == "int_1"
        assert chunks[0].meta["status"] == "in_progress"
        assert chunks[-1].meta["usage"] == EXPECTED_USAGE

        message = _aggregate_streaming_chunks(chunks, MODEL)
        assert message == ChatMessage.from_assistant(
            text="Hello world",
            meta={
                "model": MODEL,
                "interaction_id": "int_1",
                "status": "completed",
                "finish_reason": "stop",
                "usage": EXPECTED_USAGE,
            },
        )

    def test_thinking_and_parallel_tool_calls_stream(self, component_info):
        events = [
            _interaction_event("interaction.created", "in_progress"),
            _step_start(0, {"type": "thought"}),
            _step_delta(0, {"type": "thought_summary", "content": {"type": "text", "text": "Check the "}}),
            _step_delta(0, {"type": "thought_summary", "content": {"type": "text", "text": "weather."}}),
            _step_delta(0, {"type": "thought_signature", "signature": "c2lnLTE="}),
            _step_stop(0),
            _step_start(1, {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {}}),
            _step_delta(1, {"type": "arguments_delta", "arguments": '{"city": '}),
            _step_delta(1, {"type": "arguments_delta", "arguments": '"Paris"}'}),
            _step_stop(1),
            _step_start(2, {"type": "function_call", "id": "fc_2", "name": "weather", "arguments": {}}),
            _step_delta(2, {"type": "arguments_delta", "arguments": '{"city": "Rome"}'}),
            _step_stop(2),
            _interaction_event("interaction.completed", "requires_action", usage=USAGE),
        ]
        chunks = [_convert_interaction_event_to_streaming_chunk(e, component_info, MODEL) for e in events]

        assert [c.reasoning for c in chunks[1:5]] == [
            ReasoningContent(reasoning_text=""),
            ReasoningContent(reasoning_text="Check the "),
            ReasoningContent(reasoning_text="weather."),
            ReasoningContent(reasoning_text="", extra={"signature": "c2lnLTE="}),
        ]
        assert [c.tool_calls for c in chunks[6:9]] == [
            [ToolCallDelta(index=1, id="fc_1", tool_name="weather")],
            [ToolCallDelta(index=1, arguments='{"city": ')],
            [ToolCallDelta(index=1, arguments='"Paris"}')],
        ]
        assert chunks[10].tool_calls == [ToolCallDelta(index=2, id="fc_2", tool_name="weather")]
        assert [c.index for c in chunks] == [None, 0, 0, 0, 0, None, 1, 1, 1, None, 2, 2, None, None]
        assert [c.start for c in chunks] == [False, True, *[False] * 4, True, *[False] * 3, True, *[False] * 3]
        assert chunks[-1].finish_reason == "tool_calls"

        message = _aggregate_streaming_chunks(chunks, MODEL)
        assert message.reasoning == ReasoningContent(
            reasoning_text="Check the weather.", extra={"thought_steps": [THOUGHT_STEP]}
        )
        assert message.tool_calls == [
            ToolCall(tool_name="weather", arguments={"city": "Paris"}, id="fc_1"),
            ToolCall(tool_name="weather", arguments={"city": "Rome"}, id="fc_2"),
        ]
        assert message.text is None
        assert message.meta["finish_reason"] == "tool_calls"
        assert message.meta["usage"] == EXPECTED_USAGE

    def test_streaming_and_non_streaming_produce_the_same_message(self, component_info):
        steps = [
            THOUGHT_STEP,
            {"type": "model_output", "content": [{"type": "text", "text": "Checking."}]},
            {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}},
        ]
        events = [
            _step_start(0, {"type": "thought", "signature": "c2lnLTE="}),
            _step_delta(0, {"type": "thought_summary", "content": {"type": "text", "text": "Check the weather."}}),
            _step_start(1, {"type": "model_output"}),
            _step_delta(1, {"type": "text", "text": "Checking."}),
            _step_start(2, {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {"city": "Paris"}}),
            _interaction_event("interaction.completed", "requires_action", usage=USAGE),
        ]
        chunks = [_convert_interaction_event_to_streaming_chunk(e, component_info, MODEL) for e in events]

        streamed = _aggregate_streaming_chunks(chunks, MODEL)
        assert streamed == _convert_interaction_to_chat_message(_interaction(steps, status="requires_action"), MODEL)

    def test_invalid_streamed_arguments_fall_back_to_empty_dict(self, component_info):
        events = [
            _step_start(0, {"type": "function_call", "id": "fc_1", "name": "weather", "arguments": {}}),
            _step_delta(0, {"type": "arguments_delta", "arguments": '{"city": '}),
        ]
        chunks = [_convert_interaction_event_to_streaming_chunk(e, component_info, MODEL) for e in events]
        assert _aggregate_streaming_chunks(chunks, MODEL).tool_calls == [
            ToolCall(tool_name="weather", arguments={}, id="fc_1")
        ]

    def test_events_without_content_produce_meta_only_chunks(self, component_info):
        events = [
            gi.InteractionStatusUpdate.model_validate(
                {"event_type": "interaction.status_update", "interaction_id": "int_1", "status": "in_progress"}
            ),
            _step_delta(0, {"type": "text_annotation_delta", "annotations": []}),
            _step_delta(0, {"type": "thought_summary", "content": {"type": "image", "mime_type": "image/png"}}),
        ]
        chunks = [_convert_interaction_event_to_streaming_chunk(e, component_info, MODEL) for e in events]
        assert all(c.content == "" and not c.tool_calls and c.reasoning is None for c in chunks)
        assert chunks[0].meta["status"] == "in_progress"

    def test_error_event_raises(self, component_info):
        event = gi.ErrorEvent.model_validate({"event_type": "error", "error": {"code": "500", "message": "boom"}})
        with pytest.raises(RuntimeError, match="boom"):
            _convert_interaction_event_to_streaming_chunk(event, component_info, MODEL)
