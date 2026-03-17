from unittest.mock import MagicMock

import pytest
from cohere import ChatMessageEndEventDelta as MsgEndDelta
from cohere import ChatMessageStartEventDelta as MsgStartDelta
from cohere import ChatMessageStartEventDeltaMessage as MsgStartDeltaMsg
from cohere import ChatToolCallDeltaEventDelta as ToolCallDeltaCo
from cohere import ChatToolCallDeltaEventDeltaMessage as ToolCallDeltaMsg
from cohere import ChatToolCallDeltaEventDeltaMessageToolCalls as ToolCallDeltaCalls
from cohere import ChatToolCallDeltaEventDeltaMessageToolCallsFunction as ToolCallDeltaFn
from cohere import ChatToolCallStartEventDelta as ToolCallStartDelta
from cohere import ChatToolCallStartEventDeltaMessage as ToolCallStartMsg
from cohere import ChatToolPlanDeltaEventDelta as ToolPlanDelta
from cohere import ChatToolPlanDeltaEventDeltaMessage as ToolPlanDeltaMsg
from cohere import MessageEndV2ChatStreamResponse as MsgEndStream
from cohere import MessageStartV2ChatStreamResponse as MsgStartStream
from cohere import ToolCallDeltaV2ChatStreamResponse as ToolCallDeltaStream
from cohere import ToolCallEndV2ChatStreamResponse as ToolCallEndStream
from cohere import ToolCallStartV2ChatStreamResponse as ToolCallStartStream
from cohere import ToolCallV2 as ToolCall
from cohere import ToolCallV2Function as ToolCallFn
from cohere import ToolPlanDeltaV2ChatStreamResponse as ToolPlanDeltaStream
from cohere import Usage, UsageTokens
from cohere import UsageBilledUnits as BilledUnits
from haystack.dataclasses import ComponentInfo, StreamingChunk, ToolCallDelta

from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _convert_cohere_chunk_to_streaming_chunk,
)


def create_mock_cohere_chunk(chunk_type: str, index: int | None = None, **kwargs):
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
    return [
        MsgStartStream(
            type="message-start",
            id="05509383-5673-47b0-af67-e34c4cf64a5f",
            delta=MsgStartDelta(
                message=MsgStartDeltaMsg(role="assistant", content=[], tool_plan="", tool_calls=[], citations=[])
            ),
        ),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan="I"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" will"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" use"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" the"))),
        ToolPlanDeltaStream(
            type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" calculator"))
        ),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" tool"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" to"))),
        ToolPlanDeltaStream(
            type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" compute"))
        ),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" the"))),
        ToolPlanDeltaStream(
            type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" results"))
        ),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" of"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" 7"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" +"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" 2"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" and"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" 2"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" *"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan=" 2"))),
        ToolPlanDeltaStream(type="tool-plan-delta", delta=ToolPlanDelta(message=ToolPlanDeltaMsg(tool_plan="."))),
        ToolCallStartStream(
            type="tool-call-start",
            index=0,
            delta=ToolCallStartDelta(
                message=ToolCallStartMsg(
                    tool_calls=ToolCall(
                        id="calculator_mcdnh7tnn9v9",
                        type="function",
                        function=ToolCallFn(name="calculator", arguments=""),
                    )
                )
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='{"')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(
                    tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="expression"))
                )
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='":')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=' "')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="7")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=" +")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=" ")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="2")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=0,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='"}')))
            ),
        ),
        ToolCallEndStream(type="tool-call-end", index=0),
        ToolCallStartStream(
            type="tool-call-start",
            index=1,
            delta=ToolCallStartDelta(
                message=ToolCallStartMsg(
                    tool_calls=ToolCall(
                        id="calculator_yk0yf8f7fzbe",
                        type="function",
                        function=ToolCallFn(name="calculator", arguments=""),
                    )
                )
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='{"')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(
                    tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="expression"))
                )
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='":')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=' "')))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="2")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=" *")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments=" ")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments="2")))
            ),
        ),
        ToolCallDeltaStream(
            type="tool-call-delta",
            index=1,
            delta=ToolCallDeltaCo(
                message=ToolCallDeltaMsg(tool_calls=ToolCallDeltaCalls(function=ToolCallDeltaFn(arguments='"}')))
            ),
        ),
        ToolCallEndStream(type="tool-call-end", index=1),
        MsgEndStream(
            type="message-end",
            delta=MsgEndDelta(
                error=None,
                finish_reason="TOOL_CALL",
                usage=Usage(
                    billed_units=BilledUnits(
                        input_tokens=60.0, output_tokens=41.0, search_units=None, classifications=None
                    ),
                    tokens=UsageTokens(input_tokens=1462.0, output_tokens=92.0),
                    cached_tokens=0.0,
                ),
            ),
        ),
    ]


@pytest.fixture
def expected_streaming_chunks():
    common_meta = {"model": "command-a-03-2025"}
    return [
        StreamingChunk(content="", meta=common_meta, index=0),
        # TODO This one or the one above is missing the start=True
        StreamingChunk(content="I", meta=common_meta, index=0),
        StreamingChunk(content=" will", meta=common_meta, index=0),
        StreamingChunk(content=" use", meta=common_meta, index=0),
        StreamingChunk(content=" the", meta=common_meta, index=0),
        StreamingChunk(content=" calculator", meta=common_meta, index=0),
        StreamingChunk(content=" tool", meta=common_meta, index=0),
        StreamingChunk(content=" to", meta=common_meta, index=0),
        StreamingChunk(content=" compute", meta=common_meta, index=0),
        StreamingChunk(content=" the", meta=common_meta, index=0),
        StreamingChunk(content=" results", meta=common_meta, index=0),
        StreamingChunk(content=" of", meta=common_meta, index=0),
        StreamingChunk(content=" 7", meta=common_meta, index=0),
        StreamingChunk(content=" +", meta=common_meta, index=0),
        StreamingChunk(content=" 2", meta=common_meta, index=0),
        StreamingChunk(content=" and", meta=common_meta, index=0),
        StreamingChunk(content=" 2", meta=common_meta, index=0),
        StreamingChunk(content=" *", meta=common_meta, index=0),
        StreamingChunk(content=" 2", meta=common_meta, index=0),
        StreamingChunk(content=".", meta=common_meta, index=0),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025", "tool_call_id": "calculator_mcdnh7tnn9v9"},
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name="calculator", id="calculator_mcdnh7tnn9v9")],
            start=True,
        ),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments='{"')]),
        StreamingChunk(
            content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments="expression")]
        ),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments='":')]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments=' "')]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments="7")]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments=" +")]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments=" ")]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments="2")]),
        StreamingChunk(content="", meta=common_meta, index=1, tool_calls=[ToolCallDelta(index=1, arguments='"}')]),
        # TODO This chunk should not have a start=True
        StreamingChunk(content="", meta=common_meta, index=1, start=True),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025", "tool_call_id": "calculator_yk0yf8f7fzbe"},
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name="calculator", id="calculator_yk0yf8f7fzbe")],
            start=True,
        ),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments='{"')]),
        StreamingChunk(
            content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments="expression")]
        ),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments='":')]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments=' "')]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments="2")]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments=" *")]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments=" ")]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments="2")]),
        StreamingChunk(content="", meta=common_meta, index=2, tool_calls=[ToolCallDelta(index=2, arguments='"}')]),
        StreamingChunk(content="", meta=common_meta, index=2, start=True),
        StreamingChunk(
            content="",
            meta={
                "model": "command-a-03-2025",
                "finish_reason": "TOOL_CALL",
                "usage": {"prompt_tokens": 60.0, "completion_tokens": 41.0},
            },
            index=2,
        ),
    ]


class TestCohereChunkConversion:
    # TODO Probably better to replace with _parse_streaming_response
    def test_convert_cohere_chunk_to_streaming_chunk_complete_sequence(self, cohere_chunks, expected_streaming_chunks):
        global_index = 0
        for cohere_chunk, haystack_chunk in zip(cohere_chunks, expected_streaming_chunks, strict=True):
            if cohere_chunk.type in ["tool-call-start", "content-start", "citation-start"]:
                global_index += 1
            stream_chunk = _convert_cohere_chunk_to_streaming_chunk(
                chunk=cohere_chunk, model="command-a-03-2025", global_index=global_index
            )
            assert stream_chunk == haystack_chunk

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
        assert result.meta["usage"] == {"prompt_tokens": 9, "completion_tokens": 75}

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
        finish_reasons = [("COMPLETE", "stop"), ("MAX_TOKENS", "length"), ("TOOL_CALLS", "tool_calls")]

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
