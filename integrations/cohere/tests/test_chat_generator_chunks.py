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
from haystack import dataclasses
from haystack.dataclasses import ChatMessage, ComponentInfo, StreamingChunk, ToolCallDelta

from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _convert_cohere_chunk_to_streaming_chunk,
    _parse_streaming_response,
)


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
        StreamingChunk(content="I", meta=common_meta, index=0, start=True),
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
        StreamingChunk(content="", meta=common_meta, index=1),
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
        StreamingChunk(content="", meta=common_meta, index=2),
        StreamingChunk(
            content="",
            meta={
                "model": "command-a-03-2025",
                "finish_reason": "TOOL_CALL",
                "usage": {"prompt_tokens": 60.0, "completion_tokens": 41.0},
            },
            index=2,
            finish_reason="tool_calls",
        ),
    ]


class TestCohereChunkConversion:
    def test_convert_cohere_chunk_to_streaming_chunk_complete_sequence(self, cohere_chunks, expected_streaming_chunks):
        # This test has to follow the _parse_streaming_response logic to make sure the chunks conversion is correct
        # _parse_streaming_response isn't used directly b/c it returns a final ChatMessage
        original_chunks = []
        global_index = 0
        for cohere_chunk, haystack_chunk in zip(cohere_chunks, expected_streaming_chunks, strict=True):
            if cohere_chunk.type in ["tool-call-start", "content-start", "citation-start"]:
                global_index += 1
            stream_chunk = _convert_cohere_chunk_to_streaming_chunk(
                chunk=cohere_chunk,
                model="command-a-03-2025",
                global_index=global_index,
                previous_original_chunks=original_chunks,
            )
            original_chunks.append(cohere_chunk)
            assert stream_chunk == haystack_chunk

    def test_parse_response(self, cohere_chunks):
        cohere_stream = (chunk for chunk in cohere_chunks)
        chat_message = _parse_streaming_response(
            cohere_stream,
            model="command-a-03-2025",
            streaming_callback=lambda _: None,
            component_info=ComponentInfo(type="type"),
        )
        assert chat_message == ChatMessage.from_assistant(
            text="I will use the calculator tool to compute the results of 7 + 2 and 2 * 2.",
            tool_calls=[
                dataclasses.ToolCall(
                    tool_name="calculator", id="calculator_mcdnh7tnn9v9", arguments={"expression": "7 + 2"}
                ),
                dataclasses.ToolCall(
                    tool_name="calculator", id="calculator_yk0yf8f7fzbe", arguments={"expression": "2 * 2"}
                ),
            ],
            meta={
                "model": "command-a-03-2025",
                "completion_start_time": None,
                "finish_reason": "tool_calls",
                "index": 0,
                "usage": {"prompt_tokens": 60.0, "completion_tokens": 41.0},
            },
        )

    def test_convert_message_end_chunk(self):
        mock_chunk = MsgEndStream(
            type="message-end",
            delta=MsgEndDelta(
                finish_reason="COMPLETE",
                usage=Usage(
                    billed_units=BilledUnits(input_tokens=9.0, output_tokens=75.0),
                    tokens=UsageTokens(input_tokens=150.0, output_tokens=50.0),
                ),
            ),
        )

        result = _convert_cohere_chunk_to_streaming_chunk(chunk=mock_chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.index == 0
        assert result.start is False
        assert result.finish_reason == "stop"  # Mapped from "COMPLETE"
        assert result.tool_calls is None
        assert result.meta["finish_reason"] == "COMPLETE"
        assert result.meta["usage"] == {"prompt_tokens": 9, "completion_tokens": 75}

    def test_convert_unknown_chunk_type(self):
        mock_chunk = MagicMock()
        mock_chunk.type = "unknown-chunk-type"
        result = _convert_cohere_chunk_to_streaming_chunk(chunk=mock_chunk, model="command-r-08-2024")
        assert result.content == ""
        assert result.start is False
        assert result.tool_calls is None
        assert result.finish_reason is None

    def test_finish_reason_mapping(self):
        finish_reasons = [("COMPLETE", "stop"), ("MAX_TOKENS", "length"), ("TOOL_CALL", "tool_calls")]

        for cohere_reason, haystack_reason in finish_reasons:
            mock_chunk = MsgEndStream(type="message-end", delta=MsgEndDelta(finish_reason=cohere_reason, usage=None))
            result = _convert_cohere_chunk_to_streaming_chunk(chunk=mock_chunk, model="command-r-08-2024")
            assert result.finish_reason == haystack_reason
            assert result.meta["finish_reason"] == cohere_reason

    def test_usage_extraction_other_cases(self):
        # missing usage data
        mock_chunk = MsgEndStream(type="message-end", delta=MsgEndDelta(finish_reason="COMPLETE", usage=None))
        result = _convert_cohere_chunk_to_streaming_chunk(chunk=mock_chunk, model="command-r-08-2024")

        assert result.meta["usage"] == {"completion_tokens": 0.0, "prompt_tokens": 0.0}
