from unittest.mock import MagicMock

import pytest
from cohere import (
    ChatMessageEndEventDelta,
    ChatMessageStartEventDelta,
    ChatMessageStartEventDeltaMessage,
    ChatToolCallDeltaEventDelta,
    ChatToolCallDeltaEventDeltaMessage,
    ChatToolCallDeltaEventDeltaMessageToolCalls,
    ChatToolCallDeltaEventDeltaMessageToolCallsFunction,
    ChatToolCallStartEventDelta,
    ChatToolCallStartEventDeltaMessage,
    ChatToolPlanDeltaEventDelta,
    ChatToolPlanDeltaEventDeltaMessage,
    MessageEndV2ChatStreamResponse,
    MessageStartV2ChatStreamResponse,
    ToolCallDeltaV2ChatStreamResponse,
    ToolCallEndV2ChatStreamResponse,
    ToolCallStartV2ChatStreamResponse,
    ToolCallV2,
    ToolCallV2Function,
    ToolPlanDeltaV2ChatStreamResponse,
    Usage,
    UsageBilledUnits,
    UsageTokens,
)
from haystack.dataclasses import ComponentInfo, StreamingChunk, ToolCallDelta

from haystack_integrations.components.generators.cohere.chat.chat_generator import (
    _convert_cohere_chunk_to_streaming_chunk,
    _parse_streaming_response,
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
        MessageStartV2ChatStreamResponse(
            type="message-start",
            id="05509383-5673-47b0-af67-e34c4cf64a5f",
            delta=ChatMessageStartEventDelta(
                message=ChatMessageStartEventDeltaMessage(
                    role="assistant", content=[], tool_plan="", tool_calls=[], citations=[]
                )
            ),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan="I")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" will")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" use")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" the")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" calculator")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" tool")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" to")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" compute")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" the")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" results")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" of")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" 7")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" +")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" 2")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" and")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" 2")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" *")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=" 2")),
        ),
        ToolPlanDeltaV2ChatStreamResponse(
            type="tool-plan-delta",
            delta=ChatToolPlanDeltaEventDelta(message=ChatToolPlanDeltaEventDeltaMessage(tool_plan=".")),
        ),
        ToolCallStartV2ChatStreamResponse(
            type="tool-call-start",
            index=0,
            delta=ChatToolCallStartEventDelta(
                message=ChatToolCallStartEventDeltaMessage(
                    tool_calls=ToolCallV2(
                        id="calculator_mcdnh7tnn9v9",
                        type="function",
                        function=ToolCallV2Function(name="calculator", arguments=""),
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='{"')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="expression")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='":')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=' "')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="7")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=" +")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=" ")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="2")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=0,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='"}')
                    )
                )
            ),
        ),
        ToolCallEndV2ChatStreamResponse(type="tool-call-end", index=0),
        ToolCallStartV2ChatStreamResponse(
            type="tool-call-start",
            index=1,
            delta=ChatToolCallStartEventDelta(
                message=ChatToolCallStartEventDeltaMessage(
                    tool_calls=ToolCallV2(
                        id="calculator_yk0yf8f7fzbe",
                        type="function",
                        function=ToolCallV2Function(name="calculator", arguments=""),
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='{"')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="expression")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='":')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=' "')
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="2")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=" *")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments=" ")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments="2")
                    )
                )
            ),
        ),
        ToolCallDeltaV2ChatStreamResponse(
            type="tool-call-delta",
            index=1,
            delta=ChatToolCallDeltaEventDelta(
                message=ChatToolCallDeltaEventDeltaMessage(
                    tool_calls=ChatToolCallDeltaEventDeltaMessageToolCalls(
                        function=ChatToolCallDeltaEventDeltaMessageToolCallsFunction(arguments='"}')
                    )
                )
            ),
        ),
        ToolCallEndV2ChatStreamResponse(type="tool-call-end", index=1),
        MessageEndV2ChatStreamResponse(
            type="message-end",
            id=None,
            delta=ChatMessageEndEventDelta(
                error=None,
                finish_reason="TOOL_CALL",
                usage=Usage(
                    billed_units=UsageBilledUnits(
                        input_tokens=60.0, output_tokens=41.0, search_units=None, classifications=None
                    ),
                    tokens=UsageTokens(input_tokens=1462.0, output_tokens=92.0),
                    cached_tokens=0.0,
                ),
            ),
        ),
    ]


@pytest.fixture
def expected_chunks():
    return [
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="I",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" will",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" use",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" the",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" calculator",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" tool",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" to",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" compute",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" the",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" results",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" of",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" 7",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" +",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" 2",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" and",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" 2",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" *",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=" 2",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content=".",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=0,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025", "tool_call_id": "calculator_mcdnh7tnn9v9"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[
                ToolCallDelta(index=1, tool_name="calculator", arguments=None, id="calculator_mcdnh7tnn9v9", extra=None)
            ],
            tool_call_result=None,
            start=True,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments='{"', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments="expression", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments='":', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments=' "', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments="7", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments=" +", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments=" ", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments="2", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=[ToolCallDelta(index=1, tool_name=None, arguments='"}', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=1,
            tool_calls=None,
            tool_call_result=None,
            start=True,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025", "tool_call_id": "calculator_yk0yf8f7fzbe"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[
                ToolCallDelta(index=2, tool_name="calculator", arguments=None, id="calculator_yk0yf8f7fzbe", extra=None)
            ],
            tool_call_result=None,
            start=True,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments='{"', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments="expression", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments='":', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments=' "', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments="2", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments=" *", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments=" ", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments="2", id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=[ToolCallDelta(index=2, tool_name=None, arguments='"}', id=None, extra=None)],
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={"model": "command-a-03-2025"},
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=None,
            tool_call_result=None,
            start=True,
            finish_reason=None,
            reasoning=None,
        ),
        StreamingChunk(
            content="",
            meta={
                "model": "command-a-03-2025",
                "finish_reason": "TOOL_CALL",
                "usage": {"prompt_tokens": 60.0, "completion_tokens": 41.0},
            },
            component_info=ComponentInfo(
                type="haystack_integrations.components.generators.cohere.chat.chat_generator.CohereChatGenerator",
                name=None,
            ),
            index=2,
            tool_calls=None,
            tool_call_result=None,
            start=False,
            finish_reason=None,
            reasoning=None,
        ),
    ]


@pytest.fixture
def expected_streaming_chunks():
    """Fixture providing expected Haystack StreamingChunk objects for conversion testing."""

    return [
        # Chunk 1: Content delta
        StreamingChunk(content="I'll help you get the weather", index=0, meta={"model": "command-a-03-2025"}),
        # Chunk 2: Tool plan delta
        StreamingChunk(content="I need to call the weather tool", index=0, meta={"model": "command-a-03-2025"}),
        # Chunk 3: Tool call start
        StreamingChunk(
            content="",
            index=0,
            start=True,  # Tool call start
            tool_calls=[ToolCallDelta(index=0, id="call_weather_paris_123", tool_name="weather")],
            meta={"model": "command-a-03-2025", "tool_call_id": "call_weather_paris_123"},
        ),
        # Chunk 4: Tool call delta - arguments
        StreamingChunk(
            content="",
            index=0,
            # Name was set in start chunk
            tool_calls=[ToolCallDelta(index=0, arguments='{"ci')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 5: Tool call delta - more arguments
        StreamingChunk(
            content="",
            index=0,
            tool_calls=[ToolCallDelta(index=0, arguments='ty": "')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 6: Tool call delta - city name
        StreamingChunk(
            content="",
            index=0,
            tool_calls=[ToolCallDelta(index=0, arguments='Paris"')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 7: Tool call delta - closing brace
        StreamingChunk(
            content="", index=0, tool_calls=[ToolCallDelta(index=0, arguments="}")], meta={"model": "command-a-03-2025"}
        ),
        # Chunk 8: Tool call end
        StreamingChunk(content="", index=0, start=True, meta={"model": "command-a-03-2025"}),
        # Chunk 9: Tool call start - second tool
        # TODO The index in ToolCallDelta should be 1
        StreamingChunk(
            content="",
            index=0,
            start=True,  # Tool call start
            tool_calls=[ToolCallDelta(index=0, id="call_weather_berlin_456", tool_name="weather")],
            meta={"model": "command-a-03-2025", "tool_call_id": "call_weather_berlin_456"},
        ),
        # Chunk 10: Tool call delta - second tool arguments
        StreamingChunk(
            content="",
            index=0,
            tool_calls=[ToolCallDelta(index=0, arguments='{"ci')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 11: Tool call delta - more second tool arguments
        StreamingChunk(
            content="",
            index=0,
            tool_calls=[ToolCallDelta(index=0, arguments='ty": "')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 12: Tool call delta - second city name
        StreamingChunk(
            content="",
            index=0,
            tool_calls=[ToolCallDelta(index=0, arguments='Berlin"')],
            meta={"model": "command-a-03-2025"},
        ),
        # Chunk 13: Tool call delta - closing brace for second tool
        StreamingChunk(
            content="", index=0, tool_calls=[ToolCallDelta(index=0, arguments="}")], meta={"model": "command-a-03-2025"}
        ),
        # Chunk 14: Tool call end - second tool
        # TODO This shouldn't be returning start=True
        StreamingChunk(content="", index=0, start=True, meta={"model": "command-a-03-2025"}),
        # Chunk 15: Message end with finish reason and usage
        StreamingChunk(
            content="",
            index=0,
            finish_reason="tool_calls",
            meta={
                "model": "command-a-03-2025",
                "finish_reason": "TOOL_CALLS",
                "usage": {"prompt_tokens": 9, "completion_tokens": 75},
            },
        ),
    ]


class TestCohereChunkConversion:
    def test_convert_cohere_chunk_to_streaming_chunk_complete_sequence(self, cohere_chunks, expected_streaming_chunks):
        # TODO: the indexes are not correctly checked because of missing global_index
        # all the streaming chunks have index 0, but the expected indexes increase with tool calls.

        for cohere_chunk, haystack_chunk in zip(cohere_chunks, expected_streaming_chunks, strict=True):
            stream_chunk = _convert_cohere_chunk_to_streaming_chunk(chunk=cohere_chunk, model="command-a-03-2025")
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
        chunk = create_mock_cohere_chunk("tool-call-start", tool_call_id="call_123", tool_name="weather")

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
        assert result.meta["usage"] == {"prompt_tokens": 9, "completion_tokens": 75}

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
            response=mock_response, model="test-model", streaming_callback=callback, component_info=component_info
        )

        assert len(captured_chunks) == 1
        assert captured_chunks[0].component_info == component_info
        assert captured_chunks[0].content == "Hello"
        assert message.text == "Hello"
