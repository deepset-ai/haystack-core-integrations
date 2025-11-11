from typing import Any, Literal, Optional, Union, cast, get_args

from haystack.dataclasses.chat_message import (
    ChatMessage,
    ChatRole,
    ReasoningContent,
    TextContent,
    ToolCall,
    ToolCallResult,
)
from haystack.dataclasses.image_content import ImageContent
from haystack.dataclasses.streaming_chunk import (
    ComponentInfo,
    FinishReason,
    StreamingChunk,
    ToolCallDelta,
)

from anthropic.resources.messages.messages import RawMessageStreamEvent
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)

# See https://docs.anthropic.com/en/api/messages for supported formats
ImageFormat = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
IMAGE_SUPPORTED_FORMATS: list[ImageFormat] = list(get_args(ImageFormat))


# Mapping from Anthropic stop reasons to Haystack FinishReason values
FINISH_REASON_MAPPING: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
    "pause_turn": "stop",
    "tool_use": "tool_calls",
}


def _update_anthropic_message_with_tool_call_results(
    tool_call_results: list[ToolCallResult],
    content: list[
        Union[
            TextBlockParam,
            ToolUseBlockParam,
            ToolResultBlockParam,
            ImageBlockParam,
            ThinkingBlockParam,
            RedactedThinkingBlockParam,
        ]
    ],
) -> None:
    """
    Update an Anthropic message content list with tool call results.

    :param tool_call_results: The list of ToolCallResults to update the message with.
    :param content: The Anthropic message content list to update.
    """
    for tool_call_result in tool_call_results:
        if tool_call_result.origin.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
            raise ValueError(msg)

        tool_result_block = ToolResultBlockParam(
            type="tool_result",
            tool_use_id=tool_call_result.origin.id,
            content=[{"type": "text", "text": tool_call_result.result}],
            is_error=tool_call_result.error,
        )
        content.append(tool_result_block)


def _convert_tool_calls_to_anthropic_format(tool_calls: list[ToolCall]) -> list[ToolUseBlockParam]:
    """
    Convert a list of tool calls to the format expected by Anthropic Chat API.

    :param tool_calls: The list of ToolCalls to convert.
    :return: A list of ToolUseBlockParam objects in the format expected by Anthropic API.
    """
    anthropic_tool_calls = []
    for tc in tool_calls:
        if tc.id is None:
            msg = "`ToolCall` must have a non-null `id` attribute to be used with Anthropic."
            raise ValueError(msg)

        tool_use_block = ToolUseBlockParam(
            type="tool_use",
            id=tc.id,
            name=tc.tool_name,
            input=tc.arguments,
        )
        anthropic_tool_calls.append(tool_use_block)
    return anthropic_tool_calls


def _convert_messages_to_anthropic_format(
    messages: list[ChatMessage],
) -> tuple[list[TextBlockParam], list[MessageParam]]:
    """
    Convert a list of messages to the format expected by Anthropic Chat API.

    :param messages: The list of ChatMessages to convert.
    :return: A tuple of two lists:
        - A list of system message TextBlockParam objects in the format expected by Anthropic API.
        - A list of non-system MessageParam objects in the format expected by Anthropic API.
    """

    anthropic_system_messages: list[TextBlockParam] = []
    anthropic_non_system_messages: list[MessageParam] = []

    i = 0
    while i < len(messages):
        message = messages[i]
        cache_control = message.meta.get("cache_control")
        # system messages have special format requirements for Anthropic API
        # they can have only type and text fields, and they need to be passed separately
        # to the Anthropic API endpoint
        if message.is_from(ChatRole.SYSTEM) and message.text:
            sys_message = TextBlockParam(type="text", text=message.text)
            if cache_control:
                sys_message["cache_control"] = cache_control
            anthropic_system_messages.append(sys_message)
            i += 1
            continue

        content: list[
            Union[
                TextBlockParam,
                ToolUseBlockParam,
                ToolResultBlockParam,
                ImageBlockParam,
                ThinkingBlockParam,
                RedactedThinkingBlockParam,
            ]
        ] = []

        # Handle multimodal content (text and images) preserving order
        for part in message._content:
            if isinstance(part, TextContent) and part.text:
                text_block = TextBlockParam(type="text", text=part.text)
                if cache_control:
                    text_block["cache_control"] = cache_control
                content.append(text_block)
            elif isinstance(part, ReasoningContent):
                if part.extra:
                    reasoning_contents = part.extra.get("reasoning_contents", [])
                    for item in reasoning_contents:
                        if item.get("reasoning_content").get("redacted_thinking"):
                            redacted_thinking_block = RedactedThinkingBlockParam(
                                type="redacted_thinking",
                                data=str(item.get("reasoning_content").get("redacted_thinking")),
                            )
                            content.append(redacted_thinking_block)
                        elif item.get("reasoning_content").get("reasoning_text"):
                            reasoning_block = ThinkingBlockParam(
                                type="thinking",
                                thinking=item.get("reasoning_content").get("reasoning_text").get("text"),
                                signature=item.get("reasoning_content").get("reasoning_text").get("signature"),
                            )
                            content.append(reasoning_block)

            elif isinstance(part, ImageContent):
                if not message.is_from(ChatRole.USER):
                    msg = "Image content is only supported for user messages"
                    raise ValueError(msg)

                if part.mime_type not in IMAGE_SUPPORTED_FORMATS:
                    supported_formats = ", ".join(IMAGE_SUPPORTED_FORMATS)
                    msg = (
                        f"Unsupported image format: {part.mime_type}. "
                        f"Anthropic supports the following formats: {supported_formats}"
                    )
                    raise ValueError(msg)

                image_block = ImageBlockParam(
                    type="image",
                    source={
                        "type": "base64",
                        "media_type": cast(ImageFormat, part.mime_type),
                        "data": part.base64_image,
                    },
                )
                if cache_control:
                    image_block["cache_control"] = cache_control
                content.append(image_block)

        if message.tool_calls:
            tool_use_blocks = _convert_tool_calls_to_anthropic_format(message.tool_calls)
            if cache_control:
                for tool_use_block in tool_use_blocks:
                    tool_use_block["cache_control"] = cache_control
            content.extend(tool_use_blocks)

        if message.tool_call_results:
            results = message.tool_call_results.copy()
            # Handle consecutive tool call results
            while (i + 1) < len(messages) and messages[i + 1].tool_call_results:
                i += 1
                results.extend(messages[i].tool_call_results)

            _update_anthropic_message_with_tool_call_results(results, content)
            if cache_control:
                for blk in content:
                    if blk.get("type") == "tool_result":
                        # thinking block does not support cache_control
                        # we have a check to ensure that block is not a thinking block
                        # but mypy doesnt know that
                        blk["cache_control"] = cache_control  # type: ignore [typeddict-unknown-key]

        if not content:
            msg = (
                "A `ChatMessage` must contain at least one `TextContent`, `ImageContent`, "
                "`ToolCall`, or `ToolCallResult`."
            )
            raise ValueError(msg)

        # Anthropic only supports assistant and user roles in messages. User role is also used for tool messages.
        # System messages are passed separately.
        role: Union[Literal["assistant"], Literal["user"]] = "user"
        if message._role == ChatRole.ASSISTANT:
            role = "assistant"

        anthropic_message = MessageParam(role=role, content=content)
        anthropic_non_system_messages.append(anthropic_message)
        i += 1

    return anthropic_system_messages, anthropic_non_system_messages


def _get_openai_compatible_usage(response_dict: dict) -> dict:
    """
    Converts Anthropic usage metadata to OpenAI compatible format.
    """
    usage = response_dict.get("usage", {})
    if usage:
        if "input_tokens" in usage:
            usage["prompt_tokens"] = usage.pop("input_tokens")
        if "output_tokens" in usage:
            usage["completion_tokens"] = usage.pop("output_tokens")

    return usage


def _convert_chat_completion_to_chat_message(
    anthropic_response: Any, ignore_tools_thinking_messages: bool
) -> ChatMessage:
    """
    Converts the response from the Anthropic API to a ChatMessage.
    """
    tool_calls = []
    reasoning_contents = []
    reasoning_text = ""

    for block in anthropic_response.content:
        reasoning_content: dict[str, Any] = {}
        if block.type == "tool_use":
            tool_calls.append(ToolCall(tool_name=block.name, arguments=block.input, id=block.id))
        elif block.type == "thinking":
            reasoning_content["reasoning_text"] = {}
            reasoning_content["reasoning_text"]["text"] = block.thinking
            reasoning_content["reasoning_text"]["signature"] = block.signature
            reasoning_contents.append({"reasoning_content": reasoning_content})
        elif block.type == "redacted_thinking":
            reasoning_content["redacted_thinking"] = block.data
            reasoning_contents.append({"reasoning_content": reasoning_content})

    reasoning_text = ""
    for content in reasoning_contents:
        if "reasoning_text" in content["reasoning_content"]:
            reasoning_text += content["reasoning_content"]["reasoning_text"]["text"]
        elif "redacted_thinking" in content["reasoning_content"]:
            reasoning_text += "[REDACTED]"

    reasoning = ReasoningContent(reasoning_text=reasoning_text, extra={"reasoning_contents": reasoning_contents})

    # Extract and join text blocks, respecting ignore_tools_thinking_messages
    text = ""

    if not (ignore_tools_thinking_messages and tool_calls):
        text = " ".join(block.text for block in anthropic_response.content if block.type == "text")

    message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls, reasoning=reasoning)

    # Dump the chat completion to a dict
    response_dict = anthropic_response.model_dump()
    usage = _get_openai_compatible_usage(response_dict)
    message._meta.update(
        {
            "model": response_dict.get("model", None),
            "index": 0,
            "finish_reason": FINISH_REASON_MAPPING.get(response_dict.get("stop_reason" or "")),
            "usage": usage,
        }
    )
    return message


def _convert_anthropic_chunk_to_streaming_chunk(
    chunk: RawMessageStreamEvent, component_info: ComponentInfo, tool_call_index: int
) -> StreamingChunk:
    """
    Converts an Anthropic StreamEvent to a StreamingChunk.

    :param chunk: The Anthropic StreamEvent to convert.
    :param component_info: The component info.
    :param tool_call_index: The index of the tool call among the tool calls in the message.
    :returns: The StreamingChunk.
    """
    content = ""
    tool_calls = []
    start = False
    finish_reason = None
    index = getattr(chunk, "index", None)

    # starting streaming message
    if chunk.type == "message_start":
        start = True

    # start of a content block
    if chunk.type == "content_block_start":
        start = True
        if chunk.content_block.type == "tool_use":
            tool_calls.append(
                ToolCallDelta(
                    index=tool_call_index,
                    id=chunk.content_block.id,
                    tool_name=chunk.content_block.name,
                )
            )

    # delta of a content block
    elif chunk.type == "content_block_delta":
        if chunk.delta.type == "text_delta":
            content = chunk.delta.text
        elif chunk.delta.type == "input_json_delta":
            tool_calls.append(ToolCallDelta(index=tool_call_index, arguments=chunk.delta.partial_json))

    # end of streaming message
    elif chunk.type == "message_delta":
        finish_reason = FINISH_REASON_MAPPING.get(getattr(chunk.delta, "stop_reason" or ""))

    meta = chunk.model_dump()

    return StreamingChunk(
        content=content,
        index=index,
        component_info=component_info,
        start=start,
        finish_reason=finish_reason,
        tool_calls=tool_calls if tool_calls else None,
        meta=meta,
    )


def _process_reasoning_contents(chunks: list[StreamingChunk]) -> Optional[ReasoningContent]:
    """
    Process reasoning contents from a list of StreamingChunk objects into the Anthropic expected format.

    :param chunks: List of StreamingChunk objects potentially containing reasoning contents.

    :returns: List of Anthropic formatted reasoning content dictionaries
    """
    formatted_reasoning_contents = []
    current_index = None
    content_block_text = ""
    content_block_signature = None
    content_block_redacted_thinking = None
    content_block_index = None

    for chunk in chunks:
        if (delta := chunk.meta.get("delta")) is not None:
            if delta.get("type") == "thinking_delta" and delta.get("thinking") is not None:
                content_block_index = chunk.meta.get("index", None)
        if (content_block := chunk.meta.get("content_block")) is not None and content_block.get(
            "type"
        ) == "redacted_thinking":
            content_block_index = chunk.meta.get("index", None)

        # Start new group when index changes
        if current_index is not None and content_block_index != current_index:
            # Finalize current group
            if content_block_text:
                formatted_reasoning_contents.append(
                    {
                        "reasoning_content": {
                            "reasoning_text": {"text": content_block_text, "signature": content_block_signature},
                        }
                    }
                )
            if content_block_redacted_thinking:
                formatted_reasoning_contents.append(
                    {"reasoning_content": {"redacted_thinking": content_block_redacted_thinking}}
                )

            # Reset accumulators for new group
            content_block_text = ""
            content_block_signature = None
            content_block_redacted_thinking = None

        # Accumulate content for current index
        current_index = content_block_index
        if (delta := chunk.meta.get("delta")) is not None:
            if delta.get("type") == "thinking_delta" and delta.get("thinking") is not None:
                content_block_text += delta.get("thinking", "")
            if delta.get("type") == "signature_delta" and delta.get("signature") is not None:
                content_block_signature = delta.get("signature", "")
        if (content_block := chunk.meta.get("content_block")) is not None and content_block.get(
            "type"
        ) == "redacted_thinking":
            content_block_redacted_thinking = content_block.get("data", "")

    # Finalize the last group
    if current_index is not None:
        if content_block_text:
            formatted_reasoning_contents.append(
                {
                    "reasoning_content": {
                        "reasoning_text": {"text": content_block_text, "signature": content_block_signature},
                    }
                }
            )
        if content_block_redacted_thinking:
            formatted_reasoning_contents.append(
                {"reasoning_content": {"redacted_thinking": content_block_redacted_thinking}}
            )

    # Combine all reasoning texts into a single string for the main reasoning_text field
    final_reasoning_text = ""
    for content in formatted_reasoning_contents:
        if "reasoning_text" in content["reasoning_content"]:
            # mypy somehow thinks that content["reasoning_content"]["reasoning_text"]["text"] can be of type None
            final_reasoning_text += content["reasoning_content"]["reasoning_text"]["text"]  # type: ignore[operator]
        elif "redacted_thinking" in content["reasoning_content"]:
            final_reasoning_text += "[REDACTED]"

    return (
        ReasoningContent(
            reasoning_text=final_reasoning_text, extra={"reasoning_contents": formatted_reasoning_contents}
        )
        if formatted_reasoning_contents
        else None
    )
