import base64
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from botocore.eventstream import EventStream
from haystack import logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FinishReason,
    ImageContent,
    ReasoningContent,
    StreamingChunk,
    SyncStreamingCallbackT,
    TextContent,
    ToolCall,
    ToolCallDelta,
)
from haystack.tools import Tool

logger = logging.getLogger(__name__)


# see https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html for supported formats
IMAGE_SUPPORTED_FORMATS = ["png", "jpeg", "gif", "webp"]

# see https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_MessageStopEvent.html
FINISH_REASON_MAPPING: Dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
    "tool_use": "tool_calls",
}


# Haystack to Bedrock util methods
def _format_tools(tools: Optional[List[Tool]] = None) -> Optional[Dict[str, Any]]:
    """
    Format Haystack Tool(s) to Amazon Bedrock toolConfig format.

    :param tools: List of Tool objects to format
    :returns:
        Dictionary in Bedrock toolConfig format or None if no tools are provided
    """
    if not tools:
        return None

    tool_specs = []
    for tool in tools:
        tool_specs.append(
            {"toolSpec": {"name": tool.name, "description": tool.description, "inputSchema": {"json": tool.parameters}}}
        )

    return {"tools": tool_specs} if tool_specs else None


def _format_tool_call_message(tool_call_message: ChatMessage) -> Dict[str, Any]:
    """
    Format a Haystack ChatMessage containing tool calls into Bedrock format.

    :param tool_call_message: ChatMessage object containing tool calls to be formatted.
    :returns:
        Dictionary representing the tool call message in Bedrock's expected format
    """
    content: List[Dict[str, Any]] = []

    # tool call messages can contain reasoning content
    if reasoning_content := tool_call_message.reasoning:
        content.extend(_format_reasoning_content(reasoning_content=reasoning_content))

    # Tool call message can contain text
    if tool_call_message.text:
        content.append({"text": tool_call_message.text})

    for tool_call in tool_call_message.tool_calls:
        content.append(
            {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.tool_name, "input": tool_call.arguments}}
        )
    return {"role": tool_call_message.role.value, "content": content}


def _format_tool_result_message(tool_call_result_message: ChatMessage) -> Dict[str, Any]:
    """
    Format a Haystack ChatMessage containing tool call results into Bedrock format.

    :param tool_call_result_message: ChatMessage object containing tool call results to be formatted.
    :returns: Dictionary representing the tool result message in Bedrock's expected format
    """
    # Assuming tool call result messages will only contain tool results
    tool_results = []
    for result in tool_call_result_message.tool_call_results:
        try:
            json_result = json.loads(result.result)
            content = [{"json": json_result}]
        except json.JSONDecodeError:
            content = [{"text": result.result}]

        tool_results.append(
            {
                "toolResult": {
                    "toolUseId": result.origin.id,
                    "content": content,
                    **({"status": "error"} if result.error else {}),
                }
            }
        )
    # role must be user
    return {"role": "user", "content": tool_results}


def _repair_tool_result_messages(bedrock_formatted_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Repair and reorganize tool result messages to maintain proper ordering and grouping.

    Ensures tool result messages are properly grouped in the same way as their corresponding tool call messages
    and maintains the original message ordering.

    :param bedrock_formatted_messages: List of Bedrock-formatted messages that may need repair.
    :returns: List of properly organized Bedrock-formatted messages with correctly grouped tool results.
    """
    tool_call_messages = []
    tool_result_messages = []
    for idx, msg in enumerate(bedrock_formatted_messages):
        content = msg.get("content", [])
        if content:
            if any("toolUse" in c for c in content):
                tool_call_messages.append((idx, msg))
            elif any("toolResult" in c for c in content):
                tool_result_messages.append((idx, msg))

    # Determine the tool call IDs for each tool call message
    group_to_tool_call_ids: Dict[int, Any] = {idx: [] for idx, _ in tool_call_messages}
    for idx, tool_call in tool_call_messages:
        tool_use_contents = [c for c in tool_call["content"] if "toolUse" in c]
        for content in tool_use_contents:
            group_to_tool_call_ids[idx].append(content["toolUse"]["toolUseId"])

    # Regroups the tool_result_prompts based on the tool_call_prompts
    # Makes sure to:
    # - Within the new group the tool call IDs of the tool result messages are in the same order as the tool call
    #   messages
    # - The tool result messages are in the same order as the original message list
    repaired_tool_result_prompts = []
    for tool_call_ids in group_to_tool_call_ids.values():
        regrouped_tool_result = []
        original_idx = None
        for tool_call_id in tool_call_ids:
            for idx, tool_result in tool_result_messages:
                tool_result_contents = [c for c in tool_result["content"] if "toolResult" in c]
                for content in tool_result_contents:
                    if content["toolResult"]["toolUseId"] == tool_call_id:
                        regrouped_tool_result.append(content)
                        # Keep track of the original index of the last tool result message
                        original_idx = idx
        if regrouped_tool_result and original_idx is not None:
            repaired_tool_result_prompts.append((original_idx, {"role": "user", "content": regrouped_tool_result}))

    # Remove the tool result messages from bedrock_formatted_messages
    bedrock_formatted_messages_minus_tool_results: List[Tuple[int, Any]] = []
    for idx, msg in enumerate(bedrock_formatted_messages):
        # Assumes the content of tool result messages only contains 'toolResult': {...} objects (e.g. no 'text')
        if msg.get("content") and "toolResult" not in msg["content"][0]:
            bedrock_formatted_messages_minus_tool_results.append((idx, msg))

    # Add the repaired tool result messages and sort to maintain the correct order
    repaired_bedrock_formatted_messages = bedrock_formatted_messages_minus_tool_results + repaired_tool_result_prompts
    repaired_bedrock_formatted_messages.sort(key=lambda x: x[0])

    # Drop the index and return only the messages
    return [msg for _, msg in repaired_bedrock_formatted_messages]


def _format_reasoning_content(reasoning_content: ReasoningContent) -> List[Dict[str, Any]]:
    """
    Format ReasoningContent to match Bedrock's expected structure.

    :param reasoning_content: ReasoningContent object containing reasoning contents to format.
    :returns: List of formatted reasoning content dictionaries for Bedrock.
    """
    formatted_contents = []
    for content in reasoning_content.extra.get("reasoning_contents", []):
        formatted_content = {"reasoningContent": content["reasoning_content"]}
        if reasoning_text := formatted_content["reasoningContent"].pop("reasoning_text", None):
            formatted_content["reasoningContent"]["reasoningText"] = reasoning_text
        if redacted_content := formatted_content["reasoningContent"].pop("redacted_content", None):
            formatted_content["reasoningContent"]["redactedContent"] = redacted_content
        formatted_contents.append(formatted_content)
    return formatted_contents


def _format_text_image_message(message: ChatMessage) -> Dict[str, Any]:
    """
    Format a Haystack ChatMessage containing text and optional image content into Bedrock format.

    :param message: Haystack ChatMessage.
    :returns: Dictionary representing the message in Bedrock's expected format.
    :raises ValueError: If image content is found in an assistant message or an unsupported image format is used.
    """
    content_parts = message._content

    bedrock_content_blocks: List[Dict[str, Any]] = []
    # Add reasoning content if available as the first content block
    if message.reasoning:
        bedrock_content_blocks.extend(_format_reasoning_content(reasoning_content=message.reasoning))

    for part in content_parts:
        if isinstance(part, TextContent):
            bedrock_content_blocks.append({"text": part.text})

        elif isinstance(part, ImageContent):
            if message.is_from(ChatRole.ASSISTANT):
                err_msg = "Image content is not supported for assistant messages"
                raise ValueError(err_msg)

            image_format = part.mime_type.split("/")[-1] if part.mime_type else None
            if image_format not in IMAGE_SUPPORTED_FORMATS:
                err_msg = (
                    f"Unsupported image format: {image_format}. "
                    f"Bedrock supports the following image formats: {IMAGE_SUPPORTED_FORMATS}"
                )
                raise ValueError(err_msg)
            source = {"bytes": base64.b64decode(part.base64_image)}
            bedrock_content_blocks.append({"image": {"format": image_format, "source": source}})

    return {"role": message.role.value, "content": bedrock_content_blocks}


def _format_messages(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Format a list of Haystack ChatMessages to the format expected by Bedrock API.

    Processes and separates system messages from other message types and handles special formatting for tool calls
    and tool results.

    :param messages: List of ChatMessage objects to format for Bedrock API.
    :returns: Tuple containing (system_prompts, non_system_messages) in Bedrock format,
              where system_prompts is a list of system message dictionaries and
              non_system_messages is a list of properly formatted message dictionaries.
    """
    # Separate system messages, tool calls, and tool results
    system_prompts = []
    bedrock_formatted_messages = []
    for msg in messages:
        if msg.is_from(ChatRole.SYSTEM):
            # Assuming system messages can only contain text
            # Don't need to track idx since system_messages are handled separately
            system_prompts.append({"text": msg.text})
        elif msg.tool_calls:
            bedrock_formatted_messages.append(_format_tool_call_message(msg))
        elif msg.tool_call_results:
            bedrock_formatted_messages.append(_format_tool_result_message(msg))
        else:
            bedrock_formatted_messages.append(_format_text_image_message(msg))

    repaired_bedrock_formatted_messages = _repair_tool_result_messages(bedrock_formatted_messages)
    return system_prompts, repaired_bedrock_formatted_messages


def _parse_completion_response(response_body: Dict[str, Any], model: str) -> List[ChatMessage]:
    """
    Parse a Bedrock API response into Haystack ChatMessage objects.

    Extracts text content, tool calls, and metadata from the Bedrock response and converts them into the appropriate
    Haystack format.

    :param response_body: Raw JSON response from Bedrock API.
    :param model: The model ID used for generation, included in message metadata.
    :returns: List of ChatMessage objects containing the assistant's response(s) with appropriate metadata.
    """

    replies = []
    if "output" in response_body and "message" in response_body["output"]:
        message = response_body["output"]["message"]
        if message["role"] == "assistant":
            content_blocks = message["content"]

            # Common meta information
            meta = {
                "model": model,
                "index": 0,
                "finish_reason": FINISH_REASON_MAPPING.get(response_body.get("stopReason", "")),
                "usage": {
                    # OpenAI's format for usage for cross ChatGenerator compatibility
                    "prompt_tokens": response_body.get("usage", {}).get("inputTokens", 0),
                    "completion_tokens": response_body.get("usage", {}).get("outputTokens", 0),
                    "total_tokens": response_body.get("usage", {}).get("totalTokens", 0),
                },
            }
            # guardrail trace
            if "trace" in response_body:
                meta["trace"] = response_body["trace"]

            # Process all content blocks and combine them into a single message
            text_content = []
            tool_calls = []
            reasoning_contents = []
            for content_block in content_blocks:
                if "text" in content_block:
                    text_content.append(content_block["text"])
                elif "toolUse" in content_block:
                    # Convert tool use to ToolCall
                    tool_use = content_block["toolUse"]
                    tool_call = ToolCall(
                        id=tool_use.get("toolUseId"),
                        tool_name=tool_use.get("name"),
                        arguments=tool_use.get("input", {}),
                    )
                    tool_calls.append(tool_call)
                elif "reasoningContent" in content_block:
                    reasoning_content = content_block["reasoningContent"]
                    # If reasoningText is present, replace it with reasoning_text
                    if "reasoningText" in reasoning_content:
                        reasoning_content["reasoning_text"] = reasoning_content.pop("reasoningText")
                    if "redactedContent" in reasoning_content:
                        reasoning_content["redacted_content"] = reasoning_content.pop("redactedContent")
                    reasoning_contents.append({"reasoning_content": reasoning_content})

            reasoning_text = ""
            for content in reasoning_contents:
                if "reasoning_text" in content["reasoning_content"]:
                    reasoning_text += content["reasoning_content"]["reasoning_text"]["text"]
                elif "redacted_content" in content["reasoning_content"]:
                    reasoning_text += "[REDACTED]"

            # Create a single ChatMessage with combined text and tool calls
            replies.append(
                ChatMessage.from_assistant(
                    " ".join(text_content),
                    tool_calls=tool_calls,
                    meta=meta,
                    reasoning=ReasoningContent(
                        reasoning_text=reasoning_text, extra={"reasoning_contents": reasoning_contents}
                    )
                    if reasoning_contents
                    else None,
                )
            )

    return replies


def _convert_event_to_streaming_chunk(
    event: Dict[str, Any], model: str, component_info: ComponentInfo
) -> StreamingChunk:
    """
    Convert a Bedrock streaming event to a Haystack StreamingChunk.

    Handles different event types (contentBlockStart, contentBlockDelta, messageStop, metadata) and extracts relevant
    information to create StreamingChunk objects in the same format used by Haystack's OpenAIChatGenerator.

    :param event: Dictionary containing a Bedrock streaming event.
    :param model: The model ID used for generation, included in chunk metadata.
    :param component_info: ComponentInfo object
    :returns: StreamingChunk object containing the content and metadata extracted from the event.
    """

    # Initialize an empty StreamingChunk to return if no relevant event is found
    # (e.g. for messageStart and contentBlockStop)
    base_meta = {"model": model, "received_at": datetime.now(timezone.utc).isoformat()}
    streaming_chunk = StreamingChunk(content="", meta=base_meta)

    if "contentBlockStart" in event:
        # contentBlockStart always has the key "contentBlockIndex"
        block_start = event["contentBlockStart"]
        block_idx = block_start["contentBlockIndex"]
        if "start" in block_start and "toolUse" in block_start["start"]:
            tool_start = block_start["start"]["toolUse"]
            streaming_chunk = StreamingChunk(
                content="",
                index=block_idx,
                tool_calls=[
                    ToolCallDelta(
                        index=block_idx,
                        id=tool_start["toolUseId"],
                        tool_name=tool_start["name"],
                    )
                ],
                meta=base_meta,
            )

    elif "contentBlockDelta" in event:
        # contentBlockDelta always has the key "contentBlockIndex" and "delta"
        block_idx = event["contentBlockDelta"]["contentBlockIndex"]
        delta = event["contentBlockDelta"]["delta"]
        # This is for accumulating text deltas
        if "text" in delta:
            streaming_chunk = StreamingChunk(
                content=delta["text"],
                index=block_idx,
                meta=base_meta,
            )
        # This only occurs when accumulating the arguments for a toolUse
        # The content_block for this tool should already exist at this point
        elif "toolUse" in delta:
            streaming_chunk = StreamingChunk(
                content="",
                index=block_idx,
                tool_calls=[
                    ToolCallDelta(
                        index=block_idx,
                        arguments=delta["toolUse"].get("input", ""),
                    )
                ],
                meta=base_meta,
            )
        # This is for accumulating reasoning content deltas
        elif "reasoningContent" in delta:
            reasoning_content = delta["reasoningContent"]
            if "redactedContent" in reasoning_content:
                reasoning_content["redacted_content"] = reasoning_content.pop("redactedContent")
            streaming_chunk = StreamingChunk(
                content="",
                index=block_idx,
                meta={
                    **base_meta,
                    "reasoning_contents": [{"index": block_idx, "reasoning_content": reasoning_content}],
                },
            )

    elif "messageStop" in event:
        finish_reason = FINISH_REASON_MAPPING.get(event["messageStop"].get("stopReason"))
        streaming_chunk = StreamingChunk(
            content="",
            finish_reason=finish_reason,
            meta=base_meta,
        )

    elif "metadata" in event:
        event_meta = event["metadata"]
        chunk_meta: Dict[str, Any] = {**base_meta}

        if "usage" in event_meta:
            usage = event_meta["usage"]
            chunk_meta["usage"] = {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
            }
        if "trace" in event_meta:
            chunk_meta["trace"] = event_meta["trace"]

        # Only create chunk if we added usage or trace data
        if len(chunk_meta) > len(base_meta):
            streaming_chunk = StreamingChunk(content="", meta=chunk_meta)

    streaming_chunk.component_info = component_info

    return streaming_chunk


def _process_reasoning_contents(chunks: List[StreamingChunk]) -> Optional[ReasoningContent]:
    """
    Process reasoning contents from a list of StreamingChunk objects into the Bedrock expected format.

    :param chunks: List of StreamingChunk objects potentially containing reasoning contents.

    :returns: List of Bedrock formatted reasoning content dictionaries
    """
    formatted_reasoning_contents = []
    current_index = None
    reasoning_text = ""
    reasoning_signature = None
    redacted_content = None
    for chunk in chunks:
        reasoning_contents = chunk.meta.get("reasoning_contents", [])

        for reasoning_content in reasoning_contents:
            content_block_index = reasoning_content["index"]

            # Start new group when index changes
            if current_index is not None and content_block_index != current_index:
                # Finalize current group
                if reasoning_text:
                    formatted_reasoning_contents.append(
                        {
                            "reasoning_content": {
                                "reasoning_text": {"text": reasoning_text, "signature": reasoning_signature},
                            }
                        }
                    )
                if redacted_content:
                    formatted_reasoning_contents.append({"reasoning_content": {"redacted_content": redacted_content}})

                # Reset accumulators for new group
                reasoning_text = ""
                reasoning_signature = None
                redacted_content = None

            # Accumulate content for current index
            current_index = content_block_index
            reasoning_text += reasoning_content["reasoning_content"].get("text", "")
            if "redacted_content" in reasoning_content["reasoning_content"]:
                redacted_content = reasoning_content["reasoning_content"]["redacted_content"]
            if "signature" in reasoning_content["reasoning_content"]:
                reasoning_signature = reasoning_content["reasoning_content"]["signature"]

    # Finalize the last group
    if current_index is not None:
        if reasoning_text:
            formatted_reasoning_contents.append(
                {
                    "reasoning_content": {
                        "reasoning_text": {"text": reasoning_text, "signature": reasoning_signature},
                    }
                }
            )
        if redacted_content:
            formatted_reasoning_contents.append({"reasoning_content": {"redacted_content": redacted_content}})

    # Combine all reasoning texts into a single string for the main reasoning_text field
    final_reasoning_text = ""
    for content in formatted_reasoning_contents:
        if "reasoning_text" in content["reasoning_content"]:
            # mypy somehow thinks that content["reasoning_content"]["reasoning_text"]["text"] can be of type None
            final_reasoning_text += content["reasoning_content"]["reasoning_text"]["text"]  # type: ignore[operator]
        elif "redacted_content" in content["reasoning_content"]:
            final_reasoning_text += "[REDACTED]"

    return (
        ReasoningContent(
            reasoning_text=final_reasoning_text, extra={"reasoning_contents": formatted_reasoning_contents}
        )
        if formatted_reasoning_contents
        else None
    )


def _parse_streaming_response(
    response_stream: EventStream,
    streaming_callback: SyncStreamingCallbackT,
    model: str,
    component_info: ComponentInfo,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :param component_info: ComponentInfo object
    :return: List of ChatMessage objects
    """
    content_block_idxs = set()
    chunks: List[StreamingChunk] = []
    for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        content_block_idx = streaming_chunk.index
        if content_block_idx is not None and content_block_idx not in content_block_idxs:
            streaming_chunk.start = True
            content_block_idxs.add(content_block_idx)
        streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)

    reply = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    # both the reasoning content and the trace are ignored in _convert_streaming_chunks_to_chat_message
    # so we need to process them separately
    reasoning_content = _process_reasoning_contents(chunks=chunks)
    if chunks[-1].meta and "trace" in chunks[-1].meta:
        reply.meta["trace"] = chunks[-1].meta["trace"]

    reply = ChatMessage.from_assistant(
        text=reply.text,
        meta=reply.meta,
        name=reply.name,
        tool_calls=reply.tool_calls,
        reasoning=reasoning_content,
    )

    return [reply]


async def _parse_streaming_response_async(
    response_stream: EventStream,
    streaming_callback: AsyncStreamingCallbackT,
    model: str,
    component_info: ComponentInfo,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :param component_info: ComponentInfo object
    :return: List of ChatMessage objects
    """
    content_block_idxs = set()
    chunks: List[StreamingChunk] = []
    async for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        content_block_idx = streaming_chunk.index
        if content_block_idx is not None and content_block_idx not in content_block_idxs:
            streaming_chunk.start = True
            content_block_idxs.add(content_block_idx)
        await streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)
    reply = _convert_streaming_chunks_to_chat_message(chunks=chunks)
    reasoning_content = _process_reasoning_contents(chunks=chunks)
    reply = ChatMessage.from_assistant(
        text=reply.text,
        meta=reply.meta,
        name=reply.name,
        tool_calls=reply.tool_calls,
        reasoning=reasoning_content,
    )
    return [reply]


def _validate_guardrail_config(guardrail_config: Optional[Dict[str, str]] = None, streaming: bool = False) -> None:
    """
    Validate the guardrail configuration.

    :param guardrail_config: The guardrail configuration.
    :param streaming: Whether the streaming is enabled.

    :raises ValueError: If the guardrail configuration is invalid.
    """
    if guardrail_config is None:
        return

    required_fields = {"guardrailIdentifier", "guardrailVersion"}
    if not required_fields.issubset(guardrail_config):
        msg = "`guardrailIdentifier` and `guardrailVersion` fields are required in guardrail configuration."
        raise ValueError(msg)
    if not streaming and "streamProcessingMode" in guardrail_config:
        msg = "`streamProcessingMode` field is only supported for streaming (when `streaming_callback` is not None)."
        raise ValueError(msg)
