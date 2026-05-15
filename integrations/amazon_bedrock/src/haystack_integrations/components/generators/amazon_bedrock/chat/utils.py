import base64
import dataclasses
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from botocore.eventstream import EventStream
from haystack import logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ComponentInfo,
    FileContent,
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
IMAGE_MIME_TYPE_TO_FORMAT: dict[str, str] = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

# https://docs.aws.amazon.com/cli/latest/reference/bedrock-runtime/converse.html
DOCUMENT_MIME_TYPE_TO_FORMAT: dict[str, str] = {
    "application/pdf": "pdf",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/html": "html",
    "text/plain": "txt",
    "text/markdown": "md",
}

VIDEO_MIME_TYPE_TO_FORMAT: dict[str, str] = {
    "video/x-matroska": "mkv",
    "video/quicktime": "mov",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/x-ms-wmv": "wmv",
    "video/3gpp": "three_gp",
}

# see https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_MessageStopEvent.html
FINISH_REASON_MAPPING: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "guardrail_intervened": "content_filter",
    "content_filtered": "content_filter",
    "tool_use": "tool_calls",
}


# Haystack to Bedrock util methods
def _format_tools(
    tools: list[Tool] | None = None, tools_cachepoint_config: dict[str, dict[str, str]] | None = None
) -> dict[str, Any] | None:
    """
    Format Haystack Tool(s) to Amazon Bedrock toolConfig format.

    :param tools: List of Tool objects to format
    :returns:
        Dictionary in Bedrock toolConfig format or None if no tools are provided
    """
    if not tools:
        return None

    tool_specs: list[dict[str, Any]] = []
    for tool in tools:
        tool_specs.append(
            {"toolSpec": {"name": tool.name, "description": tool.description, "inputSchema": {"json": tool.parameters}}}
        )

    if tools_cachepoint_config:
        tool_specs.append(tools_cachepoint_config)

    return {"tools": tool_specs}


def _convert_image_content_to_bedrock_format(image_content: ImageContent) -> dict[str, Any]:
    """
    Convert a Haystack ImageContent to Bedrock format.
    """

    image_format = IMAGE_MIME_TYPE_TO_FORMAT.get(image_content.mime_type or "")
    if image_format is None:
        err_msg = (
            f"Unsupported image MIME type: {image_content.mime_type}. "
            f"Bedrock supports the following image formats: {list(set(IMAGE_MIME_TYPE_TO_FORMAT.values()))}"
        )
        raise ValueError(err_msg)

    source = {"bytes": base64.b64decode(image_content.base64_image)}

    return {"image": {"format": image_format, "source": source}}


def _convert_file_content_to_bedrock_format(file_content: FileContent) -> dict[str, Any]:
    """
    Convert a Haystack FileContent to Bedrock format.
    """

    if file_content.mime_type is None:
        err_msg = "MIME type is required to use FileContent in Bedrock."
        raise ValueError(err_msg)

    if doc_format := DOCUMENT_MIME_TYPE_TO_FORMAT.get(file_content.mime_type):
        source = {"bytes": base64.b64decode(file_content.base64_data)}

        name = "filename"
        if file_content.filename:
            raw_name = os.path.splitext(file_content.filename)[0]
            # Bedrock requires name to be present but is very strict about the format.
            # See https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
            sanitized_name = re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s\-\[\]()]", "", raw_name)).strip()
            if sanitized_name:
                name = sanitized_name

        doc_block = {
            "document": {
                "format": doc_format,
                "source": source,
                "name": name,
                **({"context": file_content.extra["context"]} if file_content.extra.get("context") else {}),
                **({"citations": file_content.extra["citations"]} if file_content.extra.get("citations") else {}),
            }
        }
        return doc_block

    if video_format := VIDEO_MIME_TYPE_TO_FORMAT.get(file_content.mime_type):
        source = {"bytes": base64.b64decode(file_content.base64_data)}
        video_block = {"video": {"format": video_format, "source": source}}
        return video_block

    err_msg = (
        f"Unsupported file content MIME type: {file_content.mime_type}\n"
        f"Bedrock supports the following formats:\n - Documents: {list(DOCUMENT_MIME_TYPE_TO_FORMAT.values())}\n"
        f" - Videos: {list(VIDEO_MIME_TYPE_TO_FORMAT.values())}"
    )
    raise ValueError(err_msg)


def _format_tool_call_message(tool_call_message: ChatMessage) -> dict[str, Any]:
    """
    Format a Haystack ChatMessage containing tool calls into Bedrock format.

    :param tool_call_message: ChatMessage object containing tool calls to be formatted.
    :returns:
        Dictionary representing the tool call message in Bedrock's expected format
    """
    content: list[dict[str, Any]] = []

    # tool call messages can contain reasoning content
    if reasoning_content := tool_call_message.reasoning:
        content.append(_format_reasoning_content(reasoning_content=reasoning_content))

    # Tool call message can contain text
    if tool_call_message.text:
        content.append({"text": tool_call_message.text})

    for tool_call in tool_call_message.tool_calls:
        content.append(
            {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.tool_name, "input": tool_call.arguments}}
        )
    return {"role": tool_call_message.role.value, "content": content}


def _format_tool_result_message(tool_call_result_message: ChatMessage) -> dict[str, Any]:
    """
    Format a Haystack ChatMessage containing tool call results into Bedrock format.

    :param tool_call_result_message: ChatMessage object containing tool call results to be formatted.
    :returns: Dictionary representing the tool result message in Bedrock's expected format
    """
    # Assuming tool call result messages will only contain tool results
    tool_results = []
    for tool_call_result in tool_call_result_message.tool_call_results:
        if isinstance(tool_call_result.result, str):
            try:
                json_result = json.loads(tool_call_result.result)
                content = [{"json": json_result}]
            except json.JSONDecodeError:
                content = [{"text": tool_call_result.result}]
        elif isinstance(tool_call_result.result, list):
            content = []
            for item in tool_call_result.result:
                if isinstance(item, TextContent):
                    content.append({"text": item.text})
                elif isinstance(item, ImageContent):
                    content.append(_convert_image_content_to_bedrock_format(item))
        else:
            err_msg = "Unsupported content type in tool call result"
            raise ValueError(err_msg)

        tool_results.append(
            {
                "toolResult": {
                    "toolUseId": tool_call_result.origin.id,
                    "content": content,
                    **({"status": "error"} if tool_call_result.error else {}),
                }
            }
        )
    # role must be user
    return {"role": "user", "content": tool_results}


def _repair_tool_result_messages(bedrock_formatted_messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    group_to_tool_call_ids: dict[int, Any] = {idx: [] for idx, _ in tool_call_messages}
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
                tool_result_contents = [c for c in tool_result["content"] if "toolResult" in c or "cachePoint" in c]
                for content in tool_result_contents:
                    if "toolResult" in content and content["toolResult"]["toolUseId"] == tool_call_id:
                        regrouped_tool_result.append(content)
                        # Keep track of the original index of the last tool result message
                        original_idx = idx
                    elif "cachePoint" in content and content not in regrouped_tool_result:
                        regrouped_tool_result.append(content)

        if regrouped_tool_result and original_idx is not None:
            repaired_tool_result_prompts.append((original_idx, {"role": "user", "content": regrouped_tool_result}))

    # Remove the tool result messages from bedrock_formatted_messages
    bedrock_formatted_messages_minus_tool_results: list[tuple[int, Any]] = []
    for idx, msg in enumerate(bedrock_formatted_messages):
        # Filter out messages that contain toolResult (they are handled by repaired_tool_result_prompts)
        if msg.get("content") and not any("toolResult" in c for c in msg["content"]):
            bedrock_formatted_messages_minus_tool_results.append((idx, msg))

    # Add the repaired tool result messages and sort to maintain the correct order
    repaired_bedrock_formatted_messages = bedrock_formatted_messages_minus_tool_results + repaired_tool_result_prompts
    repaired_bedrock_formatted_messages.sort(key=lambda x: x[0])

    # Drop the index and return only the messages
    return [msg for _, msg in repaired_bedrock_formatted_messages]


def _format_reasoning_content(reasoning_content: ReasoningContent) -> dict[str, Any]:
    """
    Format ReasoningContent to match Bedrock's expected structure.

    :param reasoning_content: ReasoningContent object containing reasoning contents to format.
    :returns: Dictionary representing the formatted reasoning content for Bedrock.

    """
    formatted_content = {
        "reasoningContent": {
            "reasoningText": {
                "text": reasoning_content.reasoning_text,
                **(
                    {"signature": reasoning_content.extra["signature"]}
                    if reasoning_content.extra.get("signature")
                    else {}
                ),
            }
        }
    }
    return formatted_content


def _format_user_message(message: ChatMessage) -> dict[str, Any]:
    """
    Format a Haystack user ChatMessage into Bedrock format.

    :param message: Haystack ChatMessage.
    :returns: Dictionary representing the message in Bedrock's expected format.
    """
    content_parts = message._content

    bedrock_content_blocks: list[dict[str, Any]] = []

    for part in content_parts:
        if isinstance(part, TextContent):
            bedrock_content_blocks.append({"text": part.text})

        elif isinstance(part, ImageContent):
            bedrock_content_blocks.append(_convert_image_content_to_bedrock_format(part))

        elif isinstance(part, FileContent):
            bedrock_content_blocks.append(_convert_file_content_to_bedrock_format(part))

    return {"role": message.role.value, "content": bedrock_content_blocks}


def _format_textual_assistant_message(message: ChatMessage) -> dict[str, Any]:
    """
    Format a Haystack assistant ChatMessage containing text and optionally reasoning into Bedrock format.

    :param message: Haystack ChatMessage.
    :returns: Dictionary representing the message in Bedrock's expected format.
    """
    content_parts = message._content

    bedrock_content_blocks: list[dict[str, Any]] = []
    # Add reasoning content if available as the first content block
    if message.reasoning:
        bedrock_content_blocks.append(_format_reasoning_content(reasoning_content=message.reasoning))

    for part in content_parts:
        if isinstance(part, TextContent):
            bedrock_content_blocks.append({"text": part.text})

    return {"role": message.role.value, "content": bedrock_content_blocks}


def _validate_and_format_cache_point(cache_point: dict[str, str] | None) -> dict[str, dict[str, str]] | None:
    """
    Validate and format a cache point dictionary.

    Schema available at https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html

    :param cache_point: Cache point dictionary to validate and format.
    :returns: Dictionary in Bedrock cachePoint format or None if no cache point is provided.
    :raises ValueError: If cache point is not valid.
    """
    if not cache_point:
        return None

    if "type" not in cache_point or cache_point["type"] != "default":
        err_msg = "Cache point must have a 'type' key with value 'default'."
        raise ValueError(err_msg)
    if not set(cache_point).issubset({"type", "ttl"}):
        err_msg = "Cache point can only contain 'type' and 'ttl' keys."
        raise ValueError(err_msg)
    if "ttl" in cache_point and cache_point["ttl"] not in ("5m", "1h"):
        err_msg = "Cache point 'ttl' must be one of '5m', '1h'."
        raise ValueError(err_msg)

    return {"cachePoint": cache_point}


def _format_messages(messages: list[ChatMessage]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    system_prompts: list[dict[str, Any]] = []
    bedrock_formatted_messages = []
    for msg in messages:
        cache_point = _validate_and_format_cache_point(msg.meta.get("cachePoint"))
        if msg.is_from(ChatRole.SYSTEM):
            # Assuming system messages can only contain text
            # Don't need to track idx since system_messages are handled separately
            system_prompts.append({"text": msg.text})
            if cache_point:
                system_prompts.append(cache_point)
            continue

        if msg.tool_calls:
            formatted_msg = _format_tool_call_message(msg)
        elif msg.tool_call_results:
            formatted_msg = _format_tool_result_message(msg)
        elif msg.is_from(ChatRole.USER):
            formatted_msg = _format_user_message(msg)
        elif msg.is_from(ChatRole.ASSISTANT):
            formatted_msg = _format_textual_assistant_message(msg)
        if cache_point:
            formatted_msg["content"].append(cache_point)
        bedrock_formatted_messages.append(formatted_msg)

    repaired_bedrock_formatted_messages = _repair_tool_result_messages(bedrock_formatted_messages)

    return system_prompts, repaired_bedrock_formatted_messages


def _parse_completion_response(response_body: dict[str, Any], model: str) -> list[ChatMessage]:
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
                    "cache_read_input_tokens": response_body.get("usage", {}).get("cacheReadInputTokens", 0),
                    "cache_write_input_tokens": response_body.get("usage", {}).get("cacheWriteInputTokens", 0),
                    "cache_details": response_body.get("usage", {}).get("CacheDetails", {}),
                },
            }
            # guardrail trace
            if "trace" in response_body:
                meta["trace"] = response_body["trace"]

            # Process all content blocks and combine them into a single message
            text_content = []
            tool_calls = []
            reasoning_content = None
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
                elif "citationsContent" in content_block:
                    citations_content = content_block["citationsContent"]
                    meta["citations"] = citations_content
                    if "content" in citations_content:
                        for entry in citations_content["content"]:
                            text = entry.get("text", "")
                            if text.strip():
                                text_content.append(text)

            reasoning_extra = {}
            reasoning_text = ""
            if reasoning_content:
                if "reasoningText" in reasoning_content:
                    reasoning_text = reasoning_content["reasoningText"].get("text", "")
                    signature = reasoning_content["reasoningText"].get("signature")
                    if signature:
                        reasoning_extra["signature"] = signature

            # Create a single ChatMessage with combined text and tool calls
            replies.append(
                ChatMessage.from_assistant(
                    "".join(text_content),
                    tool_calls=tool_calls,
                    meta=meta,
                    reasoning=ReasoningContent(reasoning_text=reasoning_text, extra=reasoning_extra)
                    if reasoning_text or reasoning_extra
                    else None,
                )
            )

    return replies


def _convert_event_to_streaming_chunk(
    event: dict[str, Any], model: str, component_info: ComponentInfo
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
            reasoning_text = ""
            extra = {}
            if "text" in reasoning_content:
                reasoning_text = reasoning_content["text"]
            if "signature" in reasoning_content:
                extra["signature"] = reasoning_content["signature"]
            streaming_chunk = StreamingChunk(
                content="",
                index=block_idx,
                reasoning=ReasoningContent(
                    reasoning_text=reasoning_text,
                    extra=extra,
                ),
                meta=base_meta,
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
        chunk_meta: dict[str, Any] = {**base_meta}

        if "usage" in event_meta:
            usage = event_meta["usage"]
            chunk_meta["usage"] = {
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("totalTokens", 0),
                "cache_read_input_tokens": usage.get("cacheReadInputTokens", 0),
                "cache_write_input_tokens": usage.get("cacheWriteInputTokens", 0),
                "cache_details": usage.get("cacheDetails", {}),
            }
        if "trace" in event_meta:
            chunk_meta["trace"] = event_meta["trace"]

        # Only create chunk if we added usage or trace data
        if len(chunk_meta) > len(base_meta):
            streaming_chunk = StreamingChunk(content="", meta=chunk_meta)

    streaming_chunk = dataclasses.replace(streaming_chunk, component_info=component_info)

    return streaming_chunk


def _parse_streaming_response(
    response_stream: EventStream,
    streaming_callback: SyncStreamingCallbackT,
    model: str,
    component_info: ComponentInfo,
) -> list[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :param component_info: ComponentInfo object
    :return: List of ChatMessage objects
    """
    content_block_idxs = set()
    chunks: list[StreamingChunk] = []
    for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        content_block_idx = streaming_chunk.index
        if content_block_idx is not None and content_block_idx not in content_block_idxs:
            streaming_chunk.start = True
            content_block_idxs.add(content_block_idx)
        streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)

    replies = _convert_chunks_to_messages(chunks)
    return replies


def _convert_chunks_to_messages(chunks: list[StreamingChunk]) -> list[ChatMessage]:
    reply = _convert_streaming_chunks_to_chat_message(chunks=chunks)

    # reasoning signatures are ignored in _convert_streaming_chunks_to_chat_message
    # so we need to process them separately
    if reply.reasoning:
        for chunk in reversed(chunks):
            if chunk.reasoning and chunk.reasoning.extra and "signature" in chunk.reasoning.extra:
                reply.reasoning.extra["signature"] = chunk.reasoning.extra["signature"]
                break

    # the trace are ignored in _convert_streaming_chunks_to_chat_message
    # so we need to process them separately
    last_chunk = chunks[-1] if chunks else None
    if last_chunk and last_chunk.meta and "trace" in last_chunk.meta:
        reply.meta["trace"] = last_chunk.meta["trace"]

    return [reply]


async def _parse_streaming_response_async(
    response_stream: EventStream,
    streaming_callback: AsyncStreamingCallbackT,
    model: str,
    component_info: ComponentInfo,
) -> list[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :param component_info: ComponentInfo object
    :return: List of ChatMessage objects
    """
    content_block_idxs = set()
    chunks: list[StreamingChunk] = []
    async for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        content_block_idx = streaming_chunk.index
        if content_block_idx is not None and content_block_idx not in content_block_idxs:
            streaming_chunk.start = True
            content_block_idxs.add(content_block_idx)
        await streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)

    replies = _convert_chunks_to_messages(chunks)

    return replies


def _validate_guardrail_config(guardrail_config: dict[str, str] | None = None, streaming: bool = False) -> None:
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
