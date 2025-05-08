import json
from typing import Any, Dict, List, Optional, Tuple, Union

from datetime import datetime
from botocore.eventstream import EventStream
from haystack import logging
from haystack.dataclasses import (
    ChatMessage, ChatRole, SyncStreamingCallbackT, AsyncStreamingCallbackT, StreamingChunk, ToolCall
)
from haystack.tools import Tool

logger = logging.getLogger(__name__)


def _format_tools(tools: Optional[List[Tool]] = None) -> Optional[Dict[str, Any]]:
    """
    Format Haystack Tool(s) to Amazon Bedrock toolConfig format.

    :param tools: List of Tool objects to format
    :return: Dictionary in Bedrock toolConfig format or None if no tools
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
    content = []
    # Tool call message can contain text
    if tool_call_message.text:
        content.append({"text": tool_call_message.text})

    for tool_call in tool_call_message.tool_calls:
        content.append(
            {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.tool_name, "input": tool_call.arguments}}
        )
    return {"role": tool_call_message.role.value, "content": content}


def _format_tool_result_message(tool_call_result_message: ChatMessage) -> Dict[str, Any]:
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
    tool_call_messages = []
    tool_result_messages = []
    for idx, msg in enumerate(bedrock_formatted_messages):
        content = msg.get("content", None)
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


def _format_messages(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Format a list of ChatMessages to the format expected by Bedrock API.
    Separates system messages and handles tool results and tool calls.

    :param messages: List of ChatMessages to format
    :return: Tuple of (system_prompts, non_system_messages) in Bedrock format
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
            # regular user or assistant messages with only text content
            bedrock_formatted_messages.append({"role": msg.role.value, "content": [{"text": msg.text}]})

    repaired_bedrock_formatted_messages = _repair_tool_result_messages(bedrock_formatted_messages)
    return system_prompts, repaired_bedrock_formatted_messages


def _parse_completion_response(response_body: Dict[str, Any], model: str) -> List[ChatMessage]:
    """
    Parse a Bedrock response to a list of ChatMessage objects.

    :param response_body: Raw response from Bedrock API
    :param model: The model ID used for generation
    :return: List of ChatMessage objects
    """
    replies = []
    if "output" in response_body and "message" in response_body["output"]:
        message = response_body["output"]["message"]
        if message["role"] == "assistant":
            content_blocks = message["content"]

            # Common meta information
            base_meta = {
                "model": model,
                "index": 0,
                "finish_reason": response_body.get("stopReason"),
                "usage": {
                    # OpenAI's format for usage for cross ChatGenerator compatibility
                    "prompt_tokens": response_body.get("usage", {}).get("inputTokens", 0),
                    "completion_tokens": response_body.get("usage", {}).get("outputTokens", 0),
                    "total_tokens": response_body.get("usage", {}).get("totalTokens", 0),
                },
            }

            # Process all content blocks and combine them into a single message
            text_content = []
            tool_calls = []
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

            # Create a single ChatMessage with combined text and tool calls
            replies.append(ChatMessage.from_assistant(" ".join(text_content), tool_calls=tool_calls, meta=base_meta))

    return replies


def _convert_content_blocks_to_chat_messages(
    content_blocks: Dict[str, Any],
    model: str,
    finish_reason: Optional[str],
    usage: Optional[Dict[str, Any]],
) -> List[ChatMessage]:
    ordered_content_blocks = sorted(content_blocks.items(), key=lambda x: x[0])
    tool_calls = []
    for _, tool_call in ordered_content_blocks:
        if isinstance(tool_call, dict):
            try:
                arguments = json.loads(tool_call["arguments"])
                tool_calls.append(ToolCall(id=tool_call["id"], tool_name=tool_call["name"], arguments=arguments))
            except json.JSONDecodeError:
                logger.warning(
                    "Amazon Bedrock returned a malformed JSON string for tool call arguments. This tool call will be "
                    "skipped. Tool call ID: {tool_id}, Tool name: {tool_name}, Arguments: {tool_arguments}",
                    tool_id=tool_call["id"],
                    tool_name=tool_call["name"],
                    tool_arguments=tool_call["arguments"],
                )
    replies = [
        ChatMessage.from_assistant(
            text="".join([content for _, content in ordered_content_blocks if isinstance(content, str)]),
            tool_calls=tool_calls,
            meta={
                "model": model,
                "index": 0,
                "finish_reason": finish_reason,
                "usage": usage,
            },
        )
    ]
    return replies


def _convert_event_to_streaming_chunk(event: Dict[str, Any], model: str) -> StreamingChunk:
    # We ignore messageStart and contentBlockStop events for now
    streaming_chunk = StreamingChunk(
        content="",
        meta={
            "model": model,
            "index": 0,
            "tool_calls": [],
            "finish_reason": None,
            "received_at": datetime.now().isoformat(),
        }
    )

    if "contentBlockStart" in event:
        block_start = event["contentBlockStart"]
        if "start" in block_start and "toolUse" in block_start["start"]:
            tool_start = block_start["start"]["toolUse"]
            streaming_chunk = StreamingChunk(
                content="",
                meta={
                    "model": model,
                    "index": 0,
                    # TODO Check what the expected format of the tool calls is
                    # "tool_calls": choice.delta.tool_calls,
                    "tool_calls": {
                        "id": tool_start["toolUseId"],
                        "name": tool_start["name"],
                        "arguments": "",  # Will accumulate deltas as string
                    },
                    "finish_reason": None,
                    "received_at": datetime.now().isoformat(),
                }
            )

    elif "contentBlockDelta" in event:
        block_idx = event["contentBlockDelta"]["contentBlockIndex"]
        delta = event["contentBlockDelta"]["delta"]
        # This is for accumulating text deltas
        if "text" in delta:
            streaming_chunk = StreamingChunk(
                content=delta["text"],
                meta={
                    "model": model,
                    "index": block_idx,
                    # TODO Check what empty tool_calls should be
                    "tool_calls": [],
                    "finish_reason": None,
                    "received_at": datetime.now().isoformat(),
                }
            )
        # This only occurs when accumulating the arguments for a toolUse
        # The content_block for this tool should already exist at this point
        elif "toolUse" in delta:
            streaming_chunk = StreamingChunk(
                # TODO This shouldn't be put into content but into tool_calls meta of the StreamingChunk
                content=delta["toolUse"].get("input", ""),
                meta={
                    "model": model,
                    "index": block_idx,
                    # TODO Check what empty tool_calls should be
                    "tool_calls": [],
                    "finish_reason": None,
                    "received_at": datetime.now().isoformat(),
                }
            )

    elif "messageStop" in event:
        finish_reason = event["messageStop"].get("stopReason")
        streaming_chunk = StreamingChunk(
            content="",
            meta={
                "model": model,
                "index": 0,
                # TODO Check what empty tool_calls should be
                "tool_calls": [],
                "finish_reason": finish_reason,
                "received_at": datetime.now().isoformat(),
            }
        )

    elif "metadata" in event and "usage" in event["metadata"]:
        # TODO Do I need this one to be in the StreamingChunk?
        metadata = event["metadata"]
        usage = {
            "prompt_tokens": metadata["usage"].get("inputTokens", 0),
            "completion_tokens": metadata["usage"].get("outputTokens", 0),
            "total_tokens": metadata["usage"].get("totalTokens", 0),
        }
        streaming_chunk = StreamingChunk(
            content="",
            meta={
                "model": model,
                "index": 0,
                # TODO Check what empty tool_calls should be
                "tool_calls": [],
                "finish_reason": None,
                "received_at": datetime.now().isoformat(),
            }
        )

    return streaming_chunk


def _convert_event_to_content_blocks(
    event: Dict[str, Any],
    current_content_blocks: Dict[str, Any]
) -> Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    finish_reason = None
    usage = None

    # We ignore messageStart and contentBlockStop events for now
    if "contentBlockStart" in event:
        block_start = event["contentBlockStart"]
        if "start" in block_start and "toolUse" in block_start["start"]:
            tool_start = block_start["start"]["toolUse"]
            current_content_blocks[block_start["contentBlockIndex"]] = {
                "id": tool_start["toolUseId"],
                "name": tool_start["name"],
                "arguments": "",  # Will accumulate deltas as string
            }

    elif "contentBlockDelta" in event:
        block_idx = event["contentBlockDelta"]["contentBlockIndex"]
        delta = event["contentBlockDelta"]["delta"]
        # This is for accumulating text deltas
        if "text" in delta:
            if block_idx not in current_content_blocks:
                current_content_blocks[block_idx] = delta["text"]
            else:
                current_content_blocks[block_idx] += delta["text"]
        # This only occurs when accumulating the arguments for a toolUse
        # The content_block for this tool should already exist at this point
        elif "toolUse" in delta:
            current_content_blocks[block_idx]["arguments"] += delta["toolUse"].get("input", "")

    elif "messageStop" in event:
        finish_reason = event["messageStop"].get("stopReason")

    elif "metadata" in event and "usage" in event["metadata"]:
        metadata = event["metadata"]
        usage = {
            "prompt_tokens": metadata["usage"].get("inputTokens", 0),
            "completion_tokens": metadata["usage"].get("outputTokens", 0),
            "total_tokens": metadata["usage"].get("totalTokens", 0),
        }

    return current_content_blocks, finish_reason, usage


def _convert_response_stream_to_content_blocks(
    response_stream: EventStream,
    streaming_callback: SyncStreamingCallbackT,
) -> Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    # TODO Convert each event into the corresponding StreamingChunk
    #      Match the same format as OpenAIChatGenerator
    final_usage = None
    final_finish_reason = None
    content_blocks = {}
    for event in response_stream:
        content_blocks, finish_reason, usage = _convert_event_to_content_blocks(
            event=event, current_content_blocks=content_blocks
        )
        if finish_reason is not None:
            final_finish_reason = finish_reason
        if usage is not None:
            final_usage = usage

    return content_blocks, final_finish_reason, final_usage


def _parse_streaming_response(
    response_stream: EventStream,
    streaming_callback: SyncStreamingCallbackT,
    model: str,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :return: List of ChatMessage objects
    """
    content_blocks, finish_reason, usage = _convert_response_stream_to_content_blocks(
        response_stream=response_stream,
        streaming_callback=streaming_callback,
    )
    replies = _convert_content_blocks_to_chat_messages(
        content_blocks=content_blocks,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
    )
    return replies


async def _convert_response_stream_to_content_blocks_async(
    response_stream: EventStream,
    streaming_callback: AsyncStreamingCallbackT,
) -> Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    # TODO Convert each event into the corresponding StreamingChunk
    #      Match the same format as OpenAIChatGenerator
    usage = None
    finish_reason = None
    content_blocks = {}
    async for event in response_stream:
        content_blocks, finish_reason, usage = _convert_event_to_content_blocks(
            event=event, current_content_blocks=content_blocks
        )

    return content_blocks, finish_reason, usage


async def _parse_streaming_response_async(
    response_stream: EventStream,
    streaming_callback: AsyncStreamingCallbackT,
    model: str,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :return: List of ChatMessage objects
    """
    content_blocks, finish_reason, usage = await _convert_response_stream_to_content_blocks_async(
        response_stream=response_stream,
        streaming_callback=streaming_callback,
    )

    replies = _convert_content_blocks_to_chat_messages(
        content_blocks=content_blocks,
        model=model,
        finish_reason=finish_reason,
        usage=usage,
    )
    return replies
