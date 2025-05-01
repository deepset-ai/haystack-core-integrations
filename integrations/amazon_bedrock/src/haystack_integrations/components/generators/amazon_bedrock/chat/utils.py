import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from botocore.eventstream import EventStream
from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT, StreamingChunk, ToolCall
from haystack.tools import Tool


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


def _format_messages(messages: List[ChatMessage]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Format a list of ChatMessages to the format expected by Bedrock API.
    Separates system messages and handles tool results and tool calls.

    :param messages: List of ChatMessages to format
    :return: Tuple of (system_prompts, non_system_messages) in Bedrock format
    """
    system_prompts = []
    non_system_messages = []

    for msg in messages:
        if msg.is_from(ChatRole.SYSTEM):
            system_prompts.append({"text": msg.text})
            continue

        # Handle tool results - must role these as user messages
        if msg.tool_call_results:
            tool_results = []
            for result in msg.tool_call_results:
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
            non_system_messages.append({"role": "user", "content": tool_results})
            continue

        content = []
        # Handle text content
        if msg.text:
            content.append({"text": msg.text})

        # Handle tool calls
        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                content.append(
                    {"toolUse": {"toolUseId": tool_call.id, "name": tool_call.tool_name, "input": tool_call.arguments}}
                )

        if content:  # Only add message if it has content
            non_system_messages.append({"role": msg.role.value, "content": content})

    return system_prompts, non_system_messages


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


def _parse_streaming_response(
    response_stream: EventStream,
    streaming_callback: Callable[[StreamingChunk], None],
    model: str,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :return: List of ChatMessage objects
    """
    replies = []
    current_content = ""
    current_tool_call: Optional[Dict[str, Any]] = None
    base_meta = {"model": model, "index": 0}

    for event in response_stream:
        if "contentBlockStart" in event:
            # Reset accumulators for new message
            current_content = ""
            current_tool_call = None
            block_start = event["contentBlockStart"]
            if "start" in block_start and "toolUse" in block_start["start"]:
                tool_start = block_start["start"]["toolUse"]
                current_tool_call = {
                    "id": tool_start["toolUseId"],
                    "name": tool_start["name"],
                    "arguments": "",  # Will accumulate deltas as string
                }

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                delta_text = delta["text"]
                current_content += delta_text
                streaming_chunk = StreamingChunk(content=delta_text, meta={})
                streaming_callback(streaming_chunk)
            elif "toolUse" in delta and current_tool_call:
                # Accumulate tool use input deltas
                current_tool_call["arguments"] += delta["toolUse"].get("input", "")

        elif "contentBlockStop" in event:
            if current_tool_call:
                # Parse accumulated input if it's a JSON string
                try:
                    input_json = json.loads(current_tool_call["arguments"])
                    current_tool_call["arguments"] = input_json
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass

                tool_call = ToolCall(
                    id=current_tool_call["id"],
                    tool_name=current_tool_call["name"],
                    arguments=current_tool_call["arguments"],
                )
                replies.append(ChatMessage.from_assistant("", tool_calls=[tool_call], meta=base_meta.copy()))
            elif current_content:
                replies.append(ChatMessage.from_assistant(current_content, meta=base_meta.copy()))

        elif "messageStop" in event:
            # Update finish reason for all replies
            for reply in replies:
                reply.meta["finish_reason"] = event["messageStop"].get("stopReason")

        elif "metadata" in event:
            metadata = event["metadata"]
            # Update usage stats for all replies
            for reply in replies:
                if "usage" in metadata:
                    usage = metadata["usage"]
                    reply.meta["usage"] = {
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                    }

    return replies


async def _parse_streaming_response_async(
    response_stream: EventStream,
    streaming_callback: StreamingCallbackT,
    model: str,
) -> List[ChatMessage]:
    """
    Parse a streaming response from Bedrock.

    :param response_stream: EventStream from Bedrock API
    :param streaming_callback: Callback for streaming chunks
    :param model: The model ID used for generation
    :return: List of ChatMessage objects
    """
    replies = []
    current_content = ""
    current_tool_call: Optional[Dict[str, Any]] = None
    base_meta = {"model": model, "index": 0}

    async for event in response_stream:
        if "contentBlockStart" in event:
            # Reset accumulators for new message
            current_content = ""
            current_tool_call = None
            block_start = event["contentBlockStart"]
            if "start" in block_start and "toolUse" in block_start["start"]:
                tool_start = block_start["start"]["toolUse"]
                current_tool_call = {
                    "id": tool_start["toolUseId"],
                    "name": tool_start["name"],
                    "arguments": "",  # Will accumulate deltas as string
                }

        elif "contentBlockDelta" in event:
            delta = event["contentBlockDelta"]["delta"]
            if "text" in delta:
                delta_text = delta["text"]
                current_content += delta_text
                streaming_chunk = StreamingChunk(content=delta_text, meta={})
                await streaming_callback(streaming_chunk)
            elif "toolUse" in delta and current_tool_call:
                # Accumulate tool use input deltas
                current_tool_call["arguments"] += delta["toolUse"].get("input", "")

        elif "contentBlockStop" in event:
            if current_tool_call:
                # Parse accumulated input if it's a JSON string
                try:
                    input_json = json.loads(current_tool_call["arguments"])
                    current_tool_call["arguments"] = input_json
                except json.JSONDecodeError:
                    # Keep as string if not valid JSON
                    pass

                tool_call = ToolCall(
                    id=current_tool_call["id"],
                    tool_name=current_tool_call["name"],
                    arguments=current_tool_call["arguments"],
                )
                replies.append(ChatMessage.from_assistant("", tool_calls=[tool_call], meta=base_meta.copy()))
            elif current_content:
                replies.append(ChatMessage.from_assistant(current_content, meta=base_meta.copy()))

        elif "messageStop" in event:
            # Update finish reason for all replies
            for reply in replies:
                reply.meta["finish_reason"] = event["messageStop"].get("stopReason")

        elif "metadata" in event:
            metadata = event["metadata"]
            # Update usage stats for all replies
            for reply in replies:
                if "usage" in metadata:
                    usage = metadata["usage"]
                    reply.meta["usage"] = {
                        "prompt_tokens": usage.get("inputTokens", 0),
                        "completion_tokens": usage.get("outputTokens", 0),
                        "total_tokens": usage.get("totalTokens", 0),
                    }

    return replies
