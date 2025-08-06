import base64
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from botocore.eventstream import EventStream
from haystack import logging
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ChatRole,
    ComponentInfo,
    ImageContent,
    StreamingChunk,
    SyncStreamingCallbackT,
    TextContent,
    ToolCall,
)
from haystack.tools import Tool

logger = logging.getLogger(__name__)


# see https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html for supported formats
IMAGE_SUPPORTED_FORMATS = ["png", "jpeg", "gif", "webp"]


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
    if reasoning_contents := tool_call_message.meta.get("reasoning_contents"):
        content.extend(_format_reasoning_content(reasoning_contents=reasoning_contents))

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


def _format_reasoning_content(reasoning_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format reasoning content to match Bedrock's expected structure.

    :param reasoning_contents: List of reasoning content dictionaries from Haystack ChatMessage metadata.
    :returns: Formatted reasoning content dictionary.
    """
    formatted_contents = []
    for reasoning_content in reasoning_contents:
        formatted_content = {"reasoningContent": reasoning_content["reasoning_content"]}
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
    if message.meta.get("reasoning_contents"):
        bedrock_content_blocks.extend(_format_reasoning_content(reasoning_contents=message.meta["reasoning_contents"]))

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

            # If reasoning contents were found, add them to the base meta
            base_meta.update({"reasoning_contents": reasoning_contents})

            # Create a single ChatMessage with combined text and tool calls
            replies.append(ChatMessage.from_assistant(" ".join(text_content), tool_calls=tool_calls, meta=base_meta))

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
    streaming_chunk = StreamingChunk(
        content="", meta={"model": model, "received_at": datetime.now(timezone.utc).isoformat()}
    )

    if "contentBlockStart" in event:
        # contentBlockStart always has the key "contentBlockIndex"
        block_start = event["contentBlockStart"]
        block_idx = block_start["contentBlockIndex"]
        if "start" in block_start and "toolUse" in block_start["start"]:
            tool_start = block_start["start"]["toolUse"]
            streaming_chunk = StreamingChunk(
                content="",
                meta={
                    "model": model,
                    "index": 0,
                    # We follow the same format used in the OpenAIChatGenerator
                    "tool_calls": [  # Optional[List[ChoiceDeltaToolCall]]
                        {
                            "index": block_idx,  # int
                            "id": tool_start["toolUseId"],  # Optional[str]
                            "function": {  # Optional[ChoiceDeltaToolCallFunction]
                                # Will accumulate deltas as string
                                "arguments": "",  # Optional[str]
                                "name": tool_start["name"],  # Optional[str]
                            },
                            "type": "function",  # Optional[Literal["function"]]
                        }
                    ],
                    "finish_reason": None,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                },
            )

    elif "contentBlockDelta" in event:
        # contentBlockDelta always has the key "contentBlockIndex" and "delta"
        block_idx = event["contentBlockDelta"]["contentBlockIndex"]
        delta = event["contentBlockDelta"]["delta"]
        # This is for accumulating text deltas
        if "text" in delta:
            streaming_chunk = StreamingChunk(
                content=delta["text"],
                meta={
                    "model": model,
                    "index": 0,
                    "tool_calls": None,
                    "finish_reason": None,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        # This only occurs when accumulating the arguments for a toolUse
        # The content_block for this tool should already exist at this point
        elif "toolUse" in delta:
            streaming_chunk = StreamingChunk(
                content="",
                meta={
                    "model": model,
                    "index": 0,
                    "tool_calls": [  # Optional[List[ChoiceDeltaToolCall]]
                        {
                            "index": block_idx,  # int
                            "id": None,  # Optional[str]
                            "function": {  # Optional[ChoiceDeltaToolCallFunction]
                                # Will accumulate deltas as string
                                "arguments": delta["toolUse"].get("input", ""),  # Optional[str]
                                "name": None,  # Optional[str]
                            },
                            "type": "function",  # Optional[Literal["function"]]
                        }
                    ],
                    "finish_reason": None,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        # This is for accumulating reasoning content deltas
        elif "reasoningContent" in delta:
            reasoning_content = delta["reasoningContent"]
            if "redactedContent" in reasoning_content:
                reasoning_content["redacted_content"] = reasoning_content.pop("redactedContent")
            streaming_chunk = StreamingChunk(
                content="",
                meta={
                    "model": model,
                    "index": 0,
                    "tool_calls": None,
                    "finish_reason": None,
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "reasoning_contents": [{"index": block_idx, "reasoning_content": reasoning_content}],
                },
            )

    elif "messageStop" in event:
        finish_reason = event["messageStop"].get("stopReason")
        streaming_chunk = StreamingChunk(
            content="",
            meta={
                "model": model,
                # This is always 0 b/c it represents the choice index
                "index": 0,
                "tool_calls": None,
                "finish_reason": finish_reason,
                "received_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    elif "metadata" in event and "usage" in event["metadata"]:
        metadata = event["metadata"]
        streaming_chunk = StreamingChunk(
            content="",
            meta={
                "model": model,
                # This is always 0 b/c it represents the choice index
                "index": 0,
                "tool_calls": None,
                "finish_reason": None,
                "received_at": datetime.now(timezone.utc).isoformat(),
                "usage": {
                    "prompt_tokens": metadata["usage"].get("inputTokens", 0),
                    "completion_tokens": metadata["usage"].get("outputTokens", 0),
                    "total_tokens": metadata["usage"].get("totalTokens", 0),
                },
            },
        )

    streaming_chunk.component_info = component_info

    return streaming_chunk


def _process_reasoning_contents(chunks: List[StreamingChunk]) -> List[Dict[str, Any]]:
    """
    Process reasoning contents from a list of StreamingChunk objects into the Bedrock expected format.

    :param chunks: List of StreamingChunk objects potentially containing reasoning contents.
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

    return formatted_reasoning_contents


def _convert_streaming_chunks_to_chat_message(chunks: List[StreamingChunk]) -> ChatMessage:
    """
    Converts a list of streaming chunks into a ChatMessage object.

    The function processes streaming chunks to build a ChatMessage object, including extracting and constructing
    tool calls, managing metadata such as model type, finish reason, and usage information.
    The tool call processing handles accumulating data across the chunks and attempts to parse JSON-formatted
    arguments for tool calls.

    :param chunks: A list of StreamingChunk objects representing parts of the assistant's response.

    :returns:
        A ChatMessage object constructed from the streaming chunks, containing the aggregated text, processed tool
        calls, and metadata.
    """
    # Join all text content from the chunks
    text = "".join([chunk.content for chunk in chunks])

    # If reasoning content is present in any chunk, accumulate it
    reasoning_contents = _process_reasoning_contents(chunks=chunks)

    # Process tool calls if present in any chunk
    tool_calls = []
    tool_call_data: Dict[int, Dict[str, str]] = {}  # Track tool calls by index
    for chunk_payload in chunks:
        tool_calls_meta = chunk_payload.meta.get("tool_calls")
        if tool_calls_meta is not None:
            for delta in tool_calls_meta:
                # We use the index of the tool call to track it across chunks since the ID is not always provided
                if delta["index"] not in tool_call_data:
                    tool_call_data[delta["index"]] = {"id": "", "name": "", "arguments": ""}

                # Save the ID if present
                if delta.get("id"):
                    tool_call_data[delta["index"]]["id"] = delta["id"]

                if delta.get("function"):
                    if delta["function"].get("name"):
                        tool_call_data[delta["index"]]["name"] += delta["function"]["name"]
                    if delta["function"].get("arguments"):
                        tool_call_data[delta["index"]]["arguments"] += delta["function"]["arguments"]

    # Convert accumulated tool call data into ToolCall objects
    for call_data in tool_call_data.values():
        try:
            arguments = json.loads(call_data.get("arguments", "{}")) if call_data.get("arguments") else {}
            tool_calls.append(ToolCall(id=call_data["id"], tool_name=call_data["name"], arguments=arguments))
        except json.JSONDecodeError:
            logger.warning(
                "Amazon Bedrock returned a malformed JSON string for tool call arguments. This tool call will be "
                "skipped. Tool call ID: {tool_id}, Tool name: {tool_name}, Arguments: {tool_arguments}",
                tool_id=call_data["id"],
                tool_name=call_data["name"],
                tool_arguments=call_data["arguments"],
            )

    # finish_reason can appear in different places so we look for the last one
    finish_reasons = [
        chunk.meta.get("finish_reason") for chunk in chunks if chunk.meta.get("finish_reason") is not None
    ]
    finish_reason = finish_reasons[-1] if finish_reasons else None

    # usage is usually last but we look for it as well
    usages = [chunk.meta.get("usage") for chunk in chunks if chunk.meta.get("usage") is not None]
    usage = usages[-1] if usages else None

    meta = {
        "model": chunks[-1].meta["model"],
        "index": 0,
        "finish_reason": finish_reason,
        "completion_start_time": chunks[0].meta.get("received_at"),  # first chunk received
        "usage": usage,
        "reasoning_contents": reasoning_contents,
    }

    return ChatMessage.from_assistant(text=text or None, tool_calls=tool_calls, meta=meta)


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
    chunks: List[StreamingChunk] = []
    for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)
    replies = [_convert_streaming_chunks_to_chat_message(chunks=chunks)]
    return replies


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
    chunks: List[StreamingChunk] = []
    async for event in response_stream:
        streaming_chunk = _convert_event_to_streaming_chunk(event=event, model=model, component_info=component_info)
        await streaming_callback(streaming_chunk)
        chunks.append(streaming_chunk)
    replies = [_convert_streaming_chunks_to_chat_message(chunks=chunks)]
    return replies
