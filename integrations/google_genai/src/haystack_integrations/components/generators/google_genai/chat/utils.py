# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from datetime import datetime, timezone
from typing import Any

from google.genai import types
from google.genai.types import GenerateContentResponseUsageMetadata, UsageMetadata
from haystack import logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import (
    ComponentInfo,
    FileContent,
    FinishReason,
    ImageContent,
    StreamingChunk,
    TextContent,
    ToolCall,
    ToolCallDelta,
    ToolCallResult,
)
from haystack.dataclasses.chat_message import ChatMessage, ChatRole, ReasoningContent
from haystack.tools import (
    ToolsType,
    flatten_tools_or_toolsets,
)
from jsonref import replace_refs

logger = logging.getLogger(__name__)

# Mapping from Google GenAI finish reasons to Haystack FinishReason values
FINISH_REASON_MAPPING: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "IMAGE_SAFETY": "content_filter",
}

# Google GenAI supported image MIME types based on documentation
# https://ai.google.dev/gemini-api/docs/image-understanding?lang=python#supported-formats
GOOGLE_GENAI_SUPPORTED_MIME_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",  # Common alias
    "image/webp": "webp",
    "image/heic": "heic",
    "image/heif": "heif",
}


def _process_thinking_config(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Process thinking configuration from generation_kwargs.

    Does not mutate the input dict; returns a new dict with thinking_config
    applied when applicable. Supports explicit ``include_thoughts`` in
    generation_kwargs to override the default derived from thinking_budget
    or thinking_level.

    :param generation_kwargs: The generation configuration dictionary.
    :returns: A new dict with thinking_config if applicable; caller's dict is unchanged.
    """
    generation_kwargs = dict(generation_kwargs)
    # Extract include_thoughts from generation_kwargs if explicitly set by the user.
    # This must be popped before creating ThinkingConfig so it doesn't leak as an unknown kwarg.
    explicit_include_thoughts = generation_kwargs.pop("include_thoughts", None)

    if "thinking_budget" in generation_kwargs:
        thinking_budget = generation_kwargs.pop("thinking_budget")

        # Basic type validation
        if not isinstance(thinking_budget, int):
            logger.warning(
                f"Invalid thinking_budget type: {type(thinking_budget)}. Expected int, using dynamic allocation."
            )
            # fall back to default: dynamic thinking budget allocation
            thinking_budget = -1

        # Determine include_thoughts: respect explicit user override, otherwise auto-derive
        if explicit_include_thoughts is not None:
            include_thoughts = explicit_include_thoughts
        else:
            # When thinking_budget is 0, thinking is disabled so include_thoughts must be False
            include_thoughts = thinking_budget != 0

        thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=include_thoughts)
        generation_kwargs["thinking_config"] = thinking_config

    if "thinking_level" in generation_kwargs:
        thinking_level = generation_kwargs.pop("thinking_level")

        # Basic type validation
        if not isinstance(thinking_level, str):
            logger.warning(
                f"Invalid thinking_level type: {type(thinking_level).__name__}. Expected str, "
                f"falling back to THINKING_LEVEL_UNSPECIFIED."
            )
            thinking_level = types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
        else:
            # Convert to uppercase for case-insensitive matching
            thinking_level_upper = thinking_level.upper()

            # Check if the uppercase value is a valid ThinkingLevel enum member
            valid_levels = [level.value for level in types.ThinkingLevel]
            if thinking_level_upper not in valid_levels:
                logger.warning(
                    f"Invalid thinking_level value: '{thinking_level}'. "
                    f"Must be one of: {valid_levels} (case-insensitive). "
                    "Falling back to THINKING_LEVEL_UNSPECIFIED."
                )
                thinking_level = types.ThinkingLevel.THINKING_LEVEL_UNSPECIFIED
            else:
                # Parse valid string to ThinkingLevel enum
                thinking_level = types.ThinkingLevel(thinking_level_upper)

        # Determine include_thoughts: respect explicit user override, otherwise auto-derive
        if explicit_include_thoughts is not None:
            include_thoughts = explicit_include_thoughts
        else:
            include_thoughts = thinking_level != types.ThinkingLevel.MINIMAL

        thinking_config = types.ThinkingConfig(thinking_level=thinking_level, include_thoughts=include_thoughts)
        generation_kwargs["thinking_config"] = thinking_config

    return generation_kwargs


def remove_key_from_schema(
    schema: dict[str, Any] | list[Any] | Any, target_key: str
) -> dict[str, Any] | list[Any] | Any:
    """
    Recursively traverse a schema and remove all occurrences of the target key.


    :param schema: The schema dictionary/list/value to process
    :param target_key: The key to remove from all dictionaries in the schema

    :returns: The schema with the target key removed from all nested dictionaries
    """
    if isinstance(schema, dict):
        # Create a new dict without the target key
        result = {}
        for k, v in schema.items():
            if k != target_key:
                result[k] = remove_key_from_schema(v, target_key)
        return result

    elif isinstance(schema, list):
        return [remove_key_from_schema(item, target_key) for item in schema]

    return schema


def _sanitize_tool_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitizes a tool schema to remove any keys that are not supported by Google Gen AI.

    Google Gen AI does not support additionalProperties, $schema, $defs, or $ref in the tool schema.

    :param tool_schema: The tool schema to sanitize.
    :returns: The sanitized tool schema.
    """
    # google Gemini does not support additionalProperties and $schema in the tool schema
    sanitized_schema = remove_key_from_schema(tool_schema, "additionalProperties")
    sanitized_schema = remove_key_from_schema(sanitized_schema, "$schema")
    # expand $refs in the tool schema
    expanded_schema = replace_refs(sanitized_schema)
    # and remove the $defs key leaving the rest of the schema
    final_schema = remove_key_from_schema(expanded_schema, "$defs")

    if not isinstance(final_schema, dict):
        msg = "Tool schema must be a dictionary after sanitization"
        raise ValueError(msg)

    return final_schema


def _convert_message_to_google_genai_format(message: ChatMessage) -> types.Content:
    """
    Converts a Haystack ChatMessage to Google Gen AI Content format.

    :param message: The Haystack ChatMessage to convert.
    :returns: Google Gen AI Content object.
    """
    # Check if message has content
    if not message._content:
        msg = "A `ChatMessage` must contain at least one content part."
        raise ValueError(msg)

    parts = []

    # Check if this message has thought signatures from a previous response
    # These need to be reconstructed in their original part structure
    thought_signatures = message.meta.get("thought_signatures", []) if message.meta else []

    # If we have thought signatures, we need to reconstruct the exact part structure
    # from the previous assistant response to maintain multi-turn thinking context
    if thought_signatures and message.is_from(ChatRole.ASSISTANT):
        # Track which tool calls we've used (to handle multiple tool calls)
        tool_call_index = 0

        # Reconstruct parts with their original thought signatures
        for sig_info in thought_signatures:
            part_dict: dict[str, Any] = {}

            # Check what type of content this part had
            if sig_info.get("has_text"):
                # Find the corresponding text content
                if sig_info.get("is_thought"):
                    # This was a thought part - find it in reasoning content
                    if message.reasoning:
                        part_dict["text"] = message.reasoning.reasoning_text
                        part_dict["thought"] = True
                else:
                    # Regular text part
                    part_dict["text"] = message.text or ""

            if sig_info.get("has_function_call"):
                # Find the corresponding tool call by index
                if message.tool_calls and tool_call_index < len(message.tool_calls):
                    tool_call = message.tool_calls[tool_call_index]
                    part_dict["function_call"] = types.FunctionCall(
                        id=tool_call.id, name=tool_call.tool_name, args=tool_call.arguments
                    )
                    tool_call_index += 1  # Move to next tool call for next part

            # Add the thought signature to preserve context
            part_dict["thought_signature"] = sig_info["signature"]

            parts.append(types.Part(**part_dict))

        # If we reconstructed from signatures, we're done
        if parts:
            role = "model"  # Assistant messages with signatures are always from the model
            return types.Content(role=role, parts=parts)

    # Standard processing for messages without thought signatures
    for content_part in message._content:
        if isinstance(content_part, TextContent):
            # Only add text parts that are not empty to avoid unnecessary empty text parts
            if content_part.text.strip():
                parts.append(types.Part(text=content_part.text))

        elif isinstance(content_part, (ImageContent, FileContent)):
            cls_name = content_part.__class__.__name__

            if not message.is_from(ChatRole.USER):
                msg = f"{cls_name} is only supported for user messages"
                raise ValueError(msg)

            # MIME type validation: must be provided and, for images, one of the supported types
            if not content_part.mime_type:
                msg = f"MIME type is required to use {cls_name} with GoogleGenAIChatGenerator"
                raise ValueError(msg)

            if (
                isinstance(content_part, ImageContent)
                and content_part.mime_type not in GOOGLE_GENAI_SUPPORTED_MIME_TYPES
            ):
                supported_types = list(GOOGLE_GENAI_SUPPORTED_MIME_TYPES.keys())
                msg = (
                    f"Unsupported image MIME type: {content_part.mime_type}. "
                    f"Google AI supports the following MIME types: {supported_types}"
                )
                raise ValueError(msg)

            # Use inline data approach
            try:
                base64_data = (
                    content_part.base64_data if isinstance(content_part, FileContent) else content_part.base64_image
                )
                bytes_data = base64.b64decode(base64_data)

                file_part = types.Part.from_bytes(data=bytes_data, mime_type=content_part.mime_type)
                parts.append(file_part)

            except Exception as e:
                msg = f"Failed to process {cls_name} data: {e}"
                raise RuntimeError(msg) from e

        elif isinstance(content_part, ToolCall):
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        id=content_part.id, name=content_part.tool_name, args=content_part.arguments
                    )
                )
            )

        elif isinstance(content_part, ToolCallResult):
            if isinstance(content_part.result, str):
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=content_part.origin.id,
                            name=content_part.origin.tool_name,
                            response={"result": content_part.result},
                        )
                    )
                )
            elif isinstance(content_part.result, list):
                tool_call_result_parts: list[types.FunctionResponsePart] = []
                for item in content_part.result:
                    if isinstance(item, TextContent):
                        tool_call_result_parts.append(
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    data=item.text.encode("utf-8"), mime_type="text/plain"
                                ),
                            )
                        )
                    elif isinstance(item, ImageContent):
                        tool_call_result_parts.append(
                            types.FunctionResponsePart(
                                inline_data=types.FunctionResponseBlob(
                                    data=base64.b64decode(item.base64_image), mime_type=item.mime_type
                                ),
                            )
                        )
                    else:
                        msg = (
                            "Unsupported content type in tool call result list. "
                            "Only TextContent and ImageContent are supported."
                        )
                        raise ValueError(msg)
                parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=content_part.origin.id,
                            name=content_part.origin.tool_name,
                            parts=tool_call_result_parts,
                            # the response field is mandatory, but in this case the LLM just needs multimodal parts
                            response={"result": ""},
                        )
                    )
                )
            else:
                msg = "Unsupported content type in tool call result"
                raise ValueError(msg)
        elif isinstance(content_part, ReasoningContent):
            # Reasoning content is for human transparency only, not for maintaining LLM context
            # Thought signatures (stored in message.meta) handle context preservation
            # Leave this here so we don't implement reasoning content handling in the future accidentally
            pass

    # Determine role
    if message.is_from(ChatRole.USER) or message.tool_call_results:
        role = "user"
    elif message.is_from(ChatRole.ASSISTANT):
        role = "model"
    elif message.is_from(ChatRole.SYSTEM):
        # System messages will be handled separately as system instruction
        # When we convert a list of ChatMessage to be sent to google genai,
        # we need to handle system messages separately as system instruction and we only take the first message
        # as the system instruction - if it is present.
        #
        # If we find any additional system messages, we will treat them as user messages
        role = "user"
    else:
        msg = f"Unsupported message role: {message._role}"
        raise ValueError(msg)

    return types.Content(role=role, parts=parts)


def _convert_tools_to_google_genai_format(tools: ToolsType) -> list[types.Tool]:
    """
    Converts a list of Haystack Tools, Toolsets, or a mix to Google Gen AI Tool format.

    :param tools: List of Haystack Tool and/or Toolset objects, or a single Toolset.
    :returns: List of Google Gen AI Tool objects.
    """
    # Flatten Tools and Toolsets into a single list of Tools
    flattened_tools = flatten_tools_or_toolsets(tools)

    function_declarations: list[types.FunctionDeclaration] = []
    for tool in flattened_tools:
        parameters = _sanitize_tool_schema(tool.parameters)
        function_declarations.append(
            types.FunctionDeclaration(
                name=tool.name, description=tool.description, parameters=types.Schema(**parameters)
            )
        )

    # Return a single Tool object with all function declarations as in the Google GenAI docs
    # we could also return multiple Tool objects, doesn't seem to make a difference
    # revisit this decision
    return [types.Tool(function_declarations=function_declarations)]


def _convert_usage_metadata_to_serializable(
    usage_metadata: UsageMetadata | GenerateContentResponseUsageMetadata | None,
) -> dict[str, Any]:
    """Build a JSON-serializable usage dict from a UsageMetadata object.

    Iterates over known UsageMetadata attribute names and adds each non-None value
    in serialized form. Full list of fields: https://ai.google.dev/api/generate-content#UsageMetadata
    """

    def serialize(val: Any) -> Any:
        if val is None:
            return None
        if isinstance(val, (str, int, float, bool)):
            return val
        if isinstance(val, list):
            return [serialize(item) for item in val]
        token_count = getattr(val, "token_count", None) or getattr(val, "tokenCount", None)
        if hasattr(val, "modality") and token_count is not None:
            mod = getattr(val, "modality", None)
            mod_str = getattr(mod, "value", getattr(mod, "name", str(mod))) if mod is not None else None
            return {"modality": mod_str, "token_count": token_count}
        if hasattr(val, "name"):
            return getattr(val, "value", getattr(val, "name", val))
        return val

    if not usage_metadata:
        return {}

    _usage_attr_names = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cache_tokens_details",
        "candidates_tokens_details",
        "prompt_tokens_details",
        "tool_use_prompt_token_count",
        "tool_use_prompt_tokens_details",
    )
    result: dict[str, Any] = {}
    for attr in _usage_attr_names:
        val = getattr(usage_metadata, attr, None)
        if val is not None:
            result[attr] = serialize(val)
    return result


def _convert_google_genai_response_to_chatmessage(response: types.GenerateContentResponse, model: str) -> ChatMessage:
    """
    Converts a Google Gen AI response to a Haystack ChatMessage.

    :param response: The response from Google Gen AI.
    :param model: The model name.
    :returns: A Haystack ChatMessage.
    """
    text_parts = []
    tool_calls = []
    reasoning_parts = []
    thought_signatures = []  # Store thought signatures for multi-turn context

    # Extract text, function calls, thoughts, and thought signatures from response
    finish_reason = None
    if response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        if candidate.content is not None and candidate.content.parts is not None:
            for i, part in enumerate(candidate.content.parts):
                # Check for thought signature on this part
                if hasattr(part, "thought_signature") and part.thought_signature:
                    # Store the thought signature with its part index for reconstruction
                    thought_signatures.append(
                        {
                            "part_index": i,
                            "signature": part.thought_signature,
                            "has_text": part.text is not None,
                            "has_function_call": part.function_call is not None,
                            "is_thought": hasattr(part, "thought") and part.thought,
                        }
                    )

                if part.text is not None and not (hasattr(part, "thought") and part.thought):
                    text_parts.append(part.text)
                if part.function_call is not None:
                    tool_call = ToolCall(
                        tool_name=part.function_call.name or "",
                        arguments=dict(part.function_call.args) if part.function_call.args else {},
                        id=part.function_call.id,
                    )
                    tool_calls.append(tool_call)
                # Handle thought parts for Gemini 2.5 series
                if hasattr(part, "thought") and part.thought:
                    # Extract thought content
                    if part.text:
                        reasoning_parts.append(part.text)

    # Combine text parts
    text = " ".join(text_parts) if text_parts else ""

    usage_metadata = response.usage_metadata

    # Create usage metadata including thinking tokens if available
    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0),
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0),
        "total_tokens": getattr(usage_metadata, "total_token_count", 0),
    }

    # Add thinking token count if available
    if usage_metadata and hasattr(usage_metadata, "thoughts_token_count") and usage_metadata.thoughts_token_count:
        usage["thoughts_token_count"] = usage_metadata.thoughts_token_count

    # Add cached content token count if available (implicit or explicit context caching)
    if (
        usage_metadata
        and hasattr(usage_metadata, "cached_content_token_count")
        and usage_metadata.cached_content_token_count
    ):
        usage["cached_content_token_count"] = usage_metadata.cached_content_token_count

    usage.update(_convert_usage_metadata_to_serializable(usage_metadata))

    # Create meta with reasoning content and thought signatures if available
    meta: dict[str, Any] = {
        "model": model,
        "finish_reason": FINISH_REASON_MAPPING.get(finish_reason or ""),
        "usage": usage,
    }

    # Add thought signatures to meta if present (for multi-turn context preservation)
    if thought_signatures:
        meta["thought_signatures"] = thought_signatures

    # Create ReasoningContent object if there are reasoning parts
    reasoning_content = None
    if reasoning_parts:
        reasoning_text = " ".join(reasoning_parts)
        reasoning_content = ReasoningContent(reasoning_text=reasoning_text)

    # Create ChatMessage
    message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta, reasoning=reasoning_content)

    return message


def _convert_google_chunk_to_streaming_chunk(
    chunk: types.GenerateContentResponse,
    index: int,
    component_info: ComponentInfo,
    model: str,
) -> StreamingChunk:
    """
    Convert a chunk from Google Gen AI to a Haystack StreamingChunk.

    :param chunk: The chunk from Google Gen AI.
    :param index: The index of the chunk.
    :param component_info: The component info.
    :param model: The model name.
    :returns: A StreamingChunk object.
    """
    content = ""
    tool_calls: list[ToolCallDelta] = []
    finish_reason = None
    reasoning_deltas: list[dict[str, str]] = []
    thought_signature_deltas: list[dict[str, Any]] = []  # Track thought signatures in streaming

    if chunk.candidates:
        candidate = chunk.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)

    usage_metadata = chunk.usage_metadata

    usage = {
        "prompt_tokens": getattr(usage_metadata, "prompt_token_count", 0) if usage_metadata else 0,
        "completion_tokens": getattr(usage_metadata, "candidates_token_count", 0) if usage_metadata else 0,
        "total_tokens": getattr(usage_metadata, "total_token_count", 0) if usage_metadata else 0,
    }

    # Add thinking token count if available
    if usage_metadata and hasattr(usage_metadata, "thoughts_token_count") and usage_metadata.thoughts_token_count:
        usage["thoughts_token_count"] = usage_metadata.thoughts_token_count

    if candidate.content and candidate.content.parts:
        tc_index = -1
        for part_index, part in enumerate(candidate.content.parts):
            # Check for thought signature on this part (for multi-turn context)
            if hasattr(part, "thought_signature") and part.thought_signature:
                thought_signature_deltas.append(
                    {
                        "part_index": part_index,
                        "signature": part.thought_signature,
                        "has_text": part.text is not None,
                        "has_function_call": part.function_call is not None,
                        "is_thought": hasattr(part, "thought") and part.thought,
                    }
                )

            if part.text is not None and not (hasattr(part, "thought") and part.thought):
                content += part.text

            elif part.function_call:
                tc_index += 1
                tool_calls.append(
                    ToolCallDelta(
                        # Google GenAI does not provide index, but it is required for tool calls
                        index=tc_index,
                        id=part.function_call.id,
                        tool_name=part.function_call.name or "",
                        arguments=json.dumps(part.function_call.args) if part.function_call.args else None,
                    )
                )

            # Handle thought parts for Gemini 2.5 series
            elif hasattr(part, "thought") and part.thought:
                thought_delta = {
                    "type": "reasoning",
                    "content": part.text if part.text else "",
                }
                reasoning_deltas.append(thought_delta)

    # start is only used by print_streaming_chunk. We try to make a reasonable assumption here but it should not be
    # a problem if we change it in the future.
    start = index == 0 or len(tool_calls) > 0

    # Create meta with reasoning deltas and thought signatures if available
    meta: dict[str, Any] = {
        "received_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "usage": usage,
    }

    # Add reasoning deltas to meta if available
    if reasoning_deltas:
        meta["reasoning_deltas"] = reasoning_deltas

    # Add thought signature deltas to meta if available (for multi-turn context)
    if thought_signature_deltas:
        meta["thought_signature_deltas"] = thought_signature_deltas

    return StreamingChunk(
        content="" if tool_calls else content,  # prioritize tool calls over content when both are present
        tool_calls=tool_calls,
        component_info=component_info,
        index=index,
        start=start,
        finish_reason=FINISH_REASON_MAPPING.get(finish_reason or ""),
        meta=meta,
    )


def _aggregate_streaming_chunks_with_reasoning(chunks: list[StreamingChunk]) -> ChatMessage:
    """
    Aggregate streaming chunks into a final ChatMessage with reasoning content and thought signatures.

    This method extends the standard streaming chunk aggregation to handle Google GenAI's
    specific reasoning content, thinking token usage, and thought signatures for multi-turn context.

    :param chunks: List of streaming chunks to aggregate.
    :returns: Final ChatMessage with aggregated content, reasoning, and thought signatures.
    """

    # Use the generic aggregator for standard content (text, tool calls, basic meta)
    message = _convert_streaming_chunks_to_chat_message(chunks)

    # Now enhance with Google-specific features: reasoning content, thinking token usage, and thought signatures
    reasoning_text_parts: list[str] = []
    thought_signatures: list[dict[str, Any]] = []
    thoughts_token_count = None

    for chunk in chunks:
        # Extract reasoning deltas
        if chunk.meta and "reasoning_deltas" in chunk.meta:
            reasoning_deltas = chunk.meta["reasoning_deltas"]
            if isinstance(reasoning_deltas, list):
                for delta in reasoning_deltas:
                    if delta.get("type") == "reasoning":
                        reasoning_text_parts.append(delta.get("content", ""))

        # Extract thought signature deltas (for multi-turn context preservation)
        if chunk.meta and "thought_signature_deltas" in chunk.meta:
            signature_deltas = chunk.meta["thought_signature_deltas"]
            if isinstance(signature_deltas, list):
                # Aggregate thought signatures - they should come from the final chunks
                # We'll keep the last set of signatures as they represent the complete state
                thought_signatures = signature_deltas

        # Extract thinking token usage (from the last chunk that has it)
        if chunk.meta and "usage" in chunk.meta:
            chunk_usage = chunk.meta["usage"]
            if "thoughts_token_count" in chunk_usage:
                thoughts_token_count = chunk_usage["thoughts_token_count"]

    # Add thinking token count to usage if present
    if thoughts_token_count is not None and "usage" in message.meta:
        if message.meta["usage"] is None:
            message.meta["usage"] = {}
        message.meta["usage"]["thoughts_token_count"] = thoughts_token_count

    # Add thought signatures to meta if present (for multi-turn context preservation)
    if thought_signatures:
        message.meta["thought_signatures"] = thought_signatures

    # If we have reasoning content, reconstruct the message to include it
    # Note: ChatMessage doesn't support adding reasoning after creation, reconstruction is necessary
    if reasoning_text_parts:
        reasoning_content = ReasoningContent(reasoning_text="".join(reasoning_text_parts))
        return ChatMessage.from_assistant(
            text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=reasoning_content
        )

    return message
