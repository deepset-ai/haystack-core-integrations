# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Helpers for the Gemini Interactions API.

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from google.genai.interactions import (
    ArgumentsDelta,
    ErrorEvent,
    Interaction,
    InteractionCompletedEvent,
    InteractionCreatedEvent,
    InteractionSSEEvent,
    InteractionStatusUpdate,
    StepDelta,
    StepStart,
    TextDelta,
    ThoughtSignatureDelta,
    ThoughtSummaryDelta,
    Usage,
)
from haystack import logging
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
from haystack.tools import ToolsType, flatten_tools_or_toolsets
from pydantic import BaseModel

from haystack_integrations.components.generators.google_genai.chat.utils import _sanitize_tool_schema

logger = logging.getLogger(__name__)

# Interactions report an interaction status instead of a finish reason.
# `failed` raises and `cancelled`/`in_progress` have no Haystack equivalent.
STATUS_TO_FINISH_REASON: dict[str, FinishReason] = {
    "completed": "stop",
    "requires_action": "tool_calls",
    "incomplete": "length",
    "budget_exceeded": "length",
}

# `FileContent` is sent as image, audio or video content by MIME type prefix, and as a document otherwise.
# The API validates document MIME types itself: its list grows faster than the SDK's (PDF, CSV, Office and
# OpenDocument formats at the time of writing).
_MEDIA_MIME_PREFIXES = {"image/": "image", "audio/": "audio", "video/": "video"}

# Model steps converted into Haystack content. Other step types come from built-in tools like Google Search, which
# run on Google's side.
_CONVERTED_STEP_TYPES = {"thought", "model_output", "function_call"}

# Meta key holding the steps of a turn in which built-in tools ran. Their steps carry signatures and must be sent back
# exactly as received and in their original order, interleaved with the function calls, or the API rejects the request.
_RAW_STEPS_KEY = "raw_steps_for_builtin_tools"


def _to_dict(obj: BaseModel) -> dict[str, Any]:
    """
    Convert an Interactions SDK object to a JSON-compatible dict without `None` values.

    :param obj: A Pydantic model from the google-genai SDK.
    :returns: The object as a dict.
    """
    return obj.model_dump(mode="json", exclude_none=True)


def _process_response_format(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a `response_format` given as a Pydantic model or JSON schema into the Interactions format.

    A `response_format` that is already in the Interactions format (a list, or a dict with a `mime_type`) is left
    unchanged. Does not mutate the input dict.

    :param generation_kwargs: The generation kwargs, possibly containing `response_format`.
    :returns: A new dict with `response_format` in the Interactions format.
    :raises TypeError: If `response_format` is neither a Pydantic model class, a dict, nor a list.
    """
    generation_kwargs = dict(generation_kwargs)
    response_format = generation_kwargs.get("response_format")
    if response_format is None or isinstance(response_format, list):
        return generation_kwargs

    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = response_format.model_json_schema()
    elif isinstance(response_format, dict):
        if "mime_type" in response_format:
            return generation_kwargs
        schema = response_format
    else:
        msg = (
            f"Unsupported response_format type: {type(response_format).__name__}. "
            "Expected a Pydantic model class, a JSON schema dict, or an Interactions response format."
        )
        raise TypeError(msg)

    generation_kwargs["response_format"] = {"type": "text", "mime_type": "application/json", "schema": schema}
    return generation_kwargs


def _convert_file_to_content(file: FileContent) -> dict[str, Any]:
    """
    Convert a Haystack FileContent into an Interactions content dict, picking the content type from the MIME type.

    Image, audio and video MIME types map to the matching content type; any other MIME type is sent as a document.
    The Interactions API rejects MIME types it does not support.

    :param file: The FileContent to convert.
    :returns: An Interactions `document`, `image`, `audio` or `video` content dict.
    :raises ValueError: If the FileContent has no MIME type.
    """
    if not file.mime_type:
        msg = "MIME type is required to use FileContent with the Interactions API."
        raise ValueError(msg)
    content_type = next(
        (ctype for prefix, ctype in _MEDIA_MIME_PREFIXES.items() if file.mime_type.startswith(prefix)), "document"
    )
    return {"type": content_type, "data": file.base64_data, "mime_type": file.mime_type}


def _convert_tool_call_result_to_step(result: ToolCallResult) -> dict[str, Any]:
    """
    Convert a Haystack ToolCallResult into an Interactions `function_result` step.

    :param result: The ToolCallResult to convert.
    :returns: The `function_result` step.
    :raises ValueError: If the originating tool call has no ID, or the result contains unsupported content.
    """
    if result.origin.id is None:
        msg = "The Interactions API requires an ID on the ToolCall that produced each ToolCallResult."
        raise ValueError(msg)

    step_result: str | list[dict[str, Any]]
    if isinstance(result.result, str):
        step_result = result.result
    else:
        step_result = []
        for item in result.result:
            if isinstance(item, TextContent):
                step_result.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                step_result.append({"type": "image", "data": item.base64_image, "mime_type": item.mime_type})
            else:
                msg = "Only TextContent and ImageContent are supported in tool call results for the Interactions API."
                raise ValueError(msg)

    return {
        "type": "function_result",
        "call_id": result.origin.id,
        "name": result.origin.tool_name,
        "result": step_result,
        "is_error": result.error,
    }


def _convert_assistant_message_to_steps(message: ChatMessage) -> list[dict[str, Any]]:
    """
    Convert an assistant ChatMessage into Interactions steps.

    If built-in tools ran in that turn, its steps are resent exactly as received, in their original order. Otherwise,
    the message becomes thoughts, then model output, then function calls. Thought steps are always resent exactly as
    received, which the API requires for multi-turn conversations with thinking.

    :param message: The assistant ChatMessage to convert.
    :returns: The list of steps.
    :raises ValueError: If a ToolCall has no ID.
    """
    if raw_steps := message.meta.get(_RAW_STEPS_KEY):
        return list(raw_steps)

    steps: list[dict[str, Any]] = []
    for reasoning in message.reasonings:
        thought_steps = reasoning.extra.get("thought_steps")
        if thought_steps:
            steps.extend(thought_steps)
        else:
            # Reasoning from other providers has no thought signature, which the API needs to accept it
            logger.debug("Skipping reasoning content without Interactions thought steps.")

    if message.texts:
        steps.append({"type": "model_output", "content": [{"type": "text", "text": text} for text in message.texts]})

    for tool_call in message.tool_calls:
        if tool_call.id is None:
            msg = "The Interactions API requires an ID on every ToolCall in assistant messages."
            raise ValueError(msg)
        steps.append(
            {"type": "function_call", "id": tool_call.id, "name": tool_call.tool_name, "arguments": tool_call.arguments}
        )
    return steps


def _convert_message_to_steps(message: ChatMessage) -> list[dict[str, Any]]:
    """
    Convert a Haystack ChatMessage into a list of Interactions steps.

    System messages are converted into `user_input` steps; callers move a leading system message to
    `system_instruction` beforehand.

    :param message: The ChatMessage to convert.
    :returns: The list of steps.
    :raises ValueError: If the message is empty, contains unsupported content, or has an unsupported role.
    """
    if not message._content and not message.is_from(ChatRole.ASSISTANT):
        msg = (
            f"A `ChatMessage` from `{message._role.value}` must contain at least one content part. "
            "Only assistant messages can be empty."
        )
        raise ValueError(msg)

    if message.is_from(ChatRole.ASSISTANT):
        return _convert_assistant_message_to_steps(message)

    if message.is_from(ChatRole.TOOL):
        return [_convert_tool_call_result_to_step(result) for result in message.tool_call_results]

    if not message.is_from(ChatRole.USER) and not message.is_from(ChatRole.SYSTEM):
        msg = f"Unsupported message role: {message._role}"
        raise ValueError(msg)

    content: list[dict[str, Any]] = []
    for part in message._content:
        if isinstance(part, TextContent):
            if part.text.strip():
                content.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            if not part.mime_type:
                msg = "MIME type is required to use ImageContent with the Interactions API."
                raise ValueError(msg)
            content.append({"type": "image", "data": part.base64_image, "mime_type": part.mime_type})
        elif isinstance(part, FileContent):
            content.append(_convert_file_to_content(part))
        else:
            msg = f"Unsupported content type in {message._role.value} message: {type(part).__name__}"
            raise ValueError(msg)
    return [{"type": "user_input", "content": content}]


def _convert_messages_to_interaction_input(messages: list[ChatMessage]) -> tuple[str | None, list[dict[str, Any]]]:
    """
    Convert Haystack ChatMessages into a system instruction and the list of Interactions input steps.

    Only a leading system message becomes the system instruction, matching `GoogleGenAIChatGenerator`.

    :param messages: The ChatMessages to convert.
    :returns: A tuple of the system instruction (or `None`) and the input steps.
    """
    system_instruction = None
    if messages and messages[0].is_from(ChatRole.SYSTEM):
        system_instruction = messages[0].text or ""
        messages = messages[1:]

    steps: list[dict[str, Any]] = []
    for message in messages:
        steps.extend(_convert_message_to_steps(message))
    return system_instruction, steps


def _convert_tools_to_interactions_format(tools: ToolsType) -> list[dict[str, Any]]:
    """
    Convert Haystack Tools and Toolsets into Interactions function tool declarations.

    :param tools: A list of Tool and/or Toolset objects, or a single Toolset.
    :returns: The list of function tool declarations.
    """
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": _sanitize_tool_schema(tool.parameters),
        }
        for tool in flatten_tools_or_toolsets(tools)
    ]


def _convert_usage(usage: Usage | None) -> dict[str, Any]:
    """
    Convert Interactions usage into the usage dict used by `GoogleGenAIChatGenerator`.

    :param usage: The Interactions `Usage` object or `None`.
    :returns: The usage dict.
    """
    usage_dict = _to_dict(usage) if usage is not None else {}
    result = {
        "prompt_tokens": usage_dict.get("total_input_tokens", 0),
        "completion_tokens": usage_dict.get("total_output_tokens", 0),
        "total_tokens": usage_dict.get("total_tokens", 0),
    }
    optional_keys = {
        "total_thought_tokens": "thoughts_token_count",
        "total_cached_tokens": "cached_content_token_count",
        "total_tool_use_tokens": "tool_use_prompt_token_count",
    }
    for source_key, target_key in optional_keys.items():
        if usage_dict.get(source_key) is not None:
            result[target_key] = usage_dict[source_key]
    return result


def _map_status_to_finish_reason(status: str | None, *, has_tool_calls: bool) -> FinishReason | None:
    """
    Map an interaction status to a Haystack finish reason.

    :param status: The interaction status.
    :param has_tool_calls: Whether the reply contains tool calls.
    :returns: The finish reason, or `None` if the status has no Haystack equivalent.
    """
    finish_reason = STATUS_TO_FINISH_REASON.get(status or "")
    if finish_reason == "stop" and has_tool_calls:
        return "tool_calls"
    return finish_reason


def _summary_text(summary: list[dict[str, Any]] | None) -> str:
    """
    Join the text items of a thought summary.

    :param summary: The `summary` of a thought step.
    :returns: The joined text.
    """
    return "".join(item.get("text", "") for item in summary or [] if item.get("type") == "text")


def _convert_steps_to_chat_message(
    steps: list[dict[str, Any]],
    *,
    model: str,
    interaction_id: str | None,
    status: str | None,
    usage: dict[str, Any],
) -> ChatMessage:
    """
    Convert the model-generated steps of an interaction into a Haystack ChatMessage.

    Used for both streaming and non-streaming responses. Thought steps are kept verbatim in
    `ReasoningContent.extra["thought_steps"]` so they can be sent back in later turns. Steps of built-in tools are not
    converted into content; if any are present, all steps are kept in `meta["raw_steps_for_builtin_tools"]` instead.

    :param steps: The steps as dicts.
    :param model: The model name.
    :param interaction_id: The interaction ID.
    :param status: The interaction status.
    :param usage: The usage, already converted with `_convert_usage`.
    :returns: The assistant ChatMessage.
    """
    texts: list[str] = []
    thought_steps: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []

    for step in steps:
        step_type = step.get("type")
        if step_type == "thought":
            thought_steps.append(step)
        elif step_type == "model_output":
            texts.extend(item["text"] for item in step.get("content", []) if item.get("type") == "text")
        elif step_type == "function_call":
            tool_calls.append(ToolCall(tool_name=step["name"], arguments=step.get("arguments") or {}, id=step["id"]))

    reasoning = None
    if thought_steps:
        reasoning_text = "\n\n".join(text for step in thought_steps if (text := _summary_text(step.get("summary"))))
        reasoning = ReasoningContent(reasoning_text=reasoning_text, extra={"thought_steps": thought_steps})

    meta: dict[str, Any] = {
        "model": model,
        "interaction_id": interaction_id,
        "status": status,
        "finish_reason": _map_status_to_finish_reason(status, has_tool_calls=bool(tool_calls)),
        "usage": usage,
    }
    if any(step.get("type") not in _CONVERTED_STEP_TYPES for step in steps):
        meta[_RAW_STEPS_KEY] = steps
    # `ChatMessage.text` returns only the first text part, so all model output text is merged into one
    return ChatMessage.from_assistant(
        text="".join(texts) or None, tool_calls=tool_calls, reasoning=reasoning, meta=meta
    )


def _convert_interaction_to_chat_message(interaction: Interaction, model: str) -> ChatMessage:
    """
    Convert a non-streaming Interactions response into a Haystack ChatMessage.

    :param interaction: The `Interaction` returned by `interactions.create`.
    :param model: The model name used if the response does not contain one.
    :returns: The assistant ChatMessage.
    :raises RuntimeError: If the interaction failed.
    """
    if interaction.status == "failed":
        msg = f"Interaction {interaction.id} failed: {interaction.errors}"
        raise RuntimeError(msg)
    return _convert_steps_to_chat_message(
        [_to_dict(step) for step in interaction.steps or []],
        model=interaction.model or model,
        interaction_id=interaction.id,
        status=interaction.status,
        usage=_convert_usage(interaction.usage),
    )


def _convert_step_start_to_streaming_chunk(event: StepStart, meta: dict[str, Any]) -> StreamingChunk:
    """
    Convert a `step.start` event into a StreamingChunk.

    :param event: The `step.start` event.
    :param meta: The base chunk meta.
    :returns: The StreamingChunk.
    """
    step = _to_dict(event.step)
    step_type = step.get("type")
    if step_type == "thought":
        summary = step.get("summary") or []
        extra: dict[str, Any] = {"summary": summary} if summary else {}
        if step.get("signature"):
            extra["signature"] = step["signature"]
        reasoning = ReasoningContent(reasoning_text=_summary_text(summary), extra=extra)
        return StreamingChunk(content="", index=event.index, reasoning=reasoning, meta=meta)
    if step_type == "function_call":
        arguments = step.get("arguments")
        tool_call = ToolCallDelta(
            index=event.index,
            id=step.get("id"),
            tool_name=step.get("name"),
            arguments=json.dumps(arguments) if arguments else None,
        )
        return StreamingChunk(content="", index=event.index, tool_calls=[tool_call], meta=meta)
    if step_type == "model_output":
        text = "".join(item["text"] for item in step.get("content", []) if item.get("type") == "text")
        return StreamingChunk(content=text, index=event.index, meta=meta)
    # Steps of built-in tools like Google Search have no content to show, but must be sent back on the next turn
    return StreamingChunk(content="", index=event.index, meta={**meta, "builtin_tool_step": step})


def _convert_step_delta_to_streaming_chunk(event: StepDelta, meta: dict[str, Any]) -> StreamingChunk:
    """
    Convert a `step.delta` event into a StreamingChunk.

    :param event: The `step.delta` event.
    :param meta: The base chunk meta.
    :returns: The StreamingChunk.
    """
    delta = event.delta
    if isinstance(delta, TextDelta):
        return StreamingChunk(content=delta.text or "", index=event.index, meta=meta)
    if isinstance(delta, ThoughtSummaryDelta) and delta.content is not None:
        item = _to_dict(delta.content)
        reasoning = ReasoningContent(reasoning_text=_summary_text([item]), extra={"summary": [item]})
        return StreamingChunk(content="", index=event.index, reasoning=reasoning, meta=meta)
    if isinstance(delta, ThoughtSignatureDelta):
        reasoning = ReasoningContent(reasoning_text="", extra={"signature": delta.signature})
        return StreamingChunk(content="", index=event.index, reasoning=reasoning, meta=meta)
    if isinstance(delta, ArgumentsDelta):
        tool_call = ToolCallDelta(index=event.index, arguments=delta.arguments)
        return StreamingChunk(content="", index=event.index, tool_calls=[tool_call], meta=meta)
    # A built-in tool step gets its contents (arguments or results, and the signature) in one delta of its own type.
    # Annotations are not mapped to Haystack content yet.
    return StreamingChunk(content="", index=event.index, meta={**meta, "step_delta": _to_dict(delta)})


def _convert_interaction_event_to_streaming_chunk(
    event: InteractionSSEEvent, component_info: ComponentInfo, model: str, *, previous_chunks: list[StreamingChunk]
) -> StreamingChunk:
    """
    Convert an Interactions streaming event into a Haystack StreamingChunk.

    Chunk and tool call indices are the provider's step indices, which are stable for the whole stream.
    Events without user-facing content produce chunks that only carry `meta`.

    :param event: The streaming event.
    :param component_info: The component info.
    :param model: The model name.
    :param previous_chunks: The chunks already produced for this stream, used to mark where each step starts.
    :returns: The StreamingChunk.
    :raises RuntimeError: If the event is an error event.
    """
    meta: dict[str, Any] = {"model": model, "received_at": datetime.now(UTC).isoformat()}

    if isinstance(event, ErrorEvent):
        msg = f"Error in Interactions stream: {event.error}"
        raise RuntimeError(msg)

    if isinstance(event, StepStart):
        chunk = _convert_step_start_to_streaming_chunk(event, meta)
    elif isinstance(event, StepDelta):
        chunk = _convert_step_delta_to_streaming_chunk(event, meta)
    elif isinstance(event, (InteractionCreatedEvent, InteractionCompletedEvent)):
        interaction = event.interaction
        meta.update({"interaction_id": interaction.id, "status": interaction.status})
        finish_reason = None
        if isinstance(event, InteractionCompletedEvent):
            meta["usage"] = _convert_usage(interaction.usage)
            finish_reason = _map_status_to_finish_reason(interaction.status, has_tool_calls=False)
        chunk = StreamingChunk(content="", finish_reason=finish_reason, meta=meta)
    elif isinstance(event, InteractionStatusUpdate):
        meta.update({"interaction_id": event.interaction_id, "status": event.status})
        chunk = StreamingChunk(content="", meta=meta)
    else:
        # `step.stop` and events unknown to this SDK version carry nothing to stream
        chunk = StreamingChunk(content="", meta=meta)

    # A step starts with its first chunk that carries something to show. `step.start` events and thought signatures
    # usually don't, and `print_streaming_chunk` only prints a block's label on a `start` chunk with content.
    has_content = bool(chunk.content or chunk.tool_calls or (chunk.reasoning and chunk.reasoning.reasoning_text))
    started = any(c.start and c.index == chunk.index for c in previous_chunks)
    return replace(chunk, component_info=component_info, start=has_content and not started)


def _aggregate_streaming_chunks(chunks: list[StreamingChunk], model: str) -> ChatMessage:
    """
    Rebuild the interaction steps from streaming chunks and convert them into a Haystack ChatMessage.

    Produces the same ChatMessage as the non-streaming path, including thought summary items and signatures.

    :param chunks: The streaming chunks, in the order they were received.
    :param model: The model name.
    :returns: The assistant ChatMessage.
    """
    steps: dict[int, dict[str, Any]] = {}
    interaction_id = None
    status = None
    usage: dict[str, Any] | None = None

    for chunk in chunks:
        if chunk.reasoning is not None and chunk.index is not None:
            step = steps.setdefault(chunk.index, {"type": "thought", "summary": []})
            step["summary"].extend(chunk.reasoning.extra.get("summary", []))
            if chunk.reasoning.extra.get("signature"):
                step["signature"] = chunk.reasoning.extra["signature"]
        elif chunk.tool_calls:
            for delta in chunk.tool_calls:
                step = steps.setdefault(delta.index, {"type": "function_call", "id": None, "name": "", "arguments": ""})
                step["id"] = delta.id or step["id"]
                step["name"] += delta.tool_name or ""
                step["arguments"] += delta.arguments or ""
        elif chunk.content and chunk.index is not None:
            step = steps.setdefault(chunk.index, {"type": "model_output", "text": ""})
            step["text"] += chunk.content
        elif "builtin_tool_step" in chunk.meta and chunk.index is not None:
            steps[chunk.index] = dict(chunk.meta["builtin_tool_step"])
        elif "step_delta" in chunk.meta and chunk.index in steps:
            step = steps[chunk.index]
            if step["type"] not in _CONVERTED_STEP_TYPES:
                step.update({key: value for key, value in chunk.meta["step_delta"].items() if key != "type"})

        interaction_id = chunk.meta.get("interaction_id", interaction_id)
        status = chunk.meta.get("status", status)
        usage = chunk.meta.get("usage", usage)

    step_dicts: list[dict[str, Any]] = []
    for index in sorted(steps):
        step = steps[index]
        if step["type"] == "thought":
            thought: dict[str, Any] = {"type": "thought"}
            if step.get("signature"):
                thought["signature"] = step["signature"]
            if step["summary"]:
                thought["summary"] = step["summary"]
            step_dicts.append(thought)
        elif step["type"] == "function_call":
            try:
                arguments = json.loads(step["arguments"]) if step["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse the arguments of tool call {tool_name}: {arguments}",
                    tool_name=step["name"],
                    arguments=step["arguments"],
                )
                arguments = {}
            step_dicts.append({"type": "function_call", "id": step["id"], "name": step["name"], "arguments": arguments})
        elif step["type"] == "model_output":
            step_dicts.append({"type": "model_output", "content": [{"type": "text", "text": step["text"]}]})
        else:
            step_dicts.append(step)

    return _convert_steps_to_chat_message(
        step_dicts,
        model=model,
        interaction_id=interaction_id,
        status=status,
        usage=usage if usage is not None else _convert_usage(None),
    )
