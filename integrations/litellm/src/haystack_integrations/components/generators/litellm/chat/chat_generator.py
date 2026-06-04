# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, _normalize_messages
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.streaming_chunk import (
    AsyncStreamingCallbackT,
    ComponentInfo,
    FinishReason,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

logger = logging.getLogger(__name__)


@component
class LiteLLMChatGenerator:
    """Completes chats using any of 100+ LLM providers via LiteLLM.

    LiteLLM routes to OpenAI, Anthropic, Google, AWS Bedrock, Azure, Cohere,
    Mistral, Groq, and many more through a single unified interface.

    Model names use LiteLLM format: ``provider/model-name``, e.g.
    ``anthropic/claude-sonnet-4-20250514``, ``openai/gpt-4o``,
    ``bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0``.

    See https://docs.litellm.ai/docs/providers for the full list.

    Usage example:

    ```python
    from haystack_integrations.components.generators.litellm import LiteLLMChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = LiteLLMChatGenerator(
        model="anthropic/claude-sonnet-4-20250514",
        generation_kwargs={"max_tokens": 1024, "temperature": 0.7},
    )

    messages = [
        ChatMessage.from_system("You are a helpful assistant"),
        ChatMessage.from_user("What's Natural Language Processing?"),
    ]
    result = generator.run(messages=messages)
    print(result["replies"][0].text)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret | None = None,
        model: str = "openai/gpt-4o",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
    ) -> None:
        """Create a LiteLLMChatGenerator instance.

        :param api_key:
            The API key for the provider. Optional: when not set, LiteLLM resolves
            credentials itself from the provider's standard environment variable
            (e.g. ``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``). Pass a ``Secret`` only
            when you want Haystack to manage and serialize the key explicitly.
        :param model:
            The model name in LiteLLM format (provider/model-name).
        :param streaming_callback:
            A callback function invoked with each new StreamingChunk.
        :param api_base_url:
            Custom API base URL (e.g. for a self-hosted LiteLLM proxy).
        :param generation_kwargs:
            Additional parameters passed to litellm.completion().
            See https://docs.litellm.ai/docs/completion/input for details.
        :param tools:
            A list of Tool / Toolset objects the model can prepare calls for.
        """
        self.model = model
        self.api_key = api_key
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs or {}
        self.tools = tools

    def _build_litellm_kwargs(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None,
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
    ) -> dict[str, Any]:
        merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        openai_messages = [m.to_openai_dict_format() for m in messages]

        extra: dict[str, Any] = {}
        if self.api_key:
            extra["api_key"] = self.api_key.resolve_value()
        if self.api_base_url:
            extra["api_base"] = self.api_base_url

        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        _check_duplicate_tool_names(flattened_tools)
        tool_defs = None
        if flattened_tools:
            tool_defs = [{"type": "function", "function": t.tool_spec} for t in flattened_tools]

        # User-supplied generation_kwargs go first so the framework-controlled keys below
        # (model, messages, stream, tools, drop_params, credentials) always take precedence
        # and can't be silently overridden.
        return {
            **merged_kwargs,
            "model": self.model,
            "messages": openai_messages,
            "stream": streaming_callback is not None,
            "tools": tool_defs,
            "drop_params": True,
            **extra,
        }

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Invoke chat completion via LiteLLM.

        :param messages: Input messages as ChatMessage instances.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param streaming_callback: Override the streaming callback for this call.
        :param generation_kwargs: Override generation parameters for this call.
        :param tools: Override tools for this call.
        :returns: A dict with key ``replies`` containing ChatMessage instances.
        """
        messages = _normalize_messages(messages)
        if not messages:
            return {"replies": []}

        import litellm

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        kwargs = self._build_litellm_kwargs(messages, streaming_callback, generation_kwargs, tools)
        response = litellm.completion(**kwargs)

        if streaming_callback is not None:
            return {"replies": self._handle_streaming(response, streaming_callback)}

        completions = [_build_chat_message(response, choice) for choice in response.choices]
        return {"replies": completions}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Async version of run(). Invoke chat completion via LiteLLM.

        :param messages: Input messages as ChatMessage instances.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param streaming_callback: Override the streaming callback for this call.
        :param generation_kwargs: Override generation parameters for this call.
        :param tools: Override tools for this call.
        :returns: A dict with key ``replies`` containing ChatMessage instances.
        """
        messages = _normalize_messages(messages)
        if not messages:
            return {"replies": []}

        import litellm

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        kwargs = self._build_litellm_kwargs(messages, streaming_callback, generation_kwargs, tools)
        response = await litellm.acompletion(**kwargs)

        if streaming_callback is not None:
            return {"replies": await self._ahandle_streaming(response, streaming_callback)}

        completions = [_build_chat_message(response, choice) for choice in response.choices]
        return {"replies": completions}

    def _handle_streaming(self, stream_response: Any, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in stream_response:
            stream_chunk = _convert_litellm_chunk_to_streaming_chunk(chunk, chunks, component_info)
            chunks.append(stream_chunk)
            callback(stream_chunk)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _ahandle_streaming(self, stream_response: Any, callback: AsyncStreamingCallbackT) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        async for chunk in stream_response:
            stream_chunk = _convert_litellm_chunk_to_streaming_chunk(chunk, chunks, component_info)
            chunks.append(stream_chunk)
            await callback(stream_chunk)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            api_key=self.api_key.to_dict() if self.api_key else None,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            tools=serialize_tools_or_toolset(self.tools),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiteLLMChatGenerator":
        """Deserialize a component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        if data["init_parameters"].get("streaming_callback"):
            data["init_parameters"]["streaming_callback"] = deserialize_callable(
                data["init_parameters"]["streaming_callback"]
            )
        return default_from_dict(cls, data)


# LiteLLM normalizes provider finish reasons to OpenAI's vocabulary, but we still map
# explicitly so the value matches Haystack's FinishReason literal.
_FINISH_REASON_MAPPING: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
}


def _extract_usage(obj: Any) -> dict[str, int]:
    """Pull token usage off a litellm response or chunk, tolerating its absence."""
    usage = getattr(obj, "usage", None)
    if not usage:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _build_chat_message(response: Any, choice: Any) -> ChatMessage:
    """Convert a single litellm choice into a haystack ChatMessage."""
    message = choice.message

    # Keep content as-is (None for tool-only replies) so we don't attach an empty text block,
    # matching both Haystack's OpenAI generator and our own streaming aggregation.
    text = message.content

    tool_calls = []
    if getattr(message, "tool_calls", None):
        for tc in message.tool_calls:
            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        "The LLM provider returned a malformed JSON string for tool call arguments. This tool "
                        "call will be skipped. Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                        _id=tc.id,
                        _name=tc.function.name,
                        _arguments=arguments,
                    )
                    continue
            tool_calls.append(ToolCall(tool_name=tc.function.name, arguments=arguments, id=tc.id))

    meta: dict[str, Any] = {
        "model": response.model,
        "index": choice.index,
        "finish_reason": choice.finish_reason,
        "usage": _extract_usage(response),
    }

    return ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta)


def _convert_litellm_chunk_to_streaming_chunk(
    chunk: Any, previous_chunks: list[StreamingChunk], component_info: ComponentInfo
) -> StreamingChunk:
    """Convert a single litellm streaming chunk into a Haystack ``StreamingChunk``.

    The structured fields (``tool_calls``, ``finish_reason``, ``index``) are populated so that
    ``_convert_streaming_chunks_to_chat_message`` can reconstruct tool calls and metadata; it reads
    those fields, not ``meta``. LiteLLM returns OpenAI-compatible chunks, but unlike the OpenAI SDK
    its chunk objects don't always expose a ``usage`` attribute, so it is accessed defensively.
    """
    usage = _extract_usage(chunk) or None

    # Final usage-only chunk: no choices, just aggregate token counts.
    if not chunk.choices:
        return StreamingChunk(
            content="",
            component_info=component_info,
            index=None,
            finish_reason=None,
            meta={"model": chunk.model, "usage": usage},
        )

    choice = chunk.choices[0]
    delta = choice.delta
    finish_reason = _FINISH_REASON_MAPPING.get(choice.finish_reason) if choice.finish_reason else None
    meta: dict[str, Any] = {
        "model": chunk.model,
        "index": choice.index,
        "finish_reason": choice.finish_reason,
        "usage": usage,
    }

    tool_calls = getattr(delta, "tool_calls", None) if delta else None
    if tool_calls:
        tool_call_deltas = []
        for tc in tool_calls:
            function = tc.function
            tool_call_deltas.append(
                ToolCallDelta(
                    index=tc.index,
                    id=tc.id,
                    tool_name=function.name if function else None,
                    arguments=function.arguments if function and function.arguments else None,
                )
            )
        return StreamingChunk(
            content=(delta.content or "") if delta else "",
            component_info=component_info,
            # ToolCallDelta requires an index; adopt the first one as the chunk index.
            index=tool_call_deltas[0].index,
            tool_calls=tool_call_deltas,
            start=tool_call_deltas[0].tool_name is not None,
            finish_reason=finish_reason,
            meta=meta,
        )

    content = (delta.content or "") if delta else ""
    # The opening chunk only carries role info (no content), so it isn't a content block.
    role = getattr(delta, "role", None) if delta else None
    resolved_index = None if (delta is None or delta.content is None or role is not None) else 0

    return StreamingChunk(
        content=content,
        component_info=component_info,
        index=resolved_index,
        start=len(previous_chunks) == 1,
        finish_reason=finish_reason,
        meta=meta,
    )
