# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.streaming_chunk import (
    ComponentInfo,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
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
            The API key for the provider. If not set, LiteLLM reads from the
            provider's standard environment variable (e.g. ANTHROPIC_API_KEY).
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

    def warm_up(self) -> None:
        """Verify litellm is importable."""
        import litellm  # noqa: F401

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Invoke chat completion via LiteLLM.

        :param messages: Input messages as ChatMessage instances.
        :param streaming_callback: Override the streaming callback for this call.
        :param generation_kwargs: Override generation parameters for this call.
        :param tools: Override tools for this call.
        :returns: A dict with key ``replies`` containing ChatMessage instances.
        """
        import litellm

        if not messages:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

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

        is_streaming = streaming_callback is not None

        response = litellm.completion(
            model=self.model,
            messages=openai_messages,
            stream=is_streaming,
            tools=tool_defs,
            drop_params=True,
            **merged_kwargs,
            **extra,
        )

        if is_streaming:
            return {"replies": self._handle_streaming(response, streaming_callback)}

        completions = []
        for choice in response.choices:
            completions.append(_build_chat_message(response, choice))
        return {"replies": completions}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """Async version of run()."""
        import litellm

        if not messages:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

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

        is_streaming = streaming_callback is not None

        response = await litellm.acompletion(
            model=self.model,
            messages=openai_messages,
            stream=is_streaming,
            tools=tool_defs,
            drop_params=True,
            **merged_kwargs,
            **extra,
        )

        if is_streaming:
            return {"replies": await self._ahandle_streaming(response, streaming_callback)}

        completions = []
        for choice in response.choices:
            completions.append(_build_chat_message(response, choice))
        return {"replies": completions}

    def _handle_streaming(
        self, stream_response: Any, callback: SyncStreamingCallbackT
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in stream_response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            sc = StreamingChunk(
                content=content,
                meta={
                    "model": chunk.model,
                    "finish_reason": chunk.choices[0].finish_reason,
                },
                component_info=component_info,
            )
            chunks.append(sc)
            callback(sc)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _ahandle_streaming(
        self, stream_response: Any, callback: Any
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        async for chunk in stream_response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None) or ""
            sc = StreamingChunk(
                content=content,
                meta={
                    "model": chunk.model,
                    "finish_reason": chunk.choices[0].finish_reason,
                },
                component_info=component_info,
            )
            chunks.append(sc)
            await callback(sc)
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


def _build_chat_message(response: Any, choice: Any) -> ChatMessage:
    """Convert a single litellm choice into a haystack ChatMessage."""
    message = choice.message
    tool_calls = None

    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls:
            import json

            args = tc.function.arguments
            if isinstance(args, str):
                args = json.loads(args)
            tool_calls.append(ToolCall(tool_name=tc.function.name, arguments=args, id=tc.id))

    text = message.content or ""
    meta = {
        "model": response.model,
        "index": choice.index,
        "finish_reason": choice.finish_reason,
        "usage": {},
    }
    if hasattr(response, "usage") and response.usage:
        meta["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    reply = ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta)
    return reply
