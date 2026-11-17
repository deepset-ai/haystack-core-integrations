# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from dataclasses import replace
from typing import Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import (
    _convert_chat_completion_chunk_to_streaming_chunk,
)
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
)
from haystack.tools import ToolsType
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion import Choice


def _extract_reasoning(message: Any) -> ReasoningContent | None:
    """Extract reasoning content from an OpenRouter API response message."""
    # OpenRouter attaches reasoning content as extra attributes on the standard OpenAI SDK message,
    # so we read them with getattr rather than relying on typed fields.
    reasoning_text = getattr(message, "reasoning", None) or ""
    raw_details = getattr(message, "reasoning_details", None) or []

    if not reasoning_text and not raw_details:
        return None

    details = []
    for d in raw_details:
        if isinstance(d, dict):
            details.append(d)
        elif hasattr(d, "model_dump"):
            details.append(d.model_dump())
        else:
            details.append(vars(d))

    # Some models only return structured details without a flat `reasoning` string, so we
    # reconstruct the text from the known detail types.
    if not reasoning_text and details:
        parts = []
        for d in details:
            dtype = d.get("type", "")
            if dtype == "reasoning.text":
                parts.append(d.get("text", ""))
            elif dtype == "reasoning.summary":
                parts.append(d.get("summary", ""))
        reasoning_text = "".join(parts)

    extra = {}
    if details:
        extra["reasoning_details"] = details

    return ReasoningContent(reasoning_text=reasoning_text, extra=extra)


def _convert_openrouter_chunk_to_streaming_chunks(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo
) -> list[StreamingChunk]:
    """
    Convert an OpenRouter streaming chunk to StreamingChunks, including reasoning content.

    A `StreamingChunk` holds either reasoning or content/tool calls, so a delta carrying both is split into a
    reasoning chunk followed by the content or tool call chunk.
    """
    streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
        chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
    )
    # the parent converter ignores the `reasoning`/`reasoning_details` fields OpenRouter adds to the delta
    reasoning = _extract_reasoning(chunk.choices[0].delta) if chunk.choices else None
    has_payload = bool(streaming_chunk.content or streaming_chunk.tool_calls)
    # the last chunk with reasoning, content, or tool calls tells which block the stream is currently in
    last_block = next((c for c in reversed(previous_chunks) if c.reasoning or c.content or c.tool_calls), None)

    streaming_chunks = []
    if reasoning is not None:
        details = reasoning.extra.get("reasoning_details", [])
        # adopt the index of the first reasoning detail, as the parent does with the first tool call index
        index = next((d["index"] for d in details if isinstance(d.get("index"), int)), 0)
        reasoning_chunk = StreamingChunk(
            content="",
            reasoning=reasoning,
            index=index,
            start=last_block is None or last_block.reasoning is None or last_block.index != index,
            # a reasoning-only delta replaces the parent's empty chunk, so it carries its finish reason
            finish_reason=None if has_payload else streaming_chunk.finish_reason,
            component_info=component_info,
            meta=streaming_chunk.meta,
        )
        if not has_payload:
            return [reasoning_chunk]
        streaming_chunks.append(reasoning_chunk)
        last_block = reasoning_chunk

    if streaming_chunk.content:
        # the parent converter marks the second chunk of a stream as start, assuming the first one only carries `role`,
        # and gives text chunks index None when `role` is set. OpenRouter sets `role` on every chunk and can stream text
        # in the first one, so derive start and index from the previous block instead. Text after tool calls keeps the
        # parent's start and index, so the whitespace some models stream between tool calls gets no header.
        if last_block is None:
            streaming_chunk = replace(streaming_chunk, start=True)
        elif last_block.reasoning is not None:
            streaming_chunk = replace(streaming_chunk, start=True, index=(last_block.index or 0) + 1)
        elif last_block.content:
            streaming_chunk = replace(streaming_chunk, start=False, index=last_block.index)
    elif streaming_chunk.tool_calls and last_block is not None:
        # the parent converter uses the provider's tool call index as chunk index, which collides with the reasoning
        # or text block before the tool calls, so shift the chunk index past that block. Later tool calls keep the
        # shift of the first one, even if text is streamed between them. `ToolCallDelta.index` keeps the provider's
        # index, which the tool calls are assembled by.
        previous_tool_chunk = next((c for c in reversed(previous_chunks) if c.tool_calls), None)
        if previous_tool_chunk is not None and previous_tool_chunk.tool_calls:
            offset = (previous_tool_chunk.index or 0) - previous_tool_chunk.tool_calls[0].index
        else:
            offset = (last_block.index or 0) + 1
        streaming_chunk = replace(streaming_chunk, index=streaming_chunk.tool_calls[0].index + offset)
    streaming_chunks.append(streaming_chunk)
    return streaming_chunks


def _merge_reasoning_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge streamed `reasoning_details` fragments into complete entries.

    OpenRouter streams each reasoning detail as fragments sharing its `index` and `type`. String payloads are
    concatenated in order, other fields keep their latest non-null value, such as a `signature` that only arrives
    with the last fragment.
    """
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    for detail in details:
        key = (detail.get("index"), detail.get("type"))
        if key not in merged:
            merged[key] = dict(detail)
            continue
        entry = merged[key]
        for field, value in detail.items():
            if field in ("text", "summary", "data") and isinstance(value, str):
                entry[field] = (entry.get(field) or "") + value
            elif value is not None:
                entry[field] = value
    return list(merged.values())


@component
class OpenRouterChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using OpenRouter generative models.

    For supported models, see [OpenRouter docs](https://openrouter.ai/models).

    Users can pass any text generation parameters valid for the OpenRouter chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Compatible with the OpenRouter chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the OpenRouter chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the OpenRouter chat completion endpoint.
    - **Reasoning Support**: Extracts reasoning/thinking content from models that support it
      (e.g., DeepSeek R1, Claude with extended thinking) and stores it in the `ReasoningContent`
      field on `ChatMessage`. When streaming, reasoning is also passed to the `streaming_callback`
      in `StreamingChunk.reasoning`.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the OpenRouter API, refer to the
    [OpenRouter API Docs](https://openrouter.ai/docs/quickstart).

    Usage example:
    ```python
    from haystack_integrations.components.generators.openrouter import (
        OpenRouterChatGenerator,
    )
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenRouterChatGenerator(
        model="deepseek/deepseek-r1",
        generation_kwargs={"reasoning": {"effort": "high"}},
    )
    response = client.run(messages)
    print(response["replies"][0].reasoning)  # Access reasoning content
    print(response["replies"][0].text)  # Access final answer
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OPENROUTER_API_KEY"),
        model: str = "openai/gpt-5-mini",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://openrouter.ai/api/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OpenRouterChatGenerator.

        :param api_key:
            The OpenRouter API key.
        :param model:
            The name of the OpenRouter chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The OpenRouter API Base url.
            For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the OpenRouter endpoint. See [OpenRouter API docs](https://openrouter.ai/docs/quickstart) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `safe_prompt`: Whether to inject a safety prompt before all conversations.
            - `random_seed`: The seed to use for random sampling.
            - `reasoning`: A dict to configure reasoning/thinking tokens for models that support it.
                Example: `{"effort": "high"}` or `{"max_tokens": 2000}`.
                See [OpenRouter reasoning docs](https://openrouter.ai/docs/use-cases/reasoning-tokens).
            - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance.
        :param timeout:
            The timeout for the OpenRouter API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the OpenRouter API.
            This can be useful for adding site URL or title for rankings on openrouter.ai
            For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(OpenRouterChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            tools=tools,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )
        self.extra_headers = extra_headers

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(OpenRouterChatGenerator, self).to_dict()  # noqa: UP008
        init_parameters = inspect.signature(type(self).__init__).parameters
        data["init_parameters"] = {
            key: value for key, value in data["init_parameters"].items() if key in init_parameters
        }
        data["init_parameters"]["extra_headers"] = self.extra_headers
        return data

    def _adapt_generation_kwargs(self, generation_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Send OpenRouter generation parameters in the request body with custom headers."""
        return {"extra_body": dict(generation_kwargs), "extra_headers": dict(self.extra_headers or {})}

    def _convert_message_to_api_format(self, message: ChatMessage) -> dict[str, Any]:
        """Preserve reasoning details when sending messages back to OpenRouter."""
        formatted = super(OpenRouterChatGenerator, self)._convert_message_to_api_format(message)  # noqa: UP008
        if message.reasoning and message.reasoning.extra.get("reasoning_details"):
            formatted["reasoning_details"] = message.reasoning.extra["reasoning_details"]
        return formatted

    def _convert_chat_completion_to_chat_message(
        self, completion: ChatCompletion | ParsedChatCompletion, choice: Choice
    ) -> ChatMessage:
        """Convert a completion while preserving OpenRouter reasoning content and details."""
        message = super(OpenRouterChatGenerator, self)._convert_chat_completion_to_chat_message(  # noqa: UP008
            completion=completion, choice=choice
        )
        reasoning = _extract_reasoning(choice.message)
        if reasoning is None:
            return message
        return ChatMessage.from_assistant(
            text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=reasoning
        )

    def _convert_streaming_chunks_to_chat_message(self, chunks: list[StreamingChunk]) -> ChatMessage:
        """Connect OpenRouter streaming chunks into a single ChatMessage, keeping the merged `reasoning_details`."""
        message = super(OpenRouterChatGenerator, self)._convert_streaming_chunks_to_chat_message(chunks=chunks)  # noqa: UP008
        # OpenRouter needs reasoning details back in multi-turn conversations; core only joins the text.
        details = [d for c in chunks if c.reasoning for d in c.reasoning.extra.get("reasoning_details", [])]
        if not details:
            return message
        reasoning = ReasoningContent(
            reasoning_text=message.reasoning.reasoning_text if message.reasoning else "",
            extra={"reasoning_details": _merge_reasoning_details(details)},
        )
        return ChatMessage.from_assistant(
            text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=reasoning
        )

    def _handle_stream_response(
        self, chat_completion: Stream[ChatCompletionChunk], callback: SyncStreamingCallbackT
    ) -> list[ChatMessage]:
        """Handle OpenRouter streams that can emit multiple chunks per API delta."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
            for streaming_chunk in _convert_openrouter_chunk_to_streaming_chunks(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            ):
                chunks.append(streaming_chunk)
                callback(streaming_chunk)
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream[ChatCompletionChunk], callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        """Handle async OpenRouter streams that can emit multiple chunks per API delta."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
                for streaming_chunk in _convert_openrouter_chunk_to_streaming_chunks(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                ):
                    chunks.append(streaming_chunk)
                    # haystack-ai >= 3.0 also accepts sync callbacks in async runs, so only await async ones
                    callback_result = callback(streaming_chunk)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        except asyncio.CancelledError:
            # shield the close so the stream is released even though the task is being cancelled
            await asyncio.shield(chat_completion.close())
            raise
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]
