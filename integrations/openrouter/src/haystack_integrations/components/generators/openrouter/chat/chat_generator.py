# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json
from dataclasses import replace
from typing import Any

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import (
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
)
from haystack.components.generators.utils import (
    _convert_streaming_chunks_to_chat_message,
    _normalize_messages,
    _serialize_object,
)
from haystack.dataclasses import (
    ChatMessage,
    ComponentInfo,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCall,
    select_streaming_callback,
)
from haystack.tools import ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion import Choice

logger = logging.getLogger(__name__)


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


def _convert_openrouter_completion_to_chat_message(
    completion: ChatCompletion | ParsedChatCompletion, choice: Choice
) -> ChatMessage:
    """Convert an OpenRouter chat completion to a ChatMessage, including reasoning content."""
    message = choice.message
    text = message.content
    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            func = getattr(tc, "function", None)
            if func is None:
                continue
            try:
                arguments = json.loads(func.arguments)
                tool_calls.append(ToolCall(id=tc.id, tool_name=func.name, arguments=arguments))
            except json.JSONDecodeError:
                logger.warning(
                    "OpenRouter returned a malformed JSON string for tool call arguments. "
                    "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                    _id=tc.id,
                    _name=func.name,
                    _arguments=func.arguments,
                )

    logprobs = _serialize_object(choice.logprobs) if choice.logprobs else None
    meta = {
        "model": completion.model,
        "index": choice.index,
        "finish_reason": choice.finish_reason,
        "usage": _serialize_object(completion.usage),
    }
    if logprobs:
        meta["logprobs"] = logprobs

    reasoning = _extract_reasoning(message)
    return ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta, reasoning=reasoning)


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


def _convert_openrouter_streaming_chunks_to_chat_message(chunks: list[StreamingChunk]) -> ChatMessage:
    """Connect OpenRouter streaming chunks into a single ChatMessage, keeping the merged `reasoning_details`."""
    message = _convert_streaming_chunks_to_chat_message(chunks=chunks)
    # the core helper only joins the reasoning text, but OpenRouter expects the details back in multi-turn conversations
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
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialiaze some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=serialize_tools_or_toolset(self.tools),
            extra_headers=self.extra_headers,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    def _prepare_api_call(
        self,
        *,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, Any]:
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**(self.extra_headers or {})}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)

        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)
        response_format = generation_kwargs.pop("response_format", None)

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

        # OpenRouter expects reasoning_details to be sent back in multi-turn conversations, but
        # to_openai_dict_format() strips reasoning, so we re-inject it into the formatted message dicts.
        for i, chat_msg in enumerate(messages):
            if chat_msg.reasoning and chat_msg.reasoning.extra.get("reasoning_details"):
                openai_formatted_messages[i]["reasoning_details"] = chat_msg.reasoning.extra["reasoning_details"]

        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        _check_duplicate_tool_names(flattened_tools)

        openai_tools = {}
        if flattened_tools:
            tool_definitions = []
            for t in flattened_tools:
                function_spec = {**t.tool_spec}
                if tools_strict:
                    function_spec["strict"] = True
                    function_spec["parameters"]["additionalProperties"] = False
                tool_definitions.append({"type": "function", "function": function_spec})
            openai_tools = {"tools": tool_definitions}

        base_args = {
            "model": self.model,
            "messages": openai_formatted_messages,
            "n": num_responses,
            **openai_tools,
            "extra_headers": {**extra_headers},
            "extra_body": {**generation_kwargs},
        }

        if response_format and not is_streaming:
            # for structured outputs without streaming, we use openai's parse endpoint
            # Note: `stream` cannot be passed to chat.completions.parse
            # we pass a key `openai_endpoint` as a hint to the run method to use the parse endpoint
            # this key will be removed before the API call is made
            return {**base_args, "response_format": response_format, "openai_endpoint": "parse"}

        # for structured outputs with streaming, we use openai's create endpoint
        # we pass a key `openai_endpoint` as a hint to the run method to use the create endpoint
        # this key will be removed before the API call is made
        final_args = {**base_args, "stream": is_streaming, "openai_endpoint": "create"}

        # We only set the response_format parameter if it's not None since None is not a valid value in the API.
        if response_format:
            final_args["response_format"] = response_format
        return final_args

    def _handle_stream_response(
        self, chat_completion: Stream[ChatCompletionChunk], callback: SyncStreamingCallbackT
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
            for streaming_chunk in _convert_openrouter_chunk_to_streaming_chunks(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            ):
                chunks.append(streaming_chunk)
                callback(streaming_chunk)
        return [_convert_openrouter_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream[ChatCompletionChunk], callback: StreamingCallbackT
    ) -> list[ChatMessage]:
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
        return [_convert_openrouter_streaming_chunks_to_chat_message(chunks=chunks)]

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Invokes chat completion on the OpenRouter API.

        :param messages:
            A list of ChatMessage instances representing the input messages.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These are merged per key with the
            `generation_kwargs` passed at initialization: keys provided here take precedence, keys set only
            at initialization are kept.
            For details on OpenRouter API parameters, see
            [OpenRouter docs](https://openrouter.ai/docs/quickstart).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        messages = _normalize_messages(messages)
        self.warm_up()

        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        openai_endpoint = api_args.pop("openai_endpoint")
        # with haystack-ai >= 3.0 the client is Optional and built by warm_up above
        chat_completion = getattr(self.client.chat.completions, openai_endpoint)(**api_args)  # type: ignore[union-attr]

        if streaming_callback is not None:
            completions = self._handle_stream_response(chat_completion, streaming_callback)
        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                _convert_openrouter_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}

    @component.output_types(replies=list[ChatMessage])
    async def run_async(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Asynchronously invokes chat completion on the OpenRouter API.

        :param messages:
            A list of ChatMessage instances representing the input messages.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Must be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These are merged per key with the
            `generation_kwargs` passed at initialization: keys provided here take precedence, keys set only
            at initialization are kept.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        messages = _normalize_messages(messages)
        if hasattr(self, "warm_up_async"):
            # haystack-ai >= 3.0 initializes the async client on the running event loop
            await self.warm_up_async()
        else:
            self.warm_up()

        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        openai_endpoint = api_args.pop("openai_endpoint")
        # with haystack-ai >= 3.0 the client is Optional and built by warm_up above
        chat_completion = await getattr(self.async_client.chat.completions, openai_endpoint)(**api_args)  # type: ignore[union-attr]

        if streaming_callback is not None:
            completions = await self._handle_async_stream_response(chat_completion, streaming_callback)
        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                _convert_openrouter_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}
