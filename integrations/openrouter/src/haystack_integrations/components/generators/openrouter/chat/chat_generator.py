# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import _check_finish_reason
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, _serialize_object
from haystack.dataclasses import (
    AsyncStreamingCallbackT,
    ChatMessage,
    ComponentInfo,
    FinishReason,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    ToolCall,
    ToolCallDelta,
    select_streaming_callback,
)
from haystack.tools import ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice

logger = logging.getLogger(__name__)


def _extract_reasoning(message: Any) -> ReasoningContent | None:
    """Extract reasoning content from an OpenRouter API response message or delta."""
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


def _convert_openrouter_chunk_to_streaming_chunk(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo | None = None
) -> StreamingChunk:
    """Convert an OpenRouter streaming chunk to a StreamingChunk, handling reasoning_details in deltas."""
    finish_reason_mapping: dict[str, FinishReason] = {
        "stop": "stop",
        "length": "length",
        "content_filter": "content_filter",
        "tool_calls": "tool_calls",
        "function_call": "tool_calls",
    }

    if len(chunk.choices) == 0:
        return StreamingChunk(
            content="",
            component_info=component_info,
            index=None,
            finish_reason=None,
            meta={
                "model": chunk.model,
                "received_at": datetime.now(tz=timezone.utc).isoformat(),
                "usage": _serialize_object(chunk.usage),
            },
        )

    choice: ChunkChoice = chunk.choices[0]

    raw_details = getattr(choice.delta, "reasoning_details", None) if choice.delta else None
    if raw_details:
        reasoning_parts = []
        details = []
        for d in raw_details:
            entry = d if isinstance(d, dict) else (d.model_dump() if hasattr(d, "model_dump") else vars(d))
            details.append(entry)
            dtype = entry.get("type", "")
            if dtype == "reasoning.text":
                reasoning_parts.append(entry.get("text", ""))
            elif dtype == "reasoning.summary":
                reasoning_parts.append(entry.get("summary", ""))

        reasoning_text = "".join(reasoning_parts)
        reasoning = ReasoningContent(
            reasoning_text=reasoning_text,
            extra={"reasoning_details": details} if details else {},
        )

        meta = {
            "model": chunk.model,
            "index": choice.index,
            "finish_reason": choice.finish_reason,
            "received_at": datetime.now(tz=timezone.utc).isoformat(),
            "usage": _serialize_object(chunk.usage),
        }

        return StreamingChunk(
            content="",
            reasoning=reasoning,
            component_info=component_info,
            index=0,
            start=len(previous_chunks) <= 1,
            finish_reason=finish_reason_mapping.get(choice.finish_reason) if choice.finish_reason else None,
            meta=meta,
        )

    if choice.delta and choice.delta.tool_calls:
        tool_calls_deltas = []
        for tool_call in choice.delta.tool_calls:
            function = tool_call.function
            tool_calls_deltas.append(
                ToolCallDelta(
                    index=tool_call.index,
                    id=tool_call.id,
                    tool_name=function.name if function else None,
                    arguments=function.arguments if function and function.arguments else None,
                )
            )
        return StreamingChunk(
            content=choice.delta.content or "",
            component_info=component_info,
            index=tool_calls_deltas[0].index,
            tool_calls=tool_calls_deltas,
            start=tool_calls_deltas[0].tool_name is not None,
            finish_reason=finish_reason_mapping.get(choice.finish_reason) if choice.finish_reason else None,
            meta={
                "model": chunk.model,
                "index": choice.index,
                "tool_calls": choice.delta.tool_calls,
                "finish_reason": choice.finish_reason,
                "received_at": datetime.now(tz=timezone.utc).isoformat(),
                "usage": _serialize_object(chunk.usage),
            },
        )

    if choice.delta and (choice.delta.content is None or choice.delta.role is not None):
        resolved_index = None
    else:
        resolved_index = 0

    meta = {
        "model": chunk.model,
        "index": choice.index,
        "tool_calls": choice.delta.tool_calls if choice.delta and choice.delta.tool_calls else None,
        "finish_reason": choice.finish_reason,
        "received_at": datetime.now(tz=timezone.utc).isoformat(),
        "usage": _serialize_object(chunk.usage),
    }

    logprobs = _serialize_object(choice.logprobs) if choice.logprobs else None
    if logprobs:
        meta["logprobs"] = logprobs

    content = ""
    if choice.delta and choice.delta.content:
        content = choice.delta.content

    return StreamingChunk(
        content=content,
        component_info=component_info,
        index=resolved_index,
        start=len(previous_chunks) == 1,
        finish_reason=finish_reason_mapping.get(choice.finish_reason) if choice.finish_reason else None,
        meta=meta,
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
      field on `ChatMessage`.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the OpenRouter API, refer to the
    [OpenRouter API Docs](https://openrouter.ai/docs/quickstart).

    Usage example:
    ```python
    from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenRouterChatGenerator(
        model="deepseek/deepseek-r1",
        generation_kwargs={"reasoning": {"effort": "high"}},
    )
    response = client.run(messages)
    print(response["replies"][0].reasoning)  # Access reasoning content
    print(response["replies"][0].text)       # Access final answer
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
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**(self.extra_headers or {})}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)

        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)
        response_format = generation_kwargs.pop("response_format", None)

        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

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
            return {**base_args, "response_format": response_format, "openai_endpoint": "parse"}

        final_args = {**base_args, "stream": is_streaming, "openai_endpoint": "create"}

        if response_format:
            final_args["response_format"] = response_format
        return final_args

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage],
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
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
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
        if not self._is_warmed_up:
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
        chat_completion = getattr(self.client.chat.completions, openai_endpoint)(**api_args)

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
        messages: list[ChatMessage],
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
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Must be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if not self._is_warmed_up:
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
        chat_completion = await getattr(self.async_client.chat.completions, openai_endpoint)(**api_args)

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

    def _handle_stream_response(self, chat_completion: Any, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
            chunk_delta = _convert_openrouter_chunk_to_streaming_chunk(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            )
            chunks.append(chunk_delta)
            callback(chunk_delta)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: Any, callback: AsyncStreamingCallbackT
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
                chunk_delta = _convert_openrouter_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                )
                chunks.append(chunk_delta)
                await callback(chunk_delta)
        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            raise
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]
