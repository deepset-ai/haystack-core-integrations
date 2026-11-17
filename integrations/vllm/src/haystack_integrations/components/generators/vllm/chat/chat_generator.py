# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import dataclasses
import inspect
import json
from typing import Any

from haystack import default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.openai import (
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
)
from haystack.components.generators.utils import (
    _convert_streaming_chunks_to_chat_message,
    _normalize_messages,
    _serialize_object,
)
from haystack.core.component import component
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.dataclasses.chat_message import ReasoningContent
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
from haystack.utils import Secret, deserialize_callable, serialize_callable
from openai import AsyncOpenAI, AsyncStream, OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice

from haystack_integrations.common.vllm.utils import _create_async_openai_client, _create_openai_client

logger = logging.getLogger(__name__)


def _convert_chat_history_to_vllm_format(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """
    Convert a list of ChatMessages to the OpenAI format expected by the vLLM Chat API.

    A trailing assistant message with no text and no tool calls is left out. Sent as an empty assistant turn, it makes
    some models, such as Qwen3 with thinking enabled, stop after their reasoning without writing an answer.
    """
    if messages and messages[-1].is_from(ChatRole.ASSISTANT) and not messages[-1].texts and not messages[-1].tool_calls:
        messages = messages[:-1]
    return [message.to_openai_dict_format() for message in messages]


def _convert_chat_completion_to_chat_message(completion: ChatCompletion, choice: Choice) -> ChatMessage:
    """
    Convert a vLLM chat completion response to a ChatMessage, including reasoning content if present.

    :param completion: The completion returned by the vLLM API.
    :param choice: The choice returned by the vLLM API.
    :return: The ChatMessage.
    """
    message = choice.message
    text = message.content

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if not hasattr(tc, "function"):
                continue
            try:
                arguments = json.loads(tc.function.arguments)
                tool_calls.append(ToolCall(id=tc.id, tool_name=tc.function.name, arguments=arguments))
            except json.JSONDecodeError:
                logger.warning(
                    "vLLM returned a malformed JSON string for tool call arguments. This tool call "
                    "will be skipped. Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                    _id=tc.id,
                    _name=tc.function.name,
                    _arguments=tc.function.arguments,
                )

    meta: dict[str, Any] = {
        "model": completion.model,
        "index": choice.index,
        "finish_reason": choice.finish_reason,
        "usage": _serialize_object(completion.usage),
    }

    reasoning_text = getattr(message, "reasoning", None)
    reasoning = ReasoningContent(reasoning_text=reasoning_text) if reasoning_text else None

    return ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta, reasoning=reasoning)


def _convert_vllm_chunk_to_streaming_chunks(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo
) -> list[StreamingChunk]:
    """
    Convert a vLLM streaming chunk to StreamingChunks, including reasoning content.

    A `StreamingChunk` holds at most one of `reasoning`, `content`, or `tool_calls`, so a delta carrying multiple
    payloads is split into separate chunks in `reasoning` -> `content` -> `tool_calls` order.
    """
    if not chunk.choices:
        return [
            _convert_chat_completion_chunk_to_streaming_chunk(
                chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
            )
        ]

    choice = chunk.choices[0]
    delta = choice.delta
    reasoning_text = getattr(delta, "reasoning", None)
    has_content = bool(delta.content)
    has_tool_calls = bool(delta.tool_calls)
    has_payload = has_content or has_tool_calls
    if not reasoning_text and not has_payload:
        return [
            _convert_chat_completion_chunk_to_streaming_chunk(
                chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
            )
        ]

    # the last chunk with reasoning, content, or tool calls tells which block the stream is currently in
    last_block = next((c for c in reversed(previous_chunks) if c.reasoning or c.content or c.tool_calls), None)
    streaming_chunks: list[StreamingChunk] = []

    if reasoning_text:
        meta_source_chunk = (
            chunk.model_copy(
                update={
                    "choices": [
                        choice.model_copy(
                            update={
                                "finish_reason": None,
                                "logprobs": None,
                                "delta": delta.model_copy(
                                    update={"content": None, "tool_calls": None, "refusal": None}
                                ),
                            }
                        )
                    ]
                }
            )
            if has_payload
            else chunk
        )
        base_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
            chunk=meta_source_chunk, previous_chunks=previous_chunks, component_info=component_info
        )
        reasoning_index = 0 if last_block is None or last_block.reasoning is not None else (last_block.index or 0) + 1
        reasoning_chunk = StreamingChunk(
            content="",
            reasoning=ReasoningContent(reasoning_text=reasoning_text),
            index=reasoning_index,
            start=last_block is None or last_block.reasoning is None,
            finish_reason=base_chunk.finish_reason,
            component_info=component_info,
            meta=base_chunk.meta,
        )
        if not has_payload:
            return [reasoning_chunk]
        streaming_chunks.append(reasoning_chunk)
        last_block = reasoning_chunk

    if has_content:
        content_source_chunk = (
            chunk.model_copy(
                update={
                    "choices": [
                        choice.model_copy(
                            update={
                                "finish_reason": None,
                                "delta": delta.model_copy(update={"tool_calls": None}),
                            }
                        )
                    ]
                }
            )
            if has_tool_calls
            else chunk
        )
        content_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
            chunk=content_source_chunk,
            previous_chunks=[*previous_chunks, *streaming_chunks],
            component_info=component_info,
        )
        # the parent converter marks the second chunk of a stream as start, assuming the first one only carries `role`,
        # and gives text chunks index None when `role` is set. Derive start and index from the previous block instead
        # when starting, transitioning from another block, or continuing content.
        if last_block is None:
            content_chunk = dataclasses.replace(content_chunk, start=True, index=0)
        elif last_block.content:
            content_chunk = dataclasses.replace(content_chunk, start=False, index=last_block.index)
        else:
            content_chunk = dataclasses.replace(content_chunk, start=True, index=(last_block.index or 0) + 1)
        streaming_chunks.append(content_chunk)
        last_block = content_chunk

    if has_tool_calls:
        tool_source_chunk = (
            chunk.model_copy(
                update={
                    "choices": [
                        choice.model_copy(
                            update={
                                "delta": delta.model_copy(update={"content": None}),
                            }
                        )
                    ]
                }
            )
            if has_content
            else chunk
        )
        tool_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
            chunk=tool_source_chunk,
            previous_chunks=[*previous_chunks, *streaming_chunks],
            component_info=component_info,
        )
        if last_block is not None and tool_chunk.tool_calls:
            # the parent converter uses the provider's tool call index as chunk index, which collides with the
            # reasoning or text block before the tool calls, so shift the chunk index past that block. Later tool
            # calls keep the shift of the first one, even if text is streamed between them. `ToolCallDelta.index` keeps
            # the provider's index, which the tool calls are assembled by.
            previous_tool_chunk = next((c for c in reversed(previous_chunks) if c.tool_calls), None)
            if previous_tool_chunk is not None and previous_tool_chunk.tool_calls:
                offset = (previous_tool_chunk.index or 0) - previous_tool_chunk.tool_calls[0].index
            else:
                offset = (last_block.index or 0) + 1
            tool_chunk = dataclasses.replace(tool_chunk, index=tool_chunk.tool_calls[0].index + offset)
        streaming_chunks.append(tool_chunk)

    return streaming_chunks


@component
class VLLMChatGenerator:
    """
    A component for generating chat completions using models served with [vLLM](https://docs.vllm.ai/).

    It expects a vLLM server to be running and accessible at the `api_base_url` parameter.

    ### Starting the vLLM server

    Before using this component, start a vLLM server:

    ```bash
    vllm serve Qwen/Qwen3-4B-Instruct-2507
    ```

    For reasoning models, start the server with the appropriate reasoning parser:

    ```bash
    vllm serve Qwen/Qwen3-0.6B --reasoning-parser qwen3
    ```

    For tool calling, the server must be started with `--enable-auto-tool-choice` and `--tool-call-parser`:

    ```bash
    vllm serve Qwen/Qwen3-0.6B --enable-auto-tool-choice --tool-call-parser hermes
    ```

    The available tool call parsers depend on the model. See the
    [vLLM tool calling docs](https://docs.vllm.ai/en/stable/features/tool_calling/) for the full list.

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.vllm import VLLMChatGenerator

    generator = VLLMChatGenerator(
        model="Qwen/Qwen3-0.6B",
        generation_kwargs={"max_tokens": 512, "temperature": 0.7},
    )

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]
    response = generator.run(messages=messages)
    print(response["replies"][0].text)
    ```

    ### Usage example with vLLM-specific parameters

    Pass the vLLM-specific parameters inside the `generation_kwargs`["extra_body"] dictionary.

    ```python
    from haystack_integrations.components.generators.vllm import VLLMChatGenerator

    generator = VLLMChatGenerator(
        model="Qwen/Qwen3-0.6B",
        generation_kwargs={
            "max_tokens": 512,
            "extra_body": {
                "top_k": 50,
                "min_tokens": 10,
                "repetition_penalty": 1.1,
            },
        },
    )
    ```

    ### Usage example with tool calling

    To use tool calling, start the vLLM server with `--enable-auto-tool-choice` and `--tool-call-parser`.

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack.tools import tool
    from haystack_integrations.components.generators.vllm import VLLMChatGenerator

    @tool
    def weather(city: str) -> str:
        \"\"\"Get the weather in a given city.\"\"\"
        return f"The weather in {city} is sunny"

    generator = VLLMChatGenerator(model="Qwen/Qwen3-0.6B", tools=[weather])

    messages = [ChatMessage.from_user("What is the weather in Paris?")]
    response = generator.run(messages=messages)
    print(response["replies"][0].tool_calls)
    ```

    ### Usage example with reasoning models

    To use reasoning models, start the vLLM server with `--reasoning-parser`.

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.vllm import VLLMChatGenerator

    generator = VLLMChatGenerator(model="Qwen/Qwen3-0.6B")

    messages = [ChatMessage.from_user("Solve step by step: what is 15 * 37?")]
    response = generator.run(messages=messages)
    reply = response["replies"][0]
    if reply.reasoning:
        print("Reasoning:", reply.reasoning.reasoning_text)
    print("Answer:", reply.text)
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str = "http://localhost:8000/v1",
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        tools: ToolsType | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMChatGenerator.

        :param model: The name of the model served by vLLM (e.g., "Qwen/Qwen3-0.6B").
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts
            [StreamingChunk](https://docs.haystack.deepset.ai/docs/data-classes#streamingchunk)
            as an argument.
        :param api_base_url: The base URL of the vLLM server.
        :param generation_kwargs: Additional parameters for text generation. These parameters are sent directly to
            the vLLM OpenAI-compatible endpoint. See
            [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/)
            for more details.
            Some of the supported parameters:
            - `max_tokens`: Maximum number of tokens to generate.
            - `temperature`: Sampling temperature.
            - `top_p`: Nucleus sampling parameter.
            - `n`: Number of completions to generate for each prompt.
            - `stop`: One or more sequences after which the model should stop generating tokens.
            - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the response.
            - `extra_body`: A dictionary of vLLM-specific parameters not part of the standard OpenAI API
              (e.g., `top_k`, `min_tokens`, `repetition_penalty`).
        :param timeout:
            Timeout for vLLM client calls. If not set, it defaults to the default set by the OpenAI client.
        :param max_retries:
            Maximum number of retries to attempt for failed requests. If not set, it defaults to the default
            set by the OpenAI client.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name. Not all models support tools.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """

        self.model = model
        self.api_key = api_key
        self.streaming_callback = streaming_callback
        self.api_base_url = api_base_url
        self.generation_kwargs = generation_kwargs or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.tools = tools
        self.http_client_kwargs = http_client_kwargs

        _check_duplicate_tool_names(flatten_tools_or_toolsets(self.tools))

        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

    def warm_up(self) -> None:
        """Create the synchronous OpenAI client."""
        if self._client is None:
            self._client = _create_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    async def warm_up_async(self) -> None:
        """Create the asynchronous OpenAI client."""
        if self._async_client is None:
            self._async_client = _create_async_openai_client(
                api_key=self.api_key,
                api_base_url=self.api_base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client_kwargs=self.http_client_kwargs,
            )

    def close(self) -> None:
        """Close the synchronous OpenAI client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous OpenAI client."""
        if self._async_client is not None:
            await self._async_client.close()
            self._async_client = None

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
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=serialize_tools_or_toolset(self.tools),
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VLLMChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _prepare_api_call(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None,
        generation_kwargs: dict[str, Any] | None,
        tools: ToolsType | None,
    ) -> dict[str, Any]:
        """Build the kwargs dict for the OpenAI chat completions API call."""
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        openai_formatted_messages = _convert_chat_history_to_vllm_format(messages=messages)

        flattened_tools = flatten_tools_or_toolsets(tools if tools is not None else self.tools)
        _check_duplicate_tool_names(flattened_tools)
        tool_definitions = None
        if flattened_tools:
            tool_definitions = [{"type": "function", "function": t.tool_spec} for t in flattened_tools]

        api_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_formatted_messages,
            "stream": streaming_callback is not None,
            **generation_kwargs,
        }
        if tool_definitions:
            api_kwargs["tools"] = tool_definitions

        return api_kwargs

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        """Handle a synchronous streaming response, extracting reasoning content from vLLM's reasoning chunks."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []

        for chunk in chat_completion:
            assert len(chunk.choices) <= 1  # noqa: S101
            for streaming_chunk in _convert_vllm_chunk_to_streaming_chunks(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            ):
                chunks.append(streaming_chunk)
                callback(streaming_chunk)

        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream[ChatCompletionChunk], callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        """Handle an asynchronous streaming response, extracting reasoning content from vLLM's reasoning chunks."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1  # noqa: S101
                for streaming_chunk in _convert_vllm_chunk_to_streaming_chunks(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                ):
                    chunks.append(streaming_chunk)
                    # sync callbacks are allowed in async contexts with Haystack >= 3.0, so only await async ones
                    callback_result = callback(streaming_chunk)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            raise

        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    @component.output_types(replies=list[ChatMessage])
    def run(
        self,
        messages: list[ChatMessage] | str,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the VLLM chat generator on the given input data.

        :param messages:
            A list of ChatMessage instances representing the input messages.
            If a string is provided, it is converted to a list containing a ChatMessage with user role.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These are merged per key with the
            `generation_kwargs` passed at initialization: keys provided here take precedence, keys set only
            at initialization are kept.
            For details on vLLM API parameters, see
            [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101
        messages = _normalize_messages(messages)

        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        api_kwargs = self._prepare_api_call(messages, streaming_callback, generation_kwargs, tools)
        chat_completion = self._client.chat.completions.create(**api_kwargs)

        if streaming_callback is not None:
            completions = self._handle_stream_response(chat_completion, streaming_callback)
        else:
            completions = [
                _convert_chat_completion_to_chat_message(chat_completion, choice) for choice in chat_completion.choices
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
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the VLLM chat generator on the given input data asynchronously.

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
            For details on vLLM API parameters, see
            [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        messages = _normalize_messages(messages)

        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        api_kwargs = self._prepare_api_call(messages, streaming_callback, generation_kwargs, tools)
        chat_completion = await self._async_client.chat.completions.create(**api_kwargs)

        if streaming_callback is not None:
            completions = await self._handle_async_stream_response(chat_completion, streaming_callback)
        else:
            completions = [
                _convert_chat_completion_to_chat_message(chat_completion, choice) for choice in chat_completion.choices
            ]

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}
