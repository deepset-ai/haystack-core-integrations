# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
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
    ChatRole,
    ComponentInfo,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
)
from haystack.tools import ToolsType
from haystack.utils import Secret
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion import Choice

from haystack_integrations.common.vllm.utils import _openai_client_kwargs


def _convert_vllm_chunk_to_streaming_chunks(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo
) -> list[StreamingChunk]:
    """
    Convert a vLLM streaming chunk to StreamingChunks, including reasoning content.

    A `StreamingChunk` holds only one of reasoning, text, or tool calls. vLLM can send several of them in one delta,
    for example when `</think>` falls inside a multi-token delta (`--stream-interval` > 1), so such a delta is split
    into a reasoning chunk, then a text chunk, then a tool call chunk.
    """
    choice = chunk.choices[0] if chunk.choices else None
    if choice is not None and choice.delta.content and choice.delta.tool_calls:
        # the parent converter would put both into one StreamingChunk, which rejects that, so convert them separately
        sources = [
            # text half: drop tool_calls and finish_reason (the trailing tool-call chunk carries finish_reason)
            chunk.model_copy(
                update={
                    "choices": [
                        choice.model_copy(
                            update={
                                "finish_reason": None,
                                "delta": choice.delta.model_copy(update={"tool_calls": None}),
                            }
                        )
                    ]
                }
            ),
            # tool-call half: drop content
            chunk.model_copy(
                update={
                    "choices": [choice.model_copy(update={"delta": choice.delta.model_copy(update={"content": None})})]
                }
            ),
        ]
    else:
        sources = [chunk]
    payload_chunks = [
        _convert_chat_completion_chunk_to_streaming_chunk(
            chunk=source, previous_chunks=previous_chunks, component_info=component_info
        )
        for source in sources
    ]
    # the parent converter ignores the `reasoning` field vLLM adds to the delta
    reasoning_text = getattr(choice.delta, "reasoning", None) if choice is not None else None
    has_payload = any(c.content or c.tool_calls for c in payload_chunks)
    # the last chunk with reasoning, content, or tool calls tells which block the stream is currently in
    last_block = next((c for c in reversed(previous_chunks) if c.reasoning or c.content or c.tool_calls), None)

    streaming_chunks: list[StreamingChunk] = []
    if reasoning_text:
        # reasoning that follows text or a tool call starts a new block instead of reusing index 0
        if last_block is None:
            index, start = 0, True
        elif last_block.reasoning is not None:
            index, start = last_block.index or 0, False
        else:
            index, start = (last_block.index or 0) + 1, True
        reasoning_chunk = StreamingChunk(
            content="",
            reasoning=ReasoningContent(reasoning_text=reasoning_text),
            index=index,
            start=start,
            # a reasoning-only delta replaces the parent's empty chunk, so it carries its finish reason
            finish_reason=None if has_payload else payload_chunks[0].finish_reason,
            component_info=component_info,
            # the text or tool calls that follow carry the finish reason and tool calls of the delta
            meta=payload_chunks[0].meta
            if not has_payload
            else {**payload_chunks[0].meta, "finish_reason": None, "tool_calls": None},
        )
        if not has_payload:
            return [reasoning_chunk]
        streaming_chunks.append(reasoning_chunk)
        last_block = reasoning_chunk

    for payload_chunk in payload_chunks:
        streaming_chunk = payload_chunk
        if streaming_chunk.content:
            # the parent converter marks the second chunk of a stream as start, assuming the first one only carries
            # `role`, so derive start and index from the previous block instead. Text after tool calls keeps the
            # parent's start and index.
            if last_block is None:
                streaming_chunk = replace(streaming_chunk, start=True, index=0)
            elif last_block.reasoning is not None:
                streaming_chunk = replace(streaming_chunk, start=True, index=(last_block.index or 0) + 1)
            elif last_block.content:
                streaming_chunk = replace(streaming_chunk, start=False, index=last_block.index)
        elif streaming_chunk.tool_calls and last_block is not None:
            # the parent converter uses the provider's tool call index as chunk index, which collides with the
            # reasoning or text block before the tool calls, so shift the chunk index past that block. Later tool
            # calls keep the shift of the first one. `ToolCallDelta.index` keeps the provider's index, which the tool
            # calls are assembled by.
            seen = [*previous_chunks, *streaming_chunks]
            previous_tool_chunk = next((c for c in reversed(seen) if c.tool_calls), None)
            if previous_tool_chunk is not None and previous_tool_chunk.tool_calls:
                offset = (previous_tool_chunk.index or 0) - previous_tool_chunk.tool_calls[0].index
            else:
                offset = (last_block.index or 0) + 1
            streaming_chunk = replace(streaming_chunk, index=streaming_chunk.tool_calls[0].index + offset)
        streaming_chunks.append(streaming_chunk)
        if streaming_chunk.content or streaming_chunk.tool_calls:
            last_block = streaming_chunk
    return streaming_chunks


@component
class VLLMChatGenerator(OpenAIChatGenerator):
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

        super(VLLMChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,  # type: ignore[arg-type] # vLLM accepts None, differently from OpenAIChatGenerator
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            http_client_kwargs=http_client_kwargs,
        )
        self.api_base_url: str = api_base_url

    def _client_kwargs(self) -> dict[str, Any]:
        """Build vLLM client options with optional authentication and SDK defaults."""
        return _openai_client_kwargs(
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(VLLMChatGenerator, self).to_dict()  # noqa: UP008
        init_parameters = inspect.signature(type(self).__init__).parameters
        data["init_parameters"] = {
            key: value for key, value in data["init_parameters"].items() if key in init_parameters
        }
        return data

    def _prepare_api_call(
        self,
        *,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, Any]:
        """Prepare the request, omitting a trailing assistant message without text or tool calls."""
        # An empty final assistant turn can make reasoning models stop without writing an answer.
        if (
            messages
            and messages[-1].is_from(ChatRole.ASSISTANT)
            and not messages[-1].texts
            and not messages[-1].tool_calls
        ):
            messages = messages[:-1]
        return super(VLLMChatGenerator, self)._prepare_api_call(  # noqa: UP008
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )

    def _convert_chat_completion_to_chat_message(
        self, completion: ChatCompletion | ParsedChatCompletion, choice: Choice
    ) -> ChatMessage:
        """Convert a completion while preserving vLLM reasoning content."""
        message = super(VLLMChatGenerator, self)._convert_chat_completion_to_chat_message(  # noqa: UP008
            completion=completion, choice=choice
        )
        reasoning = getattr(choice.message, "reasoning", None)
        if not reasoning:
            return message
        return ChatMessage.from_assistant(
            text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=reasoning
        )

    def _handle_stream_response(
        self, chat_completion: Stream[ChatCompletionChunk], callback: SyncStreamingCallbackT
    ) -> list[ChatMessage]:
        """Handle vLLM streams that can emit multiple chunks per API delta."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
            for streaming_chunk in _convert_vllm_chunk_to_streaming_chunks(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            ):
                chunks.append(streaming_chunk)
                callback(streaming_chunk)
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream[ChatCompletionChunk], callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        """Handle async vLLM streams that can emit multiple chunks per API delta."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
                for streaming_chunk in _convert_vllm_chunk_to_streaming_chunks(
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
