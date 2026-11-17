# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from dataclasses import replace
from typing import Any

from haystack import component
from haystack.components.generators.chat import OpenAIChatGenerator
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

    def _convert_chat_completion_chunk_to_streaming_chunk(
        self,
        chunk: ChatCompletionChunk,
        previous_chunks: list[StreamingChunk],
        component_info: ComponentInfo | None = None,
    ) -> StreamingChunk:
        """
        Convert a chunk while preserving vLLM reasoning and the start of reasoning and text.

        A combined delta is split before this runs. This method still returns reasoning alone when `reasoning` is
        set, because one StreamingChunk cannot hold reasoning together with content or tool calls.
        """
        reasoning_text = getattr(chunk.choices[0].delta, "reasoning", None) if chunk.choices else None
        if reasoning_text:
            streaming_chunk = StreamingChunk(
                content="",
                reasoning=ReasoningContent(reasoning_text=reasoning_text),
                index=0,
                start=not any(c.reasoning for c in previous_chunks),
                component_info=component_info,
                meta={
                    "model": chunk.model,
                    "index": chunk.choices[0].index,
                    "finish_reason": chunk.choices[0].finish_reason,
                },
            )
            # reasoning that follows text or a tool call continues the block index instead of restarting at 0
            last_block = next((c for c in reversed(previous_chunks) if c.reasoning or c.content or c.tool_calls), None)
            if last_block is not None and last_block.reasoning is None:
                return replace(streaming_chunk, start=True, index=(last_block.index or 0) + 1)
            if last_block is not None and last_block.reasoning is not None:
                return replace(streaming_chunk, start=False, index=last_block.index)
            return streaming_chunk
        converted = super(VLLMChatGenerator, self)._convert_chat_completion_chunk_to_streaming_chunk(  # noqa: UP008
            chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
        )
        if converted.tool_calls:
            return self._with_tool_block_index(converted, previous_chunks)
        if converted.content:
            return self._with_content_block_index(converted, previous_chunks)
        return converted

    def _with_content_block_index(
        self, converted: StreamingChunk, previous_chunks: list[StreamingChunk]
    ) -> StreamingChunk:
        """Keep streamed text in one block, including the first text after reasoning."""
        last_content = next((chunk for chunk in reversed(previous_chunks) if chunk.content), None)
        if last_content is not None:
            return replace(converted, start=False, index=last_content.index)
        if any(chunk.reasoning for chunk in previous_chunks):
            last_reasoning = next(chunk for chunk in reversed(previous_chunks) if chunk.reasoning)
            return replace(converted, start=True, index=(last_reasoning.index or 0) + 1)
        return converted

    def _with_tool_block_index(
        self, converted: StreamingChunk, previous_chunks: list[StreamingChunk]
    ) -> StreamingChunk:
        """
        Shift the chunk index past an earlier reasoning or text block.

        The parent converter uses the provider's tool call index as the chunk index, which collides with the block
        before the tool calls. `ToolCallDelta.index` keeps the provider index the tool calls are assembled by.
        """
        if not converted.tool_calls:
            return converted
        previous_tool = next((chunk for chunk in reversed(previous_chunks) if chunk.tool_calls), None)
        if previous_tool is not None and previous_tool.tool_calls:
            offset = (previous_tool.index or 0) - previous_tool.tool_calls[0].index
        else:
            last_block = next((chunk for chunk in reversed(previous_chunks) if chunk.reasoning or chunk.content), None)
            if last_block is None:
                return converted
            offset = (last_block.index or 0) + 1
        if offset == 0:
            return converted
        return replace(converted, index=converted.tool_calls[0].index + offset)

    def _convert_chat_completion_chunk_to_streaming_chunks(
        self,
        chunk: ChatCompletionChunk,
        previous_chunks: list[StreamingChunk],
        component_info: ComponentInfo | None = None,
    ) -> list[StreamingChunk]:
        """
        Convert one vLLM chunk, splitting a delta that mixes reasoning, content, and tool calls.

        A `StreamingChunk` holds at most one of those, so a combined delta is emitted as reasoning, then content,
        then tool calls. vLLM sends such deltas when `</think>` falls inside one multi-token chunk.
        """
        if not chunk.choices:
            return [
                self._convert_chat_completion_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
                )
            ]

        choice = chunk.choices[0]
        delta = choice.delta
        reasoning_text = getattr(delta, "reasoning", None)
        has_content = bool(delta.content)
        has_tool_calls = bool(delta.tool_calls)
        if not ((reasoning_text and (has_content or has_tool_calls)) or (has_content and has_tool_calls)):
            return [
                self._convert_chat_completion_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
                )
            ]

        last_block = next((c for c in reversed(previous_chunks) if c.reasoning or c.content or c.tool_calls), None)
        streaming_chunks: list[StreamingChunk] = []

        if reasoning_text:
            reasoning_source = chunk.model_copy(
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
            reasoning_chunk = self._convert_chat_completion_chunk_to_streaming_chunk(
                chunk=reasoning_source, previous_chunks=previous_chunks, component_info=component_info
            )
            streaming_chunks.append(reasoning_chunk)
            last_block = reasoning_chunk

        if has_content:
            content_update: dict[str, Any] = {"tool_calls": None}
            if reasoning_text:
                # an extra field, not a declared ChoiceDelta attribute; leaving it set makes the hook drop content
                content_update["reasoning"] = None
            content_choice_update: dict[str, Any] = {"delta": delta.model_copy(update=content_update)}
            if has_tool_calls:
                content_choice_update["finish_reason"] = None
            content_source = chunk.model_copy(update={"choices": [choice.model_copy(update=content_choice_update)]})
            content_chunk = self._convert_chat_completion_chunk_to_streaming_chunk(
                chunk=content_source,
                previous_chunks=[*previous_chunks, *streaming_chunks],
                component_info=component_info,
            )
            # the parent marks the second chunk as start, assuming the first only carries role, and gives text index
            # None when role is set. The first text of a combined delta is the start of that block.
            if last_block is None:
                content_chunk = replace(content_chunk, start=True, index=0)
            elif last_block.content:
                content_chunk = replace(content_chunk, start=False, index=last_block.index)
            else:
                content_chunk = replace(content_chunk, start=True, index=(last_block.index or 0) + 1)
            streaming_chunks.append(content_chunk)

        if has_tool_calls:
            tool_update: dict[str, Any] = {"content": None}
            if reasoning_text:
                tool_update["reasoning"] = None
            tool_source = chunk.model_copy(
                update={"choices": [choice.model_copy(update={"delta": delta.model_copy(update=tool_update)})]}
            )
            tool_chunk = self._convert_chat_completion_chunk_to_streaming_chunk(
                chunk=tool_source,
                previous_chunks=[*previous_chunks, *streaming_chunks],
                component_info=component_info,
            )
            streaming_chunks.append(tool_chunk)

        return streaming_chunks

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        """Emit every reasoning, content, and tool call payload carried by one streamed response."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
            for streaming_chunk in self._convert_chat_completion_chunk_to_streaming_chunks(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            ):
                chunks.append(streaming_chunk)
                callback(streaming_chunk)
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream, callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        """Emit every reasoning, content, and tool call payload carried by one streamed response."""
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
                for streaming_chunk in self._convert_chat_completion_chunk_to_streaming_chunks(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                ):
                    chunks.append(streaming_chunk)
                    callback_result = callback(streaming_chunk)
                    if inspect.isawaitable(callback_result):
                        await callback_result
        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            raise
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]
