# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
from dataclasses import replace
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import (
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
    _convert_chat_completion_to_chat_message,
)
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.dataclasses import ChatMessage, StreamingCallbackT, StreamingChunk
from haystack.dataclasses.chat_message import ReasoningContent
from haystack.dataclasses.streaming_chunk import ComponentInfo, SyncStreamingCallbackT, select_streaming_callback
from haystack.tools import ToolsType
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from pydantic import BaseModel


def _get_reasoning(message: ChatCompletionMessage | ChoiceDelta) -> ReasoningContent | None:
    text = getattr(message, "reasoning", None)
    return ReasoningContent(reasoning_text=text) if isinstance(text, str) and text else None


def _convert_stackit_completion_to_chat_message(completion: ChatCompletion, choice: Choice) -> ChatMessage:
    message = _convert_chat_completion_to_chat_message(completion, choice)
    return ChatMessage.from_assistant(
        text=message.text, tool_calls=message.tool_calls, meta=message.meta, reasoning=_get_reasoning(choice.message)
    )


def _convert_stackit_chunk_to_streaming_chunk(
    chunk: ChatCompletionChunk, previous_chunks: list[StreamingChunk], component_info: ComponentInfo | None = None
) -> StreamingChunk:
    streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
        chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
    )
    reasoning = _get_reasoning(chunk.choices[0].delta) if chunk.choices else None
    if reasoning:
        return replace(
            streaming_chunk,
            reasoning=reasoning,
            index=0,
            start=not any(previous.reasoning for previous in previous_chunks),
        )

    # Text after reasoning belongs to a separate content block.
    if streaming_chunk.content and any(previous.reasoning for previous in previous_chunks):
        return replace(
            streaming_chunk,
            index=1,
            start=not any(previous.content for previous in previous_chunks),
        )
    return streaming_chunk


@component
class STACKITChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using STACKIT generative models through their model serving service.

    Users can pass any text generation parameters valid for the STACKIT Chat Completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    ### Usage example
    ```python
    from haystack_integrations.components.generators.stackit import STACKITChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = STACKITChatGenerator(model="cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic")

    result = generator.run([ChatMessage.from_user("Tell me a joke.")])
    print(result)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        "Qwen/Qwen3.6-27B",
        "cortecs/Llama-3.3-70B-Instruct-FP8-Dynamic",
        "openai/gpt-oss-120b",
        "google/gemma-3-27b-it",
        "openai/gpt-oss-20b",
    ]
    """A non-exhaustive list of chat models supported by this component.
    See https://docs.stackit.cloud/products/data-and-ai/ai-model-serving/basics/available-shared-models
    for the full list."""

    def __init__(
        self,
        model: str,
        api_key: Secret = Secret.from_env_var("STACKIT_API_KEY"),
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1",
        generation_kwargs: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of STACKITChatGenerator class.

        :param model:
            The name of the chat completion model to use.
        :param api_key:
            The STACKIT API key.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The STACKIT API Base url.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the STACKIT endpoint.
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
            - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
                If provided, the output will always be validated against this
                format (unless the model returns a tool call).
                For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
                Notes:
                - For structured outputs with streaming,
                  the `response_format` must be a JSON schema and not a Pydantic model.
        :param timeout:
            Timeout for STACKIT client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact STACKIT after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(STACKITChatGenerator, self).__init__(  # noqa: UP008
            model=model,
            api_key=api_key,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        generation_kwargs = self.generation_kwargs.copy()
        response_format = generation_kwargs.get("response_format")
        # If the response format is a Pydantic model, it's converted to openai's json schema format
        # If it's already a json schema, it's left as is
        if response_format and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            json_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "strict": True,
                    "schema": to_strict_json_schema(response_format),
                },
            }

            generation_kwargs["response_format"] = json_schema

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialiaze some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=generation_kwargs,
            api_key=self.api_key.to_dict(),
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
            chunk_delta = _convert_stackit_chunk_to_streaming_chunk(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            )
            chunks.append(chunk_delta)
            callback(chunk_delta)
        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream, callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."
                chunk_delta = _convert_stackit_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                )
                chunks.append(chunk_delta)
                # Equivalent to the core helper, which is unavailable in Haystack 2.22.
                result = callback(chunk_delta)
                if inspect.isawaitable(result):
                    await result

        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            # close the stream when task is cancelled
            # asyncio.shield ensures the close operation completes
            # https://docs.python.org/3/library/asyncio-task.html#shielding-from-cancellation
            raise  # Re-raise to propagate cancellation

        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

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
        Invokes chat completion based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage instances representing the input messages. If a string is provided, it is converted
            to a list containing a ChatMessage with user role.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These are merged per key with the
            `generation_kwargs` passed at initialization: keys provided here take precedence, keys set
            only at initialization are kept.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        self.warm_up()

        if isinstance(messages, str):
            messages = [ChatMessage.from_user(messages)]

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
        assert self.client is not None  # mypy: client is built by warm_up above
        openai_endpoint_method = getattr(self.client.chat.completions, openai_endpoint)
        chat_completion = openai_endpoint_method(**api_args)

        if streaming_callback is not None:
            completions = self._handle_stream_response(
                # we cannot check isinstance(chat_completion, Stream) because some observability tools wrap Stream
                # and return a different type. See https://github.com/deepset-ai/haystack/issues/9014.
                chat_completion,
                streaming_callback,
            )

        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                _convert_stackit_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
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
        Asynchronously invokes chat completion based on the provided messages and generation parameters.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param messages:
            A list of ChatMessage instances representing the input messages. If a string is provided, it is converted
            to a list containing a ChatMessage with user role.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream. Async callbacks are
            preferred; a sync callback is accepted but will run synchronously on the event loop and may block it.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These are merged per key with the
            `generation_kwargs` passed at initialization: keys provided here take precedence, keys set
            only at initialization are kept.
            For details on OpenAI API parameters, see [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        # Haystack 2.22 creates both clients in __init__ and has no async warm-up.
        warm_up_async = getattr(self, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        else:
            self.warm_up()

        if isinstance(messages, str):
            messages = [ChatMessage.from_user(messages)]

        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )

        openai_endpoint = api_args.pop("openai_endpoint")
        assert self.async_client is not None  # mypy: async_client is built by warm_up_async above
        openai_endpoint_method = getattr(self.async_client.chat.completions, openai_endpoint)
        chat_completion = await openai_endpoint_method(**api_args)

        if streaming_callback is not None:
            completions = await self._handle_async_stream_response(
                # we cannot check isinstance(chat_completion, AsyncStream) because some observability tools wrap
                # AsyncStream and return a different type. See https://github.com/deepset-ai/haystack/issues/9014.
                chat_completion,
                streaming_callback,
            )

        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."
            completions = [
                _convert_stackit_completion_to_chat_message(chat_completion, choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}
