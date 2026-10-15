# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import replace
from typing import Any

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import (
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
    _convert_chat_completion_to_chat_message,
)
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message, _normalize_messages
from haystack.dataclasses import (
    ChatMessage,
    ChatRole,
    ComponentInfo,
    ReasoningContent,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    select_streaming_callback,
)
from haystack.dataclasses.streaming_chunk import _invoke_streaming_callback
from haystack.tools import ToolsType, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai import AsyncStream, Stream
from openai.lib._pydantic import to_strict_json_schema
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ParsedChatCompletion
from openai.types.chat.chat_completion import Choice
from pydantic import BaseModel

from haystack_integrations.common.otari.utils import _warn_if_eu_key_on_default_url


def _reasoning_content(message_or_delta: Any) -> ReasoningContent | None:
    """
    Read the reasoning that Otari adds to a response message or a streamed delta.

    Otari sends the reasoning as `reasoning`, a plain string, and provider data that has to be sent back on the next
    request, such as Anthropic's thinking signature, as `extra_content`.
    """
    reasoning = getattr(message_or_delta, "reasoning", None)
    if isinstance(reasoning, dict):
        reasoning = reasoning.get("content")
    reasoning_text = reasoning if isinstance(reasoning, str) else ""
    extra_content = getattr(message_or_delta, "extra_content", None)
    if not isinstance(extra_content, dict):
        extra_content = {}
    if not reasoning_text and not extra_content:
        return None
    return ReasoningContent(
        reasoning_text=reasoning_text, extra={"extra_content": extra_content} if extra_content else {}
    )


def _merge_extra_content(total: dict[str, Any], extra_content: dict[str, Any]) -> None:
    """Merge the `extra_content` of one streamed delta into the total, per provider namespace."""
    for namespace, values in extra_content.items():
        if isinstance(values, dict) and isinstance(total.get(namespace), dict):
            total[namespace] = {**total[namespace], **values}
        else:
            total[namespace] = values


@component
class OtariChatGenerator(OpenAIChatGenerator):
    """
    Completes chats using models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

    Otari is an OpenAI-compatible LLM gateway that routes requests to many providers and applies API keys, budgets,
    and usage tracking. By default, this component calls the otari.ai gateway. For an account in otari.ai's EU
    region, whose API keys start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`.
    To call a gateway you run yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

    Name models with a selector the gateway accepts, such as `provider:model` (`"openai:gpt-5-mini"`,
    `"anthropic:claude-sonnet-4-6"`), a model alias, or a routing policy name. For details, see the
    [Otari model docs](https://github.com/mozilla-ai/otari/blob/main/docs/models.md).

    You can pass any parameter of the Otari chat completions endpoint through `generation_kwargs`, at initialization
    or in the `run` method. Request headers, such as Otari's `Idempotency-Key`, go into
    `generation_kwargs["extra_headers"]`.

    When the gateway prices a request, `meta["usage"]` of the reply contains its cost in US dollars as `cost_usd`,
    and `pricing_source`. When streaming, pass `generation_kwargs={"stream_options": {"include_usage": True}}` to
    receive usage.

    When the model reasons, for example with `generation_kwargs={"reasoning_effort": "low"}`, the reasoning is in the
    `reasoning` of the reply and, when streaming, of the streamed chunks. If the gateway adds provider data that the
    next request needs, such as Anthropic's thinking signature, the component sends the reasoning back with the reply
    in later requests. Anthropic models need this to keep reasoning across tool calls.

    ### Usage example

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.otari import OtariChatGenerator

    # Reads the API key from the OTARI_API_KEY environment variable
    generator = OtariChatGenerator()
    response = generator.run([ChatMessage.from_user("What's Natural Language Processing? Be brief.")])
    print(response["replies"][0].text)
    # >> Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to ...
    print(response["replies"][0].meta["usage"]["cost_usd"])
    # >> 0.000279
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
        model: str = "openai:gpt-5-mini",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str = "https://api.otari.ai/api/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool = False,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OtariChatGenerator.

        :param api_key:
            The Otari API key: a key issued by your gateway, or an otari.ai API key.
            Defaults to the `OTARI_API_KEY` environment variable.
        :param model:
            The model selector, for example `"openai:gpt-5-mini"`. The gateway must have the provider configured.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
            Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
            `"https://eu.api.otari.ai/api/v1"`.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are sent directly to the Otari endpoint.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream_options`: Pass `{"include_usage": True}` to receive usage, including cost, when streaming.
            - `extra_headers`: Headers to send with the request, for example Otari's `Idempotency-Key`.
            - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
                If provided, the output will always be validated against this
                format (unless the model returns a tool call).
                For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
                Notes:
                - For structured outputs with streaming,
                  the `response_format` must be a JSON schema and not a Pydantic model.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
        :param timeout:
            The timeout for the Otari API call.
        :param max_retries:
            Maximum number of retries to contact Otari after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        # the @component decorator recreates the class, so the zero-argument form of super() cannot be used
        super(OtariChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )

    def warm_up(self) -> None:
        """
        Warm up the tools and initialize the synchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.client is None
        super(OtariChatGenerator, self).warm_up()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

    async def warm_up_async(self) -> None:
        """
        Warm up the tools and initialize the asynchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.async_client is None
        await super(OtariChatGenerator, self).warm_up_async()  # noqa: UP008
        if creates_client:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)

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
        # which would serialize some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=serialize_tools_or_toolset(self.tools),
            tools_strict=self.tools_strict,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    def _convert_chat_completion_to_chat_message(
        self, completion: ChatCompletion | ParsedChatCompletion, choice: Choice
    ) -> ChatMessage:
        """Convert a choice of a chat completion into a ChatMessage, including Otari's reasoning."""
        message = _convert_chat_completion_to_chat_message(completion=completion, choice=choice)
        reasoning = _reasoning_content(choice.message)
        if reasoning is None:
            return message
        return ChatMessage.from_assistant(
            text=message.text, meta=message.meta, name=message.name, tool_calls=message.tool_calls, reasoning=reasoning
        )

    def _convert_chat_completion_chunk_to_streaming_chunk(
        self,
        chunk: ChatCompletionChunk,
        previous_chunks: list[StreamingChunk],
        component_info: ComponentInfo | None = None,
    ) -> StreamingChunk:
        """Convert a streamed chunk into a StreamingChunk, including Otari's reasoning."""
        streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
            chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
        )
        reasoning = _reasoning_content(chunk.choices[0].delta) if chunk.choices else None
        # a StreamingChunk holds one kind of content, and Otari streams reasoning and text in separate deltas
        if reasoning is not None and not streaming_chunk.content and not streaming_chunk.tool_calls:
            # print_streaming_chunk prints its reasoning header on the first reasoning chunk
            start = not any(previous.reasoning for previous in previous_chunks)
            return replace(streaming_chunk, reasoning=reasoning, index=0, start=start)
        if streaming_chunk.content and any(previous.reasoning for previous in previous_chunks):
            # after reasoning, the parent no longer marks the first text chunk as the start of the text
            return replace(streaming_chunk, start=not any(previous.content for previous in previous_chunks))
        return streaming_chunk

    def _convert_streaming_chunks_to_chat_message(self, chunks: list[StreamingChunk]) -> ChatMessage:
        """Combine streamed chunks into a ChatMessage, keeping the `extra_content` that goes with the reasoning."""
        message = _convert_streaming_chunks_to_chat_message(chunks=chunks)
        extra_content: dict[str, Any] = {}
        for chunk in chunks:
            if chunk.reasoning and (chunk_extra_content := chunk.reasoning.extra.get("extra_content")):
                _merge_extra_content(extra_content, chunk_extra_content)
        if not extra_content:
            return message
        reasoning = ReasoningContent(
            reasoning_text=message.reasoning.reasoning_text if message.reasoning else "",
            extra={"extra_content": extra_content},
        )
        return ChatMessage.from_assistant(
            text=message.text, meta=message.meta, name=message.name, tool_calls=message.tool_calls, reasoning=reasoning
        )

    def _convert_message_to_api_format(self, message: ChatMessage) -> dict[str, Any]:
        """Convert a ChatMessage into the request format, including an assistant reply's reasoning to replay."""
        formatted = message.to_openai_dict_format()
        reasoning = message.reasoning
        # Only reasoning that comes with provider data to replay, such as Anthropic's thinking signature, goes back.
        # Some providers reject reasoning in a request: Cohere answers it with a 400.
        if reasoning is not None and message.is_from(ChatRole.ASSISTANT):
            if extra_content := reasoning.extra.get("extra_content"):
                formatted["reasoning"] = reasoning.reasoning_text
                formatted["extra_content"] = extra_content
        return formatted

    # Bridge for haystack-ai < 3.5, whose OpenAIChatGenerator doesn't call the four conversion methods above.
    # The methods below are copied from OpenAIChatGenerator on Haystack main, except that _prepare_api_call reformats
    # the messages. Remove them once the minimum haystack-ai version is 3.5.

    def _prepare_api_call(
        self,
        *,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool | None = None,
    ) -> dict[str, Any]:
        api_args = super(OtariChatGenerator, self)._prepare_api_call(  # noqa: UP008
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        api_args["messages"] = [self._convert_message_to_api_format(message) for message in messages]
        return api_args

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
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances, including the
              model's reasoning when it returns any.
        """
        self.warm_up()

        messages = _normalize_messages(messages)

        if len(messages) == 0:
            return {"replies": []}

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )
        chat_completion: Stream[ChatCompletionChunk] | ChatCompletion | ParsedChatCompletion

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        openai_endpoint = api_args.pop("openai_endpoint")
        assert self.client is not None  # noqa: S101 # mypy: client is built by warm_up above
        openai_endpoint_method = getattr(self.client.chat.completions, openai_endpoint)
        chat_completion = openai_endpoint_method(**api_args)

        if streaming_callback is not None:
            completions = self._handle_stream_response(
                # we cannot check isinstance(chat_completion, Stream) because some observability tools wrap Stream
                # and return a different type. See https://github.com/deepset-ai/haystack/issues/9014.
                chat_completion,  # type: ignore[arg-type]
                streaming_callback,
            )

        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."  # noqa: S101
            completions = [
                self._convert_chat_completion_to_chat_message(completion=chat_completion, choice=choice)
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
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
            If set, it will override the `tools_strict` parameter set during component initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances, including the
              model's reasoning when it returns any.
        """
        await self.warm_up_async()

        messages = _normalize_messages(messages)

        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )
        chat_completion: AsyncStream[ChatCompletionChunk] | ChatCompletion | ParsedChatCompletion

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
        assert self.async_client is not None  # noqa: S101 # mypy: async_client is built by warm_up_async above
        openai_endpoint_method = getattr(self.async_client.chat.completions, openai_endpoint)
        chat_completion = await openai_endpoint_method(**api_args)

        if streaming_callback is not None:
            completions = await self._handle_async_stream_response(
                # we cannot check isinstance(chat_completion, AsyncStream) because some observability tools wrap
                # AsyncStream and return a different type. See https://github.com/deepset-ai/haystack/issues/9014.
                chat_completion,  # type: ignore[arg-type]
                streaming_callback,
            )

        else:
            assert isinstance(chat_completion, ChatCompletion), "Unexpected response type for non-streaming request."  # noqa: S101
            completions = [
                self._convert_chat_completion_to_chat_message(completion=chat_completion, choice=choice)
                for choice in chat_completion.choices
            ]

        # before returning, do post-processing of the completions
        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
            chunk_delta = self._convert_chat_completion_chunk_to_streaming_chunk(
                chunk=chunk, previous_chunks=chunks, component_info=component_info
            )
            chunks.append(chunk_delta)
            callback(chunk_delta)
        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream, callback: StreamingCallbackT
    ) -> list[ChatMessage]:
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1, "Streaming responses should have at most one choice."  # noqa: S101
                chunk_delta = self._convert_chat_completion_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                )
                chunks.append(chunk_delta)
                await _invoke_streaming_callback(callback, chunk_delta)

        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            # close the stream when task is cancelled
            # asyncio.shield ensures the close operation completes
            # https://docs.python.org/3/library/asyncio-task.html#shielding-from-cancellation
            raise  # Re-raise to propagate cancellation

        return [self._convert_streaming_chunks_to_chat_message(chunks=chunks)]
