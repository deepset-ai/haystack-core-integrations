import asyncio
from typing import Any

from haystack import default_from_dict, default_to_dict
from haystack.components.generators.chat.openai import (
    OpenAIChatGenerator,
    _check_finish_reason,
    _convert_chat_completion_chunk_to_streaming_chunk,
)
from haystack.components.generators.chat.openai import (
    _convert_chat_completion_to_chat_message as _openai_convert_chat_completion_to_chat_message,
)
from haystack.components.generators.utils import _convert_streaming_chunks_to_chat_message
from haystack.core.component import component
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.chat_message import ReasoningContent
from haystack.dataclasses.streaming_chunk import (
    AsyncStreamingCallbackT,
    ComponentInfo,
    StreamingCallbackT,
    StreamingChunk,
    SyncStreamingCallbackT,
    select_streaming_callback,
)
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils import Secret, deserialize_callable, serialize_callable
from openai import AsyncStream, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice


def _convert_chat_completion_to_chat_message(completion: ChatCompletion, choice: Choice) -> ChatMessage:
    """
    Converts the non-streaming response from the vLLM API to a ChatMessage.

    Delegates to the OpenAI converter for standard fields (text, tool calls, meta) and adds
    reasoning content if present in the response.

    :param completion: The completion returned by the vLLM API.
    :param choice: The choice returned by the vLLM API.
    :return: The ChatMessage.
    """
    message = _openai_convert_chat_completion_to_chat_message(completion, choice)
    reasoning_text = getattr(choice.message, "reasoning", None)
    if not reasoning_text:
        return message
    return ChatMessage.from_assistant(
        text=message.text,
        tool_calls=message.tool_calls,
        meta=message.meta,
        reasoning=ReasoningContent(reasoning_text=reasoning_text),
    )


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

        api_key = api_key if api_key and api_key.resolve_value() else Secret.from_token("placeholder-api-key")

        super(VLLMChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            http_client_kwargs=http_client_kwargs,
        )

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

    def _handle_stream_response(self, chat_completion: Stream, callback: SyncStreamingCallbackT) -> list[ChatMessage]:
        """
        Handle a synchronous streaming response, extracting reasoning content from vLLM's reasoning chunks.
        """
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        for chunk in chat_completion:
            assert len(chunk.choices) <= 1  # noqa: S101

            reasoning_text = None
            if chunk.choices:
                reasoning_text = getattr(chunk.choices[0].delta, "reasoning", None)

            if reasoning_text:
                streaming_chunk = StreamingChunk(
                    content="",
                    reasoning=ReasoningContent(reasoning_text=reasoning_text),
                    index=0,
                    start=not any(c.reasoning for c in chunks),
                    component_info=component_info,
                    meta={
                        "model": chunk.model,
                        "index": chunk.choices[0].index,
                        "finish_reason": chunk.choices[0].finish_reason,
                    },
                )
            else:
                # delegate non-reasoning chunks to OpenAIChatGenerator converter
                streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
                    chunk=chunk, previous_chunks=chunks, component_info=component_info
                )

            chunks.append(streaming_chunk)
            callback(streaming_chunk)

        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    async def _handle_async_stream_response(
        self, chat_completion: AsyncStream[ChatCompletionChunk], callback: AsyncStreamingCallbackT
    ) -> list[ChatMessage]:
        """
        Handle an asynchronous streaming response, extracting reasoning content from vLLM's reasoning chunks.
        """
        component_info = ComponentInfo.from_component(self)
        chunks: list[StreamingChunk] = []
        try:
            async for chunk in chat_completion:
                assert len(chunk.choices) <= 1  # noqa: S101

                reasoning_text = None
                if chunk.choices:
                    reasoning_text = getattr(chunk.choices[0].delta, "reasoning", None)

                if reasoning_text:
                    streaming_chunk = StreamingChunk(
                        content="",
                        reasoning=ReasoningContent(reasoning_text=reasoning_text),
                        index=0,
                        start=not any(c.reasoning for c in chunks),
                        component_info=component_info,
                        meta={
                            "model": chunk.model,
                            "index": chunk.choices[0].index,
                            "finish_reason": chunk.choices[0].finish_reason,
                        },
                    )
                else:
                    # delegate non-reasoning chunks to OpenAIChatGenerator converter
                    streaming_chunk = _convert_chat_completion_chunk_to_streaming_chunk(
                        chunk=chunk, previous_chunks=chunks, component_info=component_info
                    )

                chunks.append(streaming_chunk)
                await callback(streaming_chunk)
        except asyncio.CancelledError:
            await asyncio.shield(chat_completion.close())
            raise

        return [_convert_streaming_chunks_to_chat_message(chunks=chunks)]

    @component.output_types(replies=list[ChatMessage])
    # tools_strict is intentionally omitted: vLLM does not support it
    def run(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the VLLM chat generator on the given input data.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on vLLM API parameters, see
            [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        resolved_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )

        # Streaming: OpenAIChatGenerator handles this correctly.
        if resolved_callback is not None:
            return super().run(messages, streaming_callback, generation_kwargs, tools=tools, tools_strict=False)

        # Non-streaming: handle reasoning.
        if not self._is_warmed_up:
            self.warm_up()

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=None,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=False,
        )
        openai_endpoint = api_args.pop("openai_endpoint")
        chat_completion = getattr(self.client.chat.completions, openai_endpoint)(**api_args)
        completions = [
            _convert_chat_completion_to_chat_message(chat_completion, choice) for choice in chat_completion.choices
        ]

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}

    @component.output_types(replies=list[ChatMessage])
    # tools_strict is intentionally omitted: vLLM does not support it
    async def run_async(  # type: ignore[override]
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        *,
        tools: ToolsType | None = None,
    ) -> dict[str, list[ChatMessage]]:
        """
        Run the VLLM chat generator on the given input data asynchronously.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            Must be a coroutine.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on vLLM API parameters, see
            [vLLM documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/).
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            If set, it will override the `tools` parameter provided during initialization.

        :returns:
            A dictionary with the following key:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        # Streaming: OpenAIChatGenerator handles this correctly.
        resolved_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )

        if resolved_callback is not None:
            return await super().run_async(
                messages, streaming_callback, generation_kwargs, tools=tools, tools_strict=False
            )

        # Non-streaming: handle reasoning.
        if not self._is_warmed_up:
            self.warm_up()

        if len(messages) == 0:
            return {"replies": []}

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=None,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=False,
        )
        openai_endpoint = api_args.pop("openai_endpoint")
        chat_completion = await getattr(self.async_client.chat.completions, openai_endpoint)(**api_args)
        completions = [
            _convert_chat_completion_to_chat_message(chat_completion, choice) for choice in chat_completion.choices
        ]

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}
