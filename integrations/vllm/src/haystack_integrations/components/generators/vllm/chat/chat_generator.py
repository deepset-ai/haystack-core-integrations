# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
)
from haystack.tools import ToolsType
from haystack.utils import Secret
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
            # vLLM accepts None; our _client_kwargs handles it without the parent's Secret resolver.
            api_key=api_key,  # type: ignore[arg-type]
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
        reasoning_text = getattr(chunk.choices[0].delta, "reasoning", None) if chunk.choices else None
        if reasoning_text:
            return StreamingChunk(
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
        converted = super(VLLMChatGenerator, self)._convert_chat_completion_chunk_to_streaming_chunk(  # noqa: UP008
            chunk=chunk, previous_chunks=previous_chunks, component_info=component_info
        )
        if (
            converted.content
            and any(c.reasoning for c in previous_chunks)
            and not any(c.content for c in previous_chunks)
        ):
            converted = replace(converted, start=True)
        return converted
