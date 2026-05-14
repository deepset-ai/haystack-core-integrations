# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, ClassVar

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.chat.openai import _check_finish_reason
from haystack.dataclasses import (
    ChatMessage,
    ReasoningContent,
    StreamingCallbackT,
    ToolCall,
    select_streaming_callback,
)
from haystack.tools import ToolsType, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _parse_mistral_content(content: Any) -> tuple[str | None, ReasoningContent | None]:
    """Parse Mistral message content which can be a string or an array of typed blocks."""
    if content is None:
        return None, None
    if isinstance(content, str):
        return content or None, None
    if not isinstance(content, list):
        return str(content), None

    text_parts: list[str] = []
    thinking_parts: list[str] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "thinking":
            for item in block.get("thinking", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    thinking_parts.append(item.get("text", ""))
        elif block_type == "text":
            text_parts.append(block.get("text", ""))

    text = "".join(text_parts) or None
    reasoning = None
    if thinking_parts:
        reasoning = ReasoningContent(reasoning_text="".join(thinking_parts))

    return text, reasoning


def _convert_mistral_response_to_chat_messages(response_data: dict[str, Any] | str) -> list[ChatMessage]:
    """Convert a raw Mistral API JSON response to a list of ChatMessages, handling array content."""
    data: dict[str, Any] = json.loads(response_data) if isinstance(response_data, str) else response_data
    completions: list[ChatMessage] = []
    usage = data.get("usage")
    model = data.get("model", "")

    for choice in data.get("choices", []):
        message = choice.get("message", {})
        text, reasoning = _parse_mistral_content(message.get("content"))

        tool_calls: list[ToolCall] = []
        for tc in message.get("tool_calls") or []:
            func = tc.get("function", {})
            try:
                arguments = json.loads(func.get("arguments", "{}"))
                tool_calls.append(ToolCall(id=tc.get("id"), tool_name=func.get("name"), arguments=arguments))
            except json.JSONDecodeError:
                logger.warning(
                    "Mistral returned malformed JSON for tool call arguments. "
                    "Tool call ID: {_id}, Tool name: {_name}, Arguments: {_arguments}",
                    _id=tc.get("id"),
                    _name=func.get("name"),
                    _arguments=func.get("arguments"),
                )

        meta: dict[str, Any] = {
            "model": model,
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason"),
            "usage": usage,
        }

        completions.append(ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta, reasoning=reasoning))

    return completions


@component
class MistralChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Mistral AI generative models.

    For supported models, see [Mistral AI docs](https://docs.mistral.ai/getting-started/models).

    Users can pass any text generation parameters valid for the Mistral Chat Completion API
    directly to this component via the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Compatible with the Mistral API Chat Completion endpoint.
    - **Streaming Support**: Supports streaming responses from the Mistral API Chat Completion endpoint.
    - **Customizability**: Supports all parameters supported by the Mistral API Chat Completion endpoint.
    - **Reasoning Support**: Extracts reasoning/thinking content from models that support it
      (e.g., mistral-small with `reasoning_effort`, magistral models) and stores it in the
      `ReasoningContent` field on `ChatMessage`.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)

    For more details on the parameters supported by the Mistral API, refer to the
    [Mistral API Docs](https://docs.mistral.ai/api/).

    Usage example:
    ```python
    from haystack_integrations.components.generators.mistral import MistralChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = MistralChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
    >> "Natural Language Processing (NLP) is a branch of artificial intelligence
    >> that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >> meaningful and useful.")], _name=None,
    >> _meta={'model': 'mistral-small-latest', 'index': 0, 'finish_reason': 'stop',
    >> 'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```

    Reasoning usage example:
    ```python
    from haystack_integrations.components.generators.mistral import MistralChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("Solve: if x + 3 = 7, what is x?")]

    client = MistralChatGenerator(
        model="mistral-small-latest",
        generation_kwargs={"reasoning_effort": "high"},
    )
    response = client.run(messages)
    print(response["replies"][0].reasoning)  # Access reasoning content
    print(response["replies"][0].text)       # Access final answer
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "mistral-medium-2505",
        "mistral-medium-2508",
        "mistral-medium-latest",
        "mistral-medium",
        "mistral-vibe-cli-with-tools",
        "open-mistral-nemo",
        "open-mistral-nemo-2407",
        "mistral-tiny-2407",
        "mistral-tiny-latest",
        "codestral-2508",
        "codestral-latest",
        "devstral-2512",
        "mistral-vibe-cli-latest",
        "devstral-medium-latest",
        "devstral-latest",
        "mistral-small-2506",
        "mistral-small-latest",
        "labs-mistral-small-creative",
        "magistral-medium-2509",
        "magistral-medium-latest",
        "magistral-small-2509",
        "magistral-small-latest",
        "voxtral-small-2507",
        "voxtral-small-latest",
        "mistral-large-2512",
        "mistral-large-latest",
        "ministral-3b-2512",
        "ministral-3b-latest",
        "ministral-8b-2512",
        "ministral-8b-latest",
        "ministral-14b-2512",
        "ministral-14b-latest",
        "mistral-large-2411",
        "pixtral-large-2411",
        "pixtral-large-latest",
        "mistral-large-pixtral-2411",
        "devstral-small-2507",
        "devstral-medium-2507",
        "labs-devstral-small-2512",
        "devstral-small-latest",
        "voxtral-mini-2507",
        "voxtral-mini-latest",
        "voxtral-mini-2602",
    ]
    """A list of models supported by Mistral AI
    see [Mistral AI docs](https://docs.mistral.ai/getting-started/models) for more information
    and send a GET HTTP request to "https://api.mistral.ai/v1/models" for a full list of model IDs."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-small-latest",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://api.mistral.ai/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of MistralChatGenerator.

        Unless specified otherwise in the `model`, this is for Mistral's `mistral-small-latest` model.

        :param api_key:
            The Mistral API key.
        :param model:
            The name of the Mistral chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Mistral API Base url.
            For more details, see Mistral [docs](https://docs.mistral.ai/api/).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Mistral endpoint. See [Mistral API docs](https://docs.mistral.ai/api/) for more details.
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
            - `reasoning_effort`: Controls reasoning/thinking tokens for models that support adjustable reasoning
                (e.g., `mistral-small-latest`, `mistral-medium`). Accepted values: `"high"`, `"none"`.
                See [Mistral reasoning docs](https://docs.mistral.ai/capabilities/reasoning/).
            - `prompt_mode`: For native reasoning models (magistral). Set to `"reasoning"` to use the default
                reasoning system prompt, or omit for the model's default behavior.
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
        :param timeout:
            The timeout for the Mistral API call. If not set, it defaults to either the `OPENAI_TIMEOUT`
            environment variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(MistralChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=None,
            generation_kwargs=generation_kwargs,
            tools=tools,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
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
        api_args = super(MistralChatGenerator, self)._prepare_api_call(  # noqa: UP008
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )

        if "response_format" in api_args and api_args["response_format"] is None:
            api_args.pop("response_format")

        extra_body: dict[str, Any] = {}
        for param in ("reasoning_effort", "prompt_mode", "safe_prompt"):
            if param in api_args:
                extra_body[param] = api_args.pop(param)
        if extra_body:
            api_args.setdefault("extra_body", {}).update(extra_body)

        for i, chat_msg in enumerate(messages):
            if chat_msg.reasoning and chat_msg.reasoning.reasoning_text:
                formatted = api_args["messages"][i]
                text_content = formatted.get("content", "") or ""
                formatted["content"] = [
                    {"type": "thinking", "thinking": [{"type": "text", "text": chat_msg.reasoning.reasoning_text}]},
                    {"type": "text", "text": text_content},
                ]

        return api_args

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
        Invokes chat completion on the Mistral API.

        :param messages:
            A list of ChatMessage instances representing the input messages.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will
            override the parameters passed during component initialization.
            For details on Mistral API parameters, see
            [Mistral docs](https://docs.mistral.ai/api/).
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

        if streaming_callback is not None:
            merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
            if merged_kwargs.get("reasoning_effort") or merged_kwargs.get("prompt_mode"):
                logger.warning(
                    "Streaming with reasoning parameters is active. Reasoning content from thinking "
                    "blocks will not be captured during streaming. Use non-streaming mode to extract "
                    "reasoning content."
                )

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        openai_endpoint = api_args.pop("openai_endpoint")

        if streaming_callback is not None:
            chat_completion = getattr(self.client.chat.completions, openai_endpoint)(**api_args)
            completions = self._handle_stream_response(chat_completion, streaming_callback)
        else:
            raw_response = getattr(self.client.chat.completions.with_raw_response, openai_endpoint)(**api_args)
            completions = _convert_mistral_response_to_chat_messages(raw_response.text)

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
        Asynchronously invokes chat completion on the Mistral API.

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

        if streaming_callback is not None:
            merged_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
            if merged_kwargs.get("reasoning_effort") or merged_kwargs.get("prompt_mode"):
                logger.warning(
                    "Streaming with reasoning parameters is active. Reasoning content from thinking "
                    "blocks will not be captured during streaming. Use non-streaming mode to extract "
                    "reasoning content."
                )

        api_args = self._prepare_api_call(
            messages=messages,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            tools=tools,
            tools_strict=tools_strict,
        )
        openai_endpoint = api_args.pop("openai_endpoint")

        if streaming_callback is not None:
            chat_completion = await getattr(self.async_client.chat.completions, openai_endpoint)(**api_args)
            completions = await self._handle_async_stream_response(chat_completion, streaming_callback)
        else:
            raw_response = await getattr(self.async_client.chat.completions.with_raw_response, openai_endpoint)(
                **api_args
            )
            completions = _convert_mistral_response_to_chat_messages(raw_response.text)

        for message in completions:
            _check_finish_reason(message.meta)

        return {"replies": completions}

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
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=serialize_tools_or_toolset(self.tools),
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
