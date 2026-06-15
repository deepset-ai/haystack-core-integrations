# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

ORCAROUTER_HEADERS = {"HTTP-Referer": "https://github.com/deepset-ai/haystack-core-integrations", "X-Title": "Haystack"}


@component
class OrcaRouterChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using OrcaRouter generative models.

    OrcaRouter is an OpenAI-compatible model routing gateway that exposes 100+ chat models from providers such as
    OpenAI, Anthropic, Google, DeepSeek, and Qwen behind a single endpoint and API key. Models are addressed with a
    `provider/model` namespace (for example `openai/gpt-4o-mini` or `anthropic/claude-opus-4.8`). The special
    `orcarouter/auto` router selects an upstream model per request according to the routing policy configured in your
    OrcaRouter console.

    For the list of supported models, see the [OrcaRouter model catalog](https://www.orcarouter.ai/models).

    Users can pass any text generation parameters valid for the OrcaRouter chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the OrcaRouter chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the OrcaRouter chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the OrcaRouter chat completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the OrcaRouter API, refer to the
    [OrcaRouter documentation](https://docs.orcarouter.ai).

    Usage example:
    ```python
    from haystack_integrations.components.generators.orcarouter import OrcaRouterChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OrcaRouterChatGenerator(model="openai/gpt-4o-mini")
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
    >>_meta={'model': 'openai/gpt-4o-mini', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("ORCAROUTER_API_KEY"),
        model: str = "openai/gpt-4o-mini",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://api.orcarouter.ai/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OrcaRouterChatGenerator.

        Unless specified otherwise, the default model is `openai/gpt-4o-mini`.

        :param api_key:
            The OrcaRouter API key.
        :param model:
            The name of the OrcaRouter chat completion model to use. Models use a `provider/model` namespace
            (for example `openai/gpt-4o-mini`). Use `orcarouter/auto` to let OrcaRouter route the request.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The OrcaRouter API Base url.
            For more details, see OrcaRouter [documentation](https://docs.orcarouter.ai).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the OrcaRouter endpoint. See OrcaRouter [API docs](https://docs.orcarouter.ai) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
            - `extra_body`: A dictionary of OrcaRouter-specific routing preferences, such as a fallback list of
                models, that is passed through to the gateway.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance.
        :param timeout:
            The timeout for the OrcaRouter API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the OrcaRouter API.
        :param max_retries:
            Maximum number of retries to contact OrcaRouter after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(OrcaRouterChatGenerator, self).__init__(  # noqa: UP008
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
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**ORCAROUTER_HEADERS, **(self.extra_headers or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API (OrcaRouter uses the same format)
        orcarouter_formatted_messages: list[dict[str, Any]] = [message.to_openai_dict_format() for message in messages]

        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        flattened_tools = flatten_tools_or_toolsets(tools or self.tools)
        _check_duplicate_tool_names(flattened_tools)

        openai_tools = {}
        if flattened_tools:
            tool_definitions = []
            for tool in flattened_tools:
                function_spec = {**tool.tool_spec}
                if tools_strict:
                    function_spec["strict"] = True
                    parameters = function_spec.get("parameters")
                    if isinstance(parameters, dict):
                        parameters["additionalProperties"] = False
                tool_definitions.append({"type": "function", "function": function_spec})
            openai_tools = {"tools": tool_definitions}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)

        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)
        response_format = generation_kwargs.pop("response_format", None)

        base_args = {
            "model": self.model,
            "messages": orcarouter_formatted_messages,
            "n": num_responses,
            **openai_tools,
            "extra_body": {**generation_kwargs},
            "extra_headers": {**extra_headers},
        }
        if response_format and not is_streaming:
            return {**base_args, "response_format": response_format, "openai_endpoint": "parse"}

        final_args = {**base_args, "stream": is_streaming, "openai_endpoint": "create"}
        if response_format:
            final_args["response_format"] = response_format
        return final_args
