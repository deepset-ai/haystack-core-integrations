# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import (
    ToolsType,
    _check_duplicate_tool_names,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "perplexity-haystack"


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


@component
class PerplexityChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using the Perplexity Agent API.

    For supported models, see [Perplexity docs](https://docs.perplexity.ai/).

    Users can pass any text generation parameters valid for the Perplexity chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the Perplexity chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the Perplexity chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the Perplexity chat completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    Usage example:
    ```python
    from haystack_integrations.components.generators.perplexity import PerplexityChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = PerplexityChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
        model: str = "sonar-pro",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://api.perplexity.ai",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of PerplexityChatGenerator.

        :param api_key:
            The Perplexity API key.
        :param model:
            The name of the Perplexity chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Perplexity API base URL.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Perplexity endpoint.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance.
        :param timeout:
            The timeout for the Perplexity API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the Perplexity API.
        :param max_retries:
            Maximum number of retries to contact Perplexity after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(PerplexityChatGenerator, self).__init__(  # noqa: UP008
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
        extra_headers = {
            **(self.extra_headers or {}),
            "X-Pplx-Integration": _attribution_header(),
        }

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)

        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)
        response_format = generation_kwargs.pop("response_format", None)

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

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
            # for structured outputs without streaming, we use openai's parse endpoint
            # Note: `stream` cannot be passed to chat.completions.parse
            # we pass a key `openai_endpoint` as a hint to the run method to use the parse endpoint
            # this key will be removed before the API call is made
            return {
                **base_args,
                "response_format": response_format,
                "openai_endpoint": "parse",
            }

        # for structured outputs with streaming, we use openai's create endpoint
        # we pass a key `openai_endpoint` as a hint to the run method to use the create endpoint
        # this key will be removed before the API call is made
        final_args = {**base_args, "stream": is_streaming, "openai_endpoint": "create"}

        # We only set the response_format parameter if it's not None since None is not a valid value in the API.
        if response_format:
            final_args["response_format"] = response_format
        return final_args
