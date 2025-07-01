# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Union

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import Tool, Toolset, _check_duplicate_tool_names
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class OpenRouterChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using OpenRouter generative models.
    For supported models, see [OpenRouter docs](https://openrouter.ai/models).

    Users can pass any text generation parameters valid for the OpenRouter chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the OpenRouter chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the OpenRouter chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the OpenRouter chat completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the OpenRouter API, refer to the
    [OpenRouter API Docs](https://openrouter.ai/docs/quickstart).

    Usage example:
    ```python
    from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = OpenRouterChatGenerator()
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
        api_key: Secret = Secret.from_env_var("OPENROUTER_API_KEY"),
        model: str = "openai/gpt-4o-mini",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = "https://openrouter.ai/api/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenRouterChatGenerator. Unless specified otherwise,
        the default model is `openai/gpt-4o-mini`.

        :param api_key:
            The OpenRouter API key.
        :param model:
            The name of the OpenRouter chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The OpenRouter API Base url.
            For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the OpenRouter endpoint. See [OpenRouter API docs](https://openrouter.ai/docs/quickstart) for more details.
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
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. This parameter can accept either a
            list of `Tool` objects or a `Toolset` instance.
        :param timeout:
            The timeout for the OpenRouter API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the OpenRouter API.
            This can be useful for adding site URL or title for rankings on openrouter.ai
            For more details, see OpenRouter [docs](https://openrouter.ai/docs/quickstart).
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(OpenRouterChatGenerator, self).__init__(  # noqa: UP008
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialiaze some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            api_key=self.api_key.to_dict(),
            tools=[tool.to_dict() for tool in self.tools] if self.tools else None,
            extra_headers=self.extra_headers,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    def _prepare_api_call(
        self,
        *,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**(self.extra_headers or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

        tools = tools or self.tools
        tools_strict = tools_strict if tools_strict is not None else self.tools_strict
        _check_duplicate_tool_names(list(tools or []))

        openai_tools = {}
        if tools:
            tool_definitions = [
                {"type": "function", "function": {**t.tool_spec, **({"strict": tools_strict} if tools_strict else {})}}
                for t in tools
            ]
            openai_tools = {"tools": tool_definitions}

        is_streaming = streaming_callback is not None
        num_responses = generation_kwargs.pop("n", 1)
        if is_streaming and num_responses > 1:
            msg = "Cannot stream multiple responses, please set n=1."
            raise ValueError(msg)

        return {
            "model": self.model,
            "messages": openai_formatted_messages,  # type: ignore[arg-type] # openai expects list of specific message types
            "stream": streaming_callback is not None,
            "n": num_responses,
            **openai_tools,
            "extra_body": {**generation_kwargs},
            "extra_headers": {**extra_headers},
        }
