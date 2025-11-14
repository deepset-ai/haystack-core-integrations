# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.tools import ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

AIMLAPI_HEADERS = {"HTTP-Referer": "https://github.com/deepset-ai/haystack-core-integrations", "X-Title": "Haystack"}


@component
class AIMLAPIChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using AIMLAPI generative models.
    For supported models, see AIMLAPI documentation.

    Users can pass any text generation parameters valid for the AIMLAPI chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the AIMLAPI chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the AIMLAPI chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the AIMLAPI chat completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the AIMLAPI API, refer to the
    AIMLAPI documentation.

    Usage example:
    ```python
    from haystack_integrations.components.generators.aimlapi import AIMLAPIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = AIMLAPIChatGenerator(model="openai/gpt-5-chat-latest")
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
    >>_meta={'model': 'openai/gpt-5-chat-latest', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("AIMLAPI_API_KEY"),
        model: str = "openai/gpt-5-chat-latest",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = "https://api.aimlapi.com/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of AIMLAPIChatGenerator. Unless specified otherwise,
        the default model is `openai/gpt-5-chat-latest`.

        :param api_key:
            The AIMLAPI API key.
        :param model:
            The name of the AIMLAPI chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The AIMLAPI API Base url.
            For more details, see AIMLAPI documentation.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the AIMLAPI endpoint. See AIMLAPI API docs for more details.
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
            The timeout for the AIMLAPI API call.
        :param extra_headers:
            Additional HTTP headers to include in requests to the AIMLAPI API.
        :param max_retries:
            Maximum number of retries to contact AIMLAPI after an internal error.
            If not set, it defaults to either the `AIMLAPI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(AIMLAPIChatGenerator, self).__init__(  # noqa: UP008
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
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        tools_strict: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        extra_headers = {**AIMLAPI_HEADERS, **(self.extra_headers or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API (AIMLAPI uses the same format)
        aimlapi_formatted_messages: List[Dict[str, Any]] = [message.to_openai_dict_format() for message in messages]

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

        response_format = generation_kwargs.pop("response_format", None)

        base_args = {
            "model": self.model,
            "messages": aimlapi_formatted_messages,
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
