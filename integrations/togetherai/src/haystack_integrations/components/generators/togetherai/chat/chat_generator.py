# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class TogetherAIChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Together AI generative models.
    For supported models, see [Together AI docs](https://docs.together.ai/docs).

    Users can pass any text generation parameters valid for the Together AI chat completion API
    directly to this component using the `generation_kwargs` parameter in `__init__` or the `generation_kwargs`
    parameter in `run` method.

    Key Features and Compatibility:
    - **Primary Compatibility**: Designed to work seamlessly with the Together AI chat completion endpoint.
    - **Streaming Support**: Supports streaming responses from the Together AI chat completion endpoint.
    - **Customizability**: Supports all parameters supported by the Together AI chat completion endpoint.

    This component uses the ChatMessage format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the ChatMessage format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    For more details on the parameters supported by the Together AI API, refer to the
    [Together AI API Docs](https://docs.together.ai/reference/chat-completions-1).

    Usage example:
    ```python
    from haystack_integrations.components.generators.togetherai import TogetherAIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = TogetherAIChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
    >>_meta={'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TOGETHER_API_KEY"),
        model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        streaming_callback: Optional[StreamingCallbackT] = None,
        api_base_url: Optional[str] = "https://api.together.xyz/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[ToolsType] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of TogetherAIChatGenerator. Unless specified otherwise,
        the default model is `meta-llama/Llama-3.3-70B-Instruct-Turbo`.

        :param api_key:
            The Together API key.
        :param model:
            The name of the Together AI chat completion model to use.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Together AI API Base url.
            For more details, see Together AI [docs](https://docs.together.ai/docs/openai-api-compatibility).
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Together AI endpoint. See [Together AI API docs](https://docs.together.ai/reference/chat-completions-1)
            for more details.
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
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param timeout:
            The timeout for the Together AI API call.
        :param max_retries:
            Maximum number of retries to contact Together AI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """
        super(TogetherAIChatGenerator, self).__init__(  # noqa: UP008
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
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
