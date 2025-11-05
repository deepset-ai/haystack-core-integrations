# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils import deserialize_callable, serialize_callable
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)


@component
class LlamaStackChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using Llama Stack framework.
    Llama Stack Server supports multiple inference providers, including Ollama, Together,
    and vLLM and other cloud providers.
    For a complete list of inference providers, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/providers/inference/index.html).

    Users can pass any text generation parameters valid for the OpenAI chat completion API
    directly to this component using the `generation_kwargs`
    parameter in `__init__` or the `generation_kwargs` parameter in `run` method.

    This component uses the `ChatMessage` format for structuring both input and output,
    ensuring coherent and contextually relevant responses in chat-based text generation scenarios.
    Details on the `ChatMessage` format can be found in the
    [Haystack docs](https://docs.haystack.deepset.ai/docs/chatmessage)

    Usage example:
    You need to setup Llama Stack Server before running this example and have a model available. For a quick start on
    how to setup server with Ollama, see [Llama Stack docs](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).

    ```python
    from haystack_integrations.components.generators.llama_stack import LlamaStackChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = LlamaStackChatGenerator(model="ollama/llama3.2:3b")
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_content=[TextContent(text='Natural Language Processing (NLP)
    is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.')], _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
    >>_meta={'model': 'ollama/llama3.2:3b', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    """

    def __init__(
        self,
        *,
        model: str,
        api_base_url: str = "http://localhost:8321/v1/openai/v1",
        organization: Optional[str] = None,
        streaming_callback: Optional[StreamingCallbackT] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        tools: Optional[ToolsType] = None,
        tools_strict: bool = False,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of LlamaStackChatGenerator. To use this chat generator,
        you need to setup Llama Stack Server with an inference provider and have a model available.

        :param model:
            The name of the model to use for chat completion.
            This depends on the inference provider used for the Llama Stack Server.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Llama Stack API base url. If not specified, the localhost is used with the default port 8321.
        :param organization: Your organization ID, defaults to `None`.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Llama Stack endpoint. See [Llama Stack API docs](https://llama-stack.readthedocs.io/) for more details.
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
        :param timeout:
            Timeout for client calls using OpenAI API. If not set, it defaults to either the
            `OPENAI_TIMEOUT` environment variable, or 30 seconds.
        :param tools:
            A list of Tool and/or Toolset objects, or a single Toolset for which the model can prepare calls.
            Each tool should have a unique name.
        :param tools_strict:
            Whether to enable strict schema adherence for tool calls. If set to `True`, the model will follow exactly
            the schema provided in the `parameters` field of the tool definition, but this may increase latency.
        :param max_retries:
            Maximum number of retries to contact OpenAI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).

        """

        # Use placeholder key for a Llama Stack server running locally
        api_key = Secret.from_token("placeholder-api-key")

        super(LlamaStackChatGenerator, self).__init__(  # noqa: UP008
            model=model,
            api_key=api_key,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            tools=tools,
            tools_strict=tools_strict,
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
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=serialize_tools_or_toolset(self.tools),
            tools_strict=self.tools_strict,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LlamaStackChatGenerator":
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
