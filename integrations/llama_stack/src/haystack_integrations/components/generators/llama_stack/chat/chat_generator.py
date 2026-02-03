# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils import deserialize_callable, serialize_callable
from haystack.utils.auth import Secret
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

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
        api_base_url: str = "http://localhost:8321/v1",
        organization: str | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: int | None = None,
        tools: ToolsType | None = None,
        tools_strict: bool = False,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
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
            - `response_format`: A JSON schema or a Pydantic model that enforces the structure of the model's response.
                If provided, the output will always be validated against this
                format (unless the model returns a tool call).
                For details, see the [OpenAI Structured Outputs documentation](https://platform.openai.com/docs/guides/structured-outputs).
                Notes:
                - For structured outputs with streaming,
                  the `response_format` must be a JSON schema and not a Pydantic model.
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

        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=generation_kwargs,
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=serialize_tools_or_toolset(self.tools),
            tools_strict=self.tools_strict,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LlamaStackChatGenerator":
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
