# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

from haystack import component, default_to_dict, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, serialize_tools_or_toolset
from haystack.utils import serialize_callable
from haystack.utils.auth import Secret
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@component
class HetznerChatGenerator(OpenAIChatGenerator):
    """
    Enables text generation using the models served by the Hetzner Inference API.

    For the list of available models, see the
    [Hetzner Inference API docs](https://docs.hetzner.com/general/company-and-policy/experiments/inference/) or query
    the `/v1/models` endpoint of the API, whose response is definitive.

    You can pass any text generation parameters valid for the Hetzner chat completion API directly to this component
    using the `generation_kwargs` parameter in `__init__` or in the `run` method.

    The served models accept images alongside text, so
    [`ImageContent`](https://docs.haystack.deepset.ai/docs/imagecontent) parts can be included in the
    [`ChatMessage`](https://docs.haystack.deepset.ai/docs/chatmessage)s passed to `run`.

    Usage example:
    ```python
    from haystack_integrations.components.generators.hetzner import HetznerChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = HetznerChatGenerator()
    response = client.run(messages)
    print(response)

    >>{'replies': [ChatMessage(_content='Natural Language Processing (NLP) is a branch of artificial intelligence
    >>that focuses on enabling computers to understand, interpret, and generate human language in a way that is
    >>meaningful and useful.', _role=<ChatRole.ASSISTANT: 'assistant'>, _name=None,
    >>_meta={'model': 'Qwen/Qwen3.6-35B-A3B-FP8', 'index': 0, 'finish_reason': 'stop',
    >>'usage': {'prompt_tokens': 15, 'completion_tokens': 36, 'total_tokens': 51}})]}
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "Qwen3.8-27B",
    ]
    """The models supported by this component while the Hetzner Inference API is in experimental status.
    The selection changes over time: query the `/v1/models` endpoint of the API for the definitive list.
    Models outside this list are not rejected and are passed on to the API as-is."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("HETZNER_API_KEY"),
        model: str = "Qwen/Qwen3.6-35B-A3B-FP8",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str | None = "https://inference.hetzner.com/api/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of HetznerChatGenerator.

        :param api_key:
            The Hetzner Inference API token.
        :param model:
            The name of the Hetzner chat completion model to use. See `SUPPORTED_MODELS`.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The Hetzner Inference API base url.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are all sent directly to
            the Hetzner endpoint.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream`: Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent
                events as they become available, with the stream terminated by a data: [DONE] message.
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
            The timeout for the Hetzner API call.
        :param max_retries:
            Maximum number of retries to contact Hetzner after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        if model not in self.SUPPORTED_MODELS:
            logger.warning(
                "Model {model} is not in the list of models known to be served by the Hetzner Inference API: "
                "{supported_models}. The request is sent anyway; check the `/v1/models` endpoint for the "
                "current selection.",
                model=model,
                supported_models=", ".join(self.SUPPORTED_MODELS),
            )

        # the @component decorator recreates the class, so the zero-argument form of super() cannot be used
        super(HetznerChatGenerator, self).__init__(  # noqa: UP008
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
        # which would serialize some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
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
