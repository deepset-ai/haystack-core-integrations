# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, logging
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

_DEFAULT_API_BASE_URL = "https://api.otari.ai/api/v1"
_EU_API_BASE_URL = "https://eu.api.otari.ai/api/v1"
_EU_API_KEY_PREFIX = "otk_v1_eu_"

_INIT_PARAMETERS: tuple[str, ...] = (
    "api_key",
    "model",
    "streaming_callback",
    "api_base_url",
    "generation_kwargs",
    "tools",
    "timeout",
    "max_retries",
    "http_client_kwargs",
)


@component
class OtariChatGenerator(OpenAIChatGenerator):
    """
    Completes chats using models served through an [Otari](https://github.com/mozilla-ai/otari) gateway.

    Otari is an OpenAI-compatible LLM gateway that routes requests to many providers and applies API keys, budgets,
    and usage tracking. By default, this component calls the otari.ai gateway. For an account in otari.ai's EU
    region, whose API keys start with `otk_v1_eu_`, set `api_base_url` to `"https://eu.api.otari.ai/api/v1"`.
    To call a gateway you run yourself, set `api_base_url`, for example to `"http://localhost:8000/api/v1"`.

    Name models with a selector the gateway accepts, such as `provider:model` (`"openai:gpt-5-mini"`,
    `"anthropic:claude-sonnet-4-6"`), a model alias, or a routing policy name. For details, see the
    [Otari model docs](https://github.com/mozilla-ai/otari/blob/main/docs/models.md).

    You can pass any parameter of the Otari chat completions endpoint through `generation_kwargs`, at initialization
    or in the `run` method. Request headers, such as Otari's `Idempotency-Key`, go into
    `generation_kwargs["extra_headers"]`.

    When the gateway prices a request, `meta["usage"]` of the reply contains its cost in US dollars as `cost_usd`,
    and `pricing_source`. When streaming, pass `generation_kwargs={"stream_options": {"include_usage": True}}` to
    receive usage.

    ### Usage example

    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.otari import OtariChatGenerator

    # Reads the API key from the OTARI_API_KEY environment variable
    generator = OtariChatGenerator()
    response = generator.run([ChatMessage.from_user("What's Natural Language Processing? Be brief.")])
    print(response["replies"][0].text)
    # >> Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to ...
    print(response["replies"][0].meta["usage"]["cost_usd"])
    # >> 0.000279
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
        model: str = "openai:gpt-5-mini",
        streaming_callback: StreamingCallbackT | None = None,
        api_base_url: str = "https://api.otari.ai/api/v1",
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OtariChatGenerator.

        :param api_key:
            The Otari API key: a key issued by your gateway, or an otari.ai API token.
            Defaults to the `OTARI_API_KEY` environment variable.
        :param model:
            The model selector, for example `"openai:gpt-5-mini"`. The gateway must have the provider configured.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url:
            The OpenAI-compatible API root of the Otari gateway, including the `/api/v1` path.
            Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
            `"https://eu.api.otari.ai/api/v1"`.
        :param generation_kwargs:
            Other parameters to use for the model. These parameters are sent directly to the Otari endpoint.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `stream_options`: Pass `{"include_usage": True}` to receive usage, including cost, when streaming.
            - `extra_headers`: Headers to send with the request, for example Otari's `Idempotency-Key`.
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
            The timeout for the Otari API call.
        :param max_retries:
            Maximum number of retries to contact Otari after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        # the @component decorator recreates the class, so the zero-argument form of super() cannot be used
        super(OtariChatGenerator, self).__init__(  # noqa: UP008
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

    def warm_up(self) -> None:
        """
        Warm up the tools and initialize the synchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.client is None
        super(OtariChatGenerator, self).warm_up()  # noqa: UP008
        if creates_client:
            self._warn_if_eu_key_on_default_url()

    async def warm_up_async(self) -> None:
        """
        Warm up the tools and initialize the asynchronous OpenAI client.

        Logs a warning if the API key belongs to otari.ai's EU region but `api_base_url` is left at its default.
        """
        creates_client = self.async_client is None
        await super(OtariChatGenerator, self).warm_up_async()  # noqa: UP008
        if creates_client:
            self._warn_if_eu_key_on_default_url()

    def _warn_if_eu_key_on_default_url(self) -> None:
        # The default host rejects keys of the EU region with a bare 401 that doesn't name the right host
        api_key = self.api_key.resolve_value()
        if self.api_base_url == _DEFAULT_API_BASE_URL and api_key and api_key.startswith(_EU_API_KEY_PREFIX):
            logger.warning(
                "The Otari API key belongs to otari.ai's EU region, which {default_url} rejects. "
                "Set api_base_url to {eu_url}.",
                default_url=_DEFAULT_API_BASE_URL,
                eu_url=_EU_API_BASE_URL,
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(OtariChatGenerator, self).to_dict()  # noqa: UP008
        # the parent also serializes OpenAI-only parameters (organization, tools_strict) that this component lacks
        data["init_parameters"] = {
            key: value for key, value in data["init_parameters"].items() if key in _INIT_PARAMETERS
        }
        return data
