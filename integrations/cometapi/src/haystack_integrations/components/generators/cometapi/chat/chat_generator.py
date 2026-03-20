from typing import Any

from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import (
    StreamingCallbackT,
)
from haystack.tools import (
    Tool,
    Toolset,
)
from haystack.utils import Secret


class CometAPIChatGenerator(OpenAIChatGenerator):
    """
    A chat generator that uses the CometAPI for generating chat responses.

    This class extends Haystack's OpenAIChatGenerator to specifically interact with the CometAPI.
    It sets the `api_base_url` to the CometAPI endpoint and allows for all the
    standard configurations available in the OpenAIChatGenerator.
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("COMET_API_KEY"),
        model: str = "gpt-5-mini",
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        tools: list[Tool | Toolset] | Toolset | None = None,
        tools_strict: bool = False,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a `CometAPIChatGenerator` instance.

        :param api_key: The API key for authenticating with the CometAPI.
        :param model: The name of the model to use for chat generation (e.g., `"gpt-5-mini"`, `"grok-3-mini"`).
        :param streaming_callback: An optional callable invoked with each chunk of a streaming response.
        :param generation_kwargs: Optional keyword arguments passed to the underlying generation API call.
        :param timeout: The maximum time in seconds to wait for a response from the API.
        :param max_retries: The maximum number of times to retry a failed API request.
        :param tools: An optional list of tools the model can use.
        :param tools_strict: If `True`, the model is forced to use one of the provided tools.
        :param http_client_kwargs: Optional keyword arguments passed to the HTTP client.
        """
        api_base_url = "https://api.cometapi.com/v1"

        super().__init__(
            api_key=api_key,
            model=model,
            api_base_url=api_base_url,
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            tools_strict=tools_strict,
            http_client_kwargs=http_client_kwargs,
        )
