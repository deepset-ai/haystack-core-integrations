from typing import Any

from haystack import component, default_to_dict
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import (
    StreamingCallbackT,
)
from haystack.tools import (
    Tool,
    Toolset,
    serialize_tools_or_toolset,
)
from haystack.utils import Secret, serialize_callable


@component
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

        # the @component decorator recreates the class, so the zero-argument form of super() cannot be used
        super(CometAPIChatGenerator, self).__init__(  # noqa: UP008
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

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None

        # if we didn't implement the to_dict method here then the to_dict method of the superclass would be used
        # which would serialize some fields that we don't want to serialize (e.g. the ones we don't have in
        # the __init__)
        # it would be hard to maintain the compatibility as superclass changes
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            streaming_callback=callback_name,
            generation_kwargs=self.generation_kwargs,
            timeout=self.timeout,
            max_retries=self.max_retries,
            tools=serialize_tools_or_toolset(self.tools),
            tools_strict=self.tools_strict,
            http_client_kwargs=self.http_client_kwargs,
        )
