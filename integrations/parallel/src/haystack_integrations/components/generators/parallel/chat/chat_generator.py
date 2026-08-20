# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any, ClassVar

from haystack import component, default_from_dict
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import StreamingCallbackT
from haystack.utils import deserialize_callable
from haystack.utils.auth import Secret

_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "parallel-haystack"

_INIT_PARAMETERS: tuple[str, ...] = (
    "api_key",
    "model",
    "api_base_url",
    "streaming_callback",
    "generation_kwargs",
    "timeout",
    "extra_headers",
    "max_retries",
    "http_client_kwargs",
)


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


def _http_client_kwargs_with_headers(
    http_client_kwargs: dict[str, Any] | None,
    extra_headers: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(http_client_kwargs or {})
    headers = {**kwargs.get("headers", {}), **(extra_headers or {})}
    headers["x-parallel-integration"] = _attribution_header()
    kwargs["headers"] = headers
    return kwargs


@component
class ParallelChatGenerator(OpenAIResponsesChatGenerator):
    """
    Completes chats using Parallel's web-research model.

    Powered by the Parallel Responses API (`POST /v1/responses`, OpenAI Responses-compatible).
    Every answer is grounded in live web research with citations; the `reasoning.effort`
    parameter selects the research tier: `low` (~5-10s), `medium` (~15-20s, default), or
    `high` (~30-60s).
    See the [Parallel Responses API quickstart](https://docs.parallel.ai/responses-api/responses-quickstart)
    for details.

    It uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format in input and output.
    Web grounding is built in, so tool calling and sampling parameters
    (`temperature`, `top_p`, ...) are not supported by the API.

    ### Usage example
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.parallel import ParallelChatGenerator

    messages = [ChatMessage.from_user("What did Parallel Web Systems announce this year?")]

    client = ParallelChatGenerator(generation_kwargs={"reasoning": {"effort": "low"}})
    response = client.run(messages)
    print(response)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = ["parallel"]
    """The Parallel Responses API models supported by this component.
    See https://docs.parallel.ai/responses-api/responses-quickstart for details."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PARALLEL_API_KEY"),
        model: str = "parallel",
        api_base_url: str | None = "https://api.parallel.ai/v1",
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the ParallelChatGenerator component.

        :param api_key:
            The Parallel API key.
        :param model:
            The Parallel Responses API model to use.
        :param api_base_url:
            The Parallel API base URL.
        :param streaming_callback:
            A callback function called when a new token is received from the stream.
        :param generation_kwargs:
            Additional parameters sent directly to the Parallel Responses API, such as
            `reasoning` (e.g. `{"effort": "low"}`) to select the research tier or
            `text` for structured output.
        :param timeout:
            Timeout for Parallel API calls.
        :param extra_headers:
            Additional HTTP headers to include in requests to the Parallel API.
        :param max_retries:
            Maximum number of retries to contact Parallel after an internal error.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        self.extra_headers = extra_headers
        super(ParallelChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=_http_client_kwargs_with_headers(http_client_kwargs, extra_headers),
        )
        # self.http_client_kwargs carries the attribution (and extra) headers so that the parent bakes them into
        # the httpx client whether it is built eagerly (haystack-ai 2.x) or at warm-up (haystack-ai >= 3.0);
        # the user-provided value is preserved for serialization
        self._http_client_kwargs = http_client_kwargs

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(ParallelChatGenerator, self).to_dict()  # noqa: UP008
        data["type"] = generate_qualified_class_name(type(self))
        data["init_parameters"]["extra_headers"] = self.extra_headers
        # serialize the user-provided value, not the internal one enriched with attribution headers
        data["init_parameters"]["http_client_kwargs"] = self._http_client_kwargs
        # the parent serializes params this component does not expose (organization, tools, ...)
        data["init_parameters"] = {
            key: value for key, value in data["init_parameters"].items() if key in _INIT_PARAMETERS
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParallelChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        serialized_callback_handler = data.get("init_parameters", {}).get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)
