# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any, ClassVar

from haystack import component, logging
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import StreamingCallbackT
from haystack.utils.auth import Secret

logger = logging.getLogger(__name__)

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

# Accepted by the API for SDK compatibility but silently ignored, so a request that sets them
# succeeds while behaving as if they were never passed.
# See https://docs.parallel.ai/responses-api/openai-compatibility
_IGNORED_GENERATION_KWARGS: tuple[str, ...] = (
    "include",
    "max_output_tokens",
    "parallel_tool_calls",
    "store",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "truncation",
    "user",
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
    Web grounding is built in, so tool calling and sampling parameters (`tools`, `temperature`,
    `top_p`, ...) are accepted for SDK compatibility but silently ignored by the API; this component
    warns when it sees them.

    Because a single call runs live research, `timeout` defaults to 120 seconds rather than the
    30 seconds inherited from the OpenAI client, so that the `high` tier fits comfortably.

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
        timeout: float | None = 120.0,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = 3,
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
            Timeout in seconds for Parallel API calls. Defaults to 120 seconds, which leaves room for
            the `high` research tier (~30-60s). Pass `None` to fall back to the OpenAI client default
            (the `OPENAI_TIMEOUT` environment variable, or 30 seconds), which is too short for most
            research calls.
        :param extra_headers:
            Additional HTTP headers to include in requests to the Parallel API.
        :param max_retries:
            Maximum number of retries to contact Parallel after an internal error. Kept low because
            every retry runs a full research call. Pass `None` to fall back to the OpenAI client
            default (the `OPENAI_MAX_RETRIES` environment variable, or 5).
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        if model not in self.SUPPORTED_MODELS:
            logger.warning(
                "Model {model} is not supported by the Parallel Responses API, which only serves "
                "{supported_models}. The request is sent anyway and the API is expected to reject it.",
                model=model,
                supported_models=", ".join(self.SUPPORTED_MODELS),
            )

        ignored = sorted(set(generation_kwargs or {}) & set(_IGNORED_GENERATION_KWARGS))
        if ignored:
            logger.warning(
                "The generation_kwargs {ignored} are accepted by the Parallel Responses API for SDK "
                "compatibility but have no effect on the response. Use `reasoning` to select the research "
                "tier and `text` for structured output instead.",
                ignored=", ".join(ignored),
            )

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
