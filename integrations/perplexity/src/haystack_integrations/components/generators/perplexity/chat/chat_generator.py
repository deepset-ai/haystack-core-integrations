# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import Any, ClassVar

from haystack import component, default_from_dict
from haystack.components.generators.chat import OpenAIResponsesChatGenerator
from haystack.core.serialization import generate_qualified_class_name
from haystack.dataclasses import StreamingCallbackT
from haystack.tools import ToolsType, deserialize_tools_or_toolset_inplace
from haystack.utils import deserialize_callable
from haystack.utils.auth import Secret

_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "perplexity-haystack"


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


def _perplexity_headers(extra_headers: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        **(extra_headers or {}),
        "X-Pplx-Integration": _attribution_header(),
    }


def _with_default_headers(client: Any, headers: dict[str, Any]) -> Any:
    with_options = getattr(client, "with_options", None)
    if with_options is not None:
        return with_options(default_headers=headers)

    client._custom_headers = {**getattr(client, "_custom_headers", {}), **headers}
    return client


@component
class PerplexityChatGenerator(OpenAIResponsesChatGenerator):
    """
    Completes chats using Perplexity models.

    Powered by the Perplexity Agent API (`POST /v1/agent`, OpenAI Responses-compatible).
    See the [Perplexity Agent API quickstart](https://docs.perplexity.ai/docs/agent-api/quickstart)
    for details.

    It uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage) format in input and output.
    You can customize generation by passing Perplexity Agent API parameters through `generation_kwargs`.

    ### Usage example
    ```python
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.components.generators.perplexity import PerplexityChatGenerator

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = PerplexityChatGenerator()
    response = client.run(messages)
    print(response)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/gpt-4o",
        "anthropic/claude-sonnet-4-6",
        "xai/grok-4-1",
        "google/gemini-3-flash-preview",
    ]
    """A non-exhaustive list of Agent API models supported by this component.
    See https://docs.perplexity.ai/docs/agent-api/models for the full and current list."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
        model: str = "openai/gpt-5.4",
        api_base_url: str | None = "https://api.perplexity.ai/v1",
        streaming_callback: StreamingCallbackT | None = None,
        organization: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[dict[str, Any]] | None = None,
        tools_strict: bool = False,
        timeout: float | None = None,
        extra_headers: dict[str, Any] | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the PerplexityChatGenerator component.

        :param api_key:
            The Perplexity API key.
        :param model:
            The Perplexity Agent API model to use.
        :param api_base_url:
            The Perplexity API base URL.
        :param streaming_callback:
            A callback function called when a new token is received from the stream.
        :param organization:
            Organization ID forwarded to the OpenAI-compatible client.
        :param generation_kwargs:
            Additional parameters sent directly to the Perplexity Agent API.
        :param tools:
            A list of Haystack tools, a Toolset, or OpenAI-compatible tool definitions.
        :param tools_strict:
            Whether to enable strict schema adherence for Haystack tool calls.
        :param timeout:
            Timeout for Perplexity API calls.
        :param extra_headers:
            Additional HTTP headers to include in requests to the Perplexity API.
        :param max_retries:
            Maximum number of retries to contact Perplexity after an internal error.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        self.extra_headers = extra_headers
        super(PerplexityChatGenerator, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            generation_kwargs=generation_kwargs,
            timeout=timeout,
            max_retries=max_retries,
            tools=tools,
            tools_strict=tools_strict,
            http_client_kwargs=http_client_kwargs,
        )

        default_headers = _perplexity_headers(extra_headers)
        self.client = _with_default_headers(self.client, default_headers)
        self.async_client = _with_default_headers(self.async_client, default_headers)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        data = super(PerplexityChatGenerator, self).to_dict()  # noqa: UP008
        data["type"] = generate_qualified_class_name(type(self))
        data["init_parameters"]["extra_headers"] = self.extra_headers
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerplexityChatGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        tools = data["init_parameters"].get("tools")
        if tools and (
            (isinstance(tools, dict) and tools.get("type") == "haystack.tools.toolset.Toolset")
            or (isinstance(tools, list) and tools[0].get("type") == "haystack.tools.tool.Tool")
        ):
            deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")

        serialized_callback_handler = data.get("init_parameters", {}).get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        return default_from_dict(cls, data)
