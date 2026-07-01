# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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


class EdenAIChatGenerator(OpenAIChatGenerator):
    """
    A chat generator that uses Eden AI's OpenAI-compatible API to generate chat responses.

    Eden AI is a unified API that gives access to 500+ AI models from many providers (OpenAI,
    Anthropic, Mistral, Google, Cohere, and more) through a single API key, with built-in
    provider fallback and EU data residency. This makes it a convenient, sovereignty-friendly
    gateway for building LLM and RAG applications with Haystack.

    This class extends Haystack's `OpenAIChatGenerator` to talk to Eden AI. It sets the
    `api_base_url` to Eden AI's OpenAI-compatible endpoint and keeps all the standard
    configurations available in the `OpenAIChatGenerator`.

    Models are selected using Eden AI's `provider/model` naming convention, for example
    `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4-5"`, or `"mistral/mistral-large-latest"`.
    See the [Eden AI models catalog](https://www.edenai.co/models) for the full list.

    Usage example:
    ```python
    from haystack_integrations.components.generators.edenai import EdenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_user("What's Natural Language Processing?")]

    client = EdenAIChatGenerator(model="mistral/mistral-large-latest")
    response = client.run(messages)
    print(response["replies"][0].text)
    ```
    """

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("EDENAI_API_KEY"),
        model: str = "openai/gpt-4o-mini",
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        tools: list[Tool | Toolset] | Toolset | None = None,
        tools_strict: bool = False,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an `EdenAIChatGenerator` instance.

        :param api_key: The Eden AI API key. Defaults to the `EDENAI_API_KEY` environment variable.
        :param model: The model to use, in Eden AI's `provider/model` format
            (e.g. `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4-5"`, `"mistral/mistral-large-latest"`).
        :param streaming_callback: An optional callable invoked with each chunk of a streaming response.
        :param generation_kwargs: Optional keyword arguments passed to the underlying generation API call,
            such as `max_tokens`, `temperature`, or `top_p`. Eden AI-specific parameters (for example a
            fallback model) are forwarded as-is to the Eden AI endpoint.
        :param timeout: The maximum time in seconds to wait for a response from the API.
        :param max_retries: The maximum number of times to retry a failed API request.
        :param tools: An optional list of tools or a Toolset the model can use for function calling.
        :param tools_strict: If `True`, enable strict schema adherence for tool calls.
        :param http_client_kwargs: Optional keyword arguments passed to the underlying HTTP client.
        """
        api_base_url = "https://api.edenai.run/v3"

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
