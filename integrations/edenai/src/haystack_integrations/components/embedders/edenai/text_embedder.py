# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret


@component
class EdenAITextEmbedder(OpenAITextEmbedder):
    """
    A component for embedding strings using Eden AI's OpenAI-compatible API.

    Eden AI routes embedding requests to many providers (OpenAI, Mistral, Cohere, Google, Jina, and
    more) through a single API key, with EU data residency. Models use Eden AI's `provider/model`
    naming convention, for example `"openai/text-embedding-3-small"` or `"mistral/mistral-embed"`.

    Usage example:
    ```python
    from haystack_integrations.components.embedders.edenai import EdenAITextEmbedder

    text_embedder = EdenAITextEmbedder(model="mistral/mistral-embed")
    print(text_embedder.run("I love pizza!"))
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "openai/text-embedding-3-small",
        "openai/text-embedding-3-large",
        "mistral/mistral-embed",
        "cohere/embed-english-v3.0",
        "google/text-embedding-004",
    ]
    """A non-exhaustive list of embedding models supported by this component.
    See the [Eden AI models catalog](https://www.edenai.co/models) for the full list."""

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: Secret = Secret.from_env_var("EDENAI_API_KEY"),
        api_base_url: str | None = "https://api.edenai.run/v3",
        prefix: str = "",
        suffix: str = "",
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an `EdenAITextEmbedder` component.

        :param model:
            The name of the Eden AI embedding model to use, in `provider/model` format.
        :param api_key:
            The Eden AI API key. Defaults to the `EDENAI_API_KEY` environment variable.
        :param api_base_url:
            The Eden AI API base URL.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param timeout:
            Timeout for the API call. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Eden AI after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(EdenAITextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=http_client_kwargs,
        )
        # We add these since they were only added in Haystack 2.14.0
        self.timeout = timeout
        self.max_retries = max_retries

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_key=self.api_key.to_dict(),
            model=self.model,
            api_base_url=self.api_base_url,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
