# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from typing import TYPE_CHECKING, Any, ClassVar

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders import OpenAITextEmbedder
from haystack.utils.auth import Secret

from haystack_integrations.components.embedders.perplexity.embedding_encoding import (
    decode_embedding,
    validate_encoding_format,
)

if TYPE_CHECKING:
    from openai.types.create_embedding_response import CreateEmbeddingResponse

_INTEGRATION_SLUG = "haystack"
_PACKAGE_NAME = "perplexity-haystack"


def _attribution_header() -> str:
    try:
        version = importlib.metadata.version(_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return f"{_INTEGRATION_SLUG}/{version}"


def _http_client_kwargs_with_attribution(
    http_client_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs = dict(http_client_kwargs or {})
    headers = dict(kwargs.get("headers", {}))
    headers["X-Pplx-Integration"] = _attribution_header()
    kwargs["headers"] = headers
    return kwargs


@component
class PerplexityTextEmbedder(OpenAITextEmbedder):
    """
    A component for embedding strings using Perplexity models.

    For supported models, see the
    [Perplexity Embeddings API reference](https://docs.perplexity.ai/api-reference/embeddings-post).

    Usage example:
     ```python
    from haystack_integrations.components.embedders.perplexity.text_embedder import PerplexityTextEmbedder

    text_to_embed = "I love pizza!"
    text_embedder = PerplexityTextEmbedder()
    print(text_embedder.run(text_to_embed))
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "pplx-embed-v1-0.6b",
        "pplx-embed-v1-4b",
    ]
    """A list of models supported by the Perplexity Embeddings API.
    See https://docs.perplexity.ai/api-reference/embeddings-post for the current list of model IDs."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("PERPLEXITY_API_KEY"),
        model: str = "pplx-embed-v1-0.6b",
        api_base_url: str | None = "https://api.perplexity.ai/v1",
        prefix: str = "",
        suffix: str = "",
        encoding_format: str = "base64_int8",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a PerplexityTextEmbedder component.

        :param api_key:
            The Perplexity API key.
        :param model:
            The name of the Perplexity embedding model to be used.
        :param api_base_url:
            The Perplexity API base URL.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param encoding_format:
            The Perplexity embedding encoding format. Supported values are `base64_int8` and `base64_binary`.
        :param timeout:
            Timeout for Perplexity client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Perplexity after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """

        self.encoding_format = validate_encoding_format(encoding_format)
        super(PerplexityTextEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
            timeout=timeout,
            max_retries=max_retries,
            http_client_kwargs=_http_client_kwargs_with_attribution(http_client_kwargs),
        )
        self.http_client_kwargs = http_client_kwargs
        self.timeout = timeout
        self.max_retries = max_retries

    def _prepare_input(self, text: str) -> dict[str, Any]:
        kwargs = OpenAITextEmbedder._prepare_input(self, text=text)
        kwargs["input"] = [kwargs["input"]]
        kwargs["encoding_format"] = self.encoding_format
        return kwargs

    def _prepare_output(self, result: "CreateEmbeddingResponse") -> dict[str, Any]:
        return {
            "embedding": decode_embedding(str(result.data[0].embedding), self.encoding_format),
            "meta": {"model": result.model, "usage": dict(result.usage)},
        }

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
            encoding_format=self.encoding_format,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerplexityTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
