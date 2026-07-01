# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret


@component
class EdenAIDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for computing Document embeddings using Eden AI's OpenAI-compatible API.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Eden AI routes embedding requests to many providers (OpenAI, Mistral, Cohere, Google, Jina, and
    more) through a single API key, with EU data residency. Models use Eden AI's `provider/model`
    naming convention, for example `"openai/text-embedding-3-small"` or `"mistral/mistral-embed"`.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.edenai import EdenAIDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = EdenAIDocumentEmbedder(model="mistral/mistral-embed")

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
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
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an `EdenAIDocumentEmbedder` component.

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
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            Whether to show a progress bar or not. Can be helpful to disable in production deployments to keep
            the logs clean.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
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
        super(EdenAIDocumentEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=None,
            api_base_url=api_base_url,
            organization=None,
            prefix=prefix,
            suffix=suffix,
            batch_size=batch_size,
            progress_bar=progress_bar,
            meta_fields_to_embed=meta_fields_to_embed,
            embedding_separator=embedding_separator,
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
            model=self.model,
            api_key=self.api_key.to_dict(),
            api_base_url=self.api_base_url,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
