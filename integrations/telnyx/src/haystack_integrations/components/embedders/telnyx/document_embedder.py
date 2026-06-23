# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret

_MODELS_WITHOUT_CUSTOM_DIMENSIONS = {"thenlper/gte-large"}


@component
class TelnyxDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for embedding Haystack Documents using Telnyx Inference embedding models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.telnyx import TelnyxDocumentEmbedder

    embedder = TelnyxDocumentEmbedder()
    result = embedder.run([Document(content="I love pizza!")])
    print(result["documents"][0].embedding)
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = ["thenlper/gte-large"]
    """A non-exhaustive list of embedding models available through Telnyx Inference."""

    def __init__(
        self,
        *,
        api_key: Secret = Secret.from_env_var("TELNYX_API_KEY"),
        model: str = "thenlper/gte-large",
        api_base_url: str | None = "https://api.telnyx.com/v2/ai/openai",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        dimensions: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a TelnyxDocumentEmbedder component.

        :param api_key:
            The Telnyx API key.
        :param model:
            The Telnyx Inference embedding model to use.
        :param api_base_url:
            The Telnyx OpenAI-compatible API base URL.
        :param prefix:
            A string to add to the beginning of each text.
        :param suffix:
            A string to add to the end of each text.
        :param batch_size:
            Number of Documents to encode at once.
        :param progress_bar:
            Whether to show a progress bar or not.
        :param meta_fields_to_embed:
            List of meta fields that should be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        :param dimensions:
            The number of dimensions the resulting output embeddings should have. Only supported by Telnyx embedding
            models that expose configurable dimensions.
        :param timeout:
            Timeout for Telnyx client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Telnyx after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client` or `httpx.AsyncClient`.
        """
        if dimensions is not None and model in _MODELS_WITHOUT_CUSTOM_DIMENSIONS:
            msg = f"The Telnyx embedding model '{model}' does not support custom dimensions."
            raise ValueError(msg)

        super(TelnyxDocumentEmbedder, self).__init__(  # noqa: UP008
            api_key=api_key,
            model=model,
            dimensions=dimensions,
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
        self.dimensions = dimensions
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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            dimensions=self.dimensions,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
