# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret
from more_itertools import batched
from openai import APIError
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from haystack_integrations.components.embedders.perplexity.embedding_encoding import (
    decode_embedding,
    validate_encoding_format,
)

if TYPE_CHECKING:
    from openai.types.create_embedding_response import CreateEmbeddingResponse

logger = logging.getLogger(__name__)

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
class PerplexityDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for computing Document embeddings using Perplexity models.

    The embedding of each Document is stored in the `embedding` field of the Document.
    For supported models, see the
    [Perplexity Embeddings API reference](https://docs.perplexity.ai/api-reference/embeddings-post).

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.perplexity import PerplexityDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = PerplexityDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)
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
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        encoding_format: str = "base64_int8",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates a PerplexityDocumentEmbedder component.

        :param api_key:
            The Perplexity API key.
        :param model:
            The name of the model to use.
        :param api_base_url:
            The Perplexity API base URL.
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
        super(PerplexityDocumentEmbedder, self).__init__(  # noqa: UP008
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
            http_client_kwargs=_http_client_kwargs_with_attribution(http_client_kwargs),
        )
        self.http_client_kwargs = http_client_kwargs
        self.timeout = timeout
        self.max_retries = max_retries

    def _decode_response_embeddings(self, response: "CreateEmbeddingResponse") -> list[list[float]]:
        return [decode_embedding(str(el.embedding), self.encoding_format) for el in response.data]

    def _embed_batch(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        """
        Embed a list of texts in batches.
        """

        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}
        for batch in tqdm(
            batched(texts_to_embed.items(), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            args: dict[str, Any] = {
                "model": self.model,
                "input": [b[1] for b in batch],
                "encoding_format": self.encoding_format,
            }

            try:
                response = self.client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                msg = "Failed embedding of documents {ids} caused by {exc}"
                logger.exception(msg, ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise exc
                continue

            embeddings = self._decode_response_embeddings(response)
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return doc_ids_to_embeddings, meta

    async def _embed_batch_async(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        """
        Embed a list of texts in batches asynchronously.
        """

        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}

        batches: Iterable[tuple[tuple[str, str], ...]] = list(batched(texts_to_embed.items(), batch_size))
        if self.progress_bar:
            batches = async_tqdm(batches, desc="Calculating embeddings")

        for batch in batches:
            args: dict[str, Any] = {
                "model": self.model,
                "input": [b[1] for b in batch],
                "encoding_format": self.encoding_format,
            }

            try:
                response = await self.async_client.embeddings.create(**args)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                msg = "Failed embedding of documents {ids} caused by {exc}"
                logger.exception(msg, ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise exc
                continue

            embeddings = self._decode_response_embeddings(response)
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))

            if "model" not in meta:
                meta["model"] = response.model
            if "usage" not in meta:
                meta["usage"] = dict(response.usage)
            else:
                meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
                meta["usage"]["total_tokens"] += response.usage.total_tokens

        return doc_ids_to_embeddings, meta

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
            encoding_format=self.encoding_format,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerplexityDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
