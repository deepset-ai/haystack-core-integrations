# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, ClassVar

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret


@component
class ForgeDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for computing Document embeddings using Forge models.

    The embedding of each Document is stored in the `embedding` field of the Document.

    Forge exposes an OpenAI-compatible embeddings API, so this component subclasses Haystack's
    built-in `OpenAIDocumentEmbedder` and points it at the Forge API base URL.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.forge import ForgeDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = ForgeDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    SUPPORTED_MODELS: ClassVar[list[str]] = [
        "forge-turbo",
        "forge-pro",
        "forge-ultra-4k",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]
    """A list of model strings accepted by the Forge embeddings API.
    The `forge-*` models are native Forge models; the `text-embedding-*` strings are accepted as
    OpenAI-compatible aliases. See [Forge](https://voxell.ai) for more information."""

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("FORGE_API_KEY"),
        model: str = "forge-pro",
        api_base_url: str | None = "https://api.voxell.ai/v1",
        dimensions: int | None = None,
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
        Creates a ForgeDocumentEmbedder component.

        :param api_key:
            The Forge API key.
        :param model:
            The name of the model to use.
        :param api_base_url:
            The Forge API base URL.
        :param dimensions:
            The number of dimensions of the resulting output embeddings. Forge models support
            Matryoshka representation learning, so a smaller dimension can be requested. If not set,
            the model's default dimensionality is used.
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
            Timeout for Forge client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Forge after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(ForgeDocumentEmbedder, self).__init__(  # noqa: UP008
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
            dimensions=self.dimensions,
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForgeDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)
