# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import component, default_to_dict
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret


@component
class MistralDocumentEmbedder(OpenAIDocumentEmbedder):
    """
    A component for computing Document embeddings using Mistral models.
    The embedding of each Document is stored in the `embedding` field of the Document.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.mistral import MistralDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = MistralDocumentEmbedder()

    result = document_embedder.run([doc])
    print(result['documents'][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("MISTRAL_API_KEY"),
        model: str = "mistral-embed",
        api_base_url: Optional[str] = "https://api.mistral.ai/v1",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        *,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates a MistralDocumentEmbedder component.

        :param api_key:
            The Mistral API key.
        :param model:
            The name of the model to use.
        :param api_base_url:
            The Mistral API Base url. For more details, see Mistral [docs](https://docs.mistral.ai/api/).
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
            Timeout for Mistral client calls. If not set, it defaults to either the `OPENAI_TIMEOUT` environment
            variable, or 30 seconds.
        :param max_retries:
            Maximum number of retries to contact Mistral after an internal error.
            If not set, it defaults to either the `OPENAI_MAX_RETRIES` environment variable, or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        super(MistralDocumentEmbedder, self).__init__(  # noqa: UP008
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

    def to_dict(self) -> Dict[str, Any]:
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
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
