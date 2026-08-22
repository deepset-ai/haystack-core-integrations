# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx
from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import SparseEmbedding
from haystack.utils import Secret
from haystack.utils.url_validation import is_valid_http_url

from .sparse_embedding import _embed_sparse, _embed_sparse_async, _request_headers


@component
class HuggingFaceAPISparseTextEmbedder:
    """
    Embeds text into a sparse vector using a Hugging Face Text Embeddings Inference (TEI) server.

    The TEI server must be running a sparse embedding model and expose the `/embed_sparse` endpoint.

    ```python
    from haystack_integrations.components.embedders.huggingface_api import HuggingFaceAPISparseTextEmbedder

    embedder = HuggingFaceAPISparseTextEmbedder(api_base_url="http://localhost:8080")
    result = embedder.run("What is sparse retrieval?")
    print(result["sparse_embedding"])
    ```
    """

    def __init__(
        self,
        *,
        api_base_url: str = "http://localhost:8080",
        token: Secret | None = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        timeout: float | None = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        """
        Create a sparse text embedder backed by TEI.

        :param api_base_url: Base URL of the TEI server.
        :param token: Token sent to TEI as HTTP bearer authorization, if set.
        :param prefix: A string to add before the text.
        :param suffix: A string to add after the text.
        :param timeout: HTTP request timeout in seconds. Set to `None` to disable it.
        :param headers: Additional HTTP headers to send with each request.
        :raises ValueError: If `api_base_url` is not a valid HTTP URL.
        """
        if not is_valid_http_url(api_base_url):
            msg = f"api_base_url must be a valid HTTP URL, but got {api_base_url}"
            raise ValueError(msg)
        self.api_base_url = api_base_url
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.timeout = timeout
        self.headers = headers or {}
        base_url = f"{self.api_base_url.rstrip('/')}/"
        client_headers = _request_headers(self.headers, self.token)
        self._client = httpx.Client(base_url=base_url, timeout=self.timeout, headers=client_headers)
        self._async_client = httpx.AsyncClient(base_url=base_url, timeout=self.timeout, headers=client_headers)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            api_base_url=self.api_base_url,
            token=self.token,
            prefix=self.prefix,
            suffix=self.suffix,
            timeout=self.timeout,
            headers=self.headers,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HuggingFaceAPISparseTextEmbedder":
        """Deserialize this component from a dictionary."""
        return default_from_dict(cls, data)

    def _prepare_input(self, text: str) -> str:
        if not isinstance(text, str):
            msg = (
                "HuggingFaceAPISparseTextEmbedder expects a string as input. "
                "To embed Documents, use HuggingFaceAPISparseDocumentEmbedder."
            )
            raise TypeError(msg)
        return self.prefix + text + self.suffix

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str) -> dict[str, SparseEmbedding]:
        """
        Embed a single string.

        :param text: Text to embed.
        :returns: The sparse embedding of the input text.
        """
        embeddings = _embed_sparse(client=self._client, inputs=self._prepare_input(text))
        return {"sparse_embedding": embeddings[0]}

    @component.output_types(sparse_embedding=SparseEmbedding)
    async def run_async(self, text: str) -> dict[str, SparseEmbedding]:
        """
        Embed a single string asynchronously.

        :param text: Text to embed.
        :returns: The sparse embedding of the input text.
        """
        embeddings = await _embed_sparse_async(client=self._async_client, inputs=self._prepare_input(text))
        return {"sparse_embedding": embeddings[0]}
