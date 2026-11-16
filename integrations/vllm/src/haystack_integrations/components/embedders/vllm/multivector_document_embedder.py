# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client
from more_itertools import batched
from tqdm import tqdm

logger = logging.getLogger(__name__)


@component
class VLLMMultivectorDocumentEmbedder:
    """
    A component for computing Document multivector embeddings using models served with [vLLM](https://docs.vllm.ai/).

    The multivector embedding of each Document is stored in its `multivector_embedding` field.
    It uses the `/pooling` endpoint of a running vLLM server with the `token_embed` task.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a model that supports token embeddings:

    ```bash
    vllm serve answerdotai/answerai-colbert-small-v1 --runner pooling --pooler-config '{"task":"token_embed"}'
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.vllm import VLLMMultivectorDocumentEmbedder

    doc = Document(content="I love pizza!")
    document_embedder = VLLMMultivectorDocumentEmbedder(model="answerdotai/answerai-colbert-small-v1")

    result = document_embedder.run([doc])
    print(result["documents"][0].multivector_embedding)
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are merged into the
    request body sent to the `/pooling` endpoint.

    ```python
    document_embedder = VLLMMultivectorDocumentEmbedder(
        model="answerdotai/answerai-colbert-small-v1",
        extra_parameters={"truncate_prompt_tokens": 256},
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000",
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        timeout: float | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        extra_parameters: dict[str, Any] | None = None,
        concurrency_limit: int = 4,
    ) -> None:
        """
        Creates an instance of VLLMMultivectorDocumentEmbedder.

        :param model: The name of the token-embedding model served by vLLM. See the
            [vLLM token embedding docs](https://docs.vllm.ai/en/stable/models/pooling_models/token_embed/)
            for information on supported models.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the server was started with `--api-key`.
        :param api_base_url: Base URL of the vLLM server. Do not append `/v1`; the native `/pooling` endpoint is used.
        :param prefix: String added before every Document text.
        :param suffix: String added after every Document text.
        :param batch_size: Number of Documents sent in each request. Must be greater than zero.
        :param progress_bar: Whether to display embedding progress.
        :param meta_fields_to_embed: Metadata fields prepended to the Document content.
        :param embedding_separator: Separator joining metadata values and Document content.
        :param timeout: Timeout in seconds for vLLM client calls. When set, this takes precedence over
            `http_client_kwargs["timeout"]`. If `None`, the timeout from `http_client_kwargs` is used;
            if neither is set, the HTTPX default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param extra_parameters: Additional parameters merged into the request body sent to the vLLM
            `/pooling` endpoint, such as `truncate_prompt_tokens`. `model`, `input`, and `task` cannot be
            overridden. See the
            [vLLM Pooling API docs](https://docs.vllm.ai/en/stable/models/pooling_models/#pooling-api).
        :param concurrency_limit: Maximum number of asynchronous batch requests in flight at once. Values below
            one use sequential execution. Synchronous embedding is unaffected.
        :raises ValueError: `batch_size` is not positive.
        """
        if batch_size <= 0:
            msg = "batch_size must be greater than 0"
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url.rstrip("/")
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.timeout = timeout
        self.http_client_kwargs = http_client_kwargs
        self.extra_parameters = extra_parameters
        self.concurrency_limit = concurrency_limit

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        headers = httpx.Headers((self.http_client_kwargs or {}).get("headers"))
        if self.api_key is not None and (resolved_key := self.api_key.resolve_value()):
            headers["Authorization"] = f"Bearer {resolved_key}"

        client_kwargs = self.http_client_kwargs.copy() if self.http_client_kwargs else {}
        client_kwargs["headers"] = headers
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return client_kwargs

    def warm_up(self) -> None:
        """Create the synchronous HTTP client."""
        if self._client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=False)
            assert client is not None  # noqa: S101
            self._client = client

    async def warm_up_async(self) -> None:
        """Create the asynchronous HTTP client."""
        if self._async_client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=True)
            assert client is not None  # noqa: S101
            self._async_client = client

    def close(self) -> None:
        """Close the synchronous HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close the asynchronous HTTP client."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def _prepare_texts_to_embed(self, documents: Iterable[Document]) -> list[str]:
        texts = []
        for document in documents:
            metadata = [
                str(document.meta[field])
                for field in self.meta_fields_to_embed
                if field in document.meta and document.meta[field] is not None
            ]
            texts.append(self.prefix + self.embedding_separator.join([*metadata, document.content or ""]) + self.suffix)
        return texts

    def _prepare_input(self, texts: list[str]) -> dict[str, Any]:
        kwargs = dict(self.extra_parameters or {})
        kwargs.update({"model": self.model, "input": texts, "task": "token_embed"})
        return kwargs

    def _embed_batch(self, documents: list[Document]) -> list[list[list[float]]]:
        assert self._client is not None  # noqa: S101 - run() guarantees warm_up() first
        embeddings: list[list[list[float]]] = []
        batches = batched(documents, self.batch_size)
        for batch in tqdm(batches, disable=not self.progress_bar, desc="Calculating multivector embeddings"):
            try:
                texts = self._prepare_texts_to_embed(batch)
                response = self._client.post(f"{self.api_base_url}/pooling", json=self._prepare_input(texts))
                response.raise_for_status()
                embeddings.extend(item["data"] for item in response.json()["data"])
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                logger.exception(
                    "Failed to embed Documents {document_ids}",
                    document_ids=", ".join(document.id for document in batch),
                )
                raise
        return embeddings

    async def _embed_batch_async(self, documents: list[Document]) -> list[list[list[float]]]:
        client = self._async_client
        assert client is not None  # noqa: S101 - run_async() guarantees warm_up_async() first
        sem = asyncio.Semaphore(max(1, self.concurrency_limit))
        total = (len(documents) + self.batch_size - 1) // self.batch_size

        with tqdm(total=total, disable=not self.progress_bar, desc="Calculating multivector embeddings") as progress:

            async def _runner(batch: tuple[Document, ...]) -> list[list[list[float]]]:
                async with sem:
                    try:
                        texts = self._prepare_texts_to_embed(batch)
                        response = await client.post(f"{self.api_base_url}/pooling", json=self._prepare_input(texts))
                        response.raise_for_status()
                        return [item["data"] for item in response.json()["data"]]
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        logger.exception(
                            "Failed to embed Documents {document_ids}",
                            document_ids=", ".join(document.id for document in batch),
                        )
                        raise
                    finally:
                        progress.update()

            results = await asyncio.gather(*(_runner(batch) for batch in batched(documents, self.batch_size)))

        return [embedding for batch_embeddings in results for embedding in batch_embeddings]

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or any(not isinstance(document, Document) for document in documents):
            msg = (
                "VLLMMultivectorDocumentEmbedder expects a list containing only Documents. To embed a string, use "
                "VLLMMultivectorTextEmbedder."
            )
            raise TypeError(msg)

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Embed Documents into token-level multivectors.

        :param documents: Documents to embed.
        :returns: A dictionary with:
            - `documents`: Documents with their `multivector_embedding` field populated.
        :raises TypeError: `documents` is not a list containing only Documents.
        :raises httpx.HTTPError: A request fails.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": []}

        self.warm_up()
        embeddings = self._embed_batch(documents)

        return {
            "documents": [
                replace(document, multivector_embedding=embedding)
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously embed Documents into token-level multivectors.

        :param documents: Documents to embed.
        :returns: A dictionary with:
            - `documents`: Documents with their `multivector_embedding` field populated.
        :raises TypeError: `documents` is not a list containing only Documents.
        :raises httpx.HTTPError: A request fails.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": []}

        await self.warm_up_async()
        embeddings = await self._embed_batch_async(documents)

        return {
            "documents": [
                replace(document, multivector_embedding=embedding)
                for document, embedding in zip(documents, embeddings, strict=True)
            ]
        }
