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
    Embed Documents into token-level multivectors using a model served with vLLM.

    Each multivector is stored in a copy of the Document at `meta["_multivector_embedding"]`; the original Document
    and its `embedding` field are not modified. The component calls vLLM's native `/pooling` API with the
    `token_embed` task. Start a server with a compatible model, for example:

    ```bash
    vllm serve answerdotai/answerai-colbert-small-v1 --runner pooling --pooler-config '{"task":"token_embed"}'
    ```

    The `/pooling` endpoint is not below the OpenAI-compatible `/v1` path, so the default `api_base_url` is
    `http://localhost:8000`. For details, see the
    [vLLM Pooling API documentation](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/#pooling-api).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.vllm import VLLMMultivectorDocumentEmbedder

    embedder = VLLMMultivectorDocumentEmbedder(model="answerdotai/answerai-colbert-small-v1")
    result = embedder.run([Document(content="A document about Rome")])
    print(result["documents"][0].meta["_multivector_embedding"])
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
        raise_on_failure: bool = False,
        extra_parameters: dict[str, Any] | None = None,
        concurrency_limit: int = 4,
    ) -> None:
        """
        Create a VLLMMultivectorDocumentEmbedder.

        :param model: Name of the token-embedding model served by vLLM.
        :param api_key: vLLM API key. Defaults to the `VLLM_API_KEY` environment variable and is only required when
            the server was started with `--api-key`.
        :param api_base_url: Base URL of the vLLM server. Do not append `/v1`; the native `/pooling` endpoint is used.
        :param prefix: String added before every Document text.
        :param suffix: String added after every Document text.
        :param batch_size: Number of Documents sent in each request. Must be greater than zero.
        :param progress_bar: Whether to display embedding progress.
        :param meta_fields_to_embed: Metadata fields prepended to the Document content.
        :param embedding_separator: Separator joining metadata values and Document content.
        :param timeout: Request timeout in seconds. When set, this takes precedence over
            `http_client_kwargs["timeout"]`.
        :param http_client_kwargs: Keyword arguments passed to `httpx.Client` and `httpx.AsyncClient`.
        :param raise_on_failure: Raise batch errors instead of logging them and continuing.
        :param extra_parameters: Additional `/pooling` request fields. `model`, `input`, and `task` cannot be
            overridden.
        :param concurrency_limit: Maximum number of asynchronous batch requests in flight at once. Values below
            one use sequential execution. Synchronous embedding is unaffected.
        :raises ValueError: If `batch_size` is not positive.
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
        self.raise_on_failure = raise_on_failure
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

    def _embed_batch(self, documents: list[Document]) -> dict[int, list[list[float]]]:
        assert self._client is not None  # noqa: S101 - run() guarantees warm_up() first
        embeddings: dict[int, list[list[float]]] = {}
        batches = batched(documents, self.batch_size)
        for batch_number, batch in enumerate(
            tqdm(batches, disable=not self.progress_bar, desc="Calculating multivector embeddings")
        ):
            try:
                texts = self._prepare_texts_to_embed(batch)
                response = self._client.post(f"{self.api_base_url}/pooling", json=self._prepare_input(texts))
                response.raise_for_status()
                batch_embeddings = {item["index"]: item["data"] for item in response.json()["data"]}
                embeddings.update(
                    {batch_number * self.batch_size + index: embedding for index, embedding in batch_embeddings.items()}
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                logger.exception(
                    "Failed to embed Documents {document_ids}",
                    document_ids=", ".join(document.id for document in batch),
                )
                if self.raise_on_failure:
                    raise
        return embeddings

    async def _embed_batch_async(self, documents: list[Document]) -> dict[int, list[list[float]]]:
        client = self._async_client
        assert client is not None  # noqa: S101 - run_async() guarantees warm_up_async() first
        sem = asyncio.Semaphore(max(1, self.concurrency_limit))
        total = (len(documents) + self.batch_size - 1) // self.batch_size

        with tqdm(total=total, disable=not self.progress_bar, desc="Calculating multivector embeddings") as progress:

            async def _runner(batch_number: int, batch: tuple[Document, ...]) -> dict[int, list[list[float]]]:
                async with sem:
                    try:
                        texts = self._prepare_texts_to_embed(batch)
                        response = await client.post(f"{self.api_base_url}/pooling", json=self._prepare_input(texts))
                        response.raise_for_status()
                        batch_embeddings = {item["index"]: item["data"] for item in response.json()["data"]}
                        return {
                            batch_number * self.batch_size + index: embedding
                            for index, embedding in batch_embeddings.items()
                        }
                    except (httpx.HTTPError, ValueError, KeyError, TypeError):
                        logger.exception(
                            "Failed to embed Documents {document_ids}",
                            document_ids=", ".join(document.id for document in batch),
                        )
                        raise
                    finally:
                        progress.update()

            results = await asyncio.gather(
                *(
                    _runner(batch_number, batch)
                    for batch_number, batch in enumerate(batched(documents, self.batch_size))
                ),
                return_exceptions=True,
            )

        embeddings: dict[int, list[list[float]]] = {}
        for result in results:
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, (httpx.HTTPError, ValueError, KeyError, TypeError)) and not self.raise_on_failure:
                    continue
                raise result
            embeddings.update(result)
        return embeddings

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
        :returns: A dictionary containing copied `documents` with multivector embeddings.
        :raises TypeError: If `documents` is not a list containing only Documents.
        :raises httpx.HTTPError: If a request fails and `raise_on_failure` is `True`.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": []}

        self.warm_up()
        embeddings = self._embed_batch(documents)

        return {
            "documents": [
                replace(document, meta={**document.meta, "_multivector_embedding": embeddings[index]})
                if index in embeddings
                else replace(document, meta=document.meta.copy())
                for index, document in enumerate(documents)
            ]
        }

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        """
        Asynchronously embed Documents into token-level multivectors.

        :param documents: Documents to embed.
        :returns: A dictionary containing copied `documents` with multivector embeddings.
        :raises TypeError: If `documents` is not a list containing only Documents.
        :raises httpx.HTTPError: If a request fails and `raise_on_failure` is `True`.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": []}

        await self.warm_up_async()
        embeddings = await self._embed_batch_async(documents)

        return {
            "documents": [
                replace(document, meta={**document.meta, "_multivector_embedding": embeddings[index]})
                if index in embeddings
                else replace(document, meta=document.meta.copy())
                for index, document in enumerate(documents)
            ]
        }
