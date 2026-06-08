# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

from haystack import Document, component, logging
from haystack.utils import Secret
from more_itertools import batched
from openai import APIError, AsyncOpenAI, OpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from haystack_integrations.common.vllm.utils import _create_openai_clients

logger = logging.getLogger(__name__)


@component
class VLLMDocumentEmbedder:
    """
    A component for computing Document embeddings using models served with [vLLM](https://docs.vllm.ai/).

    The embedding of each Document is stored in the `embedding` field of the Document.
    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    OpenAI-compatible Embeddings API exposed by vLLM.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with an embedding model:

    ```bash
    vllm serve google/embeddinggemma-300m
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.embedders.vllm import VLLMDocumentEmbedder

    doc = Document(content="I love pizza!")

    document_embedder = VLLMDocumentEmbedder(model="google/embeddinggemma-300m")

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are forwarded as `extra_body`
    to the OpenAI-compatible endpoint.

    ```python
    document_embedder = VLLMDocumentEmbedder(
        model="google/embeddinggemma-300m",
        extra_parameters={"truncate_prompt_tokens": 256, "truncation_side": "right"},
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        prefix: str = "",
        suffix: str = "",
        dimensions: int | None = None,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
        raise_on_failure: bool = False,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMDocumentEmbedder.

        :param model: The name of the model served by vLLM. Check
        [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models) for more information.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param prefix: A string to add at the beginning of each text.
        :param suffix: A string to add at the end of each text.
        :param dimensions: The number of dimensions of the resulting embedding. Only models trained with
            Matryoshka Representation Learning support this parameter. See
            [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#matryoshka-embeddings)
            for more information.
        :param batch_size: Number of documents to encode at once.
        :param progress_bar: Whether to show a progress bar.
        :param meta_fields_to_embed: List of meta fields to embed along with the document text.
        :param embedding_separator: Separator used to concatenate the meta fields to the document text.
        :param timeout: Timeout in seconds for vLLM client calls. If not set, the OpenAI client default applies.
        :param max_retries: Maximum number of retries for failed requests. If not set, the OpenAI client
            default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param raise_on_failure: Whether to raise an exception if the embedding request fails. If `False`,
            the component logs the error and continues processing the remaining documents.
        :param extra_parameters: Additional parameters forwarded as `extra_body` to the vLLM embeddings
            endpoint. Use this to pass parameters not part of the standard OpenAI Embeddings API, such as
            `truncate_prompt_tokens`, `truncation_side`, etc. See the
            [vLLM Embeddings API docs](https://docs.vllm.ai/en/stable/models/pooling_models/embed/#openai-compatible-embeddings-api).
        """
        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.prefix = prefix
        self.suffix = suffix
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client_kwargs = http_client_kwargs
        self.raise_on_failure = raise_on_failure
        self.extra_parameters = extra_parameters

        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """Create the OpenAI clients."""
        if self._is_warmed_up:
            return
        self._client, self._async_client = _create_openai_clients(
            api_key=self.api_key,
            api_base_url=self.api_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            http_client_kwargs=self.http_client_kwargs,
        )
        self._is_warmed_up = True

    def _prepare_texts_to_embed(self, documents: list[Document]) -> dict[str, str]:
        """Concatenate each Document's text with the selected meta fields."""
        texts_to_embed = {}
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            texts_to_embed[doc.id] = (
                self.prefix + self.embedding_separator.join([*meta_values_to_embed, doc.content or ""]) + self.suffix
            )
        return texts_to_embed

    def _prepare_input(self, inputs: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "input": inputs, "encoding_format": "float"}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        if self.extra_parameters:
            kwargs["extra_body"] = self.extra_parameters
        return kwargs

    @staticmethod
    def _update_meta(meta: dict[str, Any], response: Any) -> None:
        if "model" not in meta:
            meta["model"] = response.model
        if "usage" not in meta:
            meta["usage"] = dict(response.usage)
        else:
            meta["usage"]["prompt_tokens"] += response.usage.prompt_tokens
            meta["usage"]["total_tokens"] += response.usage.total_tokens

    def _embed_batch(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        assert self._client is not None  # noqa: S101
        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}

        for batch in tqdm(
            batched(texts_to_embed.items(), batch_size),
            disable=not self.progress_bar,
            desc="Calculating embeddings",
        ):
            kwargs = self._prepare_input([b[1] for b in batch])
            try:
                response = self._client.embeddings.create(**kwargs)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                logger.exception("Failed embedding of documents {ids} caused by {exc}", ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise
                continue

            embeddings = [el.embedding for el in response.data]
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))
            self._update_meta(meta, response)

        return doc_ids_to_embeddings, meta

    async def _embed_batch_async(
        self, texts_to_embed: dict[str, str], batch_size: int
    ) -> tuple[dict[str, list[float]], dict[str, Any]]:
        assert self._async_client is not None  # noqa: S101
        doc_ids_to_embeddings: dict[str, list[float]] = {}
        meta: dict[str, Any] = {}

        batches = list(batched(texts_to_embed.items(), batch_size))
        iterator = async_tqdm(batches, desc="Calculating embeddings") if self.progress_bar else batches

        for batch in iterator:
            kwargs = self._prepare_input([b[1] for b in batch])
            try:
                response = await self._async_client.embeddings.create(**kwargs)
            except APIError as exc:
                ids = ", ".join(b[0] for b in batch)
                logger.exception("Failed embedding of documents {ids} caused by {exc}", ids=ids, exc=exc)
                if self.raise_on_failure:
                    raise
                continue

            embeddings = [el.embedding for el in response.data]
            doc_ids_to_embeddings.update(dict(zip((b[0] for b in batch), embeddings, strict=True)))
            self._update_meta(meta, response)

        return doc_ids_to_embeddings, meta

    @staticmethod
    def _validate_documents(documents: list[Document]) -> None:
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            msg = (
                "VLLMDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the VLLMTextEmbedder."
            )
            raise TypeError(msg)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Embed a list of Documents.

        :param documents: Documents to embed.
        :returns: A dictionary with:
            - `documents`: The input documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        if not self._is_warmed_up:
            self.warm_up()

        texts_to_embed = self._prepare_texts_to_embed(documents)
        doc_ids_to_embeddings, meta = self._embed_batch(texts_to_embed, self.batch_size)

        new_documents = [
            replace(doc, embedding=doc_ids_to_embeddings[doc.id]) if doc.id in doc_ids_to_embeddings else replace(doc)
            for doc in documents
        ]
        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Asynchronously embed a list of Documents.

        :param documents: Documents to embed.
        :returns: A dictionary with:
            - `documents`: The input documents with their `embedding` field populated.
            - `meta`: Information about the usage of the model.
        """
        self._validate_documents(documents)
        if not documents:
            return {"documents": [], "meta": {}}

        if not self._is_warmed_up:
            self.warm_up()

        texts_to_embed = self._prepare_texts_to_embed(documents)
        doc_ids_to_embeddings, meta = await self._embed_batch_async(texts_to_embed, self.batch_size)

        new_documents = [
            replace(doc, embedding=doc_ids_to_embeddings[doc.id]) if doc.id in doc_ids_to_embeddings else replace(doc)
            for doc in documents
        ]
        return {"documents": new_documents, "meta": meta}
