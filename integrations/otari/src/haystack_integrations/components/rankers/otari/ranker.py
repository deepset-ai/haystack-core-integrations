# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

import httpx
from haystack import Document, component
from haystack.utils import Secret
from haystack.utils.http_client import init_http_client


@component
class OtariRanker:
    """
    Ranks Documents by relevance to a query using an [Otari](https://github.com/mozilla-ai/otari) gateway.

    It calls the `/rerank` endpoint of the gateway. Reranking is served only by an Otari gateway that runs in
    standalone mode, such as one you run yourself. The otari.ai gateway doesn't serve it. By default, this component
    calls a gateway at `http://localhost:8000/api/v1`.

    Name models with a selector the gateway accepts, such as `"cohere:rerank-v3.5"`.

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.otari import OtariRanker

    # Reads the API key from the OTARI_API_KEY environment variable
    ranker = OtariRanker()
    docs = [
        Document(content="The capital of Brazil is Brasilia."),
        Document(content="The capital of France is Paris."),
    ]
    result = ranker.run(query="What is the capital of France?", documents=docs)
    print(result["documents"][0].content)
    # >> The capital of France is Paris.
    ```
    """

    def __init__(
        self,
        *,
        model: str = "cohere:rerank-v3.5",
        api_key: Secret = Secret.from_env_var("OTARI_API_KEY"),
        api_base_url: str = "http://localhost:8000/api/v1",
        top_k: int | None = None,
        score_threshold: float | None = None,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
        max_tokens_per_doc: int | None = None,
        http_client_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of OtariRanker.

        :param model: The rerank model selector, for example `"cohere:rerank-v3.5"`.
            The gateway must have the provider configured.
        :param api_key: The Otari API key issued by your gateway. Defaults to the `OTARI_API_KEY` environment variable.
        :param api_base_url: The API root of the Otari gateway, including the `/api/v1` path.
        :param top_k: The maximum number of Documents to return. If `None`, all documents are returned.
        :param score_threshold: If set, documents with a relevance score below this value are dropped.
            Applied after `top_k`, so the output may contain fewer than `top_k` documents.
        :param meta_fields_to_embed: List of meta fields that should be concatenated with the document
            content before reranking.
        :param meta_data_separator: Separator used to concatenate the meta fields to the document content.
        :param max_tokens_per_doc: The maximum number of tokens of each document that the model considers.
            Longer documents are truncated. If `None`, the provider's default applies.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).

        :raises ValueError: If `top_k` is not > 0.
        """
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        self.model = model
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator
        self.max_tokens_per_doc = max_tokens_per_doc
        self.http_client_kwargs = http_client_kwargs

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    def _client_kwargs(self) -> dict[str, Any]:
        """Build the keyword arguments used to create HTTP clients."""
        headers = httpx.Headers((self.http_client_kwargs or {}).get("headers"))
        headers["Content-Type"] = "application/json"
        headers["Authorization"] = f"Bearer {self.api_key.resolve_value()}"

        client_kwargs = self.http_client_kwargs.copy() if self.http_client_kwargs else {}
        client_kwargs["headers"] = headers
        return client_kwargs

    def warm_up(self) -> None:
        """Create the synchronous HTTP client."""
        if self._client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=False)
            # init_http_client returns None only when it gets no kwargs, and _client_kwargs always sets headers
            assert client is not None  # noqa: S101
            self._client = client

    async def warm_up_async(self) -> None:
        """Create the asynchronous HTTP client."""
        if self._async_client is None:
            client = init_http_client(http_client_kwargs=self._client_kwargs(), async_client=True)
            # init_http_client returns None only when it gets no kwargs, and _client_kwargs always sets headers
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

    def _prepare_texts(self, documents: list[Document]) -> list[str]:
        """Concatenate each Document's text with the selected meta fields."""
        texts = []
        for doc in documents:
            meta_values = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            texts.append(self.meta_data_separator.join([*meta_values, doc.content or ""]))
        return texts

    def _prepare_request(self, query: str, documents: list[Document], top_k: int | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": self._prepare_texts(documents),
        }
        if top_k is not None:
            body["top_n"] = top_k
        if self.max_tokens_per_doc is not None:
            body["max_tokens_per_doc"] = self.max_tokens_per_doc
        return body

    def _parse_response(
        self,
        response: httpx.Response,
        documents: list[Document],
        score_threshold: float | None,
    ) -> dict[str, list[Document] | dict[str, Any]]:
        try:
            body = response.json()
        except ValueError:
            body = None

        if response.is_error or not isinstance(body, dict) or "results" not in body:
            # Otari reports errors as {"detail": ...}, with a message or a list of validation errors
            detail = body.get("detail") if isinstance(body, dict) else None
            msg = f"Otari rerank request failed with status code {response.status_code}: {detail or response.text}"
            raise RuntimeError(msg)

        ranked_docs: list[Document] = []
        for result in body["results"]:
            score = result["relevance_score"]
            if score_threshold is not None and score < score_threshold:
                continue
            ranked_docs.append(replace(documents[result["index"]], score=score))

        # Otari's rerank response doesn't name the model, so report the selector the request used
        meta = {"model": body.get("model") or self.model, "usage": body.get("usage") or {}}
        return {"documents": ranked_docs, "meta": meta}

    def _resolve_run_params(self, top_k: int | None, score_threshold: float | None) -> tuple[int | None, float | None]:
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)
        resolved_top_k = top_k if top_k is not None else self.top_k
        resolved_score_threshold = score_threshold if score_threshold is not None else self.score_threshold
        return resolved_top_k, resolved_score_threshold

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Returns a list of Documents ranked by their relevance to the given query.

        :param query: Query string.
        :param documents: List of Documents to rank.
        :param top_k: The maximum number of Documents to return. Overrides the value set at initialization.
        :param score_threshold: Minimum relevance score required for a document to be returned. Overrides
            the value set at initialization.
        :returns: A dictionary with:
            - `documents`: Documents sorted from most to least relevant.
            - `meta`: Information about the model and usage.

        :raises ValueError: If `top_k` is not > 0.
        :raises RuntimeError: If the gateway returns an error.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)

        self.warm_up()
        assert self._client is not None  # noqa: S101

        body = self._prepare_request(query, documents, top_k)
        response = self._client.post(f"{self.api_base_url.rstrip('/')}/rerank", json=body)
        return self._parse_response(response, documents, score_threshold)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Asynchronously returns a list of Documents ranked by their relevance to the given query.

        :param query: Query string.
        :param documents: List of Documents to rank.
        :param top_k: The maximum number of Documents to return. Overrides the value set at initialization.
        :param score_threshold: Minimum relevance score required for a document to be returned. Overrides
            the value set at initialization.
        :returns: A dictionary with:
            - `documents`: Documents sorted from most to least relevant.
            - `meta`: Information about the model and usage.

        :raises ValueError: If `top_k` is not > 0.
        :raises RuntimeError: If the gateway returns an error.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)

        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101

        body = self._prepare_request(query, documents, top_k)
        response = await self._async_client.post(f"{self.api_base_url.rstrip('/')}/rerank", json=body)
        return self._parse_response(response, documents, score_threshold)
