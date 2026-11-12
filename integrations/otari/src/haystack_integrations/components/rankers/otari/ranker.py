# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

import httpx
from haystack import Document, component
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry

from haystack_integrations.common.otari.utils import _warn_if_eu_key_on_default_url


@component
class OtariRanker:
    """
    Ranks Documents by relevance to a query using an [Otari](https://github.com/mozilla-ai/otari) gateway.

    It calls the `/rerank` endpoint of the gateway. By default, this component calls the otari.ai gateway. For an
    account in otari.ai's EU region, whose API keys start with `otk_v1_eu_`, set `api_base_url` to
    `"https://eu.api.otari.ai/api/v1"`. To call a gateway you run yourself, set `api_base_url`, for example to
    `"http://localhost:8000/api/v1"`.

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
        api_base_url: str = "https://api.otari.ai/api/v1",
        top_k: int | None = None,
        score_threshold: float | None = None,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
        max_tokens_per_doc: int | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Creates an instance of OtariRanker.

        :param model: The rerank model selector, for example `"cohere:rerank-v3.5"`.
            The gateway must have the provider configured.
        :param api_key: The Otari API key: a key issued by your gateway, or an otari.ai API key.
            Defaults to the `OTARI_API_KEY` environment variable.
        :param api_base_url: The API root of the Otari gateway, including the `/api/v1` path.
            Defaults to the otari.ai gateway. For an account in otari.ai's EU region, use
            `"https://eu.api.otari.ai/api/v1"`.
        :param top_k: The maximum number of Documents to return. If `None`, all documents are returned.
        :param score_threshold: If set, documents with a relevance score below this value are dropped.
            Applied after `top_k`, so the output may contain fewer than `top_k` documents.
        :param meta_fields_to_embed: List of meta fields that should be concatenated with the document
            content before reranking.
        :param meta_data_separator: Separator used to concatenate the meta fields to the document content.
        :param max_tokens_per_doc: The maximum number of tokens of each document that the model considers.
            Longer documents are truncated. If `None`, the provider's default applies.
        :param timeout: The timeout for each request to the gateway, in seconds.
        :param max_retries: The maximum number of retries after a failed request. Requests are retried on timeouts,
            connection errors, and the HTTP status codes 408, 418, 429, and 503.

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
        self.timeout = timeout
        self.max_retries = max_retries

        self._region_checked = False

    def _headers(self) -> dict[str, str]:
        """Build the request headers, and warn once if an EU key meets the default URL."""
        if not self._region_checked:
            _warn_if_eu_key_on_default_url(self.api_key, self.api_base_url)
            self._region_checked = True
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key.resolve_value()}"}

    def _request_kwargs(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "method": "POST",
            "url": f"{self.api_base_url.rstrip('/')}/rerank",
            "json": body,
            "headers": self._headers(),
            "timeout": self.timeout,
            "attempts": self.max_retries + 1,
        }

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

        # Otari names the model in rerank responses since 0.18.0; for older gateways, report the requested selector
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
        :raises httpx.RequestError: If the request still fails to reach the gateway after all retries.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)
        body = self._prepare_request(query, documents, top_k)
        try:
            response = request_with_retry(**self._request_kwargs(body))
        except httpx.HTTPStatusError as e:
            response = e.response
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
        :raises httpx.RequestError: If the request still fails to reach the gateway after all retries.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)
        body = self._prepare_request(query, documents, top_k)
        try:
            response = await async_request_with_retry(**self._request_kwargs(body))
        except httpx.HTTPStatusError as e:
            response = e.response
        return self._parse_response(response, documents, score_threshold)
