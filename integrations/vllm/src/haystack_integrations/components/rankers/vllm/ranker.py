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
class VLLMRanker:
    """
    Ranks Documents based on their similarity to a query using models served with [vLLM](https://docs.vllm.ai/).

    It expects a vLLM server to be running and accessible at the `api_base_url` parameter and uses the
    `/rerank` endpoint exposed by vLLM.

    ### Starting the vLLM server

    Before using this component, start a vLLM server with a reranker model:

    ```bash
    vllm serve BAAI/bge-reranker-base
    ```

    For details on server options, see the [vLLM CLI docs](https://docs.vllm.ai/en/stable/cli/serve/).

    ### Usage example

    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.vllm import VLLMRanker

    ranker = VLLMRanker(model="BAAI/bge-reranker-base")
    docs = [
        Document(content="The capital of Brazil is Brasilia."),
        Document(content="The capital of France is Paris."),
    ]
    result = ranker.run(query="What is the capital of France?", documents=docs)
    print(result["documents"][0].content)
    ```

    ### Usage example with vLLM-specific parameters

    Pass vLLM-specific parameters via the `extra_parameters` dictionary. They are merged into the
    request body sent to the `/rerank` endpoint.

    ```python
    ranker = VLLMRanker(
        model="BAAI/bge-reranker-base",
        extra_parameters={"truncate_prompt_tokens": 256},
    )
    ```
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Secret | None = Secret.from_env_var("VLLM_API_KEY", strict=False),
        api_base_url: str = "http://localhost:8000/v1",
        top_k: int | None = None,
        score_threshold: float | None = None,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
        http_client_kwargs: dict[str, Any] | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Creates an instance of VLLMRanker.

        :param model: The name of the reranker model served by vLLM. Check
            [vLLM documentation](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#supported-models) for
            information on supported models.
        :param api_key: The vLLM API key. Defaults to the `VLLM_API_KEY` environment variable.
            Only required if the vLLM server was started with `--api-key`.
        :param api_base_url: The base URL of the vLLM server.
        :param top_k: The maximum number of Documents to return. If `None`, all documents are returned.
        :param score_threshold: If set, documents with a relevance score below this value are dropped.
            Applied after `top_k`, so the output may contain fewer than `top_k` documents.
        :param meta_fields_to_embed: List of meta fields that should be concatenated with the document
            content before reranking.
        :param meta_data_separator: Separator used to concatenate the meta fields to the document content.
        :param http_client_kwargs: A dictionary of keyword arguments to configure a custom `httpx.Client` or
            `httpx.AsyncClient`. For more information, see the
            [HTTPX documentation](https://www.python-httpx.org/api/#client).
        :param extra_parameters: Additional parameters merged into the request body sent to the vLLM
            `/rerank` endpoint. Use this to pass parameters not part of the standard rerank API, such as
            `truncate_prompt_tokens`. See the
            [vLLM docs](https://docs.vllm.ai/en/stable/models/pooling_models/scoring/#rerank-api) for more information.

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
        self.http_client_kwargs = http_client_kwargs
        self.extra_parameters = extra_parameters

        self._headers = {"Content-Type": "application/json"}
        if self.api_key is not None and (resolved_key := self.api_key.resolve_value()):
            self._headers["Authorization"] = f"Bearer {resolved_key}"

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """Create the httpx clients."""
        if self._is_warmed_up:
            return

        client = init_http_client(self.http_client_kwargs, async_client=False)
        async_client = init_http_client(self.http_client_kwargs, async_client=True)
        self._client = client if client is not None else httpx.Client()
        self._async_client = async_client if async_client is not None else httpx.AsyncClient()
        self._is_warmed_up = True

    def _prepare_texts(self, documents: list[Document]) -> list[str]:
        """Concatenate each Document's text with the selected meta fields."""
        texts = []
        for doc in documents:
            meta_values = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]
            texts.append(self.meta_data_separator.join([*meta_values, doc.content or ""]))
        return texts

    def _prepare_request(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": self._prepare_texts(documents),
        }
        if top_k is not None:
            body["top_n"] = top_k
        if self.extra_parameters:
            body.update(self.extra_parameters)
        return body

    @staticmethod
    def _parse_response(
        resp: dict[str, Any],
        documents: list[Document],
        score_threshold: float | None,
    ) -> dict[str, list[Document] | dict[str, Any]]:
        if "results" not in resp:
            msg = resp.get("detail") or f"Unexpected response from vLLM rerank endpoint: {resp}"
            raise RuntimeError(msg)

        ranked_docs: list[Document] = []
        for result in resp["results"]:
            score = result["relevance_score"]
            if score_threshold is not None and score < score_threshold:
                continue
            ranked_docs.append(replace(documents[result["index"]], score=score))

        meta = {"model": resp.get("model"), "usage": resp.get("usage", {})}
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
        Returns a list of Documents ranked by their similarity to the given query.

        :param query: Query string.
        :param documents: List of Documents to rank.
        :param top_k: The maximum number of Documents to return. Overrides the value set at initialization.
        :param score_threshold: Minimum relevance score required for a document to be returned. Overrides
            the value set at initialization.
        :returns: A dictionary with:
            - `documents`: Documents sorted from most to least relevant.
            - `meta`: Information about the model and usage.

        :raises ValueError: If `top_k` is not > 0.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)

        if not self._is_warmed_up:
            self.warm_up()
        assert self._client is not None  # noqa: S101

        body = self._prepare_request(query, documents, top_k)
        url = f"{self.api_base_url.rstrip('/')}/rerank"
        response = self._client.post(url, json=body, headers=self._headers)
        return self._parse_response(response.json(), documents, score_threshold)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> dict[str, list[Document] | dict[str, Any]]:
        """
        Asynchronously returns a list of Documents ranked by their similarity to the given query.

        :param query: Query string.
        :param documents: List of Documents to rank.
        :param top_k: The maximum number of Documents to return. Overrides the value set at initialization.
        :param score_threshold: Minimum relevance score required for a document to be returned. Overrides
            the value set at initialization.
        :returns: A dictionary with:
            - `documents`: Documents sorted from most to least relevant.
            - `meta`: Information about the model and usage.

        :raises ValueError: If `top_k` is not > 0.
        """
        if not documents:
            return {"documents": [], "meta": {}}

        top_k, score_threshold = self._resolve_run_params(top_k, score_threshold)

        if not self._is_warmed_up:
            self.warm_up()
        assert self._async_client is not None  # noqa: S101

        body = self._prepare_request(query, documents, top_k)
        url = f"{self.api_base_url.rstrip('/')}/rerank"
        response = await self._async_client.post(url, json=body, headers=self._headers)
        return self._parse_response(response.json(), documents, score_threshold)
