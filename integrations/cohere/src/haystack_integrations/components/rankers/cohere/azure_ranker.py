# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any

import httpx
from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

MAX_NUM_DOCS_FOR_COHERE_RANKER = 1000


@component
class CohereAzureRanker:
    """
    Ranks Documents based on their similarity to the query using Cohere models deployed on Azure AI.

    Usage example:
    ```python
    from haystack import Document
    from haystack_integrations.components.rankers.cohere import CohereAzureRanker

    ranker = CohereAzureRanker(
        api_base_url="https://my-endpoint.cohere.models.ai.azure.com",
        api_key=Secret.from_env_var("AZURE_COHERE_API_KEY"),
        top_k=2,
    )

    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "What is the capital of germany?"
    output = ranker.run(query=query, documents=docs)
    docs = output["documents"]
    ```
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: Secret = Secret.from_env_var(["COHERE_AZURE_API_KEY", "AZURE_COHERE_API_KEY"]),
        model: str = "rerank-v3.5",
        top_k: int = 10,
        meta_fields_to_embed: list[str] | None = None,
        meta_data_separator: str = "\n",
        max_tokens_per_doc: int = 4096,
        timeout: float = 30.0,
    ) -> None:
        """
        Creates an instance of 'CohereAzureRanker'.

        :param api_base_url: Base URL of the Cohere service deployed on Azure.
        :param api_key: Cohere Azure API key.
        :param model: Model name. Check the list of supported models in Cohere/Azure docs.
        :param top_k: Maximum number of documents to return.
        :param meta_fields_to_embed: List of meta fields concatenated with document content.
        :param meta_data_separator: Separator used to concatenate meta fields.
        :param max_tokens_per_doc: Maximum number of tokens to embed for each document.
        :param timeout: HTTP request timeout in seconds.
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.top_k = top_k
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.meta_data_separator = meta_data_separator
        self.max_tokens_per_doc = max_tokens_per_doc
        self.timeout = timeout

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_base_url=self.api_base_url,
            api_key=self.api_key.to_dict() if self.api_key else None,
            model=self.model,
            top_k=self.top_k,
            meta_fields_to_embed=self.meta_fields_to_embed,
            meta_data_separator=self.meta_data_separator,
            max_tokens_per_doc=self.max_tokens_per_doc,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CohereAzureRanker":
        """
        Deserializes the component from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns: The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        return default_from_dict(cls, data)

    def _prepare_cohere_input_docs(self, documents: list[Document], top_k: int | None = None) -> tuple[list[str], int]:
        top_k = top_k or self.top_k
        if top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

        concatenated_input_list = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta.get(key)
            ]
            concatenated_input = self.meta_data_separator.join([*meta_values_to_embed, doc.content or ""])
            concatenated_input_list.append(concatenated_input)

        if len(concatenated_input_list) > MAX_NUM_DOCS_FOR_COHERE_RANKER:
            logger.warning(
                f"The Cohere reranking endpoint only supports {MAX_NUM_DOCS_FOR_COHERE_RANKER} documents. "
                f"The number of documents has been truncated to {MAX_NUM_DOCS_FOR_COHERE_RANKER} "
                f"from {len(concatenated_input_list)}."
            )
            concatenated_input_list = concatenated_input_list[:MAX_NUM_DOCS_FOR_COHERE_RANKER]

        return concatenated_input_list, top_k

    def _get_url(self) -> str:
        base = self.api_base_url.rstrip("/")
        if base.endswith("/rerank") or base.endswith("/v1/rerank"):
            return base
        return f"{base}/v1/rerank"

    def _get_headers(self) -> dict[str, str]:
        resolved_key = self.api_key.resolve_value() if self.api_key else None
        headers = {"Content-Type": "application/json"}
        if resolved_key:
            headers["api-key"] = resolved_key
        return headers

    def _build_payload(self, query: str, documents: list[str], top_k: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "query": query,
            "documents": documents,
            "top_n": top_k,
            "return_documents": False,
        }
        if self.model:
            payload["model"] = self.model
        if self.max_tokens_per_doc is not None:
            payload["max_tokens_per_doc"] = self.max_tokens_per_doc
        return payload

    def _build_result(self, response_json: dict[str, Any], documents: list[Document]) -> dict[str, list[Document]]:
        results = response_json.get("results", [])
        sorted_docs = []
        for res in results:
            idx = res["index"]
            score = res["relevance_score"]
            doc = documents[idx]
            sorted_docs.append(replace(doc, score=score))
        return {"documents": sorted_docs}

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document], top_k: int | None = None) -> dict[str, list[Document]]:
        """
        Use the Azure-hosted Cohere reranker to re-rank the list of documents based on the query.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        :param top_k:
            The maximum number of Documents you want the Ranker to return.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given query in descending order of similarity.

        :raises ValueError: If `top_k` is not > 0.
        """
        if not documents:
            return {"documents": []}

        cohere_input_docs, resolved_top_k = self._prepare_cohere_input_docs(documents, top_k)
        url = self._get_url()
        headers = self._get_headers()
        payload = self._build_payload(query, cohere_input_docs, resolved_top_k)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return self._build_result(data, documents)

    @component.output_types(documents=list[Document])
    async def run_async(
        self, query: str, documents: list[Document], top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Asynchronously re-rank the list of documents based on the query.

        This is the asynchronous version of the `run` method. It has the same parameters and return values
        but can be used with `await` in async code.

        :param query:
            Query string.
        :param documents:
            List of Documents.
        :param top_k:
            The maximum number of Documents you want the Ranker to return.
        :returns:
            A dictionary with the following keys:
            - `documents`: List of Documents most similar to the given query in descending order of similarity.

        :raises ValueError: If `top_k` is not > 0.
        """
        if not documents:
            return {"documents": []}

        cohere_input_docs, resolved_top_k = self._prepare_cohere_input_docs(documents, top_k)
        url = self._get_url()
        headers = self._get_headers()
        payload = self._build_payload(query, cohere_input_docs, resolved_top_k)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        return self._build_result(data, documents)
