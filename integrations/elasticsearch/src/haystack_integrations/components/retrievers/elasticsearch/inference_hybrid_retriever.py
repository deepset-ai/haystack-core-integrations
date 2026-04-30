# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@component
class ElasticsearchInferenceHybridRetriever:
    """
    A fully server-side hybrid retriever combining BM25 and ELSER sparse vector search via Elasticsearch RRF.

    Issues a single Elasticsearch request using the `retriever.rrf` API (ES 8.9+ for `rank.rrf`,
    ES 8.14+ for the Retriever API). No local embedding model is required and no client-side
    score merging takes place — ranking is handled entirely by Elasticsearch.

    Usage example (Elastic Cloud with ELSER deployed):

    ```python
    import os
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceHybridRetriever
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

    doc_store = ElasticsearchDocumentStore(
        hosts=os.environ["ELASTICSEARCH_URL"],
        api_key=os.environ["ELASTIC_API_KEY"],
        sparse_vector_field="sparse_vec",
    )
    retriever = ElasticsearchInferenceHybridRetriever(
        document_store=doc_store,
        inference_id=".elser-2-elasticsearch",
    )
    results = retriever.run(query="What is reinforcement learning?")
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        inference_id: str,
        filters: dict[str, Any] | None = None,
        fuzziness: str = "AUTO",
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        rank_window_size: int = 100,
        rank_constant: int = 60,
    ) -> None:
        """
        Create the ElasticsearchInferenceHybridRetriever component.

        :param document_store: An instance of ElasticsearchDocumentStore with `sparse_vector_field` configured.
        :param inference_id: The Elasticsearch inference endpoint ID used for sparse vector search e.g.
            ".elser-2-elasticsearch"
        :param filters: Filters applied to both sub-retrievers.
        :param fuzziness: Fuzziness for the BM25 multi_match query.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how runtime filters are merged with init-time filters.
        :param rank_window_size: Number of candidates each sub-retriever collects before RRF ranking.
        :param rank_constant: RRF rank constant. Higher values reduce the impact of rank position differences.
        :raises ValueError: If `document_store` is not an ElasticsearchDocumentStore or `inference_id` is empty.
        """
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        if not inference_id:
            msg = "inference_id must be provided"
            raise ValueError(msg)

        self._document_store = document_store
        self._inference_id = inference_id
        self._filters = filters or {}
        self._fuzziness = fuzziness
        self._top_k = top_k
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy
        self._rank_window_size = rank_window_size
        self._rank_constant = rank_constant

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            inference_id=self._inference_id,
            filters=self._filters,
            fuzziness=self._fuzziness,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
            rank_window_size=self._rank_window_size,
            rank_constant=self._rank_constant,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchInferenceHybridRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component instance.
        """
        data["init_parameters"]["document_store"] = ElasticsearchDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document])
    def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Run a hybrid retrieval query against Elasticsearch.

        :param query: The query string.
        :param filters: Runtime filters merged with init-time filters according to `filter_policy`.
        :param top_k: Maximum number of documents to return, overrides the init-time value.
        :returns: A dictionary with key `documents` containing the retrieved list of `Document`s.
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = self._document_store._hybrid_retrieval_inference(
            query=query,
            inference_id=self._inference_id,
            filters=filters,
            fuzziness=self._fuzziness,
            top_k=top_k or self._top_k,
            rank_window_size=self._rank_window_size,
            rank_constant=self._rank_constant,
        )
        return {"documents": docs}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Asynchronously run a hybrid retrieval query against Elasticsearch.

        :param query: The query string.
        :param filters: Runtime filters merged with init-time filters according to `filter_policy`.
        :param top_k: Maximum number of documents to return, overrides the init-time value.
        :returns: A dictionary with key `documents` containing the retrieved list of `Document`s.
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = await self._document_store._hybrid_retrieval_inference_async(
            query=query,
            inference_id=self._inference_id,
            filters=filters,
            fuzziness=self._fuzziness,
            top_k=top_k or self._top_k,
            rank_window_size=self._rank_window_size,
            rank_constant=self._rank_constant,
        )
        return {"documents": docs}
