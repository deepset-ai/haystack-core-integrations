# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.elasticsearch.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchInferenceSparseRetriever:
    """
    ElasticsearchInferenceSparseRetriever retrieves documents using Elasticsearch sparse vector inference search.

    Usage example:

    ```python
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceSparseRetriever

    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchInferenceSparseRetriever(document_store=document_store, inference_id="ELSER")

    result = retriever.run(query="Find documents about Berlin")
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        inference_id: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
    ) -> None:
        """
        Create the ElasticsearchInferenceSparseRetriever component.

        :param document_store: An instance of ElasticsearchDocumentStore.
        :param inference_id: The Elasticsearch inference model identifier used for sparse vector inference search.
        :param filters: Filters applied to the retrieved Documents.
        :param top_k: Maximum number of Documents to return.
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of ElasticsearchDocumentStore or
            `inference_id` is empty.
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
        self._top_k = top_k
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            inference_id=self._inference_id,
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElasticsearchInferenceSparseRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
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
        Retrieve documents using inference-based sparse vector search.

        :param query: Query text to use for inference-based sparse retrieval.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s most similar to the given `query`
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = self._document_store._sparse_vector_retrieval_inference(
            query=query,
            inference_id=self._inference_id,
            filters=filters,
            top_k=top_k or self._top_k,
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
        Asynchronously retrieve documents using inference-based sparse vector search.

        :param query: Query text to use for inference-based sparse retrieval.
        :param filters: Filters applied when fetching documents from the Document Store.
            The way runtime filters are applied depends on the `filter_policy` selected when initializing the Retriever.
        :param top_k: Maximum number of documents to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s most similar to the given `query`
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = await self._document_store._sparse_vector_retrieval_inference_async(
            query=query,
            inference_id=self._inference_id,
            filters=filters,
            top_k=top_k or self._top_k,
        )
        return {"documents": docs}
