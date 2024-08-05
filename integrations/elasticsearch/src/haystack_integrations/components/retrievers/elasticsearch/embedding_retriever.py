# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy
from haystack_integrations.document_stores.elasticsearch.document_store import ElasticsearchDocumentStore


@component
class ElasticsearchEmbeddingRetriever:
    """
    ElasticsearchEmbeddingRetriever retrieves documents from the ElasticsearchDocumentStore using vector similarity.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
    from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever

    document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    retriever = ElasticsearchEmbeddingRetriever(document_store=document_store)

    # Add documents to DocumentStore
    documents = [
        Document(text="My name is Carla and I live in Berlin"),
        Document(text="My name is Paul and I live in New York"),
        Document(text="My name is Silvano and I live in Matera"),
        Document(text="My name is Usagi Tsukino and I live in Tokyo"),
    ]
    document_store.write_documents(documents)

    te = SentenceTransformersTextEmbedder()
    te.warm_up()
    query_embeddings = te.run("Who lives in Berlin?")["embedding"]

    result = retriever.run(query=query_embeddings)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(
        self,
        *,
        document_store: ElasticsearchDocumentStore,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        num_candidates: Optional[int] = None,
        filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE,
    ):
        """
        Create the ElasticsearchEmbeddingRetriever component.

        :param document_store: An instance of ElasticsearchDocumentStore.
        :param filters: Filters applied to the retrieved Documents.
            Filters are applied during the approximate KNN search to ensure that top_k matching documents are returned.
        :param top_k: Maximum number of Documents to return.
        :param num_candidates: Number of approximate nearest neighbor candidates on each shard. Defaults to top_k * 10.
            Increasing this value will improve search accuracy at the cost of slower search speeds.
            You can read more about it in the Elasticsearch
            [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#tune-approximate-knn-for-speed-accuracy)
        :param filter_policy: Policy to determine how filters are applied.
        :raises ValueError: If `document_store` is not an instance of ElasticsearchDocumentStore.
        """
        if not isinstance(document_store, ElasticsearchDocumentStore):
            msg = "document_store must be an instance of ElasticsearchDocumentStore"
            raise ValueError(msg)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._num_candidates = num_candidates
        self._filter_policy = FilterPolicy.from_str(filter_policy) if isinstance(filter_policy, str) else filter_policy

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            filters=self._filters,
            top_k=self._top_k,
            num_candidates=self._num_candidates,
            filter_policy=self._filter_policy.value,
            document_store=self._document_store.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ElasticsearchEmbeddingRetriever":
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
        # Pipelines serialized with old versions of the component might not
        # have the filter_policy field.
        if filter_policy := data["init_parameters"].get("filter_policy"):
            data["init_parameters"]["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(self, query_embedding: List[float], filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = None):
        """
        Retrieve documents using a vector similarity metric.

        :param query_embedding: Embedding of the query.
        :param filters: Filters applied to the retrieved Documents. The way runtime filters are applied depends on
                        the `filter_policy` chosen at retriever initialization. See init method docstring for more
                        details.
        :param top_k: Maximum number of `Document`s to return.
        :returns: A dictionary with the following keys:
            - `documents`: List of `Document`s most similar to the given `query_embedding`
        """
        filters = apply_filter_policy(self._filter_policy, self._filters, filters)
        docs = self._document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k or self._top_k,
            num_candidates=self._num_candidates,
        )
        return {"documents": docs}
