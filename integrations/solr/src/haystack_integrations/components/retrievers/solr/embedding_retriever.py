# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy
from haystack.document_stores.types.filter_policy import apply_filter_policy

from haystack_integrations.document_stores.solr import SolrDocumentStore

logger = logging.getLogger(__name__)


@component
class SolrEmbeddingRetriever:
    """
    Fetches documents from a `SolrDocumentStore` using Solr's `{!knn}` dense vector search.

    Usage example:
    ```python
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack_integrations.document_stores.solr import SolrDocumentStore
    from haystack_integrations.components.retrievers.solr import SolrEmbeddingRetriever

    document_store = SolrDocumentStore(core="haystack", embedding_dim=384)
    embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    pipeline = Pipeline()
    pipeline.add_component("embedder", embedder)
    pipeline.add_component("retriever", SolrEmbeddingRetriever(document_store=document_store))
    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    result = pipeline.run(data={"embedder": {"text": "Apache Solr"}})
    ```
    """

    def __init__(
        self,
        *,
        document_store: SolrDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        filter_policy: str | FilterPolicy = FilterPolicy.REPLACE,
        raise_on_failure: bool = True,
    ) -> None:
        """
        Create a `SolrEmbeddingRetriever`.

        :param document_store: the document store to search.
        :param filters: filters applied to the search. Combined with the filters passed to `run`
            according to `filter_policy`. Filters act as a k-NN graph pre-filter, so the search still
            returns up to `top_k` documents.
        :param top_k: maximum number of documents to return.
        :param filter_policy: how runtime filters combine with the filters given here.
        :param raise_on_failure: whether a failing search raises, or logs and returns no documents.
        :raises ValueError: if `document_store` is not a `SolrDocumentStore`, or `top_k` is not positive.
        """
        if not isinstance(document_store, SolrDocumentStore):
            msg = "document_store must be an instance of SolrDocumentStore"
            raise ValueError(msg)
        self._validate_top_k(top_k)

        self._document_store = document_store
        self._filters = filters or {}
        self._top_k = top_k
        self._filter_policy = (
            filter_policy if isinstance(filter_policy, FilterPolicy) else FilterPolicy.from_str(filter_policy)
        )
        self._raise_on_failure = raise_on_failure

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: dictionary with serialized data.
        """
        return default_to_dict(
            self,
            document_store=self._document_store.to_dict(),
            filters=self._filters,
            top_k=self._top_k,
            filter_policy=self._filter_policy.value,
            raise_on_failure=self._raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolrEmbeddingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: dictionary to deserialize from.
        :returns: deserialized component.
        """
        init_parameters = data["init_parameters"]
        init_parameters["document_store"] = SolrDocumentStore.from_dict(init_parameters["document_store"])
        # Pipelines serialized before `filter_policy` existed omit the key entirely.
        if filter_policy := init_parameters.get("filter_policy"):
            init_parameters["filter_policy"] = FilterPolicy.from_str(filter_policy)
        return default_from_dict(cls, data)

    @staticmethod
    def _validate_top_k(top_k: int | None) -> None:
        if top_k is not None and top_k <= 0:
            msg = f"top_k must be > 0, but got {top_k}"
            raise ValueError(msg)

    def _search_kwargs(self, filters: dict[str, Any] | None, top_k: int | None) -> dict[str, Any]:
        return {
            "filters": apply_filter_policy(self._filter_policy, self._filters, filters),
            "top_k": top_k if top_k is not None else self._top_k,
        }

    @component.output_types(documents=list[Document])
    def run(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents similar to `query_embedding`.

        :param query_embedding: the query embedding.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :returns: a dictionary with a `documents` key holding the retrieved documents.
        :raises ValueError: if `top_k` is not positive.
        """
        self._validate_top_k(top_k)
        try:
            documents = self._document_store._embedding_retrieval(
                query_embedding, **self._search_kwargs(filters, top_k)
            )
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning(
                "An error occurred during embedding retrieval and will be ignored: {error}", error=str(error)
            )
            documents = []
        return {"documents": documents}

    @component.output_types(documents=list[Document])
    async def run_async(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents similar to `query_embedding`, asynchronously.

        :param query_embedding: the query embedding.
        :param filters: filters applied to the search.
        :param top_k: maximum number of documents to return.
        :returns: a dictionary with a `documents` key holding the retrieved documents.
        :raises ValueError: if `top_k` is not positive.
        """
        self._validate_top_k(top_k)
        try:
            documents = await self._document_store._embedding_retrieval_async(
                query_embedding, **self._search_kwargs(filters, top_k)
            )
        except Exception as error:
            if self._raise_on_failure:
                raise
            logger.warning(
                "An error occurred during embedding retrieval and will be ignored: {error}", error=str(error)
            )
            documents = []
        return {"documents": documents}

    def close(self) -> None:
        """Close the underlying document store connection."""
        self._document_store.close()

    async def close_async(self) -> None:
        """Close the underlying document store async connection."""
        await self._document_store.close_async()
