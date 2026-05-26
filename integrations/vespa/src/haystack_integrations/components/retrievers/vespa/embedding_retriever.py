# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component

from haystack_integrations.document_stores.vespa import DEFAULT_SEMANTIC_RANKING, VespaDocumentStore


@component
class VespaEmbeddingRetriever:
    """Retrieve documents from Vespa using dense vector similarity."""

    def __init__(
        self,
        *,
        document_store: VespaDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        ranking: str | None = DEFAULT_SEMANTIC_RANKING,
        query_tensor_name: str = "query_embedding",
        target_hits: int | None = None,
    ) -> None:
        """
        Create a Vespa embedding retriever.

        :param document_store: Configured `VespaDocumentStore` for your application, for example
            `VespaDocumentStore(url="http://localhost", schema="doc", namespace="doc")` aligned with your
            Vespa schema. See https://docs.vespa.ai/en/basics/documents.html and the integration package README.
        :param filters: Optional static Haystack metadata filters unless overridden in :meth:`run`, for example
            `{"field": "meta.category", "operator": "==", "value": "news"}`. See
            https://docs.haystack.deepset.ai/docs/metadata-filtering and https://docs.vespa.ai/en/query-language.html.
        :param top_k: Default maximum number of documents to return per query (for example `10`).
        :param ranking: Vespa rank profile used after nearest-neighbor retrieval, for example `semantic` for a
            profile that scores with `closeness(field, embedding)`. Defaults to `semantic`. Pass `None` to use the
            schema default profile. See https://docs.vespa.ai/en/basics/ranking.html.
        :param query_tensor_name: Name of the query tensor in YQL and in `input.query(...)` in your rank profile.
            For example `query_embedding` matches the default `semantic` profile. See
            https://docs.vespa.ai/en/nearest-neighbor-search.html.
        :param target_hits: Optional nearest-neighbor `targetHits` value, for example `10` or `100`: how many
            neighbors are considered per content node before first-phase ranking. See
            https://docs.vespa.ai/en/nearest-neighbor-search.html.
        :raises ValueError: If `document_store` is not an instance of VespaDocumentStore.
        """
        if not isinstance(document_store, VespaDocumentStore):
            msg = "document_store must be an instance of VespaDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.ranking = ranking
        self.query_tensor_name = query_tensor_name
        self.target_hits = target_hits

    @component.output_types(documents=list[Document])
    def run(
        self, query_embedding: list[float], filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from Vespa.

        :param query_embedding: Dense query embedding.
        :param filters: Filters applied when fetching documents from the Document Store.
        :param top_k: Maximum number of documents to return.
        :returns: Retrieved documents.
        """
        applied_filters = filters or self.filters
        documents = self.document_store._embedding_retrieval(
            query_embedding=query_embedding,
            filters=applied_filters or None,
            top_k=top_k or self.top_k,
            ranking=self.ranking,
            query_tensor_name=self.query_tensor_name,
            target_hits=self.target_hits,
        )
        return {"documents": documents}
