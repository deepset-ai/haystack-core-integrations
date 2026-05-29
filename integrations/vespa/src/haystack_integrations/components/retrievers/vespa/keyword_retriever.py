# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import Document, component

from haystack_integrations.document_stores.vespa import DEFAULT_BM25_RANKING, VespaDocumentStore


@component
class VespaKeywordRetriever:
    """Retrieve documents from Vespa using lexical search."""

    def __init__(
        self,
        *,
        document_store: VespaDocumentStore,
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
        ranking: str | None = DEFAULT_BM25_RANKING,
    ) -> None:
        """
        Create a Vespa keyword retriever.

        :param document_store: Configured `VespaDocumentStore` for your application, for example
            `VespaDocumentStore(url="http://localhost", schema="doc", namespace="doc")` so it matches the deployed
            schema and endpoint. See https://docs.vespa.ai/en/basics/documents.html and the integration package README.
        :param filters: Optional static Haystack metadata filters applied on each retrieval unless overridden in
            :meth:`run`, for example `{"field": "meta.category", "operator": "==", "value": "news"}`. See
            https://docs.haystack.deepset.ai/docs/metadata-filtering and https://docs.vespa.ai/en/query-language.html.
        :param top_k: Default maximum number of documents to return per query (for example `10`).
        :param ranking: Vespa rank profile for lexical matches, for example `bm25` for a profile that uses
            `bm25(content)`. Defaults to `bm25`. Pass `None` to use the schema default. See
            https://docs.vespa.ai/en/basics/ranking.html.
        :raises ValueError: If `document_store` is not an instance of VespaDocumentStore.
        """
        if not isinstance(document_store, VespaDocumentStore):
            msg = "document_store must be an instance of VespaDocumentStore"
            raise ValueError(msg)

        self.document_store = document_store
        self.filters = filters
        self.top_k = top_k
        self.ranking = ranking

    @component.output_types(documents=list[Document])
    def run(
        self, query: str, filters: dict[str, Any] | None = None, top_k: int | None = None
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents from Vespa.

        :param query: Query text.
        :param filters: Filters applied when fetching documents from the Document Store.
        :param top_k: Maximum number of documents to return.
        :returns: Retrieved documents.
        """
        applied_filters = filters or self.filters
        documents = self.document_store._bm25_retrieval(
            query=query,
            filters=applied_filters or None,
            top_k=top_k or self.top_k,
            ranking=self.ranking,
        )
        return {"documents": documents}
