# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from haystack import Document, component
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.lazy_imports import LazyImport

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def build_bm25_retriever(store: DocumentStore, top_k: int = 5) -> Any:
    """
    Build the matching BM25 retriever for a document store.

    :param store: The store to retrieve from.
    :param top_k: How many documents one retrieval returns.
    :returns: The retriever.
    """
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    opensearch_import.check()
    # Narrowed for the type checker: anything that is not the in-memory store is the OpenSearch one here.
    return OpenSearchBM25Retriever(document_store=cast("OpenSearchDocumentStore", store), top_k=top_k)


@component
class MultiQueryRetriever:
    """Retrieve once per expanded query and pool the results, which is what turns expansion into extra recall."""

    def __init__(self, store: DocumentStore, top_k: int = 10) -> None:
        """
        Build a retriever that runs one BM25 retrieval per query.

        :param store: The store to retrieve from.
        :param top_k: How many documents each individual retrieval returns.
        """
        self.retriever = build_bm25_retriever(store=store, top_k=top_k)

    @component.output_types(documents=list[Document])
    def run(self, queries: list[str]) -> dict[str, list[Document]]:
        """
        Retrieve for every query and pool what comes back, keeping each document once.

        :param queries: The original question followed by its expansions.
        :returns: The pooled documents, ranked by the earliest query that found them.
        """
        pooled: dict[str, Document] = {}
        for query in queries:
            for document in self.retriever.run(query=query)["documents"]:
                pooled.setdefault(document.id, document)
        return {"documents": list(pooled.values())}
