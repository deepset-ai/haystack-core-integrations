# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Document-store helpers shared by the examples.

import os

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore


def build_document_store(backend: str, corpus: str) -> DocumentStore:
    """Build an empty in-memory or OpenSearch document store for a corpus."""
    if backend == "opensearch":
        from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore  # noqa: PLC0415

        url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        return OpenSearchDocumentStore(
            hosts=url,
            index=f"advanced-rag-eval-{corpus}",
            use_ssl=url.startswith("https"),
            verify_certs=not url.startswith("https://localhost"),
        )
    return InMemoryDocumentStore()


def build_retriever(store: DocumentStore, top_k: int = 5):  # noqa: ANN201
    """Build the matching BM25 retriever for a document store."""
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever  # noqa: PLC0415

    return OpenSearchBM25Retriever(document_store=store, top_k=top_k)
