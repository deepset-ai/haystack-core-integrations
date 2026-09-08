# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os

from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DocumentStore
from haystack.lazy_imports import LazyImport

with LazyImport(message='Run "pip install opensearch-haystack" to use an OpenSearch store.') as opensearch_import:
    from haystack_integrations.components.retrievers.opensearch import OpenSearchBM25Retriever
    from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


def build_document_store(backend: str, index: str) -> DocumentStore:
    """
    Build an empty in-memory or OpenSearch document store under a named index.

    :param backend: Either "in_memory" or "opensearch".
    :param index: Names the index to open or create.
    :returns: The empty store.
    """
    if backend == "opensearch":
        opensearch_import.check()
        url = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
        return OpenSearchDocumentStore(
            hosts=url,
            index=index,
            use_ssl=url.startswith("https"),
            verify_certs=not url.startswith("https://localhost"),
        )
    return InMemoryDocumentStore(index=index)


def build_bm25_retriever(store: DocumentStore, top_k: int = 5):  # noqa: ANN201
    """Build the matching BM25 retriever for a document store."""
    if isinstance(store, InMemoryDocumentStore):
        return InMemoryBM25Retriever(document_store=store, top_k=top_k)
    opensearch_import.check()
    return OpenSearchBM25Retriever(document_store=store, top_k=top_k)
