# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

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
