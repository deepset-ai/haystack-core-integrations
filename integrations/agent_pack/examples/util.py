# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import sys
import threading
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


def preview(text: str, limit: int) -> str:
    """
    Collapse text to one line and cut it, marking the cut so a reader knows there is more.

    :param text: The text to preview.
    :param limit: How many characters to keep.
    :returns: The preview, ending in an ellipsis when anything was cut.
    """
    collapsed = " ".join((text or "").split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit]}..."


def quiet_hub_warnings() -> None:
    """
    Stop the Hugging Face Hub's unauthenticated-request notice from interleaving with a walkthrough's output.

    The corpus is public and cached after the first run, so the notice says nothing a reader can act on. It is
    written straight to the standard error descriptor rather than logged or warned, so neither a logging filter
    nor a `sys.stderr` replacement reaches it; the descriptor itself is routed through a pipe and everything but
    that one line is passed on unchanged.
    """
    noise = b"unauthenticated requests to the HF Hub"
    original = os.dup(2)
    reading, writing = os.pipe()
    os.dup2(writing, 2)
    os.close(writing)

    def forward() -> None:
        with os.fdopen(reading, "rb") as incoming:
            for line in incoming:
                if noise not in line:
                    os.write(original, line)

    pump = threading.Thread(target=forward, daemon=True)
    pump.start()

    def restore() -> None:
        """Put the descriptor back, so a final traceback is not lost to a daemon thread on the way out."""
        sys.stderr.flush()
        os.dup2(original, 2)
        pump.join(timeout=2.0)

    atexit.register(restore)
