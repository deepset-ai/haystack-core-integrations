#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: Embedding retrieval with ArcadeDB + Haystack.

Prerequisites:
    docker run -d -p 2480:2480 \
        -e JAVA_OPTS="-Darcadedb.server.rootPassword=arcadedb" \
        arcadedata/arcadedb:latest

    pip install arcadedb-haystack

Usage:
    export ARCADEDB_USERNAME=root
    export ARCADEDB_PASSWORD=arcadedb
    python examples/embedding_retrieval.py
"""

from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.arcadedb import ArcadeDBEmbeddingRetriever
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

# --- 1. Create the document store ---
store = ArcadeDBDocumentStore(
    url="http://localhost:2480",
    database="haystack_example",
    embedding_dimension=4,  # small dim for demo
    similarity_function="cosine",
    recreate_type=True,
)

# --- 2. Write some documents ---
documents = [
    Document(
        content="ArcadeDB is a multi-model database supporting graphs, documents, key-value, time-series, and vectors.",
        embedding=[1.0, 0.0, 0.0, 0.0],
        meta={"category": "database", "source": "docs"},
    ),
    Document(
        content="Haystack is an open-source framework for building RAG pipelines.",
        embedding=[0.0, 1.0, 0.0, 0.0],
        meta={"category": "framework", "source": "docs"},
    ),
    Document(
        content="HNSW (Hierarchical Navigable Small World) enables fast approximate nearest neighbor search.",
        embedding=[0.5, 0.5, 0.0, 0.0],
        meta={"category": "algorithm", "source": "paper"},
    ),
    Document(
        content="Vector databases store high-dimensional embeddings for semantic search.",
        embedding=[0.8, 0.2, 0.0, 0.0],
        meta={"category": "database", "source": "blog"},
    ),
]

written = store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
print(f"Wrote {written} documents")
print(f"Total documents: {store.count_documents()}")

# --- 3. Build a retrieval pipeline ---
pipeline = Pipeline()
pipeline.add_component("retriever", ArcadeDBEmbeddingRetriever(document_store=store, top_k=3))

# --- 4. Run a similarity search ---
query_embedding = [0.9, 0.1, 0.0, 0.0]  # close to "ArcadeDB" and "Vector databases"
result = pipeline.run({"retriever": {"query_embedding": query_embedding}})

print("\n--- Top 3 results ---")
for doc in result["retriever"]["documents"]:
    print(f"  score={doc.score:.4f}  category={doc.meta.get('category')}  content={doc.content[:80]}...")

# --- 5. Filter retrieval (only 'database' category) ---
result_filtered = pipeline.run(
    {
        "retriever": {
            "query_embedding": query_embedding,
            "filters": {"field": "meta.category", "operator": "==", "value": "database"},
        }
    }
)

print("\n--- Filtered (category=database) ---")
for doc in result_filtered["retriever"]["documents"]:
    print(f"  score={doc.score:.4f}  content={doc.content[:80]}...")
