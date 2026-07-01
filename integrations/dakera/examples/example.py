# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal end-to-end example for the Dakera Haystack integration.

Prerequisites:
    * A running Dakera server (see https://github.com/dakera-ai/dakera-deploy).
    * ``export DAKERA_API_KEY=dk-...``
    * ``pip install dakera-haystack sentence-transformers``
"""

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.dakera import DakeraEmbeddingRetriever
from haystack_integrations.document_stores.dakera import DakeraDocumentStore

document_store = DakeraDocumentStore(url="http://localhost:3000", namespace="haystack-example", dimension=384)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates self-awareness."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder", SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
)
query_pipeline.add_component("retriever", DakeraEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = query_pipeline.run({"text_embedder": {"text": "How many languages are there?"}})

print(result["retriever"]["documents"][0].content)
