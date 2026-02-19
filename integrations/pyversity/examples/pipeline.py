# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from pyversity import Strategy

from haystack_integrations.components.rankers.pyversity import PyversityReranker

# --- Index documents ---
document_store = InMemoryDocumentStore()

raw_documents = [
    Document(content="Paris is the capital of France."),
    Document(content="The Eiffel Tower is located in Paris."),
    Document(content="Berlin is the capital of Germany."),
    Document(content="The Brandenburg Gate is in Berlin."),
    Document(content="France borders Spain to the south."),
    Document(content="The Louvre is the world's largest art museum and is in Paris."),
    Document(content="Munich is the capital of Bavaria."),
    Document(content="The Rhine river flows through Germany and France."),
]

doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()
documents_with_embeddings = doc_embedder.run(raw_documents)["documents"]
document_store.write_documents(documents_with_embeddings)

# --- Build pipeline ---
pipeline = Pipeline()
pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
pipeline.add_component(
    "retriever",
    InMemoryEmbeddingRetriever(document_store=document_store, top_k=6, return_embedding=True),
)
pipeline.add_component("reranker", PyversityReranker(k=3, strategy=Strategy.MMR, diversity=0.7))

pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
pipeline.connect("retriever.documents", "reranker.documents")

# --- Run ---
result = pipeline.run({"text_embedder": {"text": "What are the famous landmarks in France?"}})

for doc in result["reranker"]["documents"]:
    print(f"{doc.score:.4f}  {doc.content}")

# 0.xxxx  Paris is the capital of France.
# 0.xxxx  The Eiffel Tower is located in Paris.
# 0.xxxx  France borders Spain to the south.
