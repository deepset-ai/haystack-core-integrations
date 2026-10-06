# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Index and retrieve documents from Azure DocumentDB using Microsoft Entra authentication."""

from haystack import Document
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.azure_documentdb import AzureDocumentDBEmbeddingRetriever
from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore

store = AzureDocumentDBDocumentStore(database_name="haystack", collection_name="documents")

# Run once after creating the collection. Match dimensions to the selected embedding model.
store.create_vector_index(dimensions=1536, kind="vector-hnsw", similarity="COS", m=16, efConstruction=64)

documents = [
    Document(content="Azure DocumentDB is a MongoDB-compatible document database."),
    Document(content="Azure DocumentDB supports integrated vector search with cosmosSearch."),
]
embedded_documents = OpenAIDocumentEmbedder(model="text-embedding-3-small").run(documents=documents)["documents"]
DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE).run(documents=embedded_documents)

query_embedding = OpenAITextEmbedder(model="text-embedding-3-small").run(text="vector database on Azure")["embedding"]
results = AzureDocumentDBEmbeddingRetriever(document_store=store).run(query_embedding=query_embedding)
for document in results["documents"]:
    print(document.content, document.score)
