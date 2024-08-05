# Install required packages for this example, including qdrant-haystack and other libraries needed
# for Markdown conversion and embeddings generation. Use the following command:
# pip install qdrant-haystack markdown-it-py mdit_plain sentence-transformers

# Download some Markdown files to index.
# git clone https://github.com/anakin87/neural-search-pills

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

# Initialize QdrantDocumentStore: for simplicity, we use an in-memory store here.
# You can also run a Qdrant instance using Docker or use Qdrant Cloud.
document_store = QdrantDocumentStore(
    ":memory:",
    index="Document",
    embedding_dim=768,
    recreate_index=True,
)

# Create the indexing Pipeline and index some documents
file_paths = glob.glob("neural-search-pills/pills/*.md")


indexing = Pipeline()
indexing.add_component("converter", MarkdownToDocument())
indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "splitter")
indexing.connect("splitter", "embedder")
indexing.connect("embedder", "writer")

indexing.run({"converter": {"sources": file_paths}})

# Create the querying Pipeline and try a query
querying = Pipeline()
querying.add_component("embedder", SentenceTransformersTextEmbedder())
querying.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store, top_k=3))
querying.connect("embedder", "retriever")

results = querying.run({"embedder": {"text": "What is a cross-encoder?"}})

for doc in results["retriever"]["documents"]:
    print(doc)
    print("-" * 10)
