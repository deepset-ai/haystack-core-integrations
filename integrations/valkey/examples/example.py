# Install the Valkey integration with example dependencies:
# uv sync --group examples
# or with pip: pip install -e ".[examples]"

# Download some markdown files to index
# git clone https://github.com/anakin87/neural-search-pills

# Run example
# uv run examples/pipeline_example.py

# Create the indexing Pipeline and index some documents

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.valkey import ValkeyEmbeddingRetriever
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

file_paths = glob.glob("neural-search-pills/pills/*.md")

document_store = ValkeyDocumentStore(
    nodes_list=[("localhost", 6379)],
    index_name="neural_search_pills",
    embedding_dim=768,
    distance_metric="cosine",
)

indexing = Pipeline()
indexing.add_component("converter", MarkdownToDocument())
indexing.add_component("splitter", DocumentSplitter(split_by="word", split_length=100))
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "splitter")
indexing.connect("splitter", "embedder")
indexing.connect("embedder", "writer")

indexing.run({"converter": {"sources": file_paths}})

# Create the querying Pipeline and try a query

querying = Pipeline()
querying.add_component("embedder", SentenceTransformersTextEmbedder())
querying.add_component("retriever", ValkeyEmbeddingRetriever(document_store=document_store, top_k=3))
querying.connect("embedder", "retriever")

results = querying.run({"embedder": {"text": "What is Question Answering?"}})

for doc in results["retriever"]["documents"]:
    print(doc)
    print("-" * 10)
