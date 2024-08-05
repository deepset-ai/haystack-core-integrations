# Install the Pinecone integration, Haystack will come as a dependency
# Install also some optional dependencies needed for Markdown conversion and text embedding
# pip install -U pinecone-haystack markdown-it-py mdit_plain "sentence-transformers>=2.2.0"

# Download some markdown files to index
# git clone https://github.com/anakin87/neural-search-pills


# Create the indexing Pipeline and index some documents

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret

from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

file_paths = glob.glob("neural-search-pills/pills/*.md")

document_store = PineconeDocumentStore(
    api_key=Secret.from_token("YOUR-PINECONE-API-KEY"),
    index="default",
    namespace="default",
    dimension=768,
    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
)

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
querying.add_component("retriever", PineconeEmbeddingRetriever(document_store=document_store, top_k=3))
querying.connect("embedder", "retriever")

results = querying.run({"embedder": {"text": "What is Question Answering?"}})

for doc in results["retriever"]["documents"]:
    print(doc)
    print("-" * 10)
