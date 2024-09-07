# Before running this example, ensure you have PostgreSQL installed with the pgvector extension.
# For a quick setup using Docker:
# docker run -d -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres
# -e POSTGRES_DB=postgres ankane/pgvector

# Install required packages for this example, including pgvector-haystack and other libraries needed
# for Markdown conversion and embeddings generation. Use the following command:
# pip install pgvector-haystack markdown-it-py mdit_plain "sentence-transformers>=2.2.0"

# Download some Markdown files to index.
# git clone https://github.com/anakin87/neural-search-pills

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.pgvector import PgvectorEmbeddingRetriever, PgvectorKeywordRetriever
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

# Set an environment variable `PG_CONN_STR` with the connection string to your PostgreSQL database.
# e.g., "postgresql://USER:PASSWORD@HOST:PORT/DB_NAME"

# Initialize PgvectorDocumentStore
document_store = PgvectorDocumentStore(
    table_name="haystack_test",
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
    search_strategy="hnsw",
)

# Create the indexing Pipeline and index some documents
file_paths = glob.glob("neural-search-pills/pills/*.md")


indexing = Pipeline()
indexing.add_component("converter", MarkdownToDocument())
indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
indexing.add_component("document_embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "splitter")
indexing.connect("splitter", "document_embedder")
indexing.connect("document_embedder", "writer")

indexing.run({"converter": {"sources": file_paths}})

# Create the querying Pipeline and try a query
querying = Pipeline()
querying.add_component("text_embedder", SentenceTransformersTextEmbedder())
querying.add_component("retriever", PgvectorEmbeddingRetriever(document_store=document_store, top_k=3))
querying.add_component("keyword_retriever", PgvectorKeywordRetriever(document_store=document_store, top_k=3))
querying.add_component(
    "joiner",
    DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=3),
)
querying.connect("text_embedder", "retriever")
querying.connect("keyword_retriever", "joiner")
querying.connect("retriever", "joiner")

query = "cross-encoder"
results = querying.run({"text_embedder": {"text": query}, "keyword_retriever": {"query": query}})

for doc in results["joiner"]["documents"]:
    print(doc)
    print("-" * 10)
