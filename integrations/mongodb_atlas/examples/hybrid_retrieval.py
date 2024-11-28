# Install required packages for this example, including mongodb-atlas-haystack and other libraries needed
# for Markdown conversion and embeddings generation. Use the following command:
#
# pip install mongodb-atlas-haystack markdown-it-py mdit_plain "sentence-transformers>=2.2.0"
#
# Download some Markdown files to index.
# git clone https://github.com/anakin87/neural-search-pills

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.mongodb_atlas import (
    MongoDBAtlasEmbeddingRetriever,
    MongoDBAtlasFullTextRetriever,
)
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

# To use the MongoDBAtlasDocumentStore, you must have a running MongoDB Atlas database.
# For details, see https://www.mongodb.com/docs/atlas/getting-started/

# Once your database is set, set the environment variable `MONGO_CONNECTION_STRING`
# with the connection string to your MongoDB Atlas database.
# format: "mongodb+srv://{mongo_atlas_username}:{mongo_atlas_password}@{mongo_atlas_host}/?{mongo_atlas_params_string}".

# Initialize the document store
document_store = MongoDBAtlasDocumentStore(
    database_name="haystack_test",
    collection_name="test_collection",
    vector_search_index="test_vector_search_index",
    full_text_search_index="test_full_text_search_index",
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
querying.add_component("embedding_retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=3))
querying.add_component("full_text_retriever", MongoDBAtlasFullTextRetriever(document_store=document_store, top_k=3))
querying.add_component(
    "joiner",
    DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=3),
)
querying.connect("text_embedder", "embedding_retriever")
querying.connect("full_text_retriever", "joiner")
querying.connect("embedding_retriever", "joiner")

query = "cross-encoder"
results = querying.run({"text_embedder": {"text": query}, "full_text_retriever": {"query": query}})

for doc in results["joiner"]["documents"]:
    print(doc)
    print("-" * 10)
