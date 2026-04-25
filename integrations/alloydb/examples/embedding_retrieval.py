# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Before running this example, ensure you have an AlloyDB instance running on Google Cloud.
# See: https://cloud.google.com/alloydb/docs/quickstart/create-and-connect
#
# Install required packages for this example:
# pip install alloydb-haystack "sentence-transformers>=2.2.0" markdown-it-py mdit_plain
#
# Download some Markdown files to index:
# git clone https://github.com/anakin87/neural-search-pills
#
# Set required environment variables:
# export ALLOYDB_INSTANCE_URI="projects/MY_PROJECT/locations/REGION/clusters/CLUSTER/instances/INSTANCE"
# export ALLOYDB_USER="my-db-user"
# export ALLOYDB_PASSWORD="my-db-password"

import glob

from haystack import Pipeline
from haystack.components.converters import MarkdownToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.retrievers.alloydb import AlloyDBEmbeddingRetriever
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore

# Initialize AlloyDBDocumentStore
document_store = AlloyDBDocumentStore(
    db="postgres",
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
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "splitter")
indexing.connect("splitter", "embedder")
indexing.connect("embedder", "writer")

indexing.run({"converter": {"sources": file_paths}})

# Create the querying Pipeline
querying = Pipeline()
querying.add_component("query_embedder", SentenceTransformersTextEmbedder())
querying.add_component("retriever", AlloyDBEmbeddingRetriever(document_store=document_store, top_k=3))
querying.connect("query_embedder.embedding", "retriever.query_embedding")

results = querying.run({"query_embedder": {"text": "What is Retrieval-Augmented Generation?"}})
for doc in results["retriever"]["documents"]:
    print(doc.content)
    print("---")
