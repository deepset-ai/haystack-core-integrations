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
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

# Provide your connection string
connection_string = input("Enter your MongoDB Atlas connection string: ")

# Initialize the document store
document_store = MongoDBAtlasDocumentStore(
    mongo_connection_string=connection_string,
    database_name="haystack_test",
    collection_name="test_collection",
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

print("Indexed documents:" + document_store.count_documents() + "\n - ".join(document_store.filter_documents()))
