import os
from pathlib import Path

from haystack.preview import Pipeline
from haystack.preview.components.file_converters import TextFileToDocument
from haystack.preview.components.writers import DocumentWriter
from haystack.utils import add_example_data

from astra_store.document_store import AstraDocumentStore
from astra_store.retriever import AstraRetriever

HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir("examples/data")]

astra_id = os.getenv("ASTRA_DB_ID", "")
astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

astra_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN", "")
collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")
keyspace_name = os.getenv("KEYSPACE_NAME", "recommender_demo")

# We support many different databases. Here, we load a simple and lightweight in-memory database.
document_store = AstraDocumentStore(
    astra_id=astra_id,
    astra_region=astra_region,
    astra_collection=collection_name,
    astra_keyspace=keyspace_name,
    astra_application_token=astra_application_token,
    embedding_dim=384,
)

print("count:")
print(document_store.count_documents())
add_example_data(document_store, "examples/data")

indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
indexing.add_component("writer", DocumentWriter(document_store))
indexing.connect("converter", "writer")
# print("Indexing data...")
# indexing.run({"converter": {"paths": file_paths}})

querying = Pipeline()
querying.add_component("retriever", AstraRetriever(document_store))
results = querying.run({"retriever": {"queries": ["Is black and white text boring?"], "top_k": 3}})

print("query:")
print(results)

print("count:")
print(document_store.count_documents())

print("filter:")
print(document_store.filter_documents({"content_type": "text"}))


print(document_store.search(["Is black and white text boring?"], 3))
print("get_document_by_id and embeddings *********")
print(document_store.get_document_by_id("539fb0d47917e832bbc661e55edb8b90"))
print("get_documents_by_ids and embeddings *********")
print(document_store.get_documents_by_id(["23dc6bb45225bade2764d856a0e1a6b3"]))

# document_store.delete_documents(["23dc6bb45225bade2764d856a0e1a6b3"])
# document_store.delete_documents(delete_all=True)

print("count:")
print(document_store.count_documents())
