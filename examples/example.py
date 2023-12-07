import os
from pathlib import Path

from haystack.preview import Document
from haystack.preview import Pipeline
from haystack.preview.components.file_converters import TextFileToDocument
from haystack.preview.components.writers import DocumentWriter
from haystack.preview.document_stores import DuplicatePolicy
from haystack.utils import add_example_data

from astra_store.document_store import AstraDocumentStore
from astra_store.retriever import AstraRetriever

from preprocessor import PreProcessor

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
    duplicates_policy=DuplicatePolicy.SKIP,
    embedding_dim=384,
)

#document_store.delete_documents(delete_all=True)
#print("count:")
#print(document_store.count_documents())
#add_example_data(document_store, "examples/data")


indexing = Pipeline()
indexing.add_component("converter", TextFileToDocument())
converter_results = indexing.run({"converter": {"paths": file_paths}})
preprocessor = PreProcessor(split_by="word", split_length=200, split_overlap=0, split_respect_sentence_boundary=True)
docs_processed = preprocessor.process(converter_results["converter"]["documents"])
document_store.write_documents(docs_processed, embed=True)


querying = Pipeline()
querying.add_component("retriever", AstraRetriever(document_store))
results = querying.run({"retriever": {"queries": ["Is black and white text boring?"], "top_k": 3}})

print("query:")
print(results)

print("count:")
print(document_store.count_documents())
assert document_store.count_documents() == 6

print("filter:")
print(document_store.filter_documents({"mime_type": "text/plain"}))


print("search without filter")
print(document_store.search(["Is black and white text boring?"], 3))
print("search with filter")
print(document_store.search(["Is black and white text boring?"], 3, {"mime_type": "text/plain"}))

print("get_document_by_id and embeddings *********")
print(document_store.get_document_by_id("92e095d5bfd66e31bb099de89bab9101474660904818fa428a8b889996c14a62"))
print("get_documents_by_ids and embeddings *********")
print(document_store.get_documents_by_id(["1df1b4b0b21e4015cf1d0976db1185b81fe9d3c07b630d6abac582ccd2b38a37", "9f11bd49e9ca4f895ac3062f01ae7332ae3e1cabd13a42ff8db41d6e83fd6479"]))

document_store.delete_documents(["7830332ffa979794b03cdaa6d3660bc0aa44f463014da9746ba7f3b987641967"])

print("count:")
print(document_store.count_documents())
assert document_store.count_documents() == 5
