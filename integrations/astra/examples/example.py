import os
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores import DuplicatePolicy

from astra_haystack.document_store import AstraDocumentStore
from astra_haystack.retriever import AstraRetriever

HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir("integrations/astra/examples/data")]
print(file_paths)

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
    duplicates_policy=DuplicatePolicy.OVERWRITE,
    embedding_dim=384,
)

# Create components and an indexing pipeline that converts txt files to documents,
# cleans and splits them, and indexes them
p = Pipeline()
p.add_component(instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router")
p.add_component(instance=TextFileToDocument(), name="text_file_converter")
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="word", split_length=150, split_overlap=30), name="splitter")
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
    name="embedder",
)
p.add_component(instance=DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP), name="writer")

p.connect("file_type_router.text/plain", "text_file_converter.sources")
p.connect("text_file_converter.documents", "cleaner.documents")
p.connect("cleaner.documents", "splitter.documents")
p.connect("splitter.documents", "embedder.documents")
p.connect("embedder.documents", "writer.documents")

p.run({"file_type_router": {"sources": file_paths}})

# Create a querying pipeline on the indexed data
q = Pipeline()
q.add_component(
    instance=SentenceTransformersTextEmbedder(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2"),
    name="embedder",
)
q.add_component("retriever", AstraRetriever(document_store))

q.connect("embedder", "retriever")

question = "This chapter introduces the manuals available with Vim"
result = q.run({"embedder": {"text": question}, "retriever": {"top_k": 1}})
print(result)

print("count:")
print(document_store.count_documents())
assert document_store.count_documents() == 9

print("filter:")
print(
    document_store.filter_documents(
        {
            "field": "meta",
            "operator": "==",
            "value": {
                "file_path": "/workspace/astra-haystack/examples/data/usr_01.txt",
                "source_id": "5b2d27de79bba97da6fc446180d0d99e1024bc7dd6a757037f0934162cfb0916",
            },
        }
    )
)

print("get_document_by_id")
print(document_store.get_document_by_id("92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10"))
print("get_documents_by_ids")
print(
    document_store.get_documents_by_id(
        [
            "92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10",
            "6f2450a51eaa3eeb9239d875402bcfe24b2d3534ff27f26c1f3fc8133b04e756",
        ]
    )
)

document_store.delete_documents(["92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10"])

print("count:")
print(document_store.count_documents())
assert document_store.count_documents() == 8
