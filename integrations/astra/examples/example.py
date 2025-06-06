import os
from pathlib import Path

from haystack import Pipeline, logging
from haystack.components.converters import TextFileToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.routers import FileTypeRouter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


HERE = Path(__file__).resolve().parent
file_paths = [HERE / "data" / Path(name) for name in os.listdir("integrations/astra/examples/data")]
logger.info(file_paths)

collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")

# Make sure ASTRA_DB_API_ENDPOINT and ASTRA_DB_APPLICATION_TOKEN environment variables are set before proceeding

# We support many different databases. Here, we load a simple and lightweight in-memory database.
document_store = AstraDocumentStore(
    collection_name=collection_name,
    duplicates_policy=DuplicatePolicy.OVERWRITE,
    embedding_dimension=384,
)

# Create components and an indexing pipeline that converts txt files to documents,
# cleans and splits them, and indexes them
p = Pipeline()
p.add_component(instance=FileTypeRouter(mime_types=["text/plain", "application/pdf"]), name="file_type_router")
p.add_component(instance=TextFileToDocument(), name="text_file_converter")
p.add_component(instance=DocumentCleaner(), name="cleaner")
p.add_component(instance=DocumentSplitter(split_by="word", split_length=150, split_overlap=30), name="splitter")
p.add_component(
    instance=SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
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
    instance=SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2"),
    name="embedder",
)
q.add_component("retriever", AstraEmbeddingRetriever(document_store))

q.connect("embedder", "retriever")

question = "This chapter introduces the manuals available with Vim"
result = q.run({"embedder": {"text": question}, "retriever": {"top_k": 1}})
logger.info(result)

ALL_DOCUMENTS_COUNT = 9
documents_count = document_store.count_documents()
logger.info("count:")
logger.info(documents_count)
if documents_count != ALL_DOCUMENTS_COUNT:
    msg = f"count mismatch, expected 9 documents, got {documents_count}"
    raise ValueError(msg)

logger.info(
    f"""filter results: {
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
    }"""
)

logger.info(
    f"""get_document_by_id {
        document_store.get_document_by_id("92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10")
    }"""
)

logger.info(
    f"""get_documents_by_ids {
        document_store.get_documents_by_id(
            [
                "92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10",
                "6f2450a51eaa3eeb9239d875402bcfe24b2d3534ff27f26c1f3fc8133b04e756",
            ]
        )
    }"""
)

document_store.delete_documents(["92ef055fbae55b2b0fc79d34cbf8a80b0ad7700ca526053223b0cc6d1351df10"])

documents_count = document_store.count_documents()
logger.info(f"count: {document_store.count_documents()}")
if documents_count != ALL_DOCUMENTS_COUNT - 1:
    msg = f"count mismatch, expected 9 documents, got {documents_count}"
    raise ValueError(msg)
