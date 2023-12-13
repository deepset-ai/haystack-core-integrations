from pathlib import Path

from haystack.pipeline_utils import build_indexing_pipeline

from astra_store.document_store import AstraDocumentStore

# We support many different databases. Here we load a simple and lightweight in-memory document store.
document_store = AstraDocumentStore()

# Let's now build indexing pipeline that indexes PDFs and text files from a test folder.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2"
)
result = indexing_pipeline.run(files=list(Path("data").iterdir()))
print(result)
