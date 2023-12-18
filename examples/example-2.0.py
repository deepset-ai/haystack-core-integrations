import os
from pathlib import Path

from haystack.document_stores import DuplicatePolicy
from haystack.pipeline_utils import build_indexing_pipeline

from astra_store.document_store import AstraDocumentStore

astra_id = os.getenv("ASTRA_DB_ID", "")
astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")

astra_application_token = os.getenv(
    "ASTRA_DB_APPLICATION_TOKEN",
    "",
)

collection_name = os.getenv("COLLECTION_NAME", "haystack_vector_search")
keyspace_name = os.getenv("KEYSPACE_NAME", "recommender_demo")

document_store = AstraDocumentStore(
    astra_id=astra_id,
    astra_region=astra_region,
    astra_application_token=astra_application_token,
    astra_keyspace=keyspace_name,
    astra_collection=collection_name,
    duplicates_policy=DuplicatePolicy.OVERWRITE,
    embedding_dim=768,
)

# Let's now build indexing pipeline that indexes PDFs and text files from a test folder.
indexing_pipeline = build_indexing_pipeline(
    document_store=document_store, embedding_model="sentence-transformers/all-mpnet-base-v2"
)
result = indexing_pipeline.run(files=list(Path("examples/data").iterdir()))
print(result)
