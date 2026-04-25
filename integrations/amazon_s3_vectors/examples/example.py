# Amazon S3 Vectors — Haystack Integration Example
#
# A serverless vector store on AWS: no database to provision, no cluster to manage.
# Just a bucket name and AWS credentials.
#
# This example indexes documents, runs unfiltered and filtered queries, and cleans up.
#
# Prerequisites:
#   pip install amazon-s3-vectors-haystack "sentence-transformers>=2.2.0"
#   AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)

import time

from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Create the document store (bucket + index are created automatically) ---
document_store = S3VectorsDocumentStore(
    vector_bucket_name="haystack-example",
    index_name="products",
    dimension=384,  # all-MiniLM-L6-v2 produces 384-dim embeddings
    distance_metric="cosine",
    region_name="us-east-1",
)

# --- Index documents with metadata ---
products = [
    Document(
        content="Lightweight running shoes with breathable mesh",
        meta={"category": "shoes", "price": 89.99, "in_stock": True},
    ),
    Document(
        content="Waterproof hiking boots with ankle support",
        meta={"category": "shoes", "price": 149.99, "in_stock": True},
    ),
    Document(
        content="Classic leather dress shoes",
        meta={"category": "shoes", "price": 199.99, "in_stock": False},
    ),
    Document(
        content="Insulated winter jacket with down filling",
        meta={"category": "jackets", "price": 249.99, "in_stock": True},
    ),
    Document(
        content="Lightweight windbreaker for running",
        meta={"category": "jackets", "price": 79.99, "in_stock": True},
    ),
]

indexing = Pipeline()
indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=MODEL))
indexing.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("embedder", "writer")

print("Indexing products...")
indexing.run({"embedder": {"documents": products}})
print(f"Indexed {document_store.count_documents()} products.\n")

# --- Build the query pipeline ---
querying = Pipeline()
querying.add_component("embedder", SentenceTransformersTextEmbedder(model=MODEL))
querying.add_component("retriever", S3VectorsEmbeddingRetriever(document_store=document_store, top_k=5))
querying.connect("embedder", "retriever")


def search(query: str, filters: dict | None = None) -> None:
    print(f"Query: '{query}'" + (f"  Filter: {filters}" if filters else ""))
    start = time.time()
    result = querying.run({"embedder": {"text": query}, "retriever": {"filters": filters}})
    elapsed = (time.time() - start) * 1000
    for i, doc in enumerate(result["retriever"]["documents"], 1):
        meta = doc.meta
        print(f"  {i}. [{doc.score:.3f}] {doc.content}")
        print(f"     category={meta['category']}  price=${meta['price']}  in_stock={meta['in_stock']}")
    print(f"  ({elapsed:.0f}ms)\n")


# Unfiltered: returns shoes AND jackets
search("something for running")

# Filtered to shoes only — server-side, inside the vector search
search("something for running", filters={"field": "meta.category", "operator": "==", "value": "shoes"})

# Combined filter: in-stock AND under $100
search(
    "something for running",
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.in_stock", "operator": "==", "value": True},
            {"field": "meta.price", "operator": "<", "value": 100.0},
        ],
    },
)

# --- Cleanup ---
document_store.delete_documents([doc.id for doc in products])
client = document_store._get_client()
client.delete_index(vectorBucketName="haystack-example", indexName="products")
client.delete_vector_bucket(vectorBucketName="haystack-example")
print("Cleaned up all AWS resources.")
