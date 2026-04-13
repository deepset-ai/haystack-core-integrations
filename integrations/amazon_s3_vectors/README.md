# amazon-s3-vectors-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/amazon-s3-vectors-haystack.svg)](https://pypi.org/project/amazon-s3-vectors-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/amazon-s3-vectors-haystack.svg)](https://pypi.org/project/amazon-s3-vectors-haystack)

---

A [Haystack](https://haystack.deepset.ai/) integration for [Amazon S3 Vectors](https://aws.amazon.com/s3/features/vectors/), providing a Document Store and Embedding Retriever backed by native vector storage in Amazon S3.

## Installation

```bash
pip install amazon-s3-vectors-haystack
```

## Usage

### Document Store

```python
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

document_store = S3VectorsDocumentStore(
    vector_bucket_name="my-vectors",
    index_name="my-index",
    dimension=768,
    distance_metric="cosine",  # or "euclidean"
    region_name="us-east-1",
)

# Write documents (embeddings are required)
docs = [
    Document(id="1", content="First document", embedding=[0.1] * 768, meta={"category": "news"}),
    Document(id="2", content="Second document", embedding=[0.2] * 768, meta={"category": "sports"}),
]
document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

# Count documents
print(document_store.count_documents())

# Delete documents
document_store.delete_documents(["1", "2"])
```

### Embedding Retriever in a Pipeline

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.retrievers.amazon_s3_vectors import S3VectorsEmbeddingRetriever
from haystack_integrations.document_stores.amazon_s3_vectors import S3VectorsDocumentStore

document_store = S3VectorsDocumentStore(
    vector_bucket_name="my-vectors",
    index_name="my-index",
    dimension=768,
)

# Index documents
doc_embedder = SentenceTransformersDocumentEmbedder()
doc_embedder.warm_up()
# ... embed and write documents ...

# Query pipeline
pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", S3VectorsEmbeddingRetriever(document_store=document_store, top_k=5))
pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = pipeline.run({"embedder": {"text": "What is the latest news?"}})
print(result["retriever"]["documents"])
```

### Filtering

The retriever supports Haystack metadata filters, which are converted to S3 Vectors filter syntax:

```python
# Filter during retrieval
result = pipeline.run({
    "embedder": {"text": "sports news"},
    "retriever": {
        "filters": {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "sports"},
                {"field": "meta.year", "operator": ">=", "value": 2024},
            ],
        }
    },
})
```

**Supported filter operators:** `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `not in`, `AND`, `OR`

### AWS Authentication

The document store uses the standard boto3 credential chain by default. You can also pass credentials explicitly:

```python
from haystack.utils.auth import Secret

document_store = S3VectorsDocumentStore(
    vector_bucket_name="my-vectors",
    index_name="my-index",
    dimension=768,
    region_name="us-east-1",
    aws_access_key_id=Secret.from_env_var("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=Secret.from_env_var("AWS_SECRET_ACCESS_KEY"),
)
```

## Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vector_bucket_name` | `str` | *required* | Name of the S3 vector bucket |
| `index_name` | `str` | *required* | Name of the vector index within the bucket |
| `dimension` | `int` | *required* | Dimensionality of the embeddings (e.g. 768, 1536) |
| `distance_metric` | `str` | `"cosine"` | `"cosine"` or `"euclidean"` |
| `region_name` | `str \| None` | `None` | AWS region (uses default from env if not set) |
| `aws_access_key_id` | `Secret \| None` | `None` | AWS access key ID |
| `aws_secret_access_key` | `Secret \| None` | `None` | AWS secret access key |
| `aws_session_token` | `Secret \| None` | `None` | AWS session token (for temporary credentials) |
| `create_bucket_and_index` | `bool` | `True` | Auto-create bucket and index if they don't exist |
| `non_filterable_metadata_keys` | `list[str] \| None` | `None` | Additional metadata keys to mark as non-filterable |

## Known Limitations & Considerations

### No Native Document Count API
S3 Vectors does not provide a dedicated count endpoint. `count_documents()` paginates through all
vector keys, which can be slow for large indexes (millions of vectors).

### `filter_documents()` Is Expensive
S3 Vectors only supports metadata filtering **during vector similarity queries** — there is no
standalone "list documents matching filter" API. As a result, `filter_documents()` must:
1. List all vectors with their data and metadata (paginated)
2. Apply filters client-side in memory

**For filtered retrieval, always prefer `S3VectorsEmbeddingRetriever` with filters**, which uses
the native `query_vectors` API with server-side filtering.

### Embedding Required
Every document written to the store **must** have an embedding. Documents without embeddings
will be rejected. This is a fundamental constraint of S3 Vectors as a pure vector store.

### No Keyword / BM25 Retrieval
S3 Vectors only supports dense vector similarity search. There is no keyword or BM25 search
capability. If you need hybrid search, consider pairing this with Amazon OpenSearch.

### Vector Data Type
Only `float32` vectors are supported. Higher-precision values are automatically downcast.

### Metadata Size Limits
- **Total metadata per vector: 40 KB** (filterable + non-filterable combined)
- **Filterable metadata per vector: 2 KB** — user `meta` fields used in filters must fit in this budget
- **Non-filterable metadata keys per index: 10** — the integration reserves 4 internal keys
  (`_content`, `_blob_data`, `_blob_meta`, `_blob_mime_type`), leaving 6 for user-defined keys
- Keys are set at index creation and **cannot be changed later**

Large content (e.g. full document text) is stored as non-filterable metadata automatically.
If you store additional large metadata fields, declare them via `non_filterable_metadata_keys`.

### Strict API Limits
- `put_vectors`: up to 500 vectors per call (handled automatically by the integration)
- `get_vectors`: up to 100 keys per call (handled automatically)
- `delete_vectors`: up to 500 keys per call (handled automatically)
- **`query_vectors`: maximum 100 results per query** — this is the hard cap on `top_k`.
  If you need more than 100 results, you must implement pagination or use a different store.
- Combined PutVectors + DeleteVectors: up to 1,000 requests/second per index

### Distance Metrics and Scoring
Only `cosine` and `euclidean` are supported. The metric is set at index creation time and cannot
be changed afterward.

S3 Vectors returns raw **distances** (lower = more similar). The integration converts these to
Haystack-convention **scores** (higher = more similar):
- **Cosine:** `score = 1.0 - distance` (1.0 = identical, 0.0 = orthogonal)
- **Euclidean:** `score = -distance` (0.0 = identical, more negative = further)

### No Embeddings Returned from Queries
The `query_vectors` API does not support returning the stored vector data alongside results.
Documents retrieved via `S3VectorsEmbeddingRetriever` will have `embedding=None`. If you need
the embedding vectors, use `filter_documents()` or fetch them separately via the boto3 client.

### Eventual Consistency
Newly written vectors may not be immediately visible in query results. S3 Vectors provides
eventual consistency for write-then-read operations.

## Running Tests

```bash
cd integrations/amazon_s3_vectors

# Unit tests (no AWS credentials required)
hatch run test:unit

# Integration tests (requires AWS credentials and S3 Vectors access)
hatch run test:integration
```

## License

`amazon-s3-vectors-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
