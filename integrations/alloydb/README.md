# AlloyDB Haystack Integration

[![PyPI - Version](https://img.shields.io/pypi/v/alloydb-haystack.svg)](https://pypi.org/project/alloydb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/alloydb-haystack.svg)](https://pypi.org/project/alloydb-haystack)

---

[AlloyDB](https://cloud.google.com/alloydb) is a fully managed, PostgreSQL-compatible database service on Google Cloud,
optimised for demanding transactional and analytical workloads.

This package provides a Haystack `DocumentStore` backed by AlloyDB with the
[pgvector extension](https://cloud.google.com/alloydb/docs/ai/work-with-embeddings), enabling both dense vector
similarity search and full-text keyword search.

Connections are established through the
[AlloyDB Python Connector](https://github.com/GoogleCloudPlatform/alloydb-python-connector),
which handles IAM-based authentication and TLS encryption without requiring manual firewall rules or IP allowlisting.

## Installation

```console
pip install alloydb-haystack
```

## Usage

```python
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore
from haystack_integrations.components.retrievers.alloydb import (
    AlloyDBEmbeddingRetriever,
    AlloyDBKeywordRetriever,
)
```

### Environment Variables

| Variable | Description |
|---|---|
| `ALLOYDB_INSTANCE_URI` | AlloyDB instance URI: `projects/P/locations/R/clusters/C/instances/I` |
| `ALLOYDB_USER` | Database user (or IAM principal for IAM auth) |
| `ALLOYDB_PASSWORD` | Database password (not required when `enable_iam_auth=True`) |

### Basic Example

```python
import os
from haystack import Document
from haystack_integrations.document_stores.alloydb import AlloyDBDocumentStore

# Requires ALLOYDB_INSTANCE_URI, ALLOYDB_USER, and ALLOYDB_PASSWORD env vars
store = AlloyDBDocumentStore(
    db="my-database",
    embedding_dimension=768,
    recreate_table=True,
)

store.write_documents([
    Document(content="Paris is the capital of France", embedding=[0.1] * 768),
    Document(content="Berlin is the capital of Germany", embedding=[0.2] * 768),
])

print(store.count_documents())  # 2
```

### IAM Authentication

When using a service account for database access:

```python
store = AlloyDBDocumentStore(
    db="my-database",
    user=Secret.from_env_var("ALLOYDB_IAM_USER"),  # e.g. "my-sa@my-project.iam"
    enable_iam_auth=True,
    embedding_dimension=768,
)
```

### Vector Similarity Search

```python
from haystack_integrations.components.retrievers.alloydb import AlloyDBEmbeddingRetriever

retriever = AlloyDBEmbeddingRetriever(document_store=store, top_k=5)
result = retriever.run(query_embedding=[0.1] * 768)
print(result["documents"])
```

### Keyword Search

```python
from haystack_integrations.components.retrievers.alloydb import AlloyDBKeywordRetriever

retriever = AlloyDBKeywordRetriever(document_store=store, top_k=5)
result = retriever.run(query="capital France")
print(result["documents"])
```

### HNSW Index

For large datasets, the HNSW index provides approximate nearest-neighbour search with significantly
better query throughput:

```python
store = AlloyDBDocumentStore(
    db="my-database",
    embedding_dimension=768,
    search_strategy="hnsw",
    hnsw_index_creation_kwargs={"m": 16, "ef_construction": 64},
    hnsw_ef_search=40,
)
```

## Integration Tests

Integration tests require a running AlloyDB instance. Set the following environment variables
before running:

```console
export ALLOYDB_INSTANCE_URI="projects/MY_PROJECT/locations/MY_REGION/clusters/MY_CLUSTER/instances/MY_INSTANCE"
export ALLOYDB_USER="my-db-user"
export ALLOYDB_PASSWORD="my-db-password"
```

Then run:

```console
cd integrations/alloydb
hatch run test:integration
```

## License

`alloydb-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
