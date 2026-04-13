---
layout: integration
name: Oracle
description: Use Oracle AI Vector Search as a Document Store with Haystack
authors:
  - name: Federico Kamelhar
    socials:
      github: fede-kamel
      linkedin: https://www.linkedin.com/in/fedekamelhar/
pypi: https://pypi.org/project/oracle-haystack/
repo: https://github.com/deepset-ai/haystack-core-integrations/tree/main/integrations/oracle
type: Document Store
report_issue: https://github.com/deepset-ai/haystack-core-integrations/issues
logo: /logos/oracle.png
version: Haystack 2.0
toc: true
---

An integration of [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/) with [Haystack](https://haystack.deepset.ai/).

This integration uses the native `VECTOR` data type available in Oracle Database 23ai and 26ai to store and retrieve document embeddings. It provides an `OracleDocumentStore` that implements the full Haystack `DocumentStore` protocol, plus an `OracleEmbeddingRetriever` component for building retrieval pipelines.

### Key Features

- **Native vector storage** using `VECTOR(dim, FLOAT32)` columns — no extensions or plugins required
- **HNSW indexing** via `CREATE VECTOR INDEX ... ORGANIZATION INMEMORY NEIGHBOR GRAPH` for fast approximate search
- **Metadata filtering** with full Haystack filter grammar translated to `JSON_VALUE` SQL expressions
- **Oracle Autonomous Database** support with wallet-based TLS connections (thin mode, no Instant Client needed)
- **Async support** for all public methods (`awrite_documents`, `afilter_documents`, `adelete_documents`, etc.)
- **Connection pooling** via `oracledb.create_pool` for production workloads

## Installation

```bash
pip install oracle-haystack
```

The only runtime dependency beyond Haystack itself is [python-oracledb](https://python-oracledb.readthedocs.io/) (thin mode — no Oracle Client libraries required).

## Usage

### Connecting to a Local Oracle Database

```python
from haystack.utils import Secret
from haystack_integrations.document_stores.oracle import OracleDocumentStore, OracleConnectionConfig

document_store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user="scott",
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn="localhost:1521/freepdb1",
    ),
    embedding_dim=1536,
)
```

### Connecting to Oracle Autonomous Database (Cloud)

For Oracle Autonomous Database on OCI, provide the wallet location for TLS authentication:

```python
document_store = OracleDocumentStore(
    connection_config=OracleConnectionConfig(
        user="admin",
        password=Secret.from_env_var("ORACLE_PASSWORD"),
        dsn="mydb_low",
        wallet_location="/path/to/wallet",
        wallet_password=Secret.from_env_var("ORACLE_WALLET_PASSWORD"),
    ),
    embedding_dim=1536,
    distance_metric="COSINE",
)
```

### Writing Documents

```python
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy

docs = [
    Document(content="Paris is the capital of France.", meta={"lang": "en", "topic": "geography"}),
    Document(content="Berlin is the capital of Germany.", meta={"lang": "en", "topic": "geography"}),
]

# Provide embeddings (e.g., from a Haystack embedder component)
for doc in docs:
    doc.embedding = [0.1, 0.2, ...]  # your embedding vector

document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
```

Three duplicate policies are supported:
- `NONE` (default) — raises `DuplicateDocumentError` if a document ID already exists
- `SKIP` — silently ignores duplicates
- `OVERWRITE` — updates existing documents

### Filtering Documents

The full Haystack filter grammar is supported, including logical operators (`AND`, `OR`, `NOT`), comparisons (`==`, `!=`, `>`, `>=`, `<`, `<=`), and set operators (`in`, `not in`):

```python
results = document_store.filter_documents(
    filters={
        "operator": "AND",
        "conditions": [
            {"field": "meta.lang", "operator": "==", "value": "en"},
            {"field": "meta.topic", "operator": "in", "value": ["geography", "history"]},
        ],
    }
)
```

### Creating an HNSW Index

For faster approximate nearest-neighbor search on large collections, create an HNSW index after writing documents:

```python
document_store.create_hnsw_index()
```

Index parameters can be tuned at construction time:

```python
document_store = OracleDocumentStore(
    connection_config=config,
    embedding_dim=1536,
    hnsw_neighbors=32,
    hnsw_ef_construction=200,
    hnsw_accuracy=95,
)
```

### Building a RAG Pipeline

Use `OracleEmbeddingRetriever` in a Haystack pipeline to perform vector similarity search:

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component(
    "retriever",
    OracleEmbeddingRetriever(document_store=document_store, top_k=5),
)
pipeline.connect("embedder.embedding", "retriever.query_embedding")

result = pipeline.run({"embedder": {"text": "What is the capital of France?"}})
documents = result["retriever"]["documents"]
```

The retriever supports runtime filter overrides that are AND-merged with any filters set at construction time:

```python
retriever = OracleEmbeddingRetriever(
    document_store=document_store,
    filters={"field": "meta.lang", "operator": "==", "value": "en"},
    top_k=10,
)

# At runtime, additional filters are AND-merged
result = retriever.run(
    query_embedding=[...],
    filters={"field": "meta.topic", "operator": "==", "value": "geography"},
)
```

## Supported Distance Metrics

| Metric | Description |
|---|---|
| `COSINE` (default) | Cosine distance — good general-purpose choice |
| `EUCLIDEAN` | L2 distance |
| `DOT` | Negative inner product |

## Requirements

- **Python** 3.10+
- **Oracle Database** 23ai or 26ai (including Autonomous Database)
- No Oracle Instant Client needed — uses `oracledb` thin mode by default

## License

`oracle-haystack` is distributed under the terms of the [Apache-2.0 license](https://spdx.org/licenses/Apache-2.0.html).
