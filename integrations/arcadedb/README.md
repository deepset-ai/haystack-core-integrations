# arcadedb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/arcadedb-haystack.svg)](https://pypi.org/project/arcadedb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/arcadedb-haystack.svg)](https://pypi.org/project/arcadedb-haystack)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.txt)

**[ArcadeDB](https://arcadedb.com)** integration for [Haystack](https://haystack.deepset.ai/) 2.x.

ArcadeDB is an open-source multi-model database that combines document storage, HNSW vector search, and SQL metadata filtering in a single engine. This integration provides a `DocumentStore` and `EmbeddingRetriever` that connect to ArcadeDB via its HTTP/JSON API using only the `requests` library -- no special drivers needed.

## Installation

```bash
pip install arcadedb-haystack
```

## Usage

Start ArcadeDB:

```bash
docker run -d -p 2480:2480 \
    -e JAVA_OPTS="-Darcadedb.server.rootPassword=arcadedb" \
    arcadedata/arcadedb:latest

export ARCADEDB_USERNAME=root
export ARCADEDB_PASSWORD=arcadedb
```

### Document Store

```python
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

store = ArcadeDBDocumentStore(
    database="myproject",
    embedding_dimension=768,
)

docs = [
    Document(
        content="ArcadeDB supports graphs, documents, and vectors.",
        embedding=[0.1] * 768,
        meta={"source": "docs", "category": "database"},
    )
]
store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
store.filter_documents(
    filters={"field": "meta.category", "operator": "==", "value": "database"}
)
```

### Pipeline with Embedding Retriever

```python
from haystack import Pipeline
from haystack_integrations.components.retrievers.arcadedb import ArcadeDBEmbeddingRetriever
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

store = ArcadeDBDocumentStore(database="myproject", embedding_dimension=768)
pipeline = Pipeline()
pipeline.add_component("retriever", ArcadeDBEmbeddingRetriever(document_store=store, top_k=10))

result = pipeline.run({"retriever": {"query_embedding": [0.1] * 768}})
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `url` | `http://localhost:2480` | ArcadeDB HTTP endpoint |
| `database` | `haystack` | Database name |
| `username` | env `ARCADEDB_USERNAME` | HTTP Basic Auth username |
| `password` | env `ARCADEDB_PASSWORD` | HTTP Basic Auth password |
| `type_name` | `Document` | Vertex type name |
| `embedding_dimension` | `768` | Vector dimension for HNSW index |
| `similarity_function` | `cosine` | `cosine`, `euclidean`, or `dot` |
| `recreate_type` | `False` | Drop and recreate type on init |
| `create_database` | `True` | Create database if it doesn't exist |

## License

`arcadedb-haystack` is distributed under the terms of the [Apache-2.0](LICENSE.txt) license.
