# Cassandra Document Store for Haystack

[![PyPI version](https://img.shields.io/pypi/v/cassandra-haystack.svg)](https://pypi.org/project/cassandra-haystack/)
[![Python versions](https://img.shields.io/pypi/pyversions/cassandra-haystack.svg)](https://pypi.org/project/cassandra-haystack/)
[![License](https://img.shields.io/github/license/deepset-ai/haystack-core-integrations.svg)](LICENSE.txt)

A Haystack document store implementation using **Apache Cassandra** or **Astra DB** as the backend.

## Features

- 📄 Full CRUD operations for documents
- 🔍 Metadata filtering support
- 🧮 Vector embeddings storage
- ⚡ Prepared statements for performance
- 🔐 Secure credential handling with Haystack Secrets
- 📦 TTL support for automatic document expiration
- 🏗️ Compatible with Haystack 2.x pipelines

## Installation

```bash
pip install cassandra-haystack
```

Requires Python 3.11+ and a running Cassandra instance (or Astra DB).

## Quick Start

```python
from haystack import Document
from haystack_integrations.document_stores.cassandra import CassandraDocumentStore

# Initialize document store
document_store = CassandraDocumentStore(
    contact_points=["127.0.0.1"],  # Cassandra nodes
    port=9042,
    keyspace="haystack",
    table="documents",
)

# Write documents
docs = [
    Document(content="Hello Haystack!", meta={"source": "greeting"}),
    Document(content="Cassandra is great", meta={"source": "tech", "rating": 5}),
]
document_store.write_documents(docs)

# Filter documents
results = document_store.filter_documents(filters={"source": "tech"})
for doc in results:
    print(f"{doc.content} - {doc.meta}")

# Count documents
print(f"Total: {document_store.count_documents()}")

# Clean up
document_store.close()
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `contact_points` | `List[str]` | `["127.0.0.1"]` | Cassandra node addresses |
| `port` | `int` | `9042` | Cassandra port |
| `username` | `str/Secret` | `None` | Username for auth |
| `password` | `str/Secret` | `None` | Password for auth |
| `keyspace` | `str` | `"haystack"` | Keyspace name |
| `table` | `str` | `"documents"` | Table name |
| `ttl_seconds` | `int` | `None` | Document TTL (optional) |
| `session` | `Session` | `None` | Pre-existing session |

## Astra DB Support

For **DataStax Astra DB**, use the secure connect bundle:

```python
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

cloud_config = {"secure_connect_bundle": "/path/to/secure-connect.zip"}
auth_provider = PlainTextAuthProvider("token", "AstraCS:...")
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

document_store = CassandraDocumentStore(
    session=session,
    keyspace="your_keyspace",
    table="documents"
)
```

## Pipeline Usage

```python
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.cassandra import CassandraDocumentStore

document_store = CassandraDocumentStore(keyspace="haystack", table="pipeline_docs")
writer = DocumentWriter(document_store=document_store)

pipeline = Pipeline()
pipeline.add_component("writer", writer)

# Run pipeline
pipeline.run({"writer": {"documents": [Document(content="From pipeline")]}})
```

## Schema

The store creates a table with the following schema:

```sql
CREATE TABLE documents (
    id text PRIMARY KEY,
    content text,
    metadata text,      -- JSON-encoded metadata
    embedding blob,     -- Vector embedding (optional)
    score float         -- Retrieval score (optional)
)
```

## Development

```bash
# Install dev dependencies
pip install hatch

# Run unit tests
hatch run unit:test

# Run integration tests (requires Cassandra)
hatch run integration:test

# Run all tests
hatch run all:test

# Type checking
hatch run types:check

# Formatting
hatch run fmt:fix
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a PR

See [CONTRIBUTING.md](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md) for detailed guidelines.

## License

Apache 2.0 - See [LICENSE.txt](LICENSE.txt) for details.