# oracle-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/oracle-haystack.svg)](https://pypi.org/project/oracle-haystack)

Haystack DocumentStore backed by [Oracle AI Vector Search](https://www.oracle.com/database/ai-vector-search/), available in Oracle Database 23ai and later.

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/oracle/CHANGELOG.md)

---

## Installation

```bash
pip install oracle-haystack
```

Requires Python 3.10+ and Oracle Database 23ai (or later). No Oracle Instant Client is needed for direct TCP connections (thin mode).

## Usage

```python
from haystack.utils import Secret
from haystack_integrations.document_stores.oracle import OracleConnectionConfig, OracleDocumentStore
from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever

# Configure the connection
config = OracleConnectionConfig(
    user="haystack",
    password=Secret.from_env_var("ORACLE_PASSWORD"),
    dsn="localhost:1521/freepdb1",
)

# Create the document store
store = OracleDocumentStore(
    connection_config=config,
    table_name="my_documents",
    embedding_dim=768,
    distance_metric="COSINE",
    create_table_if_not_exists=True,
)

# Write documents
from haystack.dataclasses import Document
store.write_documents([
    Document(content="Oracle 23ai supports native vector search."),
])

# Retrieve by embedding
retriever = OracleEmbeddingRetriever(document_store=store, top_k=5)
results = retriever.run(query_embedding=[0.1] * 768)
print(results["documents"])
```

### Connecting to Oracle Autonomous Database (ADB-S / wallet)

```python
config = OracleConnectionConfig(
    user="admin",
    password=Secret.from_env_var("ORACLE_PASSWORD"),
    dsn="mydb_high",
    wallet_location="/path/to/wallet",
    wallet_password=Secret.from_env_var("WALLET_PASSWORD"),
)
```

### Optional HNSW index

Pass `create_index=True` when constructing the store to build an HNSW vector index, which dramatically speeds up approximate nearest-neighbour search on large collections:

```python
store = OracleDocumentStore(
    connection_config=config,
    table_name="my_documents",
    embedding_dim=768,
    create_index=True,
    hnsw_neighbors=32,
    hnsw_ef_construction=200,
    hnsw_accuracy=95,
)
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

### Running tests

#### Unit tests

```bash
PYTHONPATH=src hatch run test:unit -vvv
```

#### Integration tests against a live Oracle instance

Set `ORACLE_USER`, `ORACLE_PASSWORD`, and `ORACLE_DSN` environment variables to point at your Oracle 23ai instance, then:

```bash
PYTHONPATH=src hatch run test:integration -vvv
```

#### Integration tests via Docker (local Oracle 23ai Free)

A `docker-compose.yml` is provided that runs [`gvenzl/oracle-free:23-slim`](https://hub.docker.com/r/gvenzl/oracle-free) (Oracle Database 23ai Free edition).

```bash
docker compose up -d --wait
```

`--wait` blocks until the Oracle healthcheck passes (the first boot takes 2–4 minutes while Oracle initialises its data files).

Run the full integration test suite:

```bash
PYTHONPATH=src hatch run test:integration -vvv
```