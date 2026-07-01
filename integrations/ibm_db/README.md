# ibm-db-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/ibm-db-haystack.svg)](https://pypi.org/project/ibm-db-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ibm-db-haystack.svg)](https://pypi.org/project/ibm-db-haystack)

IBM DB2 integration for Haystack, providing document storage and retrieval capabilities using IBM DB2 database with vector search support.

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/ibm_db/CHANGELOG.md)

---

## Installation

```bash
pip install ibm-db-haystack
```

## Features

- **Document Store**: Store and manage documents with embeddings in IBM DB2
- **Vector Search**: Perform similarity search using DB2's vector capabilities
- **Metadata Filtering**: Filter documents based on metadata fields
- **Async Support**: Asynchronous operations for improved performance

## Usage

### Basic Setup

```python
from haystack_integrations.document_stores.ibm_db import Db2DocumentStore, Db2ConnectionConfig
from haystack_integrations.components.retrievers.ibm_db import Db2EmbeddingRetriever
from haystack import Document

# Configure DB2 connection
config = Db2ConnectionConfig(
    database="mydb",
    hostname="localhost",
    port=50000,
    username="db2user",
    password="password",
    protocol="TCPIP"
)

# Create document store
document_store = Db2DocumentStore(
    connection_config=config,
    table_name="documents",
    embedding_dim=768,
    distance_metric="COSINE"
)

# Write documents
documents = [
    Document(content="Python is great for data science", embedding=[0.1, 0.2, ...]),
    Document(content="Machine learning with Python", embedding=[0.3, 0.4, ...])
]
document_store.write_documents(documents)

# Create retriever
retriever = Db2EmbeddingRetriever(document_store=document_store, top_k=5)

# Search for similar documents
query_embedding = [0.15, 0.25, ...]  # Your query embedding
results = retriever.run(query_embedding=query_embedding)
```

### Vector Search

The `Db2EmbeddingRetriever` performs semantic similarity search based on embeddings:

```python
from haystack_integrations.components.retrievers.ibm_db import Db2EmbeddingRetriever

# Create embedding retriever
retriever = Db2EmbeddingRetriever(
    document_store=document_store,
    top_k=10,
    filters={"field": "meta.category", "operator": "==", "value": "science"}
)

# Retrieve similar documents
results = retriever.run(query_embedding=query_embedding)
documents = results["documents"]
```

### Metadata Filtering

Filter documents based on metadata:

```python
# Filter by metadata during retrieval
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "technology"},
        {"field": "meta.year", "operator": ">=", "value": 2020}
    ]
}

results = retriever.run(query_embedding=query_embedding, filters=filters)
```

## Requirements

- IBM DB2 database (version 11.5 or later recommended)
- Python 3.10+
- ibm-db >= 3.2.8

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

## License

`ibm-db-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
