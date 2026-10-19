# azure-documentdb-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/azure-documentdb-haystack.svg)](https://pypi.org/project/azure-documentdb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/azure-documentdb-haystack.svg)](https://pypi.org/project/azure-documentdb-haystack)

Haystack document store and retrievers for [Azure DocumentDB](https://learn.microsoft.com/azure/documentdb/overview).

## Installation

```bash
pip install azure-documentdb-haystack
```

## Authentication

Microsoft Entra ID is the default and recommended authentication method. `DefaultAzureCredential` supports local Azure
CLI credentials, workload identity, and managed identity without changing application code. Assign the identity an
appropriate Azure DocumentDB data-plane role, then set the cluster name:

```bash
AZURE_DOCUMENTDB_CLUSTER_NAME=my-cluster
```

For local development and integration testing only, a connection string can be supplied through
`AZURE_DOCUMENTDB_CONNECTION_STRING`. The integration emits a warning whenever this fallback is used.

## Usage

```python
from haystack_integrations.document_stores.azure_documentdb import AzureDocumentDBDocumentStore

store = AzureDocumentDBDocumentStore(
    database_name="haystack",
    collection_name="documents",
)

# Create once. The collection must exist before initializing the store.
store.create_vector_index(dimensions=1536, kind="vector-hnsw", similarity="COS", m=16, efConstruction=64)
```

Use `AzureDocumentDBEmbeddingRetriever` for `cosmosSearch` vector queries. `AzureDocumentDBFullTextRetriever` supports
Azure DocumentDB BM25 full-text search, which is currently a gated preview and must be enabled on the cluster.

- [Azure DocumentDB vector search](https://learn.microsoft.com/azure/documentdb/vector-search)
- [Azure DocumentDB full-text search](https://learn.microsoft.com/azure/documentdb/full-text-search-overview)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/azure_documentdb/CHANGELOG.md)

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
