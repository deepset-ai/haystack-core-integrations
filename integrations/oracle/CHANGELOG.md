# Changelog

## [integrations/oracle-v0.1.0] - 2026-04-15

### 🚀 Features

- Initial release of `oracle-haystack` — Haystack DocumentStore backed by Oracle AI Vector Search (Oracle Database 23ai+)
- `OracleDocumentStore`: write, filter, count, and delete documents; embedding retrieval with COSINE/DOT_PRODUCT/EUCLIDEAN distance metrics
- `OracleEmbeddingRetriever`: synchronous and asynchronous embedding-based retrieval with `FilterPolicy` support
- `OracleConnectionConfig`: thin-mode (direct TCP) and thick-mode (wallet / ADB-S) connection configuration
- Optional HNSW vector index creation (`create_index=True`) with configurable neighbours, ef-construction, accuracy, and parallelism
- Connection pool managed via `oracledb.create_pool`; async operations offloaded with `asyncio.run_in_executor`

### 🧪 Testing

- Unit tests for `OracleDocumentStore`, `OracleEmbeddingRetriever`, and `FilterTranslator`
- Integration test suite using a live Oracle 23ai Free instance (Docker-compose provided)
- CI workflow (`.github/workflows/oracle.yml`) covering lint, unit tests, integration tests, and nightly Haystack-main compatibility checks
