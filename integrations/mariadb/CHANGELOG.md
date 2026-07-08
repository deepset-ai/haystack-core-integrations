# Changelog

## [unreleased]

### Added

- Add `MariaDBDocumentStore` with native VECTOR support (MariaDB 11.7+), MHNSW indexing, and full-text keyword search
- Add `MariaDBEmbeddingRetriever` for approximate nearest-neighbour vector search
- Add `MariaDBKeywordRetriever` for full-text search using `MATCH ... AGAINST`
- Support for Haystack metadata filtering via JSON path expressions
