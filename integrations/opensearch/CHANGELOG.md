# Changelog

## [integrations/opensearch-v6.5.1] - 2026-03-10

### 🐛 Bug Fixes

- `OpenSearchDocumentStore` AsyncAWSAuth method signature changed (#2938)

### 🧹 Chores

- Standardize author mentions (#2897)


## [integrations/opensearch-v6.5.0] - 2026-02-26

### 🐛 Bug Fixes

- OpenSearch-3.x SQL stopped supporting DSL-format SQL JSON (#2876)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)


## [integrations/opensearch-v6.4.0] - 2026-02-11

### 🚀 Features

- Support additional `search_kwargs` with `OpenSearchEmbeddingRetriever` (#2825)


## [integrations/opensearch-v6.3.0] - 2026-01-30

### 🚀 Features

- Add an `SQLRetriever` to `OpenSearchDocumentStore` (#2746)
- Adding a `MetadataRetriever` to `OpenSearchDocumentStore` (#2771)


## [integrations/opensearch-v6.2.0] - 2026-01-19

### 🚀 Features

- Updating `opensearch-py` dependency (#2763)


## [integrations/opensearch-v6.1.2] - 2026-01-16

### 🐛 Bug Fixes

- Docs: fixing docstring in `OpenSearchDocumentStore` (#2758)


## [integrations/opensearch-v6.1.1] - 2026-01-16

### 🐛 Bug Fixes

- Docs: fixing docstrings in `OpenSearchDocumentStore` (#2757)


## [integrations/opensearch-v6.1.0] - 2026-01-16

### 🚀 Features

- Adding  count with filtering operations to `OpenSearchDocumentStore` (#2653)


## [integrations/opensearch-v6.0.0] - 2026-01-13

### 🧪 Testing

- Add `pytest-xdist` to execute tests in parallel on CI (and/or locally) (#2672)

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Opensearch - drop Python 3.9 and use X|Y typing (#2744)


## [integrations/opensearch-v5.1.0] - 2026-01-08

### 🌀 Miscellaneous

- Feat: Add routing support for OpenSearchDocumentStore (#2624)

## [integrations/opensearch-v5.0.0] - 2026-01-07

### 🚀 Features

- [**breaking**] Expose refresh parameter in `OpenSearchDocumentStore` (#2623)


## [integrations/opensearch-v4.6.0] - 2026-01-05

### 🐛 Bug Fixes

- Fix failing tests for opensearch and ollama (#2500)
- OpenSearch async client initialization (#2645)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 5) (#2528)

## [integrations/opensearch-v4.5.0] - 2025-10-27

### 🚀 Features

- Add delete by filter and update by filer to OpenSearchDocumentStore (#2407)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)
- Fix docstrings to avoid errors in API reference generation (#2423)


## [integrations/opensearch-v4.4.0] - 2025-10-09

### 🚀 Features

- Allow `OpenSearch` embedders to query a different `DocumentStore` at runtime (#2361)


## [integrations/opensearch-v4.3.0] - 2025-10-07

### 🚀 Features

- Adding the operation `delete_all_documents` to the `OpenSearchDocumentStore` (#2321)

### 🧹 Chores

- Fix linting for ruff 0.12.0 (#1969)
- Remove black (#1985)
- Standardize readmes - part 2 (#2205)
- Fix linting in tests for opensearch (#2259)


## [integrations/opensearch-v4.2.0] - 2025-06-12

### 🐛 Bug Fixes

- Fix passing filters and topks to OpenSearchHybridRetriever at runtime (#1936)


## [integrations/opensearch-v4.1.0] - 2025-06-11

### 🐛 Bug Fixes

- Fix Opensearch types + add py.typed (#1925)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/opensearch-v4.0.2] - 2025-05-30

### 🚜 Refactor

- OpenSearchHybridRetriever use `deserialize_chatgenerator_inplace` (#1870)

### 🌀 Miscellaneous

- Docs: add OpenSearchHybridRetriever to API Reference (#1868)

## [integrations/opensearch-v4.0.1] - 2025-05-28

### 🚀 Features

- Adding an `HybridRetriever` as a `Supercomponent` having `OpenSearch` as the document store (#1701)

### 🚜 Refactor

- Use `component_to_dict` in OpenSearchHybridRetriever (#1866)

### 📚 Documentation

- Add usage example to OpenSearchDocumentStore docstring (#1690)

### 🧪 Testing

- OpenSearch - reorganize test suite (#1563)

## [integrations/opensearch-v4.0.0] - 2025-03-26


### 🌀 Miscellaneous

- Feat!: OpenSearch - apply `return_embedding` to `filter_documents` (#1562)

## [integrations/opensearch-v3.1.1] - 2025-03-21

### 🐛 Bug Fixes

- OpenSearch custom_query use without filters (#1554)


### ⚙️ CI

- Review testing workflows (#1541)

## [integrations/opensearch-v3.1.0] - 2025-03-12

### 🚀 Features

- AWS IAM Auth support for OpenSearch async (#1527)

### 🐛 Bug Fixes

- OpenSearch - call _ensure_index_exists only at initialization (#1522)


## [integrations/opensearch-v3.0.0] - 2025-03-11

### 🐛 Bug Fixes

- OpenSearchDocumentStore depends on async opensearch-py (#1438)

### 📚 Documentation

- Add docstore description to docstring (#1446)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Chore: OpenSearch - pin haystack and remove dataframe checks (#1513)

## [integrations/opensearch-v2.1.0] - 2025-02-18

### 🚀 Features

- OpenSearch - async support (#1414)


## [integrations/opensearch-v2.0.0] - 2025-02-14

### 🚀 Features

- Add Secret handling in OpenSearchDocumentStore (#1288)

### 🧹 Chores

- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)
- [**breaking**] OpenSearch - remove dataframe support (#1378)

### 🌀 Miscellaneous

- Chore: OpenSearch - manually fix changelog (#1299)

## [integrations/opensearch-v1.2.0] - 2024-12-12

### 🧹 Chores

- Update docstring and type of fuzziness (#1243)


## [integrations/opensearch-v1.1.0] - 2024-10-29

### 🚀 Features

- Efficient knn filtering support for OpenSearch (#1134)

### 📚 Documentation

- Update opensearch retriever docstrings (#1035)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- OpenSearch - remove legacy filter support (#1067)
- Update changelog after removing legacy filters (#1083)
- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Docs: Update OpenSearchEmbeddingRetriever docstrings (#947)
- Update BM25 docstrings (#945)
- Chore: opensearch - ruff update, don't ruff tests (#988)

## [integrations/opensearch-v0.9.0] - 2024-08-01

### 🚀 Features

- Support aws authentication with OpenSearchDocumentStore (#920)


## [integrations/opensearch-v0.8.1] - 2024-07-15

### 🚀 Features

- Add raise_on_failure param to OpenSearch retrievers (#852)
- Add filter_policy to opensearch integration (#822)

### 🐛 Bug Fixes

- `OpenSearch` - Fallback to default filter policy when deserializing retrievers without the init parameter (#895)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Chore: Minor retriever pydoc fix (#884)

## [integrations/opensearch-v0.7.1] - 2024-06-27

### 🐛 Bug Fixes

- Serialization for custom_query in OpenSearch retrievers (#851)
- Support legacy filters with OpenSearchDocumentStore (#850)


## [integrations/opensearch-v0.7.0] - 2024-06-25

### 🚀 Features

- Defer the database connection to when it's needed (#753)
- Improve `OpenSearchDocumentStore.__init__` arguments (#739)
- Return_embeddings flag for opensearch (#784)
- Add create_index option to OpenSearchDocumentStore (#840)
- Add custom_query param to OpenSearch retrievers (#841)

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)
- Fixing opensearch docstrings (#521)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🌀 Miscellaneous

- Generate API docs (#324)
- Make tests show coverage (#566)
- Refactor tests (#574)
- Fix opensearch errors bulk write (#594)
- Remove references to Python 3.7 (#601)
- [Elasticsearch] fix: Filters not working with metadata that contain a space or capitalization (#639)
- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)

## [integrations/opensearch-v0.2.0] - 2024-01-17

### 🐛 Bug Fixes

- Fix links in docstrings (#188)

### 🚜 Refactor

- Use `hatch_vcs` to manage integrations versioning (#103)

### 🌀 Miscellaneous

- Fix opensearch test badge (#97)
- Move package under haystack_integrations/* (#212)

## [integrations/opensearch-v0.1.1] - 2023-12-05

### 🐛 Bug Fixes

- Document Stores: fix protocol import (#77)

## [integrations/opensearch-v0.1.0] - 2023-12-04

### 🐛 Bug Fixes

- Fix license headers

### 🌀 Miscellaneous

- Remove Document Store decorator (#76)

## [integrations/opensearch-v0.0.2] - 2023-11-30

### 🚀 Features

- Extend OpenSearch params support (#70)

### 🌀 Miscellaneous

- Bump OpenSearch integration version to 0.0.2 (#71)

## [integrations/opensearch-v0.0.1] - 2023-11-30

### 🚀 Features

- [OpenSearch] add document store, BM25Retriever and EmbeddingRetriever (#68)

<!-- generated by git-cliff -->
