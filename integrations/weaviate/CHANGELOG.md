# Changelog

## [integrations/weaviate-v7.4.0] - 2026-03-17

### 🐛 Bug Fixes

- Weaviate - stop ignoring _split_overlap meta field (#2966)


## [integrations/weaviate-v7.3.0] - 2026-03-11

### 🚀 Features

- Add missing async methods for `WeaviateDocumentStore` (#2929)


## [integrations/weaviate-v7.2.0] - 2026-03-06

### 🐛 Bug Fixes

- Remove unnecessary connection test and add `close`/`close_async` methods to `WeaviateDocumentStore` (#2891)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)
- Fix Weaviate tests to include grpc_config (#2871)


## [integrations/weaviate-v7.1.0] - 2026-01-29

### 🚀 Features

- Adding count with filtering operations to `WeaviateDocumentStore` (#2767)


## [integrations/weaviate-v7.0.0] - 2026-01-13

### 🐛 Bug Fixes

- `WeaviateDocumentStore `_to_document()` and `to_data_object()` should be static methods (#2669)

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Weaviate - drop Python 3.9 and use X|Y typing (#2733)


## [integrations/weaviate-v6.5.0] - 2026-01-08

### 🚀 Features

- Adding `delete_by_filter` and `update_by_filter` to `WeaviateDocumentStore` (#2656)


## [integrations/weaviate-v6.4.0] - 2025-12-16

### 🚀 Features

- Add `run_async` to all Weaviate retrievers (#2607)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 5) (#2528)

## [integrations/weaviate-v6.3.0] - 2025-10-03

### 🚀 Features

- Implement `delete_all_documents` for weaviate integration (#2354)


### 🌀 Miscellaneous

- Fix: Update version of weaviate in docker-compose file (#2347)

## [integrations/weaviate-v6.2.0] - 2025-09-18

### 🚀 Features

- Weaviate Hybrid Retrieval (#2276)

### 🧹 Chores

- Remove black (#1985)
- Standardize readmes - part 2 (#2205)


## [integrations/weaviate-v6.1.0] - 2025-06-20

### 🐛 Bug Fixes

- Weaviate - fix types + add py.typed (#1977)


### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)

## [integrations/weaviate-v6.0.0] - 2025-03-11


### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Weaviate - pin haystack and remove dataframe checks (#1521)

### 🌀 Miscellaneous

- Docs: Update document store descriptions for deepset Pipeline Builder (#1447)

## [integrations/weaviate-v5.0.0] - 2025-02-17

### 🧹 Chores

- Fix linting/isort (#1215)
- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)
- [**breaking**] Weaviate - remove dataframe support (#1406)


## [integrations/weaviate-v4.0.2] - 2024-11-13

### 🐛 Bug Fixes

- Dependency for weaviate document store (#1186)


## [integrations/weaviate-v4.0.1] - 2024-11-11

### 🌀 Miscellaneous

- Fix: Weaviate - skip `_split_overlap` meta field (#1173)

## [integrations/weaviate-v4.0.0] - 2024-10-18

### 🐛 Bug Fixes

- Compatibility with Weaviate 4.9.0 (#1143)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Weaviate - remove legacy filter support (#1070)
- Update changelog after removing legacy filters (#1083)
- Update ruff linting scripts and settings (#1105)


## [integrations/weaviate-v2.2.1] - 2024-09-07

### 🚀 Features

- Add filter_policy to weaviate integration (#824)

### 🐛 Bug Fixes

- Weaviate filter error (#811)
- Fix connection to Weaviate Cloud Service (#624)
- Pin weaviate-client (#1046)
- Weaviate - fix connection issues with some WCS URLs (#1058)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Handle connection to WCS and add tests
- Revert "Handle connection to WCS and add tests"

This reverts commit f48802b2ce612896fd06a13cf33dffd9f77a8859.
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Minor retriever pydoc fix (#884)
- Fix: `weaviate` - Fallback to default filter policy when deserializing retrievers without the init parameter  (#903)
- Fix: Weaviate: Use collections.list_all instead of collections._get_all (#921)
- Chore: weaviate - ruff update, don't ruff tests (#986)

## [integrations/weaviate-v2.1.0] - 2024-06-10

### 🚀 Features

- Defer the database connection to when it's needed (#802)

### 🐛 Bug Fixes

- Weaviate schema class name conversion which preserves PascalCase (#707)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)

## [integrations/weaviate-v2.0.0] - 2024-03-25

### 📚 Documentation

- Disable-class-def (#556)
- Fix docstrings (#586)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Migrate from weaviate python client v3  to v4 (#463)
- Refactor tests (#574)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)
- Fix Weaviate retrievers returning no score (#621)
- Update Weaviate docstrings (#622)

## [integrations/weaviate-v1.0.2] - 2024-02-27

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Weaviate: fix auth tests (#488)

### 📚 Documentation

- Update category slug (#442)

### 🌀 Miscellaneous

- Make retrievers return dicts (#491)

## [integrations/weaviate-v1.0.0] - 2024-02-15

### 🚀 Features

- Generate weaviate API docs (#351)

### 🌀 Miscellaneous

- WeaviateDocumentStore initialization and serialization (#187)
- Move package under haystack_integrations (#214)
- Add `collection_name` parameter and creation (#215)
- Support more collection settings when creating a new `WeaviateDocumentStore` (#260)
- Implement `count_document` for WeaviateDocumentStore (#267)
- Add methods to convert from Document to Weaviate data object and viceversa (#269)
- Add filter, write and delete documents in Weaviate (#270)
- Implement filtering for `WeaviateDocumentStore` (#278)
- Add `WeaviateBM25Retriever` (#410)
- Add `WeaviateEmbeddingRetriever` (#412)
- Update Weaviate docs configs (#414)
- Update WeaviateDocumentStore authentication to use new Secret class (#425)

## [integrations/weaviate-v0.0.0] - 2024-01-10

### 🌀 Miscellaneous

- Setup everything to start working on the Weaviate integration (#186)

<!-- generated by git-cliff -->
