# Changelog

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

- Update document store descriptions (#1447)

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

- Weaviate - skip writing _split_overlap meta field (#1173)

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
- Install pytest-rerunfailures; change test-cov script (#845)
- Minor retriever pydoc fix (#884)
- Add defensive check for filter_policy deserialization (#903)
- Use collections.list_all instead of collections._get_all (#921)
- Ruff update, don't ruff tests (#986)

## [integrations/weaviate-v2.1.0] - 2024-06-10

### 🚀 Features

- Defer the database connection to when it's needed (#802)

### 🐛 Bug Fixes

- Weaviate schema class name conversion which preserves PascalCase (#707)

### 🌀 Miscellaneous

- Add license classifiers (#680)
- Change the pydoc renderer class (#718)

## [integrations/weaviate-v2.0.0] - 2024-03-25

### 📚 Documentation

- Disable-class-def (#556)
- Fix docstrings (#586)

### 🌀 Miscellaneous

- Make tests show coverage (#566)

* make tests show coverage

* rm duplicate coverage definition
- Migrate from weaviate python client v3  to v4 (#463)
- Refactor tests (#574)

* first refactorings

* separate unit tests in pgvector

* small change to weaviate

* fix format

* usefixtures when possible
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip SparseEmbedding (#606)
- Fix Weaviate retrievers returning no score (#621)

* Update testing Docker image

* Fix not score being returned for embedding and bm25 retrieval
- Update Weaviate docstrings (#622)

* Update Weaviate docstrings

* Fix linting

* Simplify count_documents docstring

Co-authored-by: Madeesh Kannan <shadeMe@users.noreply.github.com>

---------

Co-authored-by: Madeesh Kannan <shadeMe@users.noreply.github.com>

## [integrations/weaviate-v1.0.2] - 2024-02-27

### 🐛 Bug Fixes

- Fix order of API docs (#447)

This PR will also push the docs to Readme
- Fix weaviate auth tests (#488)

### 📚 Documentation

- Update category slug (#442)

### 🌀 Miscellaneous

- Make retrievers return dicts (#491)

## [integrations/weaviate-v1.0.0] - 2024-02-15

### 🚀 Features

- Generate weaviate API docs (#351)

### 🌀 Miscellaneous

- WeaviateDocumentStore initialization and serialization (#187)

* Update weaviate dependencies

* Implement WeaviateDocumentStore initialization and serialization

* Fix linting
- Move package under haystack_integrations (#214)

* Move package under haystack_integrations

* Fix linting

* Fix linting again?
- Add `collection_name` parameter and creation (#215)

* Add collection_name parameter

* Fix linting
- Support more collection settings when creating a new `WeaviateDocumentStore` (#260)

* Add docker-compose.yml

* Accept more collection settings when initializing WeaviateDocumentStore

* Linting
- Implement `count_document` for WeaviateDocumentStore (#267)

* Implement count_document for WeaviateDocumentStore

* Start container in test workflow

* Ditch Windows and Mac on Weaviate CI as Docker images are not provided
- Add methods to convert from Document to Weaviate data object and viceversa (#269)

* Add methods to convert from Document to Weaviate data object and viceversa

* Add tests
- Add filter, write and delete documents in Weaviate (#270)

* Add filter, write and delete documents in Weaviate

* Fix linting

* Fix typo
- Implement filtering for `WeaviateDocumentStore` (#278)

* Add filters logic

* Set null values as indexable by default so filters work as expected

* Save flattened Documents to properly support filters

* Add filters support in filter_documents

* Add filter_documents tests

* Handle inversion of NOT filters

* Update tests and skip one

* Add some documentation

* Fix linting

* Remove override decorator

* Replace datetime.fromisoformat with python-dateutil function

* Move field check when parsing comparisons

* Add utility function that returns filter that matches no documents
- Add WeaviateBM25Retriever (#410)
- Add WeaviateEmbeddingRetriever (#412)
- Update Weaviate docs (#414)
- Update WeaviateDocumentStore authentication to use new Secret class (#425)

* Update WeaviateDocumentStore authentication to use new Secret class

* Fix linting

* Update docs config

* Export auth classes

* Change expires_in to non secret

* Use enum for serialization types

* Freeze dataclasses

* Fix linting

* Fix failing tests

## [integrations/weaviate-v0.0.0] - 2024-01-10

### 🌀 Miscellaneous

- Scaffolding for Weaviate integration (#186)

<!-- generated by git-cliff -->
