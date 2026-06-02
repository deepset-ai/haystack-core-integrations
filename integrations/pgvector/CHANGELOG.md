# Changelog

## [integrations/pgvector-v6.3.1] - 2026-06-01

### 🐛 Bug Fixes

- Replace in place dataclass mutations for pgvector #11087 (#3142)
- *(pgvector)* Order retrieval by distance operator (#3370)

### 🚜 Refactor

- *(pgvector)* Use async DocumentStore mixin tests (#3094)

### 🧪 Testing

- `PGVectorDocumentStore`e use Mixin tests (#3003)
- Test compatible integrations with python 3.14; update pyproject (#3001)
- Track test coverage for all integrations (#3065)
- Add tests for pgvector document store (#3156)

### 🧹 Chores

- Add ANN ruff ruleset to optimum, paddleocr, pgvector, pinecone, pyversity, qdrant, ragas, snowflake (#2992)
- Enforce ruff docstring rules in integrations 31-40 (openrouter, opensearch, optimum, paddleocr, pgvector, pinecone, pyversity, qdrant, ragas, snowflake) (#3011)


## [integrations/pgvector-v6.3.0] - 2026-03-17

### 🐛 Bug Fixes

- PgVectorDocumentStore `_treat_meta_field` and comparison functions now return Composed - string escaping done by `psycopg` (#2964)


## [integrations/pgvector-v6.2.0] - 2026-03-02

### 🐛 Bug Fixes

- Add metadata field-name validation using regex in `PGVectorDocumentStore` filters to prevent SQL injection vectors. (#2881)
- Remove NUL bytes when converting from haystack to pg documents (#2892)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)


## [integrations/pgvector-v6.1.0] - 2026-01-28

### 🚀 Features

- Adding count, filtering and metadata related operations to `PGVectorDocumentStore` (#2768)


## [integrations/pgvector-v6.0.0] - 2026-01-12

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Pgvector - drop Python 3.9 and use X|Y typing (#2722)

### 🌀 Miscellaneous

- Fix: Fix doc links (#2661)

## [integrations/pgvector-v5.5.0] - 2026-01-02

### 🚀 Features

- Adding `update_by_filter` and `delete_by_filter` to PgVector document store (#2647)


## [integrations/pgvector-v5.4.0] - 2025-12-05

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Pgvector - document expected connection string and raise informative error (#2583)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 5) (#2528)

## [integrations/pgvector-v5.3.0] - 2025-10-17

### 🚀 Features

- `PgvectorDocumentStore` now supports `delete_all_documents` and  `delete_all_documents_async` (#2394)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/pgvector-v5.2.1] - 2025-07-03

### 🐛 Bug Fixes

- Pgvector - ensure DB is initialized when calling `delete_table` (#2055)


### 🧹 Chores

- Remove black (#1985)
- Pgvector - update docker image in examples (#1995)


## [integrations/pgvector-v3.4.1] - 2025-06-20

### 🐛 Bug Fixes

- Pgvector - do not pass null `meta` to `Document.from_dict` (#1980)


## [integrations/pgvector-v3.4.0] - 2025-06-10

### 🐛 Bug Fixes

- Fix pgvector types + add py.typed (#1914)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/pgvector-v3.3.1] - 2025-05-28

### 🌀 Miscellaneous

- Add pins for Pgvector (#1849)

## [integrations/pgvector-v3.3.0] - 2025-05-06

### 🚀 Features

- Pgvector - make error messages more informative (#1684)


## [integrations/pgvector-v3.2.0] - 2025-04-11

### 🚀 Features

- Add halfvec support for storing high-dimensional embeddings in half-precision (#1607)


## [integrations/pgvector-v3.1.0] - 2025-03-20

### 🚀 Features

- Pgvector - async support (+ refactoring) (#1547)


### ⚙️ CI

- Review testing workflows (#1541)

## [integrations/pgvector-v3.0.1] - 2025-03-12


### 🌀 Miscellaneous

- Fix: pgvector - improve `_create_table_if_not_exists` to be used without admin rights (#1490)

## [integrations/pgvector-v3.0.0] - 2025-03-11


### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)
- Pgvector - pin haystack and remove dataframe checks (#1518)

## [integrations/pgvector-v2.0.0] - 2025-02-13

### 🧹 Chores

- Pgvector - remove support for dataframe (#1370)


## [integrations/pgvector-v1.3.0] - 2025-02-03

### 🚀 Features

- Pgvector - add like and not like filters (#1341)

### 🧹 Chores

- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)


## [integrations/pgvector-v1.2.1] - 2025-01-10

### 🐛 Bug Fixes

- PgvectorDocumentStore - use appropriate schema name if dropping index (#1277)


## [integrations/pgvector-v1.2.0] - 2024-11-22

### 🚀 Features

- Add `create_extension` parameter to control vector extension creation (#1213)


## [integrations/pgvector-v1.1.0] - 2024-11-21

### 🚀 Features

- Add filter_policy to pgvector integration (#820)
- Add schema support to pgvector document store. (#1095)
- Pgvector - recreate the connection if it is no longer valid (#1202)

### 🐛 Bug Fixes

- `PgVector` - Fallback to default filter policy when deserializing retrievers without the init parameter (#900)

### 📚 Documentation

- Explain different connection string formats in the docstring (#1132)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)
- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- PgVector - remove legacy filter support (#1068)
- Update changelog after removing legacy filters (#1083)
- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Minor retriever pydoc fix (#884)
- Chore: Update pgvector test for the new `apply_filter_policy` usage (#970)
- Chore: pgvector ruff update, don't ruff tests (#984)

## [integrations/pgvector-v0.4.0] - 2024-06-20

### 🚀 Features

- Defer the database connection to when it's needed (#773)
- Add customizable index names for pgvector (#818)

### 🌀 Miscellaneous

- Docs: add missing api references (#728)
- [deepset-ai/haystack-core-integrations#727] (#738)

## [integrations/pgvector-v0.2.0] - 2024-05-08

### 🚀 Features

- `MongoDBAtlasEmbeddingRetriever` (#427)
- Implement keyword retrieval for pgvector integration (#644)

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Pgvector - review docstrings and API reference (#502)
- Refactor tests (#574)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)
- Chore: add license classifiers (#680)
- Type hints in pgvector document store updated for 3.8 compability (#704)
- Chore: change the pydoc renderer class (#718)

## [integrations/pgvector-v0.1.0] - 2024-02-14

### 🐛 Bug Fixes

- Pgvector: fix linting (#328)

### 🌀 Miscellaneous

- Pgvector Document Store - minimal implementation (#239)
- Pgvector - filters (#257)
- Pgvector - embedding retrieval (#298)
- Pgvector - Embedding Retriever (#320)
- Pgvector: generate API docs (#325)
- Pgvector: add an example (#334)
- Adopt `Secret` to pgvector (#402)

<!-- generated by git-cliff -->
