# Changelog

## [integrations/pinecone-v6.2.0] - 2026-05-08

### 🐛 Bug Fixes

- Replace in-place dataclass mutations in document stores (#3114)

### 🚜 Refactor

- *(pinecone)* Make get_metadata_field_min_max more consistent; use async DocumentStore mixin tests (#3231)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)
- Test compatible integrations with python 3.14; update pyproject (#3001)
- `PineconeDocumentStore` use Mixin tests (#3020)
- Track test coverage for all integrations (#3065)
- Pinecone - add unit tests (#3218)
- Pinecone - remove namespaces when tests ends (#3248)

### 🧹 Chores

- Exposing `PineconeDocumentStore` `show_progress` bar parameter and disabling it for tests (#2799)
- Remove unused allow-direct-references (#2866)
- Add ANN ruff ruleset to optimum, paddleocr, pgvector, pinecone, pyversity, qdrant, ragas, snowflake (#2992)
- Enforce ruff docstring rules in integrations 31-40 (openrouter, opensearch, optimum, paddleocr, pgvector, pinecone, pyversity, qdrant, ragas, snowflake) (#3011)
- Pinecone - fix types for Pinecone 9 (#3277)


## [integrations/pinecone-v6.1.2] - 2026-01-28

### 🐛 Bug Fixes

- Docs: fixing docstring parsing error for `PineConeDocumentStore` (#2790)


## [integrations/pinecone-v6.1.1] - 2026-01-28

### 📚 Documentation

- Fixing docstring parsing error on `PineConeDocumentStore` (#2788)


## [integrations/pinecone-v6.1.0] - 2026-01-28

### 🚀 Features

- Add operations to PineConeDocumentStore (#2772)

### 🧪 Testing

- Pinecone - improve flaky tests (#2787)


## [integrations/pinecone-v6.0.0] - 2026-01-13

### 🧹 Chores

- [**breaking**] Pinecone - drop Python 3.9 and use X|Y typing (#2723)


## [integrations/pinecone-v5.5.0] - 2026-01-08

### 🚀 Features

- Adding `delete_by_filter` and update_by_filter to `PineconeDocumentStore` (#2655)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Fix Pinecone types and make sure that tests run (#2658)
- Make fmt command more forgiving (#2671)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 5) (#2528)
- Fix: Fix doc links (#2661)

## [integrations/pinecone-v5.4.0] - 2025-11-05

### 🚀 Features

- Add delete all documents to Pinecone DocumentStore (#2403)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### 🧪 Testing

- Pinecone - relax flaky test (#2360)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)
- Change pytest command (#2475)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/pinecone-v5.3.0] - 2025-07-30

### 🚀 Features

- Pinecone - add methods to close resources (#1972)


## [integrations/pinecone-v5.2.0] - 2025-06-26

### 🐛 Bug Fixes

- Fix Pinecone types + add py.typed (#1993)

### 🧪 Testing

- Pinecone - use uuid for namespace names (#1984)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)
- Remove black (#1985)

### 🌀 Miscellaneous

- Test: Pinecone - reenable tests for `comparison_not_equal` and `or_operator` (#1871)

## [integrations/pinecone-v5.1.1] - 2025-05-28


### 🌀 Miscellaneous

- Add pins for Pinecone (#1851)

## [integrations/pinecone-v5.1.0] - 2025-03-25

### 🚀 Features

- Pinecone -- async support (#1560)


### ⚙️ CI

- Review testing workflows (#1541)

## [integrations/pinecone-v5.0.0] - 2025-03-11


### 🧪 Testing

- Skip tests that require credentials on PRs from forks for some integrations (#1485)

### 🧹 Chores

- Use Haystack logging across integrations (#1484)

### 🌀 Miscellaneous

- Increase sleep time (#1478)
- Chore: Pinecone - pin haystack and remove dataframe checks (#1514)

## [integrations/pinecone-v4.0.0] - 2025-02-25

### 🐛 Bug Fixes

- Replace `pinecone-client` dependency with `pinecone` (#1431)


### 🧹 Chores

- Remove Python 3.8 support (#1421)

## [integrations/pinecone-v3.0.0] - 2025-02-14

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Pinecone - remove legacy filter support (#1069)
- Update changelog after removing legacy filters (#1083)
- Update ruff linting scripts and settings (#1105)
- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)

### 🌀 Miscellaneous

- Test: Pinecone - increase sleep time (#1307)
- Chore!: pinecone - remove dataframe support (#1372)

## [integrations/pinecone-v1.2.3] - 2024-08-29

### 🚀 Features

- Add filter_policy to pinecone integration (#821)

### 🐛 Bug Fixes

- `pinecone` - Fallback to default filter policy when deserializing retrievers without the init parameter (#901)
- Skip unsupported meta fields in PineconeDB (#1009)
- Converting `Pinecone` metadata fields from float back to int (#1034)

### 🧪 Testing

- Pinecone - fix `test_serverless_index_creation_from_scratch` (#806)
- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: add retry config to Pinecone test (#848)
- Chore: Minor retriever pydoc fix (#884)
- Chore: Pinecone - fix import in conftest (#914)

## [integrations/pinecone-v1.1.0] - 2024-06-11

### 🚀 Features

- Defer the database connection to when it's needed (#804)


## [integrations/pinecone-v1.0.0] - 2024-06-10

### 🚀 Features

- [**breaking**] Pinecone - support for the new API (#793)

### 🌀 Miscellaneous

- Pinecone - Skip `test_comparison_not_equal_with_dataframe` (#647)
- Pinecone - increase sleep time (#673)
- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)

## [integrations/pinecone-v0.4.1] - 2024-04-02

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Correctly deserialize Pinecone docstore in embedding retriever (#636)

### 📚 Documentation

- Update category slug (#442)
- Disable-class-def (#556)

### ⚙️ CI

- Generate API docs for Pinecone (#359)

### 🌀 Miscellaneous

- Pinecone - decrease concurrency in tests (#323)
- Rename retriever (#396)
- Fix imports in example (#398)
- [Pinecone] Use Haystack Secrets (#420)
- Pinecone - review docstrings and API reference (#503)
- Make tests show coverage (#566)
- Temporarily skip failing tests (#593)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)
- Skip `test_write_documents_duplicate_overwrite` test (#637)

## [integrations/pinecone-v0.2.1] - 2024-01-31

### 🐛 Bug Fixes

- Fix: fix linter (#281)

### 🌀 Miscellaneous

- Increase pinecone sleep time (#288)
- Pinecone - change dummy vector (#301)

## [integrations/pinecone-v0.2.0] - 2024-01-23

### 🌀 Miscellaneous

- Pinecone - update import for beta5 (#236)
- Change import paths (#256)

## [integrations/pinecone-v0.1.0] - 2024-01-17

### 🌀 Miscellaneous

- [Pinecone] Add example notebook (#149)
- Optimize API key reading (#162)
- Pin `pinecone-client<3` (#224)

## [integrations/pinecone-v0.0.1] - 2023-12-22

### 🌀 Miscellaneous

- Pinecone Document Store - minimal implementation (#81)
- Pinecone - filters (#133)
- Pinecone - dense retriever (#145)

<!-- generated by git-cliff -->
