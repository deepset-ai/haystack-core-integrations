# Changelog

## [integrations/astra-v4.3.0] - 2026-04-27

### 🚀 Features

- Astra - improve filters and add unit tests (#3194)

### 🧪 Testing

- Track test coverage for all integrations (#3065)


## [integrations/astra-v4.2.0] - 2026-03-24

### 🚀 Features

- *(astra)* Add run_async to AstraEmbeddingRetriever (#3016)

### 🐛 Bug Fixes

- Converting to `staticmethod` some tests in `AstraDocumentStore` (#2928)

### 🧪 Testing

- `AstraDocumentStore` use Mixin tests (#3027)

### 🧹 Chores

- Add missing -> None return type annotations to astra __init__ methods (#2974)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in first 10 integrations (#3008)
- Increase lower pins for 3.14 support in some integrations + test with 3.14 (#3033)


## [integrations/integrations/astra-v4.1.0] - 2026-03-06

### 🚀 Features

- Add Astra document store operations (#2904)

### 📚 Documentation

- Fix some invalid raises directives (#2847)
- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)
- Standardize author mentions (#2897)


## [integrations/astra-v4.0.0] - 2026-01-12

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Astra - drop Python 3.9 and use X|Y typing (#2694)


## [integrations/astra-v3.1.0] - 2025-12-30

### 🚀 Features

- Adding `delete_by_filter` and `update_by_filter` (#2631)


### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)

### 🌀 Miscellaneous

- Adopt PEP 585 type hinting (part 1) (#2509)

## [integrations/astra-v3.0.0] - 2025-10-21

### 🚀 Features

- Add support for `delete_all_documents` for astra client (#2362)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### 🧹 Chores

- Standardize readmes - part 1 (#2202)


## [integrations/astra-v2.1.0] - 2025-06-27

### 🐛 Bug Fixes

- Astra - fix types + add py.typed (#2011)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Fix linting for ruff 0.12.0 (#1969)
- Remove black (#1985)


## [integrations/astra-v2.0.1] - 2025-04-03


### ⚙️ CI

- Review testing workflows (#1541)

### 🌀 Miscellaneous

- Fix: astra-haystack remove init files to make them namespace packages (#1595)

## [integrations/astra-v2.0.0] - 2025-03-11

### 🧹 Chores

- Use Haystack logging across integrations (#1484)
- Astra - pin `haystack-ai>=2.11.0` and remove dataframe checks (#1492)


## [integrations/astra-v1.0.0] - 2025-02-17

### 🧹 Chores

- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)
- [**breaking**] Astra - remove dataframe support (#1376)

## [integrations/astra-v0.1.2] - 2024-11-25


## [integrations/astra-v0.9.4] - 2024-11-25

### 🌀 Miscellaneous

- Fix: Astra - fix embedding retrieval top-k limit (#1210)

## [integrations/astra-v0.10.0] - 2024-10-22

### 🚀 Features

- Update astradb integration for latest client library (#1145)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Fix: #1047 Remove count_documents from delete_documents (#1049)

## [integrations/astra-v0.9.3] - 2024-09-12

### 🐛 Bug Fixes

- Astra DB, improved warnings and guidance about indexing-related mismatches (#932)
- AstraDocumentStore filter by id (#1053)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)


## [integrations/astra-v0.9.2] - 2024-07-22

### 🌀 Miscellaneous

- Normalize logical filter conditions (#874)

## [integrations/astra-v0.9.1] - 2024-07-15

### 🚀 Features

- Defer the database connection to when it's needed (#769)
- Add filter_policy to astra integration (#827)

### 🐛 Bug Fixes

- Fix astra nightly
- Fix typing checks
- `Astra` - Fallback to default filter policy when deserializing retrievers without the init parameter (#896)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Fix: Incorrect astra not equal operator (#868)
- Chore: Minor retriever pydoc fix (#884)

## [integrations/astra-v0.7.0] - 2024-05-15

### 🐛 Bug Fixes

- Make unit tests pass (#720)

### 🌀 Miscellaneous

- Chore: change the pydoc renderer class (#718)
- [Astra DB] Explicit projection when reading from Astra DB (#733)

## [integrations/astra-v0.6.0] - 2024-04-24

### 🐛 Bug Fixes

- Pass namespace in the docstore init (#683)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Bug fix for document_store.py (#618)

## [integrations/astra-v0.5.1] - 2024-04-09

### 🐛 Bug Fixes

- Fix `haystack-ai` pins (#649)

### 🌀 Miscellaneous

- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)

## [integrations/astra-v0.5.0] - 2024-03-18

### 📚 Documentation

- Review `integrations.astra` (#498)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Fix example code for Astra DB pipeline (#481)
- Make tests show coverage (#566)
- Astra DB: Add integration usage tracking (#568)

## [integrations/astra-v0.4.2] - 2024-02-21

### 🌀 Miscellaneous

- Proper name for the sort param (#454)

## [integrations/astra-v0.4.1] - 2024-02-20

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Astra: fix integration tests (#450)

## [integrations/astra-v0.4.0] - 2024-02-20

### 📚 Documentation

- Update category slug (#442)

### 🌀 Miscellaneous

- Update the Astra DB Integration to fit latest conventions (#428)

## [integrations/astra-v0.3.0] - 2024-02-15

### 🌀 Miscellaneous

- Model_name_or_path > model (#418)
- [Astra] Change authentication parameters (#423)

## [integrations/astra-v0.2.0] - 2024-02-13

### 🌀 Miscellaneous

- [**breaking**] Change import paths (#277)
- Generate api docs (#327)
- Astra: rename retriever (#399)

## [integrations/astra-v0.1.1] - 2024-01-18

### 🌀 Miscellaneous

- Update the import paths for beta5 (#235)

## [integrations/astra-v0.1.0] - 2024-01-11

### 🌀 Miscellaneous

- Adding AstraDB as a DocumentStore (#144)

<!-- generated by git-cliff -->
