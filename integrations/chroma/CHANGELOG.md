# Changelog

## [integrations/chroma-v4.3.0] - 2026-04-29

### 🚀 Features

- *(chroma)* Support DuplicatePolicy in write_documents and use async mixin tests (#3245)

### 🚜 Refactor

- Weaviate, chroma, elasticsearch, opensearch, azure_ai_search use `_normalize_metadata_field_name` from haystack.utils (#2953)

### 🧪 Testing

- `ChromaDocumentStore` use Mixin tests (#3026)
- Track test coverage for all integrations (#3065)
- Better categorize some Document Stores tests (#3085)
- Chroma - add unit tests (#3175)

### 🧹 Chores

- Add missing -> None return type annotations to chroma __init__ methods (#2976)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in first 10 integrations (#3008)
- Increase lower pins for 3.14 support in some integrations + test with 3.14 (#3033)


## [integrations/chroma-v4.2.0] - 2026-03-11

### 🚀 Features

- Add support for metadata that contains lists of supported types (#2877)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)
- Chroma - remove tests for invalid Settings (now ignored) (#2935)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)
- Standardize author mentions (#2897)


## [integrations/chroma-v4.1.1] - 2026-02-10

### 📚 Documentation

- Fixing curly brackets (#2824)


## [integrations/chroma-v4.1.0] - 2026-02-10

### 🌀 Miscellaneous

- Add metadata query operations to ChromaDocumentStore (#2819)

## [integrations/chroma-v4.0.0] - 2026-01-12

### 🚀 Features

- Add `client_settings` to `ChromaDocumentStore` (#2651)

### 📚 Documentation

- Fix: doc links (#2662)

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Chroma - drop Python 3.9 and use X|Y typing (#2701)

### 🌀 Miscellaneous

- Fix: Fix doc links (#2661)

## [integrations/chroma-v3.5.0] - 2026-01-07

### 🚀 Features

- Adding `delete_by_filter()` and `update_by_filter()` to `ChromaDocumentStore` (#2649)


### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)

### 🌀 Miscellaneous

- Adopt PEP 585 type hinting (part 1) (#2509)

## [integrations/chroma-v3.4.0] - 2025-10-27

### 🚀 Features

- Adding delete_all_docs to ChromaDB document store (#2399)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)

### 🧹 Chores

- Update md files for new hatch scripts (#1911)
- Remove black (#1985)
- Standardize readmes - part 1 (#2202)


## [integrations/chroma-v3.3.0] - 2025-06-09

### 🚀 Features

- Chroma - fix typing + ship types by adding py.typed files (#1910)


## [integrations/chroma-v3.2.0] - 2025-06-06

### 🚀 Features

- Add async support for Chroma + improve typing (#1697)


### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)


## [integrations/chroma-v2.0.3] - 2025-04-04

### 🐛 Bug Fixes

- Update test and unpin chroma (#1618)
- Make chroma backwards compatible (#1619)


## [integrations/chroma-v2.0.2] - 2025-04-04


### ⚙️ CI

- Review testing workflows (#1541)

### 🌀 Miscellaneous

- Fix: Pin chroma (#1608)

## [integrations/chroma-v3.0.0] - 2025-03-11


### 🧹 Chores

- Use Haystack logging across integrations (#1484)

### 🌀 Miscellaneous

- Fix: Chroma - relax dataframe/blob test (#1404)
- Pin haystack and remove dataframe checks (#1501)

## [integrations/chroma-v2.0.1] - 2025-02-14

### 🌀 Miscellaneous

- Better handle discarded fields (#1373)

## [integrations/chroma-v1.0.1] - 2025-02-14

### 🧹 Chores

- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)

### 🌀 Miscellaneous

- Fix: Chroma - make `filter_documents` method return embeddings (#1361)

## [integrations/chroma-v2.0.0] - 2025-01-02

### 🧹 Chores

- Fix linting/isort (#1215)
- Chroma - pin `tokenizers` (#1223)

### 🌀 Miscellaneous

- Unpin tokenizers (#1233)
- Fix: updates for Chroma 0.6.0 (#1270)

## [integrations/chroma-v1.0.0] - 2024-11-06

### 🐛 Bug Fixes

- Fixing Chroma tests due `chromadb` update behaviour change (#1148)
- Adapt our implementation to breaking changes in Chroma 0.5.17  (#1165)

### ⚙️ CI

- Adopt uv as installer (#1142)


## [integrations/chroma-v0.22.1] - 2024-09-30

### 🌀 Miscellaneous

- Empty filters should behave as no filters (#1117)

## [integrations/chroma-v0.22.0] - 2024-09-30

### 🚀 Features

- Chroma - allow remote HTTP connection (#1094)
- Chroma - defer the DB connection (#1107)

### 🐛 Bug Fixes

- Refactor: fix chroma linting; do not use numpy (#1063)
- Filters in chroma integration (#1072)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### 🧹 Chores

- Chroma - ruff update, don't ruff tests (#983)
- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Chore: ChromaDocumentStore lint fix (#1065)

## [integrations/chroma-v0.21.1] - 2024-07-17

### 🐛 Bug Fixes

- `ChromaDocumentStore` - discard `meta` items when the type of their value is not supported in Chroma (#907)


## [integrations/chroma-v0.21.0] - 2024-07-16

### 🚀 Features

- Add metadata parameter to ChromaDocumentStore. (#906)


## [integrations/chroma-v0.20.1] - 2024-07-15

### 🚀 Features

- Added distance_function property to ChromadocumentStore (#817)
- Add filter_policy to chroma integration (#826)

### 🐛 Bug Fixes

- Allow search in ChromaDocumentStore without metadata (#863)
- `Chroma` - Fallback to default filter policy when deserializing retrievers without the init parameter (#897)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Minor retriever pydoc fix (#884)

## [integrations/chroma-v0.18.0] - 2024-05-31

### 🌀 Miscellaneous

- Chore: pin chromadb>=0.5.0 (#777)

## [integrations/chroma-v0.17.0] - 2024-05-10

### 🌀 Miscellaneous

- Chore: change the pydoc renderer class (#718)
- Implement filters for chromaQueryTextRetriever via existing haystack filters logic (#705)

## [integrations/chroma-v0.16.0] - 2024-05-02

### 📚 Documentation

- Small consistency improvements (#536)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Refactor tests (#574)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)
- Pin databind-core (#619)
- Chore: add license classifiers (#680)
- Feature/bump chromadb dep to 0.5.0 (#700)

## [integrations/chroma-v0.15.0] - 2024-03-01

### 🌀 Miscellaneous

- Release chroma on python 3.8 (#512)

## [integrations/chroma-v0.14.0] - 2024-02-29

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Serialize the path to the local db (#506)

### 📚 Documentation

- Update category slug (#442)
- Review chroma integration (#501)

### 🌀 Miscellaneous

- Small improvements (#443)
- Fix: make write_documents compatible with the DocumentStore protocol (#505)

## [integrations/chroma-v0.13.0] - 2024-02-13

### 🌀 Miscellaneous

- Chroma: rename retriever (#407)

## [integrations/chroma-v0.12.0] - 2024-02-06

### 🚀 Features

- Generate API docs (#262)

### 🌀 Miscellaneous

- Add typing_extensions pin to Chroma integration (#295)
- Allows filters and persistent document stores for Chroma (#342)

## [integrations/chroma-v0.11.0] - 2024-01-18

### 🐛 Bug Fixes

- Chroma DocumentStore creation for pre-existing collection name (#157)

### 🌀 Miscellaneous

- Mount chroma integration under `haystack_integrations.*` (#193)
- Remove ChromaSingleQueryRetriever (#240)

## [integrations/chroma-v0.9.0] - 2023-12-20

### 🐛 Bug Fixes

- Fix project URLs (#96)

### 🚜 Refactor

- Use `hatch_vcs` to manage integrations versioning (#103)

### 🌀 Miscellaneous

- Chore: pin chroma version (#104)
- Fix: update to the latest Document format (#127)

## [integrations/chroma-v0.8.1] - 2023-12-05

### 🐛 Bug Fixes

- Document Stores: fix protocol import (#77)

## [integrations/chroma-v0.8.0] - 2023-12-04

### 🐛 Bug Fixes

- Fix license headers

### 🌀 Miscellaneous

- Reorganize repository (#62)
- Update import paths (#64)
- Patch chroma filters tests (#67)
- Remove Document Store decorator (#76)

<!-- generated by git-cliff -->
