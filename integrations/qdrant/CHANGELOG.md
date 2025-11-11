# Changelog

## [integrations/qdrant-v9.3.0] - 2025-11-11

### 🚀 Features

- Adding `delete_all_docs` to Qdrant document store (#2363)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove black (#1985)
- Standardize readmes - part 2 (#2205)


## [integrations/qdrant-v9.2.0] - 2025-06-12

### 🐛 Bug Fixes

- Fix Qdrant types + add py.typed (#1919)


### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)

### 🌀 Miscellaneous

- Add pins for Qdrant (#1853)

## [integrations/qdrant-v9.1.2] - 2025-05-27

### 🐛 Bug Fixes

- Fix exposing Qdrant api-key in `metadata` field when running `to_dict` (#1813)


## [integrations/qdrant-v9.1.1] - 2025-03-20


### ⚙️ CI

- Review testing workflows (#1541)

### 🌀 Miscellaneous

- Fix: `TypeError` in `QdrantDocumentStore` when handling duplicate documents (#1551)

## [integrations/qdrant-v9.1.0] - 2025-03-14

### 🚀 Features

- Qdrant -- async support (#1480)


## [integrations/qdrant-v9.0.0] - 2025-03-11


### 🧹 Chores

- Use Haystack logging across integrations (#1484)
- Qdrant - pin haystack and remove dataframe checks (#1519)

## [integrations/qdrant-v8.1.0] - 2025-03-07

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Docs: Update document store descriptions for deepset Pipeline Builder (#1447)
- Refactor: Qdrant - raise error if existing collection is not compatible with Haystack (#1481)

## [integrations/qdrant-v8.0.0] - 2025-02-19

### 🧹 Chores

- Fix linting/isort (#1215)
- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)
- [**breaking**] Qdrant - remove dataframe support (#1403)


## [integrations/qdrant-v7.0.0] - 2024-10-29

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Refactor!: Qdrant - remove `index` parameter from methods (#1160)

## [integrations/qdrant-v6.0.0] - 2024-09-13

### 🌀 Miscellaneous

- Remove support for deprecated legacy filters in Qdrant (#1084)

## [integrations/qdrant-v5.1.0] - 2024-09-12

### 🚀 Features

- Qdrant - Add group_by and group_size optional parameters to Retrievers (#1054)


## [integrations/qdrant-v5.0.0] - 2024-09-02

### 🌀 Miscellaneous

- Fix!: fix type errors in `QdrantDocumentStore`; rename `ids` (parameter of `delete_documents`) to `document_ids` (#1041)

## [integrations/qdrant-v4.2.0] - 2024-08-27

### 🚜 Refactor

- Qdrant Query API (#1025)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### 🌀 Miscellaneous

- Chore: Update Qdrant tests for the new `apply_filter_policy` usage (#969)
- Chore: qdrant - ruff update, don't ruff tests (#989)

## [integrations/qdrant-v4.1.2] - 2024-07-15

### 🐛 Bug Fixes

- `qdrant` - Fallback to default filter policy when deserializing retrievers without the init parameter (#902)


## [integrations/qdrant-v4.1.1] - 2024-07-10

### 🚀 Features

- Add filter_policy to qdrant integration (#819)

### 🐛 Bug Fixes

- Errors in convert_filters_to_qdrant (#870)

### 🌀 Miscellaneous

- Chore: Minor retriever pydoc fix (#884)

## [integrations/qdrant-v4.1.0] - 2024-07-03

### 🚀 Features

- Add `score_threshold` to Qdrant Retrievers (#860)
- Qdrant - add support for BM42 (#864)


## [integrations/qdrant-v4.0.0] - 2024-07-02

### 🚜 Refactor

- [**breaking**] Qdrant - remove unused init parameters: `content_field`, `name_field`, `embedding_field`, and `duplicate_documents` (#861)
- [**breaking**] Qdrant - set `scale_score` default value to `False` (#862)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)

## [integrations/qdrant-v3.8.1] - 2024-06-20

### 📚 Documentation

- Added docstrings for QdrantDocumentStore (#808)


## [integrations/qdrant-v3.8.0] - 2024-06-06

### 🚀 Features

- Add force_disable_check_same_thread init param for Qdrant local client (#779)

## [integrations/qdrant-v3.7.0] - 2024-05-24

### 🚀 Features

- Make get_distance and recreate_collection public, replace deprecated recreate_collection function (#754)

## [integrations/qdrant-v3.6.0] - 2024-05-24

### 🚀 Features

- Defer database connection to the first usage (#748)

### 🌀 Miscellaneous

- Qdrant - improve docstrings for retrievers (#687)
- Chore: change the pydoc renderer class (#718)
- Allow vanilla qdrant filters (#692)

## [integrations/qdrant-v3.5.0] - 2024-04-24

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Qdrant - add hybrid retriever (#675)

## [integrations/qdrant-v3.4.0] - 2024-04-23

### 🌀 Miscellaneous

- Add embedding retrieval example (#666)
- Rename `QdrantSparseRetriever` to `QdrantSparseEmbeddingRetriever` (#681)

## [integrations/qdrant-v3.3.1] - 2024-04-12

### 🌀 Miscellaneous

- Add migration utility function for Sparse Embedding support (#659)

## [integrations/qdrant-v3.3.0] - 2024-04-12

### 🚀 Features

- *(Qdrant)* Start to work on sparse vector integration (#578)

## [integrations/qdrant-v3.2.1] - 2024-04-09

### 🐛 Bug Fixes

- Fix `haystack-ai` pins (#649)

## [integrations/qdrant-v3.2.0] - 2024-03-27

### 🚀 Features

- *(Qdrant)* Allow payload indexing + on disk vectors (#553)
- Qdrant datetime filtering support (#570)

### 🐛 Bug Fixes

- Fix: fix linter errors (#282)
- Fix order of API docs (#447)
- Doc: fixing docstrings for qdrant (#518)

### 🚜 Refactor

- [**breaking**] Qdrant - update secret management (#405)

### 📚 Documentation

- Update category slug (#442)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### ⚙️ CI

- Generate API docs for Qdrant (#361)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)

## [integrations/qdrant-v3.0.0] - 2024-01-22

### 🌀 Miscellaneous

- [**breaking**] Change import paths (#255)

## [integrations/qdrant-v2.0.1] - 2024-01-18

### 🚀 Features

- Add Qdrant integration (#98)

### 🐛 Bug Fixes

- Fix import paths for beta5 (#237)

### 🚜 Refactor

- Use `hatch_vcs` to manage integrations versioning (#103)

### 🌀 Miscellaneous

- Renamed QdrntRetriever to QdrntEmbeddingRetriever (#174)

<!-- generated by git-cliff -->
