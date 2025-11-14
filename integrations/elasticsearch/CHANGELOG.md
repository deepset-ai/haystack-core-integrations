# Changelog

## [integrations/elasticsearch-v4.1.0] - 2025-10-09

### 🚀 Features

- Adding the operation `delete_all_documents` to the `ElasticSearchDocumentStore` (#2320)


## [integrations/elasticsearch-v4.0.0] - 2025-09-24

### 🚀 Features

- [**breaking**] Adding `api_token` and `apit_token_id` support to `ElasticSearchDocumentStore` (#2292)

### 🧹 Chores

- Remove black (#1985)
- Standardize readmes - part 1 (#2202)
- Standardize readmes - part 2 (#2205)


## [integrations/elasticsearch-v3.1.0] - 2025-06-12

### 🐛 Bug Fixes

- Fix Elasticsearch types + add py.typed (#1923)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/elasticsearch-v3.0.1] - 2025-05-27


### ⚙️ CI

- Review testing workflows (#1541)

### 🌀 Miscellaneous

- Pining lower versions of haystack and `aiohttp` for `ElasticSearch` (#1827)

## [integrations/elasticsearch-v3.0.0] - 2025-03-11


### 🧹 Chores

- Use Haystack logging across integrations (#1484)
- Elasticsearch - pin haystack and remove dataframe checks; add `aiohttp` dependency (#1502)

### 🌀 Miscellaneous

- Docs: Update document store descriptions for deepset Pipeline Builder (#1447)

## [integrations/elasticsearch-v2.1.0] - 2025-02-26

### 🚀 Features

- Adding async support to ElasticSearch retrievers and document store (#1429)

### 🧹 Chores

- Remove Python 3.8 support (#1421)


## [integrations/elasticsearch-v2.0.0] - 2025-02-14

### 🧹 Chores

- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)
- [**breaking**] Elasticsearch - remove dataframe support (#1377)


## [integrations/elasticsearch-v1.0.1] - 2024-10-28

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update changelog after removing legacy filters (#1083)
- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Fix: Elasticsearch - allow passing headers (#1156)

## [integrations/elasticsearch-v1.0.0] - 2024-09-12

### 🚀 Features

- Defer the database connection to when it's needed (#766)
- Add filter_policy to elasticsearch integration (#825)

### 🐛 Bug Fixes

- `ElasticSearch` - Fallback to default filter policy when deserializing retrievers without the init parameter (#898)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- ElasticSearch - remove legacy filters elasticsearch (#1078)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Minor retriever pydoc fix (#884)
- Chore: elasticsearch - ruff update, don't ruff tests (#999)

## [integrations/elasticsearch-v0.5.0] - 2024-05-24

### 🐛 Bug Fixes

- Add support for custom mapping in ElasticsearchDocumentStore (#721)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)

## [integrations/elasticsearch-v0.4.0] - 2024-04-03

### 📚 Documentation

- Docstring update  (#525)
- Review Elastic (#541)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Refactor tests (#574)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)
- [Elasticsearch] fix: Filters not working with metadata that contain a space or capitalization (#639)

## [integrations/elasticsearch-v0.3.0] - 2024-02-23

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)

### 🌀 Miscellaneous

- Generate api docs (#322)
- Add filters to run function in retrievers of elasticsearch (#440)
- Add user-agent header (#457)

## [integrations/elasticsearch-v0.2.0] - 2024-01-19

### 🌀 Miscellaneous

- Mount import paths under haystack_integrations (#244)

## [integrations/elasticsearch-v0.1.3] - 2024-01-18

### 🌀 Miscellaneous

- Added top_k argument in the run function of ElasticSearcBM25Retriever (#130)
- Add more docstrings for `ElasticsearchDocumentStore` and `ElasticsearchBM25Retriever` (#184)
- Elastic - update imports for beta5 (#238)

## [integrations/elasticsearch-v0.1.2] - 2023-12-20

### 🐛 Bug Fixes

- Fix project URLs (#96)

### 🚜 Refactor

- Use `hatch_vcs` to manage integrations versioning (#103)

### 🌀 Miscellaneous

- Update elasticsearch test badge (#79)
- [Elasticsearch] - BM25 retrieval: not all terms must mandatorily match (#125)

## [integrations/elasticsearch-v0.1.1] - 2023-12-05

### 🐛 Bug Fixes

- Document Stores: fix protocol import (#77)

## [integrations/elasticsearch-v0.1.0] - 2023-12-04

### 🐛 Bug Fixes

- Fix license headers

### 🌀 Miscellaneous

- Remove Document Store decorator (#76)

## [integrations/elasticsearch-v0.0.2] - 2023-11-29

### 🌀 Miscellaneous

- Reorganize repository (#62)
- Update `ElasticSearchDocumentStore` to use latest `haystack-ai` version (#63)
- Bump elasticsearch_haystack to 0.0.2

<!-- generated by git-cliff -->
