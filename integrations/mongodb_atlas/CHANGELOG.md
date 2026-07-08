# Changelog

## [integrations/mongodb_atlas-v4.2.0] - 2026-06-18

### 🚀 Features

- *(mongodb-atlas)* Use async DocumentStore mixin tests (#3249)

### 🐛 Bug Fixes

- Serialize MongoDBAtlasDocumentStore embedding_field and content_field (#3446)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧪 Testing

- Replacing each `DocumentStore` specific tests and used the generalised ones from `haystack.testing.document_store` (#2812)
- Test compatible integrations with python 3.14; update pyproject (#3001)
- `MongodbDocumentStore` use Mixin tests (#3022)
- Track test coverage for all integrations (#3065)
- Better categorize some Document Stores tests (#3085)
- Mongodb - refactor test suite (#3171)
- Fix mongodb async connection teardown (#3253)

### 🧹 Chores

- Add ANN ruff ruleset to llama_cpp, llama_stack, mcp, meta_llama, mistral, mongodb_atlas, nvidia, ollama, openrouter, opensearch (#2991)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 21-30 (#3010)


## [integrations/mongodb_atlas-v4.1.0] - 2026-02-12

### 🚀 Features

- Add metadata exploration methods to MongoDBAtlasDocumentStore (#2820)


## [integrations/mongodb_atlas-v4.0.0] - 2026-01-12

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)
- [**breaking**] Mongodb_atlas - drop Python 3.9 and use X|Y typing (#2718)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)
- Fix: Fix doc links (#2661)

## [integrations/mongodb_atlas-v3.5.0] - 2025-11-06


### ⚙️ CI

- Change pytest command (#2475)

### 🌀 Miscellaneous

- Feat: add filter methods to MongoDB DocumentStore (#2474)

## [integrations/mongodb_atlas-v3.4.0] - 2025-10-29

### 🚀 Features

- Add delete_all_documents() to MongoDB DocumentStore (#2401)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### 🧹 Chores

- Remove black (#1985)
- Standardize readmes - part 2 (#2205)


## [integrations/mongodb_atlas-v3.3.0] - 2025-06-18

### 🐛 Bug Fixes

- Fix MongoDB types + add py.typed (#1967)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/mongodb_atlas-v3.2.2] - 2025-05-27

### 🌀 Miscellaneous

- Pin version for `pymongo` and `haystack` in MongoDB integration (#1832)

## [integrations/mongodb_atlas-v3.2.1] - 2025-05-13

### 🚀 Features

- Support custom content field (#1721)

### 🐛 Bug Fixes

- Fulltext retrieval (#1730)

### 📚 Documentation


## [integrations/mongodb_atlas-v3.1.2] - 2025-05-07

### 🐛 Bug Fixes

- Hard coded embedding field for mongodb (#1708)


### 🌀 Miscellaneous

- Docs: add MongoDBAtlasFullTextRetriever to API Reference (#1669)

## [integrations/mongodb_atlas-v3.1.1] - 2025-04-11

### 🚀 Features

- Mongodb async (#1590)

### 🐛 Bug Fixes

- Async failing in mongodb-integrations (#1633)


### ⚙️ CI

- Review testing workflows (#1541)

## [integrations/mongodb_atlas-v3.0.0] - 2025-03-11


### 🧪 Testing

- Skip tests that require credentials on PRs from forks for some integrations (#1485)

### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)

### 🌀 Miscellaneous

- Docs: Update document store descriptions for deepset Pipeline Builder (#1447)
- Chore: Mongo - pin haystack and remove dataframe checks (#1512)

## [integrations/mongodb_atlas-v2.0.0] - 2025-02-14

### 🚀 Features

- Defer the database connection to when it's needed (#770)
- Add filter_policy to mongodb_atlas integration (#823)

### 🐛 Bug Fixes

- Pass empty dict to filter instead of None (#775)
- `Mongo` - Fallback to default filter policy when deserializing retrievers without the init parameter (#899)


### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)
- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- Update mongodb test for the new `apply_filter_policy` usage (#971)
- MongoDB - remove legacy filter support (#1066)
- Update ruff linting scripts and settings (#1105)
- Inherit from `FilterDocumentsTestWithDataframe` in Document Stores (#1290)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Minor retriever pydoc fix (#884)
- Fix: remove exposure of connection_string (#937)
- Chore: mongo - ruff update, don't ruff tests (#991)
- Mongodb keyword search (#1228)
- Chore!: Mongodb - remove dataframe support (#1398)

## [integrations/mongodb_atlas-v0.2.1] - 2024-04-09

### 🐛 Bug Fixes

- Fix `haystack-ai` pins (#649)

### 📚 Documentation

- Disable-class-def (#556)

### 🌀 Miscellaneous

- Refactor tests (#574)
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip `SparseEmbedding` (#606)

## [integrations/mongodb_atlas-v0.2.0] - 2024-03-09

### 📚 Documentation

- Mongo atlas (#534)
- Final API docs touches (#538)

### 🌀 Miscellaneous

- Improve example (#546)
- MongoDB Atlas: filters (#542)

## [integrations/mongodb_atlas-v0.1.0] - 2024-02-23

### 🚀 Features

- MongoDBAtlas Document Store (#413)
- `MongoDBAtlasEmbeddingRetriever` (#427)

### 🐛 Bug Fixes

- Remove filters from `MongoDBAtlasDocumentStore.count()` method (#430)
- Fix order of API docs (#447)
- Fix: `pyproject.toml` typo for MongoDB Atlas (#478)

### 📚 Documentation

- Update category slug (#442)

### 🌀 Miscellaneous

- Fix: typo in `mongodb_atlas/pyproject.toml` (#426)

<!-- generated by git-cliff -->
