# Changelog

## [integrations/pgvector-v3.0.0] - 2025-03-11

### 📚 Documentation

- Update changelog for integrations/pgvector (#1374)

### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)
- Pgvector - pin haystack and remove dataframe checks (#1518)

## [integrations/pgvector-v2.0.0] - 2025-02-13

### 🧹 Chores

- Pgvector - remove support for dataframe (#1370)

### 🌀 Miscellaneous

- Update changelog for integrations/pgvector (#1345)

Co-authored-by: anakin87 <44616784+anakin87@users.noreply.github.com>

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

- Install pytest-rerunfailures; change test-cov script (#845)
- Minor retriever pydoc fix (#884)
- Update test for the new apply_filter_policy usage (#970)
- Ruff update, don't ruff tests (#984)

Co-authored-by: Madeesh Kannan <shadeMe@users.noreply.github.com>

## [integrations/pgvector-v0.4.0] - 2024-06-20

### 🚀 Features

- Defer the database connection to when it's needed (#773)
- Add customizable index names for pgvector (#818)

### 🌀 Miscellaneous

- Missing api references (#728)
- [deepset-ai/haystack-core-integrations#727] (#738)

* hybrid retrieval ex

* Update integrations/pgvector/examples/hybrid_retrieval.py

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* suggested updates

* suggested updates

* suggested updates

---------

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

## [integrations/pgvector-v0.2.0] - 2024-05-08

### 🚀 Features

- `MongoDBAtlasEmbeddingRetriever` (#427)
- Implement keyword retrieval for pgvector integration (#644)

### 🐛 Bug Fixes

- Fix order of API docs (#447)

This PR will also push the docs to Readme

### 📚 Documentation

- Update category slug (#442)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Pgvector - review docstrings and API reference (#502)

* pgvector - docstrings and api ref

* rm os.environ from usage example
- Refactor tests (#574)

* first refactorings

* separate unit tests in pgvector

* small change to weaviate

* fix format

* usefixtures when possible
- Remove references to Python 3.7 (#601)
- Make Document Stores initially skip SparseEmbedding (#606)
- Add license classifiers (#680)
- Type hints in pgvector document store updated for 3.8 compability (#704)
- Change the pydoc renderer class (#718)

## [integrations/pgvector-v0.1.0] - 2024-02-14

### 🐛 Bug Fixes

- Fix linting (#328)

### 🌀 Miscellaneous

- Pgvector Document Store - minimal implementation (#239)

* very first draft

* setup integration folder and workflow

* update readme

* making progress!

* mypy overrides

* making progress on index

* drop sqlalchemy in favor of psycopggit add tests/test_document_store.py !

* good improvements!

* docstrings

* improve definition

* small improvements

* more test cases

* standardize

* inner_product

* explicit create statement

* address feedback

* change embedding_similarity_function to vector_function

* explicit insert and update statements

* remove useless condition

* unit tests for conversion functions
- Pgvector - filters (#257)

* very first draft

* setup integration folder and workflow

* update readme

* making progress!

* mypy overrides

* making progress on index

* drop sqlalchemy in favor of psycopggit add tests/test_document_store.py !

* good improvements!

* docstrings

* improve definition

* small improvements

* more test cases

* standardize

* start working on filters

* inner_product

* explicit create statement

* address feedback

* tests separation

* filters - draft

* change embedding_similarity_function to vector_function

* explicit insert and update statements

* remove useless condition

* unit tests for conversion functions

* tests change

* simplify!

* progress!

* better error messages and more

* cover also complex cases

* fmt

* make things work again

* progress on simplification

* further simplification

* filters simplification

* fmt

* rm print

* uncomment line

* fix name

* mv check filters is a dict in filter_documents

* f-strings

* NO_VALUE constant

* handle nested logical conditions in _parse_logical_condition

* add examples to _treat_meta_field

* fix fmt

* ellipsis fmt

* more tests for unhappy paths

* more tests for internal methods

* black

* log debug query and params
- Pgvector - embedding retrieval (#298)

* squash

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* fix fmt

---------

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>
- Pgvector - Embedding Retriever (#320)

* squash

* squash

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/document_stores/pgvector/document_store.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* fix fmt

* adjust docstrings

* Update integrations/pgvector/src/haystack_integrations/components/retrievers/pgvector/embedding_retriever.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* Update integrations/pgvector/src/haystack_integrations/components/retrievers/pgvector/embedding_retriever.py

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>

* improve docstrings

* fmt

---------

Co-authored-by: Massimiliano Pippi <mpippi@gmail.com>
- Api docs (#325)
- Add example (#334)
- Adopt `Secret` to pgvector (#402)

* initial import

* adding Secret support and fixing tests

* completing docs

* code formating

* linting and typing

* fixing tests

* adding custom from_dict

* adding test coverage

* use deserialize_secrets_inplace()

<!-- generated by git-cliff -->
