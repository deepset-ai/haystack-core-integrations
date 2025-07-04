# Changelog

## [integrations/ollama-v3.4.0] - 2025-07-04

### 🚀 Features

- Pass `component_info` to `StreamingChunk` in `OllamaChatGenerator` (#2039)
- Add "think" parameter for Ollama (#1948)

### 🧹 Chores

- Remove black (#1985)


## [integrations/ollama-v3.3.0] - 2025-06-12

### 🐛 Bug Fixes

- Fix Ollama types + add py.typed (#1922)


## [integrations/ollama-v3.2.0] - 2025-06-10

### 🚀 Features

- Add run_async support for OllamaDocumentEmbedder (#1878)


## [integrations/ollama-v3.1.0] - 2025-06-10

### 🚀 Features

- Add run_async support for OllamaTextEmbedder (#1877)

### 🐛 Bug Fixes

- Ollama tools with streaming (#1906)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/ollama-v3.0.1] - 2025-05-22

### 🚀 Features

- OllamaChatGenerator - add Toolset support (#1765)


## [integrations/ollama-v3.0.0] - 2025-05-16

### 🚀 Features

- Adapt `OllamaGenerator` metadata to OpenAI format (#1753)

### 🧪 Testing

- Ollama - make test_run_with_response_format more robust (#1757)

### 🌀 Miscellaneous

- Removing import try for 2.2.12 `deserialize_tools_or_toolset_inplace` (#1713)

## [integrations/ollama-v2.4.2] - 2025-05-07

### 🐛 Bug Fixes

- Ollama streaming metadata 1686 (#1698)


## [integrations/ollama-v2.4.1] - 2025-04-10

### 🚀 Features

- Add `streaming_callback` to run methods of OllamaGenerator and OllamaChatGenerator (#1636)

### 🧹 Chores

- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)


## [integrations/ollama-v2.4.0] - 2025-04-04

### 🚀 Features

- Adapt Ollama metadata to OpenAI format; support Ollama in Langfuse (#1577)

### 🧪 Testing

- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)

## [integrations/ollama-v2.3.0] - 2025-01-29

### 🚀 Features

- Add `response_format` param to `OllamaChatGenerator` (#1326)


## [integrations/ollama-v2.2.0] - 2025-01-16

### 🚀 Features

- Ollama - add support for tools (#1294)


## [integrations/ollama-v2.1.2] - 2024-12-18

### 🐛 Bug Fixes

- Make Ollama Chat Generator compatible with new ChatMessage (#1256)


## [integrations/ollama-v2.1.1] - 2024-12-10

### 🌀 Miscellaneous

- Use  instead of  for  in Ollama (#1239)

## [integrations/ollama-v2.1.0] - 2024-11-28

### 🚀 Features

- `OllamaDocumentEmbedder` - allow batching embeddings (#1224)

### 🌀 Miscellaneous

- Add ollama missing changelog (#1214)
- Use class methods to create ChatMessage (#1222)

## [integrations/ollama-v2.0.0] - 2024-11-22

### 🐛 Bug Fixes

- Adapt to Ollama client 0.4.0 (#1209)

### ⚙️ CI

- Adopt uv as installer (#1142)


## [integrations/ollama-v1.1.0] - 2024-10-11

### 🚀 Features

- Add `keep_alive` parameter to Ollama Generators (#1131)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)


## [integrations/ollama-v1.0.1] - 2024-09-26

### 🐛 Bug Fixes

- Ollama Chat Generator - add missing `to_dict` and `from_dict` methods (#1110)


## [integrations/ollama-v1.0.0] - 2024-09-07

### 🐛 Bug Fixes

- Chat roles for model responses in chat generators (#1030)

### 🚜 Refactor

- [**breaking**] Use ollama python library instead of calling the API with `requests` (#1059)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Install pytest-rerunfailures; change test-cov script (#845)
- Ruff update, don't ruff tests (#985)

## [integrations/ollama-v0.0.7] - 2024-05-31

### 🚀 Features

- Add streaming support to OllamaChatGenerator (#757)

### 🌀 Miscellaneous

- Add license classifiers (#680)
- Change the pydoc renderer class (#718)

## [integrations/ollama-v0.0.6] - 2024-04-18

### 📚 Documentation

- Disable-class-def (#556)

### 🧹 Chores

- Update docstrings (#499)

### 🌀 Miscellaneous

- Update API docs (#494)
- Change testing workflow (#551)
- Remove references to Python 3.7 (#601)
- Add ollama embedder example (#669)
- Squash (#670)

Co-authored-by: anakin87 <stefanofiorucci@gmail.com>

## [integrations/ollama-v0.0.5] - 2024-02-28

### 🐛 Bug Fixes

- Fix order of API docs (#447)

This PR will also push the docs to Readme

### 📚 Documentation

- Update category slug (#442)

### 🧹 Chores

- Use `serialize_callable` instead of `serialize_callback_handler` in Ollama (#461)

### 🌀 Miscellaneous

- Ollama document embedder (#400)

* Added ollama document embedder and tests

* Cleaning of non-used variables and batch restrictions

* Fixed issue with test_document_embedder.py import_text_in_embedder test, test was incorrect

* Fixed lint issues and tests

* chore: Exculde evaluator private classes in API docs (#392)

* rename astraretriever (#399)

* rename retriever (#407)

* test patch- documents embedding wasn't working as expected

---------

Co-authored-by: Madeesh Kannan <shadeMe@users.noreply.github.com>
Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>
- Changed Default Ollama Embedding models to supported model: nomic-embed-text (#490)

* Changed Embedding model to supported model: nomic-embed-text

* Updated workflow yml with support for llm and embedding model

## [integrations/ollama-v0.0.4] - 2024-02-12

### 🌀 Miscellaneous

- Add license (#219)
- Generate api docs (#332)
- Ollama Text Embedder with new format (#252)

* add tests

* add ollama text embedder

* add init for text embedder

* format with black

* lint with ruff

* add meta to return message
- Support for streaming ollama generator (#280)

* added support for streaming ollama generator

* fixed linting errors

* more linting issues

* implemented serializing/deserializing for OllamaGenerator

* linting changes

* minor refactoring

* formating

## [integrations/ollama-v0.0.3] - 2024-01-16

### 🌀 Miscellaneous

- Ollama docstrings update (#171)
- Add example of OllamaGenerator (#170)

* add example of OllamaGenerator

* fix example with ruff

* change example to reference the greatest politician of all time - Super Mario

* add comments on how to set up and expected output
- Ollama Chat Generator (#176)

* update inits to expose ollamachatgenerator

* add ollama chat generator

* add tests for ollama chat generator

* add tests for init method

* Change order of chat history to chronological

* add test for chat history

* add return type to _build_message

* refactor message_to_dict to one liner

* add return types to fixtures

* add test for unavailable model

* drop streaming references for now

* drop streaming callback from tests

* Update integrations/ollama/src/ollama_haystack/chat/chat_generator.py

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* Update integrations/ollama/src/ollama_haystack/chat/chat_generator.py

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* Update integrations/ollama/src/ollama_haystack/chat/chat_generator.py

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* drop _chat_history_to_dict

* drop intermediate ollama to haystack response methods

* change metadata to meta

* lint with black

* refactor chat message fixture into one list

* add chat generator example

* rename example -> generator example

* add new chat generator example

* Update integrations/ollama/src/ollama_haystack/chat/chat_generator.py

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* update test for new timeout

* Update test_chat_generator.py

* increase generator timeout

* add docstrings

* fix

---------

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>
- Improve test (#191)
- Mount Ollama in haystack_integrations (#216)

* mount ollama in haystack_integrations

* fix fmt

## [integrations/ollama-v0.0.1] - 2024-01-03

### 🌀 Miscellaneous

- Ollama Integration (#132)

* add ollama generator

* add test skeleton

* add __inits__

* add __inits__

* add __inits__

* add pyproject

* add version

* Add instructions for building local environment

* Add integration tests using docker ollama service

* Add integration test market in pyproject.toml

* lint with black

* Delete integrations/__init__.py

* drop unused dunders

* update pyproject based on elasticsearch integration

* rename metadata to meta

* Use full /generate URL from ollama in init

* add docstrings and additional post arguments

* add github action

* change single to double quotes in tests

* fix init paths

* make timeout argument explicit in POST

* lint with black

* fix typo from jina to ollama in hatch lint

* ignore pytest and haystack typing issues

* correct type hint in output type

* add assertion for replies and meta in tests

* update labeler with ollama

* try to install and run ollama wo docker

* try to see if the issue with ports is related to concurrency

* another try

* better try to run ollama

* Update ollama.yml

* Update ollama.yml

* simplify docker-compose

* try to pull the model when the container is already running

* add -d

* Update ollama.yml

* Delete integrations/ollama/docker-compose.yml

* Update integrations/ollama/pyproject.toml

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

* Add timeout to POST Request

* Remove unpacking of dictionary arguments to pass mypy

* Add docstring to run function

* add test for init defaults, delete telemetry tests

* drop use of dataclass for housing response data

* modify type hints in protected methods

* update readme with new docker testing regime

* refactor post_args to json payload only

* try using a github service for Ollama

* modify post_args test to new json_payload

* add ollama integration to general README

* refinements

* fix tests

* rm unused fixture

---------

Co-authored-by: Stefano Fiorucci <stefanofiorucci@gmail.com>

<!-- generated by git-cliff -->
