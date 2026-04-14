# Changelog

## [integrations/ollama-v6.3.0] - 2026-04-10

### 🐛 Bug Fixes

- Replace in-place dataclass mutations with dataclasses.replace() (#3112)
- Ollama - better reasoning streaming support (#3131)

### 🧪 Testing

- Track test coverage for all integrations (#3065)

### 🧹 Chores

- Add ANN ruff ruleset to llama_cpp, llama_stack, mcp, meta_llama, mistral, mongodb_atlas, nvidia, ollama, openrouter, opensearch (#2991)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 21-30 (#3010)
- Increase lower pins for 3.14 support in some integrations + test with 3.14 (#3033)


## [integrations/ollama-v6.2.0] - 2026-03-09

### 🚀 Features

- *(ollama)* Add max_retries to chat generator (#2899)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)


## [integrations/ollama-v6.1.0] - 2026-02-19

### 🚀 Features

- Use StreamingChunk.reasoning field in Ollama chat generator (#2850)

### 📚 Documentation

- Fix some invalid raises directives (#2847)
- Simplify pydoc configs (#2855)


## [integrations/ollama-v6.0.0] - 2026-01-12

### 🐛 Bug Fixes

- Fix failing tests for opensearch and ollama (#2500)
- Fix ollama unit test (#2523)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)
- [**breaking**] Ollama - drop Python 3.9 and use X|Y typing (#2719)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 5) (#2528)

## [integrations/ollama-v5.3.0] - 2025-10-22

### 🚀 Features

- OllamaChatGenerator update tools param to ToolsType (#2432)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)


## [integrations/ollama-v5.2.0] - 2025-09-26

### 🚀 Features

- Add run_async support for OllamaChatGenerator (#2279)


## [integrations/ollama-v5.1.0] - 2025-08-28

### 🚀 Features

- Adding optional keepalive parameter to Ollama embedders (#2228)


## [integrations/ollama-v5.0.0] - 2025-08-25

### 🚜 Refactor

- [**breaking**] `OllamaChatGenerator` - refine reasoning support + refactoring (#2200)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/ollama-v4.1.0] - 2025-08-06

### 🚀 Features

- `OllamaChatGenerator` - image support (#2147)


## [integrations/ollama-v4.0.0] - 2025-08-05

### 🚀 Features

- Adopt new `StreamingChunk` in Ollama (#2109)


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

- Chore: use `text` instead of `content` for `ChatMessage` in Ollama (#1239)

## [integrations/ollama-v2.1.0] - 2024-11-28

### 🚀 Features

- `OllamaDocumentEmbedder` - allow batching embeddings (#1224)

### 🌀 Miscellaneous

- Chore: update changelog for `ollama-haystack==2.0.0` (#1214)
- Chore: use class methods to create `ChatMessage` (#1222)

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

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: ollama - ruff update, don't ruff tests (#985)

## [integrations/ollama-v0.0.7] - 2024-05-31

### 🚀 Features

- Add streaming support to OllamaChatGenerator (#757)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)

## [integrations/ollama-v0.0.6] - 2024-04-18

### 📚 Documentation

- Disable-class-def (#556)

### 🧹 Chores

- Update docstrings (#499)

### 🌀 Miscellaneous

- Update API docs adding embedders (#494)
- Change testing workflow (#551)
- Remove references to Python 3.7 (#601)
- Add ollama embedder example (#669)
- Fix: change ollama output name to 'meta' (#670)

## [integrations/ollama-v0.0.5] - 2024-02-28

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)

### 🧹 Chores

- Use `serialize_callable` instead of `serialize_callback_handler` in Ollama (#461)

### 🌀 Miscellaneous

- Ollama document embedder (#400)
- Changed Default Ollama Embedding models to supported model: nomic-embed-text (#490)

## [integrations/ollama-v0.0.4] - 2024-02-12

### 🌀 Miscellaneous

- Ollama: add license (#219)
- Generate api docs (#332)
- Ollama Text Embedder with new format (#252)
- Support for streaming ollama generator (#280)

## [integrations/ollama-v0.0.3] - 2024-01-16

### 🌀 Miscellaneous

- Docs: Ollama docstrings update (#171)
- Add example of OllamaGenerator (#170)
- Ollama Chat Generator (#176)
- Ollama: improve test (#191)
- Mount Ollama in haystack_integrations (#216)

## [integrations/ollama-v0.0.1] - 2024-01-03

### 🌀 Miscellaneous

- Ollama Integration (#132)

<!-- generated by git-cliff -->
