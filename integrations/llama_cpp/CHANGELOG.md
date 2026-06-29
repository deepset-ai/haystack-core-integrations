# Changelog

## [integrations/llama_cpp-v2.3.0] - 2026-06-04

### 🚀 Features

- Llama.cpp - accept str as ChatGenerator input; deprecate generator; migrate generator example to chat generator (#3400)

### 📚 Documentation

- Generate missing API reference for Chat Generator (#2842)
- *(llama_cpp)* Remove explicit warm_up calls (#2852)
- Simplify pydoc configs (#2855)

### 🧪 Testing

- Test compatible integrations with python 3.14; update pyproject (#3001)
- Track test coverage for all integrations (#3065)

### ⚙️ CI

- Refactor some workflows (#3074)
- Llama.cpp - simplify test setup (#3113)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)
- Add ANN ruff ruleset to llama_cpp, llama_stack, mcp, meta_llama, mistral, mongodb_atlas, nvidia, ollama, openrouter, opensearch (#2991)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 21-30 (#3010)


## [integrations/llama_cpp-v2.2.0] - 2026-02-16

### 🚀 Features

- Add run_async to LlamaCppChatGenerator (#2821)

### 🧹 Chores

- Llama.cpp - pin transformers test dependency; fix type error (#2784)


## [integrations/llama_cpp-v2.1.0] - 2026-01-14

### 🚀 Features

- Update Llama CPP components to auto call run `warm_up` (#2748)


## [integrations/llama_cpp-v2.0.0] - 2026-01-12

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)
- [**breaking**] Llama_cpp - drop Python 3.9 and use X|Y typing (#2710)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)

## [integrations/llama_cpp-v1.4.0] - 2025-10-23

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Download pre-built wheels for llama-cpp-python on macOS (#2235)

### 🧹 Chores

- Fix llama.cpp types (#2271)

### 🌀 Miscellaneous

- Feat: `LlamaCppChatGenerator` update tools param to ToolsType (#2438)

## [integrations/llama_cpp-v1.3.0] - 2025-08-22

### 🚀 Features

- Add image support to LlamaCppChatGenerator (#2197)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/llama_cpp-v1.2.0] - 2025-07-28

### 🚀 Features

- `LlamaCppChatGenerator` streaming support (#2108)

### 🧹 Chores

- Remove black (#1985)


## [integrations/llama_cpp-v1.1.0] - 2025-06-19

### 🐛 Bug Fixes

- Fix llama.cpp types; add py.typed; Toolset support (#1973)

### 🧪 Testing

- Test llama.cpp with python 3.12 (#1601)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)
- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)
- Align core-integrations Hatch scripts (#1898)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)

## [integrations/llama_cpp-v1.0.0] - 2025-02-07

### 🚀 Features

- [**breaking**] Llama.cpp - unified support for tools + refactoring (#1357)


## [integrations/llama_cpp-v0.4.4] - 2025-01-16

### 🧹 Chores

- Llama.cpp - gently handle the removal of ChatMessage.from_function (#1298)


## [integrations/llama_cpp-v0.4.3] - 2024-12-19

### 🐛 Bug Fixes

- Make llama.cpp Chat Generator compatible with new `ChatMessage` (#1254)


## [integrations/llama_cpp-v0.4.2] - 2024-12-10

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)
- Unpin `llama-cpp-python` (#1115)
- Fix linting/isort (#1215)
- Use text instead of content for ChatMessage in Llama.cpp, Langfuse and Mistral (#1238)

### 🌀 Miscellaneous

- Chore: lamma_cpp - ruff update, don't ruff tests (#998)
- Fix: pin `llama-cpp-python<0.3.0` (#1111)

## [integrations/llama_cpp-v0.4.1] - 2024-08-08

### 🐛 Bug Fixes

- Replace DynamicChatPromptBuilder with ChatPromptBuilder (#940)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- Pin `llama-cpp-python>=0.2.87` (#955)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Fix: pin llama-cpp-python to an older version (#943)
- Refactor: introduce `_convert_message_to_llamacpp_format` utility function (#939)

## [integrations/llama_cpp-v0.4.0] - 2024-05-13

### 🐛 Bug Fixes

- Llama.cpp: change wrong links and imports (#436)
- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### 🧹 Chores

- [**breaking**] Rename model_path to model in the Llama.cpp integration (#243)

### 🌀 Miscellaneous

- Generate api docs (#353)
- Model_name_or_path > model (#418)
- Llama.cpp - review docstrings (#510)
- Llama.cpp - update examples (#511)
- Make tests show coverage (#566)
- Remove references to Python 3.7 (#601)
- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Basic implementation of llama.cpp chat generation (#723)

## [integrations/llama_cpp-v0.2.1] - 2024-01-18

### 🌀 Miscellaneous

- Update import paths for beta5 (#233)

## [integrations/llama_cpp-v0.2.0] - 2024-01-17

### 🌀 Miscellaneous

- Mount llama_cpp in haystack_integrations (#217)

## [integrations/llama_cpp-v0.1.0] - 2024-01-09

### 🚀 Features

- Add Llama.cpp Generator (#179)

<!-- generated by git-cliff -->
