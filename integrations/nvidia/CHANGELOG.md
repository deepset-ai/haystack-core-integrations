# Changelog

## [integrations/nvidia-v1.1.1] - 2026-04-14

### 🐛 Bug Fixes

- Replace in-place dataclass mutations with dataclasses.replace() (#3112)

### 🧪 Testing

- Track test coverage for all integrations (#3065)
- Add unit tests for Nvidia integration (#3162)


## [integrations/nvidia-v1.1.0] - 2026-03-30

### 🐛 Bug Fixes

- Nvidia - fix structured output syntax (#3058)

### 📚 Documentation

- *(nvidia)* Remove explicit warm_up from examples (#2843)
- Simplify pydoc configs (#2855)

### 🧪 Testing

- Test compatible integrations with python 3.14; update pyproject (#3001)

### 🧹 Chores

- Add ANN ruff ruleset to llama_cpp, llama_stack, mcp, meta_llama, mistral, mongodb_atlas, nvidia, ollama, openrouter, opensearch (#2991)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 21-30 (#3010)

## [integrations/nvidia-v1.0.0] - 2026-01-13

### 🧹 Chores

- [**breaking**] Nvidia - drop Python 3.9 and use X|Y typing; fix default reranking model; improve tests (#2736)


## [integrations/nvidia-v0.5.0] - 2026-01-13

### 🚀 Features

- Update Nvidia components to auto call run `warm_up` and don't modify Documents in place (#2680)

### 📚 Documentation

- Add NvidiaChatGenerator to pydoc configs (#2448)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)

## [integrations/nvidia-v0.4.0] - 2025-10-23

### 🚀 Features

- `NvidiaChatGenerator` add integration tests for mixing Tool/Toolset (#2422)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)
- Fix docstrings to avoid errors in API reference generation (#2423)

### 🧪 Testing

- Make tests successfully run from forks (#2203)
- Remove deprecated NV-Embed-QA model from Nvidia tests (#2370)

### ⚙️ CI

- Temporarily install `click<8.3.0` (#2288)

### 🧹 Chores

- Remove black (#1985)
- Standardize readmes - part 2 (#2205)

### 🌀 Miscellaneous

- Add structured output support in `NvidiaChatGenerator` (#2405)

## [integrations/nvidia-v0.3.0] - 2025-06-20

### 🐛 Bug Fixes

- Fix Nvidia types + add py.typed (#1970)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)

## [integrations/nvidia-v0.2.0] - 2025-06-05

### 🚀 Features

- Add NvidiaChatGenerator based on OpenAIChatGenerator (#1776)


## [integrations/nvidia-v0.1.8] - 2025-05-28

### 🌀 Miscellaneous

- Add pins for Nvidia (#1846)

## [integrations/nvidia-v0.1.7] - 2025-04-03


### 🧪 Testing

- Reduce Nvidia API calls in integration tests (#1432)
- Add test cases for all utils methods for Nvidia integration (#1458)
- Add unit tests for Nvidia NimBackend (#1546)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Fix: nvidia-haystack remove init files to make them namespace packages (#1594)

## [integrations/nvidia-v0.1.6] - 2025-02-11

### 🚀 Features

- Add nvidia latest embedding models (#1364)


## [integrations/nvidia-v0.1.5] - 2025-02-04

### 🌀 Miscellaneous

- Client Reject Incompatible models (#1056)
- Base url validation fix and cleanup (#1349)

## [integrations/nvidia-v0.1.4] - 2025-01-08

### 🌀 Miscellaneous

- Feat: add nv-rerank-qa-mistral-4b:1 reranker (#1278)

## [integrations/nvidia-v0.1.3] - 2025-01-02

### 🚀 Features

- Improvements to NvidiaRanker and adding user input timeout (#1193)
- Add model `nvidia/llama-3.2-nv-rerankqa-1b-v2` to `_MODEL_ENDPOINT_MAP` (#1260)

### 🧹 Chores

- Fix linting/isort (#1215)


## [integrations/nvidia-v0.1.1] - 2024-11-14

### 🐛 Bug Fixes

- Fixes to NvidiaRanker (#1191)


## [integrations/nvidia-v0.1.0] - 2024-11-13

### 🚀 Features

- Update default embedding model to nvidia/nv-embedqa-e5-v5 (#1015)
- Add NVIDIA NIM ranker support (#1023)
- Raise error when attempting to embed empty documents/strings with Nvidia embedders (#1118)

### 🐛 Bug Fixes

- Lints in `nvidia-haystack` (#993)
- Missing Nvidia embedding truncate mode (#1043)

### 🚜 Refactor

- Remove deprecated Nvidia Cloud Functions backend and related code. (#803)

### 📚 Documentation

- Update Nvidia API docs (#1031)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)
- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Fix: make hosted nim default (#734)
- Fix: align tests and docs on NVIDIA_API_KEY (instead of NVIDIA_CATALOG_API_KEY) (#731)
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Raise warning for base_url ../embeddings .../completions .../rankings (#922)
- Update NvidiaGenerator docstrings (#966)
-  Add default model for NVIDIA HayStack local NIM endpoints (#915)
- Feat: add nvidia/llama-3.2-nv-rerankqa-1b-v1 to set of known ranking models (#1183)

## [integrations/nvidia-v0.0.3] - 2024-05-22

### 📚 Documentation

- Update docstrings of Nvidia integrations (#599)

### ⚙️ CI

- Add generate docs to Nvidia workflow (#603)

### 🌀 Miscellaneous

- Remove references to Python 3.7 (#601)
- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Update Nvidia integration to support new endpoints (#701)
- Docs: add missing api references (#728)
- Update _nim_backend.py (#744)

## [integrations/nvidia-v0.0.2] - 2024-03-18

### 📚 Documentation

- Disable-class-def (#556)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Add NIM backend support (#597)

## [integrations/nvidia-v0.0.1] - 2024-03-07

### 🚀 Features

- Add `NvidiaTextEmbedder`, `NvidiaDocumentEmbedder` and co. (#537)

### 🐛 Bug Fixes

- `nvidia-haystack`- Handle non-strict env var secrets correctly (#543)

### 🌀 Miscellaneous

- Add `NvidiaGenerator` (#557)
- Add missing import in NvidiaGenerator docstring (#559)

## [integrations/nvidia-v0.0.0] - 2024-03-01

### 🌀 Miscellaneous

- Add Nvidia integration scaffold (#515)

<!-- generated by git-cliff -->
