# Changelog

## [integrations/google_vertex-v5.3.1] - 2026-04-07

### 🐛 Bug Fixes

- Replace in-place dataclass mutations with dataclasses.replace() (#3112)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)

## [integrations/google_vertex-v5.3.0.post1] - 2025-11-18

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)
- Chore: archive google-ai and google-vertex integrations, migrate to google-genai (#2521)

## [integrations/google_vertex-v5.3.0] - 2025-07-29

### 🐛 Bug Fixes

- *(vertex-document)* Correct f-string in token-limit log + logger.info -> logger.debug (#2052)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)
- Remove black (#1985)
- Google Vertex - suggest users to switch to Google GenAI (#2105)


## [integrations/google_vertex-v5.2.1] - 2025-05-27

### 📚 Documentation

- Fix VertexAI Embedders docstrings (#1735)

### 🌀 Miscellaneous

- Add pins for Vertex (#1810)

## [integrations/google_vertex-v5.2.0] - 2025-05-02

### 🚀 Features

- Add vertexai document and text embedders (#1683)


## [integrations/google_vertex-v5.1.0] - 2025-04-08

### 🚀 Features

- Add run_async for VertexAIGeminiChatGenerator (#1574)

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)

### 🧪 Testing

- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)
- Upgrade gemini models (#1617)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)
- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)

## [integrations/google_vertex-v5.0.2] - 2025-01-30

### 🐛 Bug Fixes

- VertexAIGeminiChatGenerator - do not create messages with empty text (#1337)


## [integrations/google_vertex-v5.0.0] - 2025-01-29

### 🚀 Features

- [**breaking**] Google Vertex - support for Tool + general refactoring (#1327)


## [integrations/google_vertex-v4.0.2] - 2025-01-16

### 🌀 Miscellaneous

- Handle function role removal (#1296)

## [integrations/google_vertex-v4.0.1] - 2024-12-19

### 🐛 Bug Fixes

- Make Google Vertex Chat Generator compatible with new ChatMessage (#1255)


## [integrations/google_vertex-v4.0.0] - 2024-12-11

### 🐛 Bug Fixes

- Fix: Google Vertex - fix the content type of `ChatMessage` `content` from function (#1242)

### 🧹 Chores

- Fix linting/isort (#1215)

### 🌀 Miscellaneous

- Chore: use class methods to create `ChatMessage` (#1222)

## [integrations/google_vertex-v3.0.0] - 2024-11-14

### 🐛 Bug Fixes

- VertexAIGeminiGenerator - remove support for tools and change output type (#1180)

### 🧹 Chores

- Fix Vertex tests (#1163)


## [integrations/google_vertex-v2.2.0] - 2024-10-23

### 🐛 Bug Fixes

- Make "project-id" parameter optional during initialization (#1141)
- Make project-id optional in all VertexAI generators (#1147)

### ⚙️ CI

- Adopt uv as installer (#1142)


## [integrations/google_vertex-v2.1.0] - 2024-10-04

### 🚀 Features

- Add chatrole tests and meta for GeminiChatGenerators (#1090)
- Add custom params to VertexAIGeminiGenerator and VertexAIGeminiChatGenerator (#1100)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Update docstrings to remove vertexai preview package (#1074)
- Chore: Unpin protobuf dependency in Google Vertex integration (#1085)
- Chore: pin `google-cloud-aiplatform>=1.61` and fix tests (#1124)

## [integrations/google_vertex-v2.0.0] - 2024-09-09

### 🚀 Features

- Enable streaming for VertexAIGeminiChatGenerator (#1014)
- Add tests for VertexAIGeminiGenerator and enable streaming (#1012)

### 🐛 Bug Fixes

- Remove the use of deprecated gemini models (#1032)
- Chat roles for model responses in chat generators (#1030)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)
- Add tests for VertexAIChatGeminiGenerator and migrate from preview package in vertexai (#1042)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Ping `protobuf` to `<5.28` to fix Google Vertex Components serialization (#1050)

## [integrations/google_vertex-v1.1.0] - 2024-03-28

### 🌀 Miscellaneous

- Add pyarrow as required dependency (#629)

## [integrations/google_vertex-v1.0.0] - 2024-03-27

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)
- Review google vertex integration (#535)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Create api docs (#355)
- Make tests show coverage (#566)
- Remove references to Python 3.7 (#601)
- Google Generators: change `answers` to `replies` (#626)

## [integrations/google_vertex-v0.2.0] - 2024-01-26

### 🌀 Miscellaneous

- Refact!: change import paths (#273)

## [integrations/google_vertex-v0.1.0] - 2024-01-03

### 🐛 Bug Fixes

- The default model of VertexAIImagegenerator (#158)

### 🧹 Chores

- Replace - with _ (#114)

### 🌀 Miscellaneous

- Change metadata to meta (#152)
- Add VertexAI prefix to GeminiGenerator and GeminiChatGenerator components (#166)

<!-- generated by git-cliff -->
