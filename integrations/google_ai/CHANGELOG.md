# Changelog

## [integrations/google_ai-v5.4.0.post1] - 2025-11-18

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)
- Fix linting google_ai (#2256)

### 🌀 Miscellaneous

- Fix: Update default model to `gemini-2.0-flash` in `GoogleAIGeminiChatGenerator` (#2324)
- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)
- Chore: archive google-ai and google-vertex integrations, migrate to google-genai (#2521)

## [integrations/google_ai-v5.4.0] - 2025-07-28

### 🧹 Chores

- Remove black (#1985)
- Google AI - suggest users to switch to Google GenAI (#2106)

## [integrations/google_ai-v5.3.0] - 2025-06-24


### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)

## [integrations/google_ai-v5.2.1] - 2025-05-28


### 🌀 Miscellaneous

- Add pins for Google AI (#1828)

## [integrations/google_ai-v5.2.0] - 2025-05-26

### 🚀 Features

- Add run_async for GoogleAIGeminiChatGenerator (#1543)

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)

### 🧪 Testing

- Change model from unsupported gemini-pro to gemini-1.5-pro (#1436)
- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)
- Upgrade gemini models (#1617)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Use Haystack logging across integrations (#1484)
- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)
- Ci: Update to `GoogleAIGeminiGenerator` tests (#1648)

## [integrations/google_ai-v5.1.0] - 2025-01-30

### 🚀 Features

- Google AI Chat Generator - add tool config + improvements (#1333)


## [integrations/google_ai-v5.0.1] - 2025-01-23

### 🚜 Refactor

- GoogleAIGeminiGenerator - make some attributes public (#1317)


## [integrations/google_ai-v5.0.0] - 2025-01-23

### 🚀 Features

- [**breaking**] Google AI - support for Tool + general refactoring (#1316)


## [integrations/google_ai-v4.1.0] - 2025-01-16

### 🧹 Chores

- Google-ai - gently handle the removal of function role (#1297)


## [integrations/google_ai-v4.0.1] - 2024-12-19

### 🐛 Bug Fixes

- Make GoogleAI Chat Generator compatible with new `ChatMessage`; small fixes to Cohere tests (#1253)


## [integrations/google_ai-v4.0.0] - 2024-12-10

### 🐛 Bug Fixes

- GoogleAI - fix the content type of `ChatMessage` `content` from function (#1241)

### 🧹 Chores

- Fix linting/isort (#1215)

### 🌀 Miscellaneous

- Chore: use class methods to create `ChatMessage` (#1222)

## [integrations/google_ai-v3.0.2] - 2024-11-19

### 🐛 Bug Fixes

- Fix missing usage metadata in GoogleAIGeminiChatGenerator (#1195)

## [integrations/google_ai-v3.0.1] - 2024-11-19


## [integrations/google_ai-v3.0.0] - 2024-11-12

### 🐛 Bug Fixes

- `GoogleAIGeminiGenerator` - remove support for tools and change output type (#1177)

### ⚙️ CI

- Adopt uv as installer (#1142)


## [integrations/google_ai-v2.0.1] - 2024-10-15

### 🚀 Features

- Add chatrole tests and meta for GeminiChatGenerators (#1090)

### 🐛 Bug Fixes

- Chat roles for model responses in chat generators (#1030)
- Make sure that streaming works with function calls - (drop python3.8) (#1137)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)


## [integrations/google_ai-v2.0.0] - 2024-08-29

### 🐛 Bug Fixes

- Remove the use of deprecated gemini models (#1032)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### 🌀 Miscellaneous

- Update GeminiGenerator docstrings (#964)
- Update GoogleChatGenerator docstrings (#962)
- Feat: enable streaming in GoogleAIGemini (#1016)

## [integrations/google_ai-v1.1.1] - 2024-07-17

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Fix Google AI tests failing (#885)

## [integrations/google_ai-v1.1.0] - 2024-06-05

### 🐛 Bug Fixes

- Handle `TypeError: Could not create Blob` in `GoogleAIGeminiChatGenerator` (#772)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Fix Google AI integration tests (#786)
- Test: Fix tests skipping for Google AI integration (#788)

## [integrations/google_ai-v1.0.0] - 2024-03-27

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)
- Disable-class-def (#556)

### 🌀 Miscellaneous

- Google AI - review docstrings (#533)
- Make tests show coverage (#566)
- Remove references to Python 3.7 (#601)
- Google Generators: change `answers` to `replies` (#626)

## [integrations/google_ai-v0.2.0] - 2024-02-15

### 🌀 Miscellaneous

- Create api docs (#354)
- Google AI - new secrets management (#424)

## [integrations/google_ai-v0.1.0] - 2024-01-25

### 🌀 Miscellaneous

- Add docstrings for `GoogleAIGeminiGenerator` and `GoogleAIGeminiChatGenerator` (#175)
- [**breaking**] Adjust import paths (#268)

## [integrations/google_ai-v0.0.1] - 2024-01-03

### 🌀 Miscellaneous

- Gemini with Makersuite (#156)
- Fix google_ai integration versioning

<!-- generated by git-cliff -->
