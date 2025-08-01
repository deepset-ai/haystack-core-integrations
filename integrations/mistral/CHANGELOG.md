# Changelog

## [integrations/mistral-v0.3.0] - 2025-06-27

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Remove black (#1985)
- Mistral - improve type checking + add py.typed (#2009)

### 🌀 Miscellaneous

- Test: Remove `test_check_abnormal_completions` - already tested in Haystack (#1842)

## [integrations/mistral-v0.2.0] - 2025-05-23

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)
- Bring Mistral integration up to date with changes made to OpenAIChatGenerator and OpenAI Embedders (#1774)

### 📚 Documentation

- ChatMessage examples (#1752)

### 🧪 Testing

- Mistral/Stackit - add pytz test dependency (#1504)
- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)
- Remove not needed tool_choice (#1651)
- Add async tests for MistralChatGenerator (#1674)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Mark test as integration (#1422)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)
- Fix: Fix Mistral sede tests (#1666)

## [integrations/mistral-v0.1.1] - 2025-02-12

### 🧹 Chores

- Remove robust tool chunk stream handling - added in Haystack 2.10 (#1367)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)

## [integrations/mistral-v0.1.0] - 2025-02-10

### 🐛 Bug Fixes

- Replace DynamicChatPromptBuilder with ChatPromptBuilder (#940)
- Lints in `mistral-haystack` (#994)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)
- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)
- Update ruff linting scripts and settings (#1105)
- Use text instead of content for ChatMessage in Llama.cpp, Langfuse and Mistral (#1238)
- Mistral - pin haystack-ai>=2.9.0 and simplify test (#1293)

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Test: adapt Mistral to OpenAI refactoring (#1271)
- Adding tools to MistralChatGenerator - based on OpenAIChatGenerator (#1358)

## [integrations/mistral-v0.0.2] - 2024-04-09

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Fix `haystack-ai` pins (#649)

### 🚜 Refactor

- Mistral: refactor tests (#487)

### 📚 Documentation

- Update category slug (#442)
- Disable-class-def (#556)

### 🧹 Chores

- Update docstrings (#497)

### 🌀 Miscellaneous

- Make tests show coverage (#566)
- Remove references to Python 3.7 (#601)

## [integrations/mistral-v0.0.1] - 2024-02-16

### 🌀 Miscellaneous

- Adding mistral generator and embedders (#409)

<!-- generated by git-cliff -->
