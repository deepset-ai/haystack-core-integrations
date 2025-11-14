# Changelog

## [integrations/anthropic-v4.5.1]

### 🧹 Chores

- Adopt PEP 585 type hinting


## [integrations/anthropic-v4.5.0] - 2025-10-22

### 🚀 Features

- `AnthropicChatGenerator` update tools param to ToolsType  (#2417)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

## [integrations/anthropic-v4.4.0] - 2025-09-15

### 🚀 Features

- `AnthropicChatGenerator`- refine reasoning support (#2230)


### 🧹 Chores

- Standardize readmes - part 1 (#2202)
- Standardize readmes - part 2 (#2205)
- Fix linting in Anthropic (#2251)

## [integrations/anthropic-v4.3.0] - 2025-08-18

### 🚀 Features

- Add multimodal support to AnthropicChatGenerator (#2186)


## [integrations/anthropic-v4.2.1] - 2025-08-11

### 🐛 Bug Fixes

- Add `input_tokens` in usage of Anthropic messages (#2173)


## [integrations/anthropic-v4.2.0] - 2025-08-07

### 🐛 Bug Fixes

- Error in `ToolCallDelta.index` for parallel tool calling (#2154)


## [integrations/anthropic-v4.1.0] - 2025-08-05

### 🚀 Features

- Adopt new `StreamingChunk` in Anthropic (#2096)

### 🌀 Miscellaneous

- Fix/prompt caching support (#2051)

## [integrations/anthropic-v3.1.0] - 2025-07-04

### 🚀 Features

- Pass `component_info`to `StreamingChunk` in `AnthropicChatGenerator` (#2056)


## [integrations/anthropic-v3.0.0] - 2025-06-30

### 🚀 Features

- [**breaking**] Anthopic model update to `claude-sonnet-4-20250514` (#2022)

### 🐛 Bug Fixes

- Anthropic reports input tokens in first message delta (#2001)

### 🧹 Chores

- Improve typing for select_streaming_callback (#2008)


## [integrations/anthropic-v2.7.0] - 2025-06-25

### 🚀 Features

- Add `timeout`, `max_retries` to all generators and async support to `AnthropicVertexChatGenerator` (#1952)

### 🧹 Chores

- Remove black (#1985)


## [integrations/anthropic-v2.6.1] - 2025-06-16

### 🌀 Miscellaneous

- Fix: `AnthropicChatGenerator` now properly can call tools that have no arguments when streaming is enabled (#1950)

## [integrations/anthropic-v2.6.0] - 2025-06-13

### 🐛 Bug Fixes

- Fix Anthropic types + add py.typed (#1940)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/anthropic-v2.5.0] - 2025-05-28

### 🚀 Features

- AnthropicChatGenerator - add Toolset support (#1787)


## [integrations/anthropic-v2.4.2] - 2025-05-27

### 📚 Documentation

- ChatMessage examples (#1752)

### 🧪 Testing

- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)
- Anthropic - add `server_tool_use` to `usage` (#1717)
- Add service_tier to test_convert_anthropic_chunk_to_streaming_chunk (#1778)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)

### 🌀 Miscellaneous

- Improve `streaming_callback` type and use async version in `run_async` (#1579)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)
- Add pins for Anthropic (#1811)

## [integrations/anthropic-v2.4.0] - 2025-03-06

### 🚀 Features

- Support extended thinking mode with AnthropicGenerator (#1455)


## [integrations/anthropic-v2.3.0] - 2025-03-05

### 🚀 Features

- Support thinking parameter in Anthropic generators (#1473)


## [integrations/anthropic-v2.2.1] - 2025-03-05

### 🐛 Bug Fixes

- Apply ddtrace workaround to chat generator (#1470)


## [integrations/anthropic-v2.2.0] - 2025-03-05

### 🚀 Features

- Adding async run to `AnthropicChatGenerator` (#1461)

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)


## [integrations/anthropic-v2.1.0] - 2025-03-03

### 🐛 Bug Fixes

- Workaround for Anthropic streaming with ddtrace (#1454) (#1456)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Test: remove tests involving serialization of lambdas (#1281)
- Test: remove more tests involving serialization of lambdas (#1285)
- Feat: Anthropic - support for Tools + refactoring (#1300)
- Chore: remove `jsonschema` dependency from `default` environment (#1368)

## [integrations/anthropic-v1.2.1] - 2024-12-18

### 🐛 Bug Fixes

- Make Anthropic compatible with new `ChatMessage`; fix prompt caching tests (#1252)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)
- Fix linting/isort (#1215)

### 🌀 Miscellaneous

- Add AnthropicVertexChatGenerator component (#1192)
- Docs: add AnthropicVertexChatGenerator to pydoc (#1221)
- Chore: use `text` instead of `content` for `ChatMessage` in Cohere and Anthropic (#1237)

## [integrations/anthropic-v1.1.0] - 2024-09-20

### 🚀 Features

- Add Anthropic prompt caching support, add example (#1006)

### 🌀 Miscellaneous

- Chore: Update Anthropic example, use ChatPromptBuilder properly (#978)

## [integrations/anthropic-v1.0.0] - 2024-08-12

### 🐛 Bug Fixes

- Replace DynamicChatPromptBuilder with ChatPromptBuilder (#940)

### 🚜 Refactor

- Change meta data fields (#911)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)


## [integrations/anthropic-v0.4.1] - 2024-07-17

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Add meta deprecration warning (#910)

## [integrations/anthropic-v0.4.0] - 2024-06-21

### 🚀 Features

- Update Anthropic/Cohere for tools use (#790)
- Update Anthropic default models, pydocs (#839)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🌀 Miscellaneous

- Remove references to Python 3.7 (#601)
- Chore: add license classifiers (#680)
- Chore: change the pydoc renderer class (#718)
- Docs: add missing api references (#728)

## [integrations/anthropic-v0.2.0] - 2024-03-15

### 🌀 Miscellaneous

- Docs: Replace amazon-bedrock with anthropic in readme (#584)
- Chore: Use the correct sonnet model name (#587)

## [integrations/anthropic-v0.1.0] - 2024-03-15

### 🚀 Features

- Add AnthropicGenerator and AnthropicChatGenerator (#573)

<!-- generated by git-cliff -->
