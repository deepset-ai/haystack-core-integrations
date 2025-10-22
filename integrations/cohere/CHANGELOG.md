# Changelog

## [integrations/cohere-v6.1.0] - 2025-10-22

### 🚀 Features

- `CohereChatGenerator` update tools param to ToolsType (#2416)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)


## [integrations/cohere-v6.0.0] - 2025-09-17

### 🧹 Chores

- Fix linting cohere (#2252)
- Revert default model to `command-r-08-2024` (#2282)


## [integrations/cohere-v5.5.0] - 2025-08-29

### 🚀 Features

- Add image support to Cohere Chat Generator (#2223)


## [integrations/cohere-v5.4.0] - 2025-08-26

### 🚀 Features

- Add `StreamingChunk` to Cohere ChatGenerator  (#2157)

### 🧪 Testing

- Make tests successfully run from forks (#2203)

### 🧹 Chores

- Standardize readmes - part 1 (#2202)
- Standardize readmes - part 2 (#2205)

### 🌀 Miscellaneous

- Docs: add Cohere document image embedder to pydocs (#2196)

## [integrations/cohere-v5.3.0] - 2025-08-18

### 🚀 Features

- Add Cohere Image Document Embedder (#2190)

### 🧹 Chores

- CohereChatGenerator - make finish_reason and usage handling more robust (#2149)


## [integrations/cohere-v5.2.0] - 2025-07-10

### 🐛 Bug Fixes

- Update Cohere to client 5.16.0 (#2086)


## [integrations/cohere-v5.1.1] - 2025-07-10

### 🌀 Miscellaneous

- Fix: Pin cohere to less than 5.16 (#2081)

## [integrations/cohere-v5.1.0] - 2025-07-02

### 🚀 Features

- Cohere - add support for ComponentInfo + refactor streaming (#2043)

### 🧹 Chores

- Remove black (#1985)
- Improve typing for select_streaming_callback (#2008)


## [integrations/cohere-v5.0.0] - 2025-06-17

### 🚀 Features

- Add run_async support for CohereTextEmbedder (#1873)
- Add async support for CohereDocumentEmbedder (#1876)

### 🚜 Refactor

- [**breaking**] Cohere: remove `use_async_client` parameter; fix types and add py.typed  (#1946)

### 🧹 Chores

- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/cohere-v4.2.1] - 2025-05-27

### 🌀 Miscellaneous

- Add pins for Cohere (#1817)

## [integrations/cohere-v4.2.0] - 2025-05-26

### 🚀 Features

- Add run_async for CohereChatGenerator (#1689)


## [integrations/cohere-v4.1.0] - 2025-05-05

### 🚀 Features

- Add Toolset support to CohereChatGenerator (#1700)

### 🧹 Chores

- Upgrade default model to command-r-08-2024 (#1691)


## [integrations/cohere-v4.0.0] - 2025-05-02

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)

### 🧪 Testing

- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)
- Use Haystack logging across integrations (#1484)
- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)
- [**breaking**] `CohereRanker` - change default model and remove `max_chunks_per_doc parameter` (#1687)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)
- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)

## [integrations/cohere-v3.0.0] - 2025-01-28

### 🚀 Features

- CohereChatGenerator - add tools support  (#1318)

### 🚜 Refactor

- Migrate Cohere to V2 (#1321)


## [integrations/cohere-v2.0.2] - 2025-01-15

### 🐛 Bug Fixes

- Make GoogleAI Chat Generator compatible with new `ChatMessage`; small fixes to Cohere tests (#1253)
- Cohere - fix chat message creation (#1289)

### 🌀 Miscellaneous

- Test: remove tests involving serialization of lambdas (#1281)
- Test: remove more tests involving serialization of lambdas (#1285)

## [integrations/cohere-v2.0.1] - 2024-12-09

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)
- Fix linting/isort (#1215)

### 🌀 Miscellaneous

- Chore: use class methods to create `ChatMessage` (#1222)
- Chore: use `text` instead of `content` for `ChatMessage` in Cohere and Anthropic (#1237)

## [integrations/cohere-v2.0.0] - 2024-09-16

### 🚀 Features

- Update Anthropic/Cohere for tools use (#790)
- Update Cohere default LLMs, add examples and update unit tests (#838)
- Cohere LLM - adjust token counting meta to match OpenAI format (#1086)

### 🐛 Bug Fixes

- Lints in `cohere-haystack` (#995)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Update CohereChatGenerator docstrings (#958)
- Update CohereGenerator docstrings (#960)

## [integrations/cohere-v1.1.1] - 2024-06-12

### 🌀 Miscellaneous

- Chore: `CohereGenerator` - remove warning about `generate` API (#805)

## [integrations/cohere-v1.1.0] - 2024-05-24

### 🐛 Bug Fixes

- Remove support for generate API (#755)

### 🌀 Miscellaneous

- Chore: change the pydoc renderer class (#718)

## [integrations/cohere-v1.0.0] - 2024-05-03

### 🌀 Miscellaneous

- Follow up: update Cohere integration to use Cohere SDK v5 (#711)

## [integrations/cohere-v0.7.0] - 2024-05-02

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Update Cohere integration to use Cohere SDK v5 (#702)

## [integrations/cohere-v0.6.0] - 2024-04-08

### 🚀 Features

- Add Cohere ranker (#643)

## [integrations/cohere-v0.5.0] - 2024-03-29

### 🌀 Miscellaneous

- Add the Cohere client name to cohere requests  (#362)

## [integrations/cohere-v0.4.1] - 2024-03-21

### 🐛 Bug Fixes

- Fix order of API docs (#447)
- Fix tests (#561)

### 📚 Documentation

- Update category slug (#442)
- Review cohere integration (#500)
- Small consistency improvements (#536)
- Disable-class-def (#556)

### 🧹 Chores

- Update Cohere integration to use new generic callable (de)serializers for their callback handlers (#453)
- Use `serialize_callable` instead of `serialize_callback_handler` in Cohere  (#460)

### 🌀 Miscellaneous

- Choere - remove matching error message from tests (#419)
- Fix linting (#509)
- Make tests show coverage (#566)
- Refactor tests (#574)
- Test: relax test constraints (#591)
- Remove references to Python 3.7 (#601)
- Fix: Pin cohere version (#609)

## [integrations/cohere-v0.4.0] - 2024-02-12

### 🐛 Bug Fixes

- Fix Cohere tests (#337)
- Cohere inconsistent embeddings and documents lengths (#284)

### 🚜 Refactor

- [**breaking**] Use `Secret` for API keys in Cohere components (#386)

### 🧪 Testing

- Fix failing `TestCohereChatGenerator.test_from_dict_fail_wo_env_var` test (#393)

### 🌀 Miscellaneous

- Cohere: generate api docs (#321)
- Fix: update to latest haystack-ai version (#348)

## [integrations/cohere-v0.3.0] - 2024-01-25

### 🐛 Bug Fixes

- Fix project URLs (#96)
- Cohere namespace reorg (#271)

### 🚜 Refactor

- Use `hatch_vcs` to manage integrations versioning (#103)

### 🧹 Chores

- [**breaking**] Rename `model_name` to `model` in the Cohere integration (#222)
- Cohere namespace change (#247)

### 🌀 Miscellaneous

- Cohere: remove unused constant (#91)
- Change default 'input_type' for CohereTextEmbedder (#99)
- Change metadata to meta (#152)
- Add cohere chat generator (#88)
- Optimize API key reading (#162)
- Cohere - change metadata to meta (#178)

## [integrations/cohere-v0.2.0] - 2023-12-11

### 🚀 Features

- Add support for V3 Embed models to CohereEmbedders (#89)

### 🌀 Miscellaneous

- Cohere: increase version to prepare release (#92)

## [integrations/cohere-v0.1.1] - 2023-12-07

### 🌀 Miscellaneous

- [cohere] Add text and document embedders (#80)
- [cohere] fix cohere pypi version badge and add Embedder note (#86)

## [integrations/cohere-v0.0.1] - 2023-12-04

### 🌀 Miscellaneous

- Add `cohere_haystack` integration package (#75)

<!-- generated by git-cliff -->
