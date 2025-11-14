# Changelog

## [integrations/amazon_bedrock-v5.2.0] - 2025-10-22

### 🚀 Features

- `AmazonBedrockChatGenerator` update tools param to ToolsType (#2415)

## [integrations/amazon_bedrock-v5.1.0] - 2025-09-29

### 🚀 Features

- S3Downloader - add `s3_key_generation_function` param to customize S3 key generation (#2343)

## [integrations/amazon_bedrock-v5.0.0] - 2025-09-22

### 🧹 Chores

- [**breaking**] Remove deprecated `BedrockRanker` (use `AmazonBedrockRanker` instead) (#2287)

## [integrations/amazon_bedrock-v4.2.0] - 2025-09-19

### 🚀 Features

- Add a new `S3Downloader` component (#2192)

## [integrations/amazon_bedrock-v4.1.0] - 2025-09-19

### 🚀 Features

- Support AWS Bedrock Guardrails in `AmazonBedrockChatGenerator` (#2284)

### 🧹 Chores

- Bedrock - remove unused `stop_words` init parameter (#2275)

### 🌀 Miscellaneous

- Chore: Fix linting aws bedrock (#2249)

## [integrations/amazon_bedrock-v4.0.0] - 2025-08-29

### 🚀 Features

- [**breaking**] Update AmazonBedrockChatGenerator to use the new fields in `StreamingChunk` (#2216)
- [**breaking**] Use `ReasoningContent` to store reasoning content in `ChatMessage` instead of `ChatMessage.meta` (#2226)


### 🧹 Chores

- Standardize readmes - part 2 (#2205)

## [integrations/amazon_bedrock-v3.11.0] - 2025-08-21

### 🚀 Features

- Add `AmazonBedrockDocumentImageEmbedder` component (#2185)

### 🧹 Chores

- Add framework name into UserAgent header for bedrock integration (#2168)
- Standardize readmes - part 1 (#2202)


## [integrations/amazon_bedrock-v3.10.0] - 2025-08-06

### 🚀 Features

- Add `reasoning_contents` to meta of BedrockChatGenerator to support normal thinking and redacted thinking (#2153)

### 🌀 Miscellaneous

- Adds support for thinking when using Claude with BedrockChatGenerator in multi-turn conversations (#2094)

## [integrations/amazon_bedrock-v3.9.1] - 2025-07-31

### 🐛 Bug Fixes

- `AmazonBedrockChatGenerator` - fix bug with streaming + tool calls with no arguments (#2121)


## [integrations/amazon_bedrock-v3.9.0] - 2025-07-29

### 🚀 Features

- Amazon Bedrock - multimodal support (#2114)


## [integrations/amazon_bedrock-v3.8.0] - 2025-07-04

### 🚀 Features

- Pass component_info to StreamingChunk in AmazonBedrockChatGenerator (#2042)

### 🧹 Chores

- Remove black (#1985)
- Improve typing for select_streaming_callback (#2008)


## [integrations/amazon_bedrock-v3.7.0] - 2025-06-11

### 🐛 Bug Fixes

- Fix Bedrock types + add py.typed (#1912)
- Bedrock - do not assume connection issues in case of ClientError (#1921)

### ⚙️ CI

- Bedrock - improve worfklow; skip tests from CI (#1773)

### 🧹 Chores

- Update bedrock_ranker_example.py (#1740)
- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)


## [integrations/amazon_bedrock-v3.6.2] - 2025-05-13

### 🧹 Chores

- Extend error message for unknown model family in AmazonBedrockGenerator (#1733)


## [integrations/amazon_bedrock-v3.6.1] - 2025-05-13

### 🚜 Refactor

- Add AmazonBedrockRanker and keep BedrockRanker as alias (#1732)


## [integrations/amazon_bedrock-v3.6.0] - 2025-05-09

### 🚀 Features

- Add Toolset support to AmazonBedrockChatGenerator (#1707)

### 🐛 Bug Fixes

- Fix BedrockChatGenerator to be able to handle multiple tool calls from one response (#1711)

### 🚜 Refactor

- Refactor tests of AmazonBedrock Integration (#1671)


### 🧹 Chores

- Update ChatGenerators with `deserialize_tools_or_toolset_inplace` (#1623)

### 🌀 Miscellaneous

- Chore: Update docstrings, add more types, remove haystack 2.12 check and pin >=2.13.1 (#1706)

## [integrations/amazon_bedrock-v3.5.0] - 2025-04-03

### 🌀 Miscellaneous

- Test: Update tests to check for `outputs_to_string` in Tool when running haystack-ai>=2.12 (#1585)
- Fix: amazon-bedrock-haystack remove init files to make them namespace packages (#1593)

## [integrations/amazon_bedrock-v3.4.0] - 2025-04-02

### 🚀 Features

- AmazonBedrockGenerator - return response metadata (#1584)


### 🧪 Testing

- Update tool serialization in tests to include `inputs_from_state` and `outputs_to_state` (#1581)

### 🌀 Miscellaneous

- Improve streaming_callback type and use async version in run_async (#1582)

## [integrations/amazon_bedrock-v3.3.0] - 2025-03-18

### 🚀 Features

- Support new Amazon Bedrock rerank API (#1529)

### ⚙️ CI

- Review testing workflows (#1541)


## [integrations/amazon_bedrock-v3.2.1] - 2025-03-13

### 🐛 Bug Fixes

- Update serialization/deserialization tests to add new parameter `connection_type_validation` (#1464)

### 🚜 Refactor

- Update AWS Bedrock with improved docstrings and warning message (#1532)


### 🧹 Chores

- Use Haystack logging across integrations (#1484)

## [integrations/amazon_bedrock-v3.2.0] - 2025-02-27

### 🚀 Features

- Adding async to `AmazonChatGenerator` (#1445)


## [integrations/amazon_bedrock-v3.1.1] - 2025-02-26

### 🐛 Bug Fixes

- Avoid thinking end tag on first content block (#1442)


## [integrations/amazon_bedrock-v3.1.0] - 2025-02-26

### 🚀 Features

- Support thinking parameter for Claude 3.7 in AmazonBedrockGenerator (#1439)

### 🧹 Chores

- Remove Python 3.8 support (#1421)

### 🌀 Miscellaneous

- Chore: remove `jsonschema` dependency from `default` environment (#1368)

## [integrations/amazon_bedrock-v3.0.1] - 2025-01-30

### 🐛 Bug Fixes

- Chore: Bedrock - manually fix changelog (#1319)
- Fix error when empty document list (#1325)


## [integrations/amazon_bedrock-v3.0.0] - 2025-01-23

### 🚀 Features

- *(AWS Bedrock)* Add Cohere Reranker (#1291)
- AmazonBedrockChatGenerator - add tools support (#1304)

### 🚜 Refactor

- [**breaking**] AmazonBedrockGenerator - remove truncation  (#1314)


## [integrations/amazon_bedrock-v2.1.3] - 2025-01-21

### 🧹 Chores

- Bedrock - pin `transformers!=4.48.*` (#1306)


## [integrations/amazon_bedrock-v2.1.2] - 2025-01-20

### 🌀 Miscellaneous

- Fix: Bedrock - pin `transformers!=4.48.0` (#1302)

## [integrations/amazon_bedrock-v2.1.1] - 2024-12-18

### 🐛 Bug Fixes

- Fixes to Bedrock Chat Generator for compatibility with the new ChatMessage (#1250)


## [integrations/amazon_bedrock-v2.1.0] - 2024-12-11

### 🚀 Features

- Support model_arn in AmazonBedrockGenerator (#1244)


## [integrations/amazon_bedrock-v2.0.0] - 2024-12-10

### 🚀 Features

- Update AmazonBedrockChatGenerator to use Converse API (BREAKING CHANGE) (#1219)


## [integrations/amazon_bedrock-v1.1.1] - 2024-12-03

### 🐛 Bug Fixes

- AmazonBedrockChatGenerator with Claude raises moot warning for stream… (#1205)
- Allow passing boto3 config to all  AWS Bedrock classes (#1166)

### 🧹 Chores

- Fix linting/isort (#1215)

### 🌀 Miscellaneous

- Chore: use class methods to create `ChatMessage` (#1222)

## [integrations/amazon_bedrock-v1.1.0] - 2024-10-23

### 🚜 Refactor

- Avoid downloading tokenizer if `truncate` is `False` (#1152)

### ⚙️ CI

- Adopt uv as installer (#1142)


## [integrations/amazon_bedrock-v1.0.5] - 2024-10-17

### 🚀 Features

- Add prefixes to supported model patterns to allow cross region model ids (#1127)


## [integrations/amazon_bedrock-v1.0.4] - 2024-10-16

### 🐛 Bug Fixes

- Avoid bedrock read timeout (add boto3_config param) (#1135)


## [integrations/amazon_bedrock-v1.0.3] - 2024-10-04

### 🐛 Bug Fixes

- *(Bedrock)* Allow tools kwargs for AWS Bedrock Claude model (#976)
- Chat roles for model responses in chat generators (#1030)

### 🚜 Refactor

- Remove usage of deprecated `ChatMessage.to_openai_format` (#1007)

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Modify regex to allow cross-region inference in bedrock  (#1120)

## [integrations/amazon_bedrock-v1.0.1] - 2024-08-19

### 🚀 Features

- Make truncation optional for bedrock chat generator (#967)

### 🐛 Bug Fixes

- Normalising ChatGenerators output (#973)


## [integrations/amazon_bedrock-v1.0.0] - 2024-08-12

### 🚜 Refactor

- Change meta data fields (#911)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)


## [integrations/amazon_bedrock-v0.10.0] - 2024-08-12

### 🐛 Bug Fixes

- Support streaming_callback param in amazon bedrock generators (#927)

### 🌀 Miscellaneous

- Update AmazonBedrockChatGenerator docstrings (#949)
- Update AmazonBedrockGenerator docstrings (#956)

## [integrations/amazon_bedrock-v0.9.3] - 2024-07-17

### 🚀 Features

- Use non-gated tokenizer as fallback for mistral in AmazonBedrockChatGenerator (#843)
- Made truncation optional for BedrockGenerator (#833)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- Update ruff invocation to include check parameter (#853)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Add meta deprecration warning (#910)

## [integrations/amazon_bedrock-v0.9.0] - 2024-06-14

### 🚀 Features

- Support Claude v3, Llama3 and Command R models on Amazon Bedrock (#809)

### 🧪 Testing

- Amazon Bedrock - skip integration tests from forks (#801)

## [integrations/amazon_bedrock-v0.8.0] - 2024-05-23

### 🐛 Bug Fixes

- Max_tokens typo in Mistral Chat (#740)

### 🌀 Miscellaneous

- Chore: change the pydoc renderer class (#718)
- Adding support of "amazon.titan-embed-text-v2:0" (#735)

## [integrations/amazon_bedrock-v0.7.1] - 2024-04-24

### 🌀 Miscellaneous

- Chore: add license classifiers (#680)
- Fix: Fix streaming_callback serialization in AmazonBedrockChatGenerator (#685)

## [integrations/amazon_bedrock-v0.7.0] - 2024-04-16

### 🚀 Features

- Add Mistral Amazon Bedrock support  (#632)

### 📚 Documentation

- Disable-class-def (#556)

### 🌀 Miscellaneous

- Remove references to Python 3.7 (#601)
- [Bedrock] Added Amazon Bedrock examples (#635)

## [integrations/amazon_bedrock-v0.6.0] - 2024-03-11

### 🚀 Features

- AmazonBedrockChatGenerator - migrate Anthropic chat models to use messaging API (#545)

### 📚 Documentation

- Small consistency improvements (#536)
- Review integrations bedrock (#550)

### 🌀 Miscellaneous

- Docs updates + two additional unit tests (#513)

## [integrations/amazon_bedrock-v0.5.1] - 2024-02-22

### 🚀 Features

- Add Amazon Bedrock chat model support (#333)

### 🐛 Bug Fixes

- Fix order of API docs (#447)

### 📚 Documentation

- Update category slug (#442)

### 🧹 Chores

- Update Amazon Bedrock integration to use new generic callable (de)serializers for their callback handlers (#452)
- Use `serialize_callable` instead of `serialize_callback_handler` in Bedrock (#459)

### 🌀 Miscellaneous

- Amazon bedrock: generate api docs (#326)
- Adopt Secret to Amazon Bedrock (#416)
- Bedrock - remove `supports` method (#456)
- Bedrock refactoring (#455)
- Bedrock Text Embedder (#466)
- Bedrock Document Embedder (#468)

## [integrations/amazon_bedrock-v0.3.0] - 2024-01-30

### 🧹 Chores

- [**breaking**] Rename `model_name` to `model` in `AmazonBedrockGenerator` (#220)
- Amazon Bedrock subproject refactoring (#293)
- Adjust amazon bedrock helper classes names (#297)

## [integrations/amazon_bedrock-v0.1.0] - 2024-01-03

### 🌀 Miscellaneous

- [Amazon Bedrock] Add AmazonBedrockGenerator (#153)

<!-- generated by git-cliff -->
