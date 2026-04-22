# Changelog

## [integrations/langfuse-v5.0.0] - 2026-04-17

### 🚀 Features

- [**breaking**] Migrate to Langfuse Python SDK v4 (#3126)

### 🧪 Testing

- Track test coverage for all integrations (#3065)


## [integrations/langfuse-v4.1.0] - 2026-03-26

### 🚀 Features

- Log generation_kwargs in Langfuse spans (#3017)

### 🐛 Bug Fixes

- *(langfuse)* Replace MagicMock with Mock to prevent RAM exhaustion (#2543) (#2798)

### 📚 Documentation

- Simplify pydoc configs (#2855)

### 🧹 Chores

- Remove unused allow-direct-references (#2866)
- Add ANN type annotations to google_genai, hanlp, jina, langfuse, lara (#2990)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 21-30 (#3010)


## [integrations/langfuse-v4.0.1] - 2026-01-27

### 🐛 Bug Fixes

- *(langfuse)* Apply trace_id from tracing context to Langfuse traces (#2781)


## [integrations/langfuse-v4.0.0] - 2026-01-12

### 🧹 Chores

- Make fmt command more forgiving (#2671)
- [**breaking**] Langfuse - drop Python 3.9 and use X|Y typing (#2709)


## [integrations/langfuse-v3.3.1] - 2025-12-10

### 🚀 Features

- *(langfuse)* Add embedder usage metrics for langfuse (#2542)

### 🐛 Bug Fixes

- Correct token usage accounting, fix trace polling in tests (#2594)

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)


## [integrations/langfuse-v3.3.0] - 2025-11-21

### 🚀 Features

- *(langfuse)* Embedder, retriever and generator as obs. type (#2497)

### 🌀 Miscellaneous

- Enhancement: Adopt PEP 585 type hinting (part 4) (#2527)
- *(langfuse)* Log levels (#2522)

## [integrations/langfuse-v3.2.1] - 2025-11-07

### 🌀 Miscellaneous

- Chore: Upgrade langfuse dep, observation types require version>=3.3.1 (#2493)

## [integrations/langfuse-v3.2.0] - 2025-11-07

### 🐛 Bug Fixes

- Flatten usage_details dict (#2491)

### ⚙️ CI

- Change pytest command (#2475)

### 🌀 Miscellaneous

- Usage_details instead usage (#2481)
- Feat: Langfuse - add support for tool and agent observation types (#2490)

## [integrations/langfuse-v3.1.0] - 2025-10-24

### 🐛 Bug Fixes

- Langfuse - add py.typed; fix testing with lowest deps (#2458)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)

### 🧹 Chores

- Remove ruff exclude and fix linting in Langfuse integration (#2257)


## [integrations/langfuse-v3.0.0] - 2025-09-19

### 🌀 Miscellaneous

- Migrate langfuse to v3 (#2247)

## [integrations/langfuse-v2.3.0] - 2025-08-25

### 🐛 Bug Fixes

- Avoid mixed Langfuse traces in async envs (#2207)

### 🧪 Testing

- Make tests successfully run from forks (#2203)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/langfuse-v2.2.1] - 2025-08-07

### 🚀 Features

- Add AmazonBedrockChatGenerator to supported chat generators in Langfuse (#2164)


## [integrations/langfuse-v2.2.0] - 2025-07-03

### 🚀 Features

- Simpler generation spans, use Haystack's to_openai_dict_format (#2044)

### 🐛 Bug Fixes

- Properly cleanup Langfuse tracing context after pipeline run failures (#1999)


### 🧹 Chores

- Pin langfuse<3.0.0 (#1904)
- Align core-integrations Hatch scripts (#1898)
- Update md files for new hatch scripts (#1911)
- Remove black (#1985)


## [integrations/langfuse-v2.0.1] - 2025-06-02

### 🚀 Features

- Use Langfuse local to_openai_dict_format function to serialize messages (#1885)

### 🌀 Miscellaneous

- Feat: Add detailed tracing for GoogleGenAIChatGenerator (#1887)

## [integrations/langfuse-v2.0.0] - 2025-05-27

### 🌀 Miscellaneous

- Add Langfuse pins (#1837)

## [integrations/langfuse-v1.1.2] - 2025-05-16

### 🌀 Miscellaneous

- Test: Update how test skipping works in Langfuse (#1756)
- `OllamaGenerator` support in Langfuse (#1759)

## [integrations/langfuse-v1.1.1] - 2025-05-08

### 🚀 Features

- Add more options to LangfuseConnector (#1657)


## [integrations/langfuse-v1.1.0] - 2025-05-06

### 🚀 Features

- Enhance Langfuse ToolInvoker span naming (#1682)


## [integrations/langfuse-v1.0.1] - 2025-04-28

### 🐛 Bug Fixes

- Langfuse - remove warning "Creating a new trace without a parent span is not recommended"


## [integrations/langfuse-v1.0.0] - 2025-04-11

### 🐛 Bug Fixes

- [**breaking**] Make sure to JSON serialize objects before setting content tags (#1627)


## [integrations/langfuse-v0.10.1] - 2025-04-11

### 🚀 Features

- Adapt Ollama metadata to OpenAI format; support Ollama in Langfuse (#1577)
- Unify traces of sub-pipelines within pipelines with Langfuse (#1624)



## [integrations/langfuse-v0.9.0] - 2025-04-04

### 🚀 Features

- Correctly set pipeline input and output by updating DefaultHandler (#1589)
- Add trace_id to output of LangfuseConnector (#1587)

### 📚 Documentation

- Fixing typo in Langfuse API docstring

### 🧪 Testing

- Langfuse - wait before retrieving the trace in `test_custom_span_handler` (#1359)
- Langfuse - make polling more robust in tests (#1375)

### ⚙️ CI

- Review testing workflows (#1541)

### 🧹 Chores

- Remove Python 3.8 support (#1421)


## [integrations/langfuse-v0.8.0] - 2025-01-28

### 🚀 Features

- Add custom Langfuse span handling support (#1313)


## [integrations/langfuse-v0.7.0] - 2025-01-21

### 🚀 Features

- LangfuseConnector - add httpx.Client init param (#1308)

### 🐛 Bug Fixes

- End langfuse generation spans properly (#1301)


## [integrations/langfuse-v0.6.4] - 2025-01-17

### 🚀 Features

- Add LangfuseConnector secure key management and serialization  (#1287)


## [integrations/langfuse-v0.6.3] - 2025-01-15

### 🌀 Miscellaneous

- Chore: Langfuse - pin `haystack-ai>=2.9.0` and simplify message conversion (#1292)

## [integrations/langfuse-v0.6.2] - 2025-01-02

### 🚀 Features

- Warn if LangfuseTracer initialized without tracing enabled (#1231)

### 🧹 Chores

- Use text instead of content for ChatMessage in Llama.cpp, Langfuse and Mistral (#1238)

### 🌀 Miscellaneous

- Chore: Fix tracing_context_var lint errors (#1220)
- Fix messages conversion to OpenAI format (#1272)

## [integrations/langfuse-v0.6.0] - 2024-11-18

### 🚀 Features

- Add support for ttft (#1161)

### ⚙️ CI

- Adopt uv as installer (#1142)

### 🌀 Miscellaneous

- Fixed TypeError in LangfuseTrace (#1184)

## [integrations/langfuse-v0.5.0] - 2024-10-01

### 🧹 Chores

- Update ruff linting scripts and settings (#1105)

### 🌀 Miscellaneous

- Fix: Add delay to flush the Langfuse traces (#1091)
- Add invocation_context to identify traces (#1089)

## [integrations/langfuse-v0.4.0] - 2024-09-17

### 🚀 Features

- Langfuse - support generation span for more LLMs (#1087)

### 🚜 Refactor

- Remove usage of deprecated `ChatMessage.to_openai_format` (#1001)

### 📚 Documentation

- Add link to langfuse in LangfuseConnector (#981)

### 🧪 Testing

- Do not retry tests in `hatch run test` command (#954)

### ⚙️ CI

- Retry tests to reduce flakyness (#836)

### 🧹 Chores

- `Langfuse` - replace DynamicChatPromptBuilder with ChatPromptBuilder (#925)
- Remove all `DynamicChatPromptBuilder` references in Langfuse integration (#931)

### 🌀 Miscellaneous

- Ci: install `pytest-rerunfailures` where needed; add retry config to `test-cov` script (#845)
- Chore: Update Langfuse README to avoid common initialization issues (#952)
- Chore: langfuse - ruff update, don't ruff tests (#992)

## [integrations/langfuse-v0.2.0] - 2024-06-18

### 🌀 Miscellaneous

- Feat: add support for Azure generators (#815)

## [integrations/langfuse-v0.1.0] - 2024-06-13

### 🚀 Features

- Langfuse integration (#686)

### 🐛 Bug Fixes

- Performance optimizations and value error when streaming in langfuse (#798)

### 🧹 Chores

- Use ChatMessage to_openai_format, update unit tests, pydocs (#725)

### 🌀 Miscellaneous

- Chore: change the pydoc renderer class (#718)
- Docs: add missing api references (#728)

<!-- generated by git-cliff -->
