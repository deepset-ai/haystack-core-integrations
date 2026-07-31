# Changelog

## [integrations/google_genai-v4.5.0] - 2026-07-03

### 🧪 Testing

- Trust test modules under Haystack 3.0's deserialization allowlist (#3537)

### 🧹 Chores

- Improve consistency of integrations folder structure (#3430)
- Support sync streaming callbacks in async contexts for Haystack 2.x/3.x compatibility (#3534)


## [integrations/google_genai-v4.4.1] - 2026-06-08

### 🐛 Bug Fixes

- Async streaming chunk indices in GoogleGenAIChatGenerator start at 0, not 1 (#3410)


## [integrations/google_genai-v4.4.0] - 2026-06-08

### 🚀 Features

- *(google_genai)* Add timeout and max_retries to embedder components (#3412)


## [integrations/google_genai-v4.3.0] - 2026-06-05

### 🚀 Features

- Google GenAI - accept str as ChatGenerator input (#3394)

### 🧪 Testing

- Remove old integration tests using deprecated model (#3381)


## [integrations/google_genai-v4.2.0] - 2026-04-29

### 🚜 Refactor

- Google GenAI embedders - adaptations for Gemini Embedding 2 general availability (#3251)


## [integrations/google_genai-v4.1.0] - 2026-04-24

### 🐛 Bug Fixes

- *(google-genai)* Include cached_content_token_count in streaming responses (#3177)


## [integrations/google_genai-v4.0.1] - 2026-04-13

### 🐛 Bug Fixes

- Fix GoogleGenAIMultimodalDocumentEmbedder input format (#3136)

### 🧪 Testing

- Google GenAI - test_live_run_with_parallel_tools should use tool_calls finish reason (#3125)


## [integrations/google_genai-v4.0.0] - 2026-04-09

### 🐛 Bug Fixes

- *(google-genai)* Remap finish_reason to tool_calls when response contains tool calls (#3102)

### 🧪 Testing

- Test compatible integrations with python 3.14; update pyproject (#3001)
- Track test coverage for all integrations (#3065)
- Add unit tests for Google GenAI integration (#3092)

### 🧹 Chores

- Add ANN type annotations to google_genai, hanlp, jina, langfuse, lara (#2990)
- Enforce ruff docstring rules (D102/D103/D205/D209/D213/D417/D419) in integrations 11-20 (#3009)


## [integrations/google_genai-v3.11.0] - 2026-03-12

### 🚀 Features

- Add support for structured output (response_format) in GoogleGenAIChatGenerator (#2946)


## [integrations/google_genai-v3.10.1] - 2026-03-11

### 🐛 Bug Fixes

- Fix wrong batching in Google GenAI Document Embedder (#2951)


## [integrations/google_genai-v3.10.0] - 2026-03-10

### 🚀 Features

- *(google-genai)* Add timeout and max_retries to chat generator (#2875)
- GoogleGenAIChatGenerator provides supported models list (#2930)
- Add GoogleGenAIMultimodalDocumentEmbedder to support gemini-embedding-2 (#2944)

### 🌀 Miscellaneous


## [integrations/google_genai-v3.8.0] - 2026-03-02

### 🚀 Features

- Use reasoning field in StreamingChunk for Google GenAI (#2900)


## [integrations/google_genai-v3.7.0] - 2026-02-27

### 🐛 Bug Fixes

- Set include_thoughts=False when thinking_budget is 0 (#2853)


## [integrations/google_genai-v3.6.0] - 2026-02-24

### 🚀 Features

- Google GenAI - add FileContent support + refactoring (#2860)


## [integrations/google_genai-v3.5.0] - 2026-02-18

### 🚀 Features

- Include token usage count in `GoogleGenAIChatGenerator` response metadata (#2851)

### 📚 Documentation

- Simplify pydoc configs (#2855)


## [integrations/google_genai-v3.4.0] - 2026-02-09

### 🧹 Chores

- Google GenAI - switch default embedding model to gemini-embedding-001 (#2823)


## [integrations/google_genai-v3.3.0] - 2026-02-03

### 🚀 Features

- Google GenAI - support images in tool results (#2809)

### 🧪 Testing

- Google GenAI - make test_live_run_with_parallel_tools more robust (#2779)


## [integrations/google_genai-v3.2.0] - 2026-01-19

### 🐛 Bug Fixes

- *(google_genai)* Use dataclass replace to avoid modifying input documents (#2762)


## [integrations/google_genai-v3.1.0] - 2026-01-13

### 🚀 Features

- Support google genai thinking_level (#2737)


## [integrations/google_genai-v3.0.0] - 2026-01-12

### 🧹 Chores

- Remove Readme API CI workflow and configs (#2573)
- Make fmt command more forgiving (#2671)
- [**breaking**] Google_genai - drop Python 3.9 and use X|Y typing (#2706)

### 🌀 Miscellaneous

- Fix: Update google genai streaming test (#2630)

## [integrations/google_genai-v2.3.0] - 2025-11-26

### ⚙️ CI

- Change pytest command (#2475)

### 🧹 Chores

- Update GoogleGenAIChatGenerator default model to gemini-2.5-flash (#2554)

### 🌀 Miscellaneous

- Adopt PEP 585 type hinting (part 3) (#2510)

## [integrations/google_genai-v2.2.0] - 2025-10-22

### 🚀 Features

- `GoogleGenAIChatGenerator` update tools param to ToolsType (#2419)

### 📚 Documentation

- Add pydoc configurations for Docusaurus (#2411)
- Fix docstrings to avoid errors in API reference generation (#2423)

### 🧪 Testing

- Replace deprecated `gemini-1.5` model with `gemini-2.0-flash` in Google GenAI test (#2323)

### ⚙️ CI

- Install dependencies in the `test` environment when testing with lowest direct dependencies and Haystack main (#2418)

### 🧹 Chores

- *(deps)* Bump actions/setup-python from 5 to 6 (#2241)


## [integrations/google_genai-v2.1.2] - 2025-08-28

### 🚀 Features

- Add Gemini "Thinking" support to GoogleGenAIChatGenerator (#2212)

### 🧹 Chores

- Standardize readmes - part 2 (#2205)


## [integrations/google_genai-v2.1.1] - 2025-08-07

### 🐛 Bug Fixes

- Add optional `aiohttp` to google-genai integration (#2160)


## [integrations/google_genai-v2.1.0] - 2025-08-06

### 🚀 Features

- Image support in GoogleGenAIChatGenerator (#2150)


## [integrations/google_genai-v2.0.0] - 2025-08-04

### 🚀 Features

- Adopt new `StreamingChunk` in Google GenAI (#2078)

### 🌀 Miscellaneous

- Fix!: `GoogleGenAIChatGenerator` - align `finish_reason` between streaming and non-streaming (#2142)

## [integrations/google_genai-v1.3.0] - 2025-07-07

### 🚀 Features

- Google GenAI - add support for Vertex API (#2058)


## [integrations/google_genai-v1.2.0] - 2025-06-27

### 🐛 Bug Fixes

- Fix Google GenAI types + add py.typed (#2005)

### 🧹 Chores

- Remove black (#1985)


## [integrations/google_genai-v1.1.0] - 2025-06-24

### 🚀 Features

- Add asynchronous embedding methods for GoogleGenAIDocumentEmbedder and GoogleGenAITextEmbedder (#1983)

### 🧹 Chores

- Fix linting for ruff 0.12.0 (#1969)


## [integrations/google_genai-v1.0.2] - 2025-06-12

### 🚀 Features

- Add GoogleAITextEmbedder and GoogleAIDocumentEmbedder components (#1783)

### 🐛 Bug Fixes

- Fix types in the Google Gen AI embedders (#1916)

### 🧹 Chores

- Add GoogleGenAIChatGenerator examples, set safety_settings (#1901)
- Align core-integrations Hatch scripts (#1898)
- Use GEMINI_API_KEY as default env var for the api key (in addition to GOOGLE_API_KEY) (#1928)

### 🌀 Miscellaneous

- Style: Update to linting to allow function calls in default arguments (#1899)

## [integrations/google_genai-v1.0.0] - 2025-06-02

### 🚀 Features

- Add Google GenAI GoogleGenAIChatGenerator (#1875)

<!-- generated by git-cliff -->
