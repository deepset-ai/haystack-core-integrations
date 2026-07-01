# edenai-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/edenai-haystack.svg)](https://pypi.org/project/edenai-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/edenai-haystack.svg)](https://pypi.org/project/edenai-haystack)

[Eden AI](https://www.edenai.co/) is a unified, OpenAI-compatible API that gives access to 500+ AI
models from many providers (OpenAI, Anthropic, Mistral, Google, Cohere, and more) through a single
API key, with built-in provider fallback and **EU data residency**. This makes it a convenient,
sovereignty-friendly gateway for building LLM and RAG applications with Haystack.

- [Integration page](https://haystack.deepset.ai/integrations/edenai)
- [Eden AI models catalog](https://www.edenai.co/models)
- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/edenai/CHANGELOG.md)

## Installation

```console
pip install edenai-haystack
```

## Authentication

Create an API key in your [Eden AI account](https://app.edenai.run/) and expose it as an
environment variable:

```bash
export EDENAI_API_KEY="<your-eden-ai-api-key>"
```

You can also pass it explicitly with Haystack's `Secret` API:

```python
from haystack.utils import Secret
from haystack_integrations.components.generators.edenai import EdenAIChatGenerator

generator = EdenAIChatGenerator(api_key=Secret.from_env_var("EDENAI_API_KEY"))
```

## Model naming

Models use Eden AI's `provider/model` convention, for example:

| Model string                     | Provider  | Notes                          |
|----------------------------------|-----------|--------------------------------|
| `openai/gpt-4o-mini`             | OpenAI    | Default model                  |
| `mistral/mistral-large-latest`   | Mistral   | EU-hosted                      |
| `google/gemini-2.5-flash`        | Google    |                                |
| `cohere/command-r-plus`          | Cohere    |                                |

Browse the full, always-up-to-date list in the [Eden AI models catalog](https://www.edenai.co/models).
If a model is unavailable on your account, Eden AI returns a clear `400 Model(s) not found or inactive`
error, surfaced directly by the component.

## Usage

### Basic chat

```python
from haystack_integrations.components.generators.edenai import EdenAIChatGenerator
from haystack.dataclasses import ChatMessage

generator = EdenAIChatGenerator(model="mistral/mistral-large-latest")
result = generator.run([ChatMessage.from_user("What's the capital of France?")])
print(result["replies"][0].text)
```

### Streaming

```python
from haystack.components.generators.utils import print_streaming_chunk

generator = EdenAIChatGenerator(streaming_callback=print_streaming_chunk)
generator.run([ChatMessage.from_user("Write a haiku about sovereignty.")])
```

### Tool / function calling

```python
from haystack.tools import Tool

def weather(city: str) -> str:
    return f"The weather in {city} is sunny and 32°C"

tool = Tool(
    name="weather",
    description="Get the weather for a given city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    function=weather,
)

generator = EdenAIChatGenerator(tools=[tool])
result = generator.run([ChatMessage.from_user("What's the weather in Paris?")])
print(result["replies"][0].tool_calls)
```

### Async

```python
import asyncio

async def main():
    generator = EdenAIChatGenerator()
    result = await generator.run_async([ChatMessage.from_user("Hello!")])
    print(result["replies"][0].text)

asyncio.run(main())
```

### Generation parameters and fallback

Standard generation parameters are passed via `generation_kwargs`. Eden AI-specific parameters
(such as a fallback model) are forwarded as-is to the Eden AI endpoint:

```python
generator = EdenAIChatGenerator(
    model="openai/gpt-4o-mini",
    generation_kwargs={"temperature": 0.2, "max_tokens": 512},
)
```

### Embeddings

Eden AI also exposes embeddings through the same OpenAI-compatible API. Use `EdenAITextEmbedder`
to embed a query string and `EdenAIDocumentEmbedder` to embed `Document`s for indexing.

```python
from haystack import Document
from haystack_integrations.components.embedders.edenai import (
    EdenAIDocumentEmbedder,
    EdenAITextEmbedder,
)

# Embed documents (e.g. before writing them to a document store)
doc_embedder = EdenAIDocumentEmbedder(model="mistral/mistral-embed")
docs = doc_embedder.run([Document(content="I love pizza!")])["documents"]

# Embed a query at search time
text_embedder = EdenAITextEmbedder(model="mistral/mistral-embed")
query_embedding = text_embedder.run("What food do I love?")["embedding"]
```

This lets you build a fully sovereign RAG stack — retrieval *and* generation — on EU-hosted models
through a single Eden AI key.

### In a RAG pipeline

See [`examples/edenai_rag_pipeline.py`](examples/edenai_rag_pipeline.py) for a complete, runnable
retrieval-augmented-generation pipeline that uses `EdenAIChatGenerator` as the LLM.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run the tests:

```console
# unit tests (no API key needed)
hatch run test:unit

# integration tests (hit the live Eden AI API)
export EDENAI_API_KEY="<your-eden-ai-api-key>"
hatch run test:integration
```

## License

`edenai-haystack` is distributed under the terms of the [Apache-2.0](LICENSE.txt) license.
