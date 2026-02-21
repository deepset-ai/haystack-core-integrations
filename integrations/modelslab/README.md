# modelslab-haystack

[![PyPI](https://img.shields.io/pypi/v/modelslab-haystack)](https://pypi.org/project/modelslab-haystack/)

[ModelsLab](https://modelslab.com) integration for [Haystack](https://haystack.deepset.ai/) — providing uncensored Llama 3.1 chat models with 128K context windows for RAG pipelines, agents, and LLM applications.

## Installation

```bash
pip install modelslab-haystack
```

## Setup

```bash
export MODELSLAB_API_KEY="your-api-key"
```

Get your key at [modelslab.com](https://modelslab.com).

## Usage

### Basic chat

```python
from modelslab_haystack import ModelsLabChatGenerator
from haystack.dataclasses import ChatMessage

generator = ModelsLabChatGenerator(model="llama-3.1-8b-uncensored")

messages = [ChatMessage.from_user("Explain retrieval-augmented generation.")]
result = generator.run(messages=messages)
print(result["replies"][0].text)
```

### In a RAG pipeline

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from modelslab_haystack import ModelsLabChatGenerator

pipeline = Pipeline()
pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=store))
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
pipeline.add_component("llm", ModelsLabChatGenerator(model="llama-3.1-70b-uncensored"))

pipeline.connect("retriever.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")

result = pipeline.run({"retriever": {"query": "What is RAG?"}})
```

### Streaming

```python
from haystack.dataclasses import ChatMessage

def print_chunk(chunk):
    print(chunk.content, end="", flush=True)

generator = ModelsLabChatGenerator(streaming_callback=print_chunk)
generator.run(messages=[ChatMessage.from_user("Write a haiku about AI.")])
```

## Models

| Model | Context | Notes |
|---|---|---|
| `llama-3.1-8b-uncensored` | 128K | Default — fast, no content restrictions |
| `llama-3.1-70b-uncensored` | 128K | Higher quality, deeper reasoning |

## API Reference

- ModelsLab docs: https://docs.modelslab.com/uncensored-chat
- Haystack docs: https://docs.haystack.deepset.ai
