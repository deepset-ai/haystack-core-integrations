# dakera-haystack

[Haystack](https://haystack.deepset.ai/) integration for [Dakera](https://dakera.ai) — self-hosted, decay-weighted vector memory for AI pipelines.

## What is Dakera?

Dakera is a persistent memory server you run on your own infrastructure. It scores memories by recency and access frequency so the most relevant context always surfaces first. Unlike stateless RAG pipelines, agents using Dakera remember what happened across sessions without any external cloud dependency.

## Installation

```bash
pip install dakera-haystack
```

## Prerequisites

```bash
docker run -d -p 3000:3000 -e DAKERA_API_KEY=demo dakera/dakera:latest
```

## Usage

```python
from haystack import Pipeline
from haystack_integrations.components.memory.dakera import DakeraMemoryStore, DakeraMemoryRetriever, DakeraMemoryWriter

store = DakeraMemoryStore(base_url="http://localhost:3000", api_key="demo")

# Write pipeline
write_pipeline = Pipeline()
write_pipeline.add_component("writer", DakeraMemoryWriter(store=store, session_id="session-1"))

# Retrieval pipeline
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("retriever", DakeraMemoryRetriever(store=store, session_id="session-1", top_k=5))

# Store a memory
write_pipeline.run({"writer": {"memories": ["The user prefers concise answers."]}})

# Recall relevant memories
result = retrieval_pipeline.run({"retriever": {"query": "How should I format responses?"}})
print(result["retriever"]["memories"])
```

## Components

| Class | Type | Description |
|-------|------|-------------|
| `DakeraMemoryStore` | Client | REST client for the Dakera API |
| `DakeraMemoryRetriever` | `@component` | Semantic recall via `POST /v1/memories/search` |
| `DakeraMemoryWriter` | `@component` | Stores memories via `POST /v1/memories` |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:3000` | Dakera server URL (or `DAKERA_API_URL` env var) |
| `api_key` | `""` | API key (or `DAKERA_API_KEY` env var / Haystack `Secret`) |
| `session_id` | `"default"` | Groups memories by conversation or user |
| `top_k` | `5` | Number of memories to retrieve per query |

## License

Apache 2.0
