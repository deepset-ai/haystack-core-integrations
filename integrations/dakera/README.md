# dakera-haystack

[Haystack](https://haystack.deepset.ai/) integration for [Dakera](https://dakera.ai) — self-hosted, decay-weighted vector memory for AI pipelines.

## What is Dakera?

Dakera is a persistent memory server you run on your own infrastructure. It scores memories by recency and access frequency so the most relevant context always surfaces first. Unlike stateless RAG pipelines, agents using Dakera remember what happened across sessions without any external cloud dependency.

## Installation

```bash
pip install dakera-haystack
```

## Prerequisites

Start a Dakera server and export the API key the client will send:

```bash
docker run -d -p 3300:3300 -e DAKERA_API_KEY=demo ghcr.io/dakera-ai/dakera:latest
export DAKERA_API_KEY=demo
```

## Usage

```python
from haystack import Pipeline
from haystack.utils import Secret
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore
from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter

store = DakeraMemoryStore(
    base_url="http://localhost:3300",
    api_key=Secret.from_env_var("DAKERA_API_KEY"),
)

# Write pipeline — persist memories
write_pipeline = Pipeline()
write_pipeline.add_component("writer", DakeraMemoryWriter(memory_store=store))

# Retrieval pipeline — recall relevant memories
retrieval_pipeline = Pipeline()
retrieval_pipeline.add_component("retriever", DakeraMemoryRetriever(memory_store=store, top_k=5))

# Store a memory
write_pipeline.run({"writer": {"messages": ["The user prefers concise answers."], "user_id": "alice"}})

# Recall relevant memories
result = retrieval_pipeline.run({"retriever": {"query": "How should I format responses?", "user_id": "alice"}})
print(result["retriever"]["memories"])
```

## Components

| Class | Type | Description |
|-------|------|-------------|
| `DakeraMemoryStore` | Client | REST client for the Dakera API |
| `DakeraMemoryRetriever` | `@component` | Semantic recall via `POST /v1/memories/search` |
| `DakeraMemoryWriter` | `@component` | Stores memories via `POST /v1/memories` |

## Configuration

`DakeraMemoryStore` constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:3300` | Dakera server URL (falls back to the `DAKERA_API_URL` env var) |
| `api_key` | `Secret.from_env_var("DAKERA_API_KEY", strict=False)` | API key as a Haystack `Secret` |
| `default_agent_id` | `"haystack"` | Agent/namespace used to isolate memories |
| `timeout` | `10.0` | HTTP request timeout in seconds |

Per-call arguments accepted by both `DakeraMemoryWriter.run()` and `DakeraMemoryRetriever.run()`
(`user_id`, `agent_id`, `session_id`) scope reads and writes; `DakeraMemoryRetriever` additionally
takes `top_k` (default `5`) at construction time.

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

To run the tests locally:

```bash
cd integrations/dakera
hatch run test:unit
```

## License

`dakera-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
