# dakera-haystack

[Haystack](https://haystack.deepset.ai/) integration for [Dakera](https://dakera.ai) — self-hosted, decay-weighted vector memory for AI pipelines.

## What is Dakera?

Dakera is a persistent memory server you run on your own infrastructure. It scores memories by recency and access frequency so the most relevant context always surfaces first. Unlike stateless RAG pipelines, agents using Dakera remember what happened across sessions without any external cloud dependency.

## Installation

```bash
pip install dakera-haystack
```

## Prerequisites

Run a Dakera server (see [dakera-deploy](https://github.com/dakera-ai/dakera-deploy) for Docker Compose / Kubernetes / Helm). The REST API listens on port **3000** by default:

```bash
docker run -d -p 3000:3000 -e DAKERA_ROOT_API_KEY=demo ghcr.io/dakera-ai/dakera:latest
export DAKERA_API_KEY=demo   # the key the Haystack client sends as X-API-Key
```

`DAKERA_ROOT_API_KEY` is the server's bootstrap key; the Haystack client authenticates with the same value via the `X-API-Key` header.

## Usage

The writer stores `ChatMessage` objects and the retriever returns recalled memories as `ChatMessage` objects, so both connect directly to Haystack chat components.

```python
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore
from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter

store = DakeraMemoryStore(
    base_url="http://localhost:3000",
    api_key=Secret.from_env_var("DAKERA_API_KEY"),
)

# Persist a memory
writer = DakeraMemoryWriter(memory_store=store)
writer.run(messages=[ChatMessage.from_user("The user prefers concise answers.")], session_id="session-1")

# Recall relevant memories (returns a list of ChatMessage)
retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
result = retriever.run(query="How should I format responses?", session_id="session-1")
for message in result["memories"]:
    print(message.text, "→", message.meta["score"])
```

Both components are `@component`-decorated and can be wired into a `Pipeline` — e.g. connect the retriever's `memories` output to a `ChatPromptBuilder` to inject persistent context before generation.

## Components

| Class | Type | Description |
|-------|------|-------------|
| `DakeraMemoryStore` | Client | REST client for the Dakera API (`X-API-Key` auth) |
| `DakeraMemoryRetriever` | `@component` | Decay-weighted semantic recall via `POST /v1/memory/recall`; outputs `memories: list[ChatMessage]` |
| `DakeraMemoryWriter` | `@component` | Persists `ChatMessage` text via `POST /v1/memory/store`; outputs `memories_written: int` |

## Configuration

`DakeraMemoryStore` constructor parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:3000` | Dakera server URL (falls back to the `DAKERA_API_URL` env var) |
| `api_key` | `Secret.from_env_var("DAKERA_API_KEY", strict=False)` | API key as a Haystack `Secret`, sent as `X-API-Key` |
| `default_agent_id` | `"haystack"` | Agent namespace used to isolate memories (Dakera requires an `agent_id` on every call) |
| `timeout` | `10.0` | HTTP request timeout in seconds |

`DakeraMemoryWriter.run()` and `DakeraMemoryRetriever.run()` accept `agent_id`, `session_id`, and `tags` to scope reads and writes; the retriever also accepts a per-call `top_k` (default `5` at construction time).

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

Run the unit tests (no live server needed):

```bash
cd integrations/dakera
hatch run test:unit
```

Integration tests run against a live Dakera server — set `DAKERA_API_URL` (and `DAKERA_API_KEY`) first:

```bash
export DAKERA_API_URL=http://localhost:3000 DAKERA_API_KEY=demo
hatch run test:integration
```

## License

`dakera-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
