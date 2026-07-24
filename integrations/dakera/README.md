# dakera-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dakera-haystack.svg)](https://pypi.org/project/dakera-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dakera-haystack.svg)](https://pypi.org/project/dakera-haystack)

[Haystack](https://haystack.deepset.ai/) integration for [Dakera](https://dakera.ai), a self-hosted
memory server with persistent, decay-weighted vector recall. This package exposes two complementary
integrations that share one Dakera server:

- **Memory** — `DakeraMemoryStore`, `DakeraMemoryRetriever`, `DakeraMemoryWriter`: conversational,
  decay-weighted memory over Dakera's memory API. Works with `ChatMessage` objects; the server
  handles embedding and importance scoring.
- **Document store** — `DakeraDocumentStore`, `DakeraEmbeddingRetriever`: a standard Haystack
  `DocumentStore` over Dakera's vector-namespace API, for embedding-based document retrieval (RAG)
  with metadata filtering. Embeddings are supplied by any Haystack embedder.

## Installation

```console
pip install dakera-haystack
```

## Running Dakera

Dakera is self-hosted. The canonical way to run it is the
[`dakera-deploy`](https://github.com/dakera-ai/dakera-deploy) docker-compose stack, which starts the
Dakera server (default REST port `3000`) together with the MinIO object store it depends on:

```console
git clone https://github.com/dakera-ai/dakera-deploy
cd dakera-deploy
docker compose up -d
```

The Haystack client authenticates with the `DAKERA_API_KEY` environment variable.

## Memory usage

Store conversation turns and recall decay-weighted context as `ChatMessage` objects:

```python
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.memory_stores.dakera import DakeraMemoryStore
from haystack_integrations.components.retrievers.dakera import DakeraMemoryRetriever
from haystack_integrations.components.writers.dakera import DakeraMemoryWriter

store = DakeraMemoryStore(base_url="http://localhost:3000", api_key=Secret.from_env_var("DAKERA_API_KEY"))

# Persist a memory
writer = DakeraMemoryWriter(memory_store=store)
writer.run(messages=[ChatMessage.from_user("The user prefers concise answers.")], session_id="session-1")

# Recall relevant memories (list[ChatMessage])
retriever = DakeraMemoryRetriever(memory_store=store, top_k=5)
result = retriever.run(query="How should I format responses?", session_id="session-1")
for message in result["memories"]:
    print(message.text, "→", message.meta["score"])
```

## Document-store usage

Index embedded documents and retrieve them by dense similarity:

```python
import os

from haystack import Document, Pipeline
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.dakera import DakeraEmbeddingRetriever
from haystack_integrations.document_stores.dakera import DakeraDocumentStore

os.environ["DAKERA_API_KEY"] = "dk-..."

document_store = DakeraDocumentStore(url="http://localhost:3000", namespace="my-docs", dimension=768)

# Index some documents
documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates self-awareness."),
]
document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)["documents"]
document_store.write_documents(documents_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

# Query
query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", DakeraEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = query_pipeline.run({"text_embedder": {"text": "How many languages are there?"}})
print(result["retriever"]["documents"][0].content)
```

## Components

| Class | Kind | Description |
|-------|------|-------------|
| `DakeraMemoryStore` | Client | Memory API client (`POST /v1/memory/store`, `POST /v1/memory/recall`); `X-API-Key` auth |
| `DakeraMemoryWriter` | `@component` | Persists `ChatMessage` text as memories |
| `DakeraMemoryRetriever` | `@component` | Decay-weighted recall; outputs `memories: list[ChatMessage]` |
| `DakeraDocumentStore` | `DocumentStore` | Vector-namespace document store backed by the `dakera` SDK |
| `DakeraEmbeddingRetriever` | `@component` | Dense document retrieval with metadata filtering |

## Configuration

`DakeraMemoryStore`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:3000` | Dakera server URL (or the `DAKERA_API_URL` env var) |
| `api_key` | `DAKERA_API_KEY` env var | API key as a Haystack `Secret`, sent as `X-API-Key` |
| `default_agent_id` | `"haystack"` | Agent namespace used to isolate memories |
| `timeout` | `10.0` | HTTP request timeout in seconds |

`DakeraDocumentStore`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `DAKERA_API_KEY` env var | The Dakera API key (a `dk-...` token) |
| `url` | `http://localhost:3000` | Base URL of the Dakera server |
| `namespace` | `default` | Namespace documents are written to and read from |
| `dimension` | `768` | Embedding dimension. Only used when the namespace is created |
| `metric` | `cosine` | Distance metric (`cosine`, `euclidean`, `dot_product`) at creation time |
| `batch_size` | `100` | Number of documents per upsert request |

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

```bash
cd integrations/dakera
hatch run test:unit          # mocked, runs in CI
hatch run test:integration   # requires a live Dakera server (set DAKERA_API_URL / DAKERA_URL)
```

## License

`dakera-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
