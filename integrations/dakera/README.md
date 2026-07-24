# dakera-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/dakera-haystack.svg)](https://pypi.org/project/dakera-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dakera-haystack.svg)](https://pypi.org/project/dakera-haystack)

Haystack integration for [Dakera](https://dakera.ai), a self-hosted memory server that provides
persistent, decay-weighted vector recall. This package adds a `DakeraDocumentStore` and a
`DakeraEmbeddingRetriever` so Haystack pipelines can store and retrieve documents from a Dakera
namespace.

## Installation

```console
pip install dakera-haystack
```

## Running Dakera

Dakera is self-hosted. The canonical way to run it is the
[`dakera-deploy`](https://github.com/dakera-ai/dakera-deploy) docker-compose stack, which starts the
Dakera server (default port `3000`) together with the MinIO object store it depends on:

```console
git clone https://github.com/dakera-ai/dakera-deploy
cd dakera-deploy
docker compose up -d
```

## Usage

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

document_store = DakeraDocumentStore(
    url="http://localhost:3000",
    namespace="my-docs",
    dimension=768,
)

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

## Configuration

`DakeraDocumentStore` accepts the following parameters:

| Parameter    | Default                  | Description                                                              |
| ------------ | ------------------------ | ------------------------------------------------------------------------ |
| `api_key`    | `DAKERA_API_KEY` env var | The Dakera API key (a `dk-...` token).                                    |
| `url`        | `http://localhost:3000`  | Base URL of the Dakera server.                                           |
| `namespace`  | `default`                | The namespace documents are written to and read from.                    |
| `dimension`  | `768`                    | Embedding dimension. Used only when the namespace is created.            |
| `metric`     | `cosine`                 | Distance metric (`cosine`, `euclidean`, `dot_product`) at creation time. |
| `batch_size` | `100`                    | Number of documents per upsert request.                                  |

## License

`dakera-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
