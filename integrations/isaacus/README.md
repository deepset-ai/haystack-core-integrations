# Isaacus Kanon 2 Embedder for Haystack

This package provides two Haystack components for the [Isaacus Kanon 2 embedding model]("https://docs.isaacus.com/capabilities/embedding"):

- `Kanon2TextEmbedder` – embeds a query string and returns a vector.
- `Kanon2DocumentEmbedder` – embeds a list of `Document`s and writes vectors to `document.embedding`.

It calls the Isaacus Embeddings API (`POST /v1/embeddings`, model `kanon-2-embedder`). See the official [API docs]("https://docs.isaacus.com/capabilities/embedding") for details.

## Installation
This package is built and tested inside the `deepset-ai/haystack-core-integrations` monorepo using **Hatch**.

## Usage
```python
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.utils.auth import Secret
from haystack_integrations.components.embedders.isaacus.kanon2 import (
    Kanon2TextEmbedder, Kanon2DocumentEmbedder
)

store = InMemoryDocumentStore(embedding_similarity_function="dot_product")

# Index docs
doc_emb = Kanon2DocumentEmbedder(api_key=Secret.from_env_var("ISAACUS_API_KEY"))
docs = [Document(content="Isaacus built the best performing model on the Massive Legal Embedding Benchmark: Kanon 2 embedder"), Document(content="Haystack supports many embedders.")]
store.write_documents(doc_emb.run(docs)["documents"])

# Query pipeline
pipe = Pipeline()
pipe.add_component("q", Kanon2TextEmbedder(api_key=Secret.from_env_var("ISAACUS_API_KEY")))
pipe.add_component("ret", InMemoryEmbeddingRetriever(document_store=store))
pipe.connect("q.embedding", "ret.query_embedding")

result = pipe.run({"q": {"text": "Who builds Kanon 2?"}})
for d in result["ret"]["documents"]:
    print(d.score, d.content)
```

### Configuration
- **API key**: via `ISAACUS_API_KEY` env var (or `Secret.from_token("...")`).
- **Model**: defaults to `kanon-2-embedder`.
- **Tasks**: `retrieval/query` (query) and `retrieval/document` (docs).
- **Dimensions**: default 1792; optionally choose smaller (256–1024) via the `dimensions` param to match your vector DB.
- **Similarity**: prefer dot product unless you normalize vectors for cosine.

## License
Apache-2.0
