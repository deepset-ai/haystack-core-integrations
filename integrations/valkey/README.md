# valkey-haystack

Document Store for [Haystack](https://haystack.deepset.ai/) using [Valkey](https://valkey.io/) with vector search capabilities.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Using pip

```console
pip install valkey-haystack
```

### Using uv

```console
uv add valkey-haystack
```

### Using hatch

```console
hatch add valkey-haystack
```

## Usage

To use Valkey as your document store, initialize a `ValkeyDocumentStore` with the details of your Valkey instance:

```python
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore

document_store = ValkeyDocumentStore(
    nodes_list=[("localhost", 6379)],
    index_name="my_documents",
    embedding_dim=768,
    distance_metric="cosine"
)
```

### Writing Documents

To write documents to your `ValkeyDocumentStore`, create a indexing pipeline, or use the `write_documents()` function.

```python
from haystack import Document

documents = [
    Document(content="This is first", embedding=[0.1, 0.2, ...]),
    Document(content="This is second", embedding=[0.3, 0.4, ...])
]

document_store.write_documents(documents)
```

### Using ValkeyEmbeddingRetriever

```python
from haystack_integrations.components.retrievers.valkey import ValkeyEmbeddingRetriever

retriever = ValkeyEmbeddingRetriever(document_store=document_store)
result = retriever.run(query_embedding=[0.1, 0.1, ...])
```

## Examples

You can find a code example showing how to use the Document Store and the Retriever under the `examples/` folder of this repo or in the [Haystack Tutorials](https://haystack.deepset.ai/tutorials).

### Running Tests

To run integration tests locally, you need a running Valkey instance. You can start one using Docker:

```bash
docker run -d -p 6379:6379 valkey/valkey-bundle:latest
```

Navigate to the integration directory and set up environment variables:

```bash
cd integrations/valkey

# Sync dependencies including test dependencies
uv sync --group test

# Run unit tests only
hatch run test:unit

# Run integration tests only (requires Valkey instance)
hatch run test:integration

# Run all tests
hatch run test:all
```
## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).

## License

`valkey-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
