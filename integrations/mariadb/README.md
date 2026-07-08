# MariaDB Document Store for Haystack

[![PyPI - Version](https://img.shields.io/pypi/v/mariadb-haystack.svg)](https://pypi.org/project/mariadb-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mariadb-haystack.svg)](https://pypi.org/project/mariadb-haystack)

An integration of [MariaDB 11.7+](https://mariadb.org/) with [Haystack](https://github.com/deepset-ai/haystack) that uses MariaDB's native `VECTOR` datatype and `MHNSW` indexing for vector search, and `MATCH ... AGAINST` for full-text keyword search.

## Requirements

- MariaDB 11.7 or later
- Python 3.10+

## Installation

```bash
pip install mariadb-haystack
```

## Usage

### Document Store

```python
import os
from haystack.dataclasses import Document
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore

os.environ["MARIADB_USER"] = "root"
os.environ["MARIADB_PASSWORD"] = "password"

store = MariaDBDocumentStore(
    host="localhost",
    port=3306,
    database="haystack",
    embedding_dimension=768,
)

docs = [
    Document(content="Haystack is an open-source framework for building NLP pipelines."),
    Document(content="MariaDB is a community-developed fork of MySQL."),
]
store.write_documents(docs)
print(store.count_documents())  # 2
```

### Embedding Retriever

```python
from haystack_integrations.components.retrievers.mariadb import MariaDBEmbeddingRetriever

retriever = MariaDBEmbeddingRetriever(document_store=store, top_k=5)
results = retriever.run(query_embedding=[0.1] * 768)
```

### Keyword Retriever

```python
from haystack_integrations.components.retrievers.mariadb import MariaDBKeywordRetriever

retriever = MariaDBKeywordRetriever(document_store=store, top_k=5)
results = retriever.run(query="NLP pipelines")
```

## Running MariaDB locally with Docker

```bash
docker run -d \
  --name mariadb-haystack \
  -e MARIADB_ROOT_PASSWORD=password \
  -e MARIADB_DATABASE=haystack \
  -p 3306:3306 \
  mariadb:11.7
```

## License

`mariadb-haystack` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.
