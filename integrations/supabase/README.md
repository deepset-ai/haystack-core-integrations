# supabase-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/supabase-haystack.svg)](https://pypi.org/project/supabase-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/supabase-haystack.svg)](https://pypi.org/project/supabase-haystack)

- [Changelog](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/supabase/CHANGELOG.md)

---

## Installation

```bash
pip install supabase-haystack
```

## Usage

This integration provides a `SupabasePgvectorDocumentStore` and retrievers for use with
[Supabase](https://supabase.com/) PostgreSQL databases with the pgvector extension.

It is a thin wrapper around the [pgvector-haystack](https://pypi.org/project/pgvector-haystack/) integration
with Supabase-specific defaults:

- Reads the connection string from the `SUPABASE_DB_URL` environment variable.
- Defaults `create_extension` to `False` since pgvector is pre-installed on Supabase.

### Connection string

Set the `SUPABASE_DB_URL` environment variable with your Supabase database connection string:

```bash
export SUPABASE_DB_URL="postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres"
```

You can find this in your Supabase project dashboard under **Settings > Database > Connection string > URI**.

**Connection modes:** Supabase offers two pooler ports — transaction mode (6543) and session mode (5432).
For best compatibility with pgvector operations, use session mode (port 5432) or a direct connection.

### Document Store

```python
from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore

document_store = SupabasePgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
)
```

### Embedding Retriever

```python
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.document_stores import DuplicatePolicy
from haystack import Document

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
from haystack_integrations.components.retrievers.supabase import SupabasePgvectorEmbeddingRetriever

document_store = SupabasePgvectorDocumentStore(
    embedding_dimension=768,
    vector_function="cosine_similarity",
    recreate_table=True,
)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_embedder = SentenceTransformersDocumentEmbedder()
document_embedder.warm_up()
documents_with_embeddings = document_embedder.run(documents)

document_store.write_documents(documents_with_embeddings.get("documents"), policy=DuplicatePolicy.OVERWRITE)

query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
query_pipeline.add_component("retriever", SupabasePgvectorEmbeddingRetriever(document_store=document_store))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

result = query_pipeline.run({"text_embedder": {"text": "How many languages are there?"}})
```

### Keyword Retriever

```python
from haystack import Document
from haystack.document_stores import DuplicatePolicy

from haystack_integrations.document_stores.supabase import SupabasePgvectorDocumentStore
from haystack_integrations.components.retrievers.supabase import SupabasePgvectorKeywordRetriever

document_store = SupabasePgvectorDocumentStore(
    embedding_dimension=768,
    recreate_table=True,
)

documents = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates a high level of self-awareness."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

retriever = SupabasePgvectorKeywordRetriever(document_store=document_store)

result = retriever.run(query="languages")
```

## Testing

### Unit tests

```bash
hatch run test:unit
```

### Integration tests

Integration tests run against a real PostgreSQL instance with the pgvector extension.

Start a local PostgreSQL container:

```bash
docker run -d --name supabase-test \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

Then run the integration tests:

```bash
export SUPABASE_DB_URL="postgresql://postgres:postgres@localhost:5432/postgres"
hatch run test:integration
```

## Contributing

Refer to the general [Contribution Guidelines](https://github.com/deepset-ai/haystack-core-integrations/blob/main/CONTRIBUTING.md).
