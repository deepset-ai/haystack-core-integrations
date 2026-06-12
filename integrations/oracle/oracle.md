---
title: Oracle AI Vector Search
id: integrations-oracle
description: Oracle AI Vector Search integration for Haystack
---

<a id="haystack_integrations.components.document_stores.oracle.document_store"></a>

# haystack\_integrations.components.document\_stores.oracle.document\_store

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleVectorizerPreference"></a>

## OracleVectorizerPreference Objects

```python
class OracleVectorizerPreference()
```

Manage DBMS_VECTOR_CHAIN vectorizer preferences for Oracle hybrid indexes.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore"></a>

## OracleDocumentStore Objects

```python
class OracleDocumentStore()
```

A document store using Oracle as the backend.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.__init__"></a>

#### \_\_init\_\_

```python
def __init__(connection_params: dict[str, Any],
             table_name: str = "documents",
             *,
             use_connection_pool: bool = False,
             embedding_dim: Optional[int] = None,
             support_sparse_embeddings: bool = True,
             create_vector_index: bool = False,
             vector_index_params: dict[str, Any] | None = None,
             vector_index_embedding_field: EmbeddingField = "embedding",
             vector_index_distance_strategy: DistanceStrategy = "cosine",
             sparse_vector_index: dict[str, Any] | None = None)
```

Create a new OracleDocumentStore instance.

:param connection_params: Connection parameters for python-oracledb. These are passed to
    `oracledb.connect()`, `oracledb.connect_async()`, `oracledb.create_pool()`, or
    `oracledb.create_pool_async()` depending on the selected mode.
:param table_name: Oracle table name used to store Haystack documents.
:param use_connection_pool: If `True`, create and use an Oracle connection pool.
:param embedding_dim: Optional dense and sparse embedding dimension for Oracle VECTOR columns.
    If omitted, the VECTOR columns are created with flexible dimensions.
:param support_sparse_embeddings: If `True`, create support for sparse embeddings in the table schema
    and allow sparse retrieval and writes.
:param create_vector_index: If `True`, create a vector index during initialization.
:param vector_index_params: Optional Oracle vector index parameters. Supported index types are `HNSW` and `IVF`.
:param vector_index_embedding_field: VECTOR column to index. Must be either `embedding`
    or `sparse_embedding`.
:param vector_index_distance_strategy: Distance strategy to use for vector indexing and retrieval.
    Must be one of `dot`, `euclidean`, or `cosine`.
:param sparse_vector_index: Optional sparse vector index configuration. Supported keys are
    `enabled`, `distance_strategy`, and `params`.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.count_documents"></a>

#### count\_documents

```python
@_handle_exceptions
def count_documents() -> int
```

Returns how many documents are present in the document store.

:returns: how many documents are present in the document store.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.count_documents_async"></a>

#### count\_documents\_async

```python
@_handle_exceptions_async
async def count_documents_async() -> int
```

Asynchronously returns how many documents are present in the document store.

:returns: how many documents are present in the document store.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.filter_documents"></a>

#### filter\_documents

```python
@_handle_exceptions
def filter_documents(
        filters: Optional[dict[str, Any]] = None) -> list[Document]
```

Returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/docs/metadata-filtering).

:param filters: the filters to apply to the document list.
:returns: a list of Documents that match the given filters.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.filter_documents_async"></a>

#### filter\_documents\_async

```python
@_handle_exceptions_async
async def filter_documents_async(
        filters: Optional[dict[str, Any]] = None) -> list[Document]
```

Asynchronously returns the documents that match the filters provided.

For a detailed specification of the filters,
refer to the [documentation](https://docs.haystack.deepset.ai/v2.0/docs/metadata-filtering).

:param filters: the filters to apply to the document list.
:returns: a list of Documents that match the given filters.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.write_documents"></a>

#### write\_documents

```python
@_handle_exceptions
def write_documents(documents: list[Document],
                    policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Writes (or overwrites) documents into the store.

:param documents:
    A list of documents to write into the document store.
:param policy:
    Not supported at the moment.

:raises ValueError:
    When input is not valid.

:returns:
    The number of documents written

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.write_documents_async"></a>

#### write\_documents\_async

```python
@_handle_exceptions_async
async def write_documents_async(
        documents: list[Document],
        policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int
```

Asynchronously writes (or overwrites) documents into the store.

:param documents:
    A list of documents to write into the document store.
:param policy:
    Not supported at the moment.

:raises ValueError:
    When input is not valid.

:returns:
    The number of documents written

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.delete_documents"></a>

#### delete\_documents

```python
@_handle_exceptions
def delete_documents(document_ids: list[str]) -> None
```

Deletes all documents with a matching document_ids from the document store.

:param document_ids: the document ids to delete

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.delete_documents_async"></a>

#### delete\_documents\_async

```python
@_handle_exceptions_async
async def delete_documents_async(document_ids: list[str]) -> None
```

Asynchronously deletes all documents with a matching document_ids from the document store.

:param document_ids: the document ids to delete

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OracleDocumentStore"
```

Deserializes the component from a dictionary.

:param data:
    Dictionary to deserialize from.
:returns:
    Deserialized component.

<a id="haystack_integrations.components.document_stores.oracle.document_store.OracleDocumentStore.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

:returns:
    Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.oracle.text_embedder"></a>

# haystack\_integrations.components.embedders.oracle.text\_embedder

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder"></a>

## OracleTextEmbedder Objects

```python
@component
class OracleTextEmbedder()
```

A component for embedding strings using Oracle Database.

It connects to Oracle Database and retrieves embeddings for input text using the configured
provider/model parameters.

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder.__init__"></a>

#### \_\_init\_\_

```python
def __init__(connection_params: dict[str, Any],
             embedding_params: dict[str, Any],
             *,
             use_connection_pool: bool = False,
             proxy: Optional[str])
```

Creates a new OracleTextEmbedder component.

:param connection_params: Connection parameters for python-oracledb. Required.
    See the python-oracledb docs (https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html).
:param embedding_params: Embedding parameters passed to Oracle embeddings (for example, provider, model, etc.).
    See the Oracle embedding docs (https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/utl_to_embedding-and-utl_to_embeddings-dbms_vector.html)
    for accepted values.
:param use_connection_pool: If True, use a python-oracledb connection pool for connections. Defaults to False.
:param proxy: Optional HTTP proxy to set via UTL_HTTP.set_proxy for outbound calls in the database session.

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OracleTextEmbedder"
```

Deserializes the component from a dictionary.

:param data:
    Dictionary to deserialize from.
:returns:
    Deserialized component.

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

:returns:
    Dictionary with serialized data.

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder.run"></a>

#### run

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
def run(text: str) -> dict[str, Any]
```

Compute an embedding for a single text string.

:param text: The string to embed.
:returns: A dictionary with:
    - embedding: The embedding of the input string.
    - meta: The embedding parameters used for the call (for example, provider, model, etc.).
:raises TypeError: If the input is not a string.

<a id="haystack_integrations.components.embedders.oracle.text_embedder.OracleTextEmbedder.run_async"></a>

#### run\_async

```python
@component.output_types(embedding=list[float], meta=dict[str, Any])
async def run_async(text: str) -> dict[str, Any]
```

Asynchronously compute an embedding for a single text string.

:param text: The string to embed.
:returns: A dictionary with:
    - embedding: The embedding of the input string.
    - meta: The embedding parameters used for the call (for example, provider, model, etc.).
:raises TypeError: If the input is not a string.

<a id="haystack_integrations.components.embedders.oracle.document_embedder"></a>

# haystack\_integrations.components.embedders.oracle.document\_embedder

Oracle Document Embedder component.

This module provides OracleDocumentEmbedder, a Haystack component that computes vector embeddings
for lists of Haystack Documents using Oracle Database vector capabilities. It extends
OracleTextEmbedder by handling Document objects, optional inclusion of selected metadata fields,
and synchronous/asynchronous execution.

<a id="haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder"></a>

## OracleDocumentEmbedder Objects

```python
@component
class OracleDocumentEmbedder(OracleTextEmbedder)
```

Embed Haystack Documents with Oracle Database.

This component concatenates selected metadata fields with the Document content and
requests embeddings from Oracle Database. The resulting vectors are assigned back
to the corresponding Document.embedding fields.

<a id="haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder.__init__"></a>

#### \_\_init\_\_

```python
def __init__(connection_params: dict[str, Any],
             embedding_params: dict[str, Any],
             *,
             use_connection_pool: bool = False,
             proxy: Optional[str],
             meta_fields_to_embed: list[str] = [],
             embedding_separator: str = "\n")
```

Create an OracleDocumentEmbedder component.

        :param connection_params: Connection parameters for python-oracledb. Required.
            See the python-oracledb docs (https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html).
        :param embedding_params: Embedding parameters passed to Oracle embeddings (for example, provider, model, etc.).
            See the Oracle embedding docs (https://docs.oracle.com/en/database/oracle/oracle-database/26/vecse/utl_to_embedding-and-utl_to_embeddings-dbms_vector.html)
            for accepted values.
        :param use_connection_pool: If True, use a python-oracledb connection pool for connections. Defaults to False.
        :param proxy: Optional HTTP proxy to set via UTL_HTTP.set_proxy for outbound calls in the database session.
        :param meta_fields_to_embed: Optional list of keys from Document.meta whose values will be concatenated with the
            Document content before embedding. Keys missing in a Document or with None values are skipped.
            If None or empty, only the Document content is used.
        :param embedding_separator: String used to join selected metadata values and the Document content. Defaults to "
".

<a id="haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder.run"></a>

#### run

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
def run(documents: list[Document]) -> dict[str, Any]
```

Compute embeddings for a list of Documents.

Each Document's embedding field is set in-place. The text passed to the Oracle embedding
function is constructed from selected metadata fields and the Document content:

    "{meta_field_1}{separator}{meta_field_2}{separator}...{separator}{content}"

Where the set of metadata fields comes from meta_fields_to_embed and the separator is embedding_separator.

:param documents: List of Haystack Documents to embed. If a Document has no content, an empty string is used.
:returns: A dictionary with:
    - documents: The same list of Documents with their embedding fields populated.
    - meta: The embedding parameters used for the call (for example, provider, model, etc.).
:raises TypeError: If the input is not a list of Documents.

<a id="haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder.run_async"></a>

#### run\_async

```python
@component.output_types(documents=list[Document], meta=dict[str, Any])
async def run_async(documents: list[Document]) -> dict[str, Any]
```

Asynchronously compute embeddings for a list of Documents.

Behavior matches run(), but uses the async Oracle client.

:param documents: List of Haystack Documents to embed. If a Document has no content, an empty string is used.
:returns: A dictionary with:
    - documents: The same list of Documents with their embedding fields populated.
    - meta: The embedding parameters used for the call (for example, provider, model, etc.).
:raises TypeError: If the input is not a list of Documents.

<a id="haystack_integrations.components.embedders.oracle.document_embedder.OracleDocumentEmbedder.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

:returns:
    Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever"></a>

# haystack\_integrations.components.retrievers.oracle.embedding\_retriever

Oracle Embedding Retriever component.

Retrieves Documents from OracleDocumentStore using vector distance functions on embeddings.
Provides synchronous and asynchronous interfaces, supports metadata filtering with
FilterPolicy, and configurable distance strategies ("dot", "euclidean", "cosine").

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever"></a>

## OracleEmbeddingRetriever Objects

```python
@component
class OracleEmbeddingRetriever()
```

Retrieve documents from an OracleDocumentStore based on dense embedding similarity.

This component delegates retrieval to OracleDocumentStore, which executes a vector
similarity query in Oracle using the configured distance strategy. Runtime filters
are merged with those defined at initialization using the selected FilterPolicy.

Example:
```python
import os
from haystack import Document, Pipeline
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.document_stores.oracle import OracleDocumentStore
from haystack_integrations.components.embedders.oracle import (
    OracleTextEmbedder,
    OracleDocumentEmbedder,
)
from haystack_integrations.components.retrievers.oracle import OracleEmbeddingRetriever

# Create the document store (adjust connection params)
store = OracleDocumentStore(
    connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
    table_name="documents",
    embedding_dim=768,
    create_vector_index=True,  # optional but recommended
    vector_index_distance_strategy="cosine",
)

# Prepare and write documents with embeddings
docs = [
    Document(content="There are over 7,000 languages spoken around the world today."),
    Document(content="Elephants have been observed to behave in a way that indicates..."),
    Document(content="In certain places, you can witness the phenomenon of bioluminescent waves."),
]

doc_embedder = OracleDocumentEmbedder(
    connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
    embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
    proxy=None,
    use_connection_pool=False,
    meta_fields_to_embed=None,
)
docs_with_embeddings = doc_embedder.run(docs)["documents"]
store.write_documents(docs_with_embeddings, policy=DuplicatePolicy.OVERWRITE)

# Build a pipeline that embeds the query and retrieves similar documents
pipe = Pipeline()
pipe.add_component(
    "text_embedder",
    OracleTextEmbedder(
        connection_params={"dsn": os.environ["ORACLE_DB_DSN"]},
        embedding_params={"provider": "database", "model": "ALL_MINILM_L12_V2"},
        proxy=None,
        use_connection_pool=False,
    ),
)
pipe.add_component("retriever", OracleEmbeddingRetriever(document_store=store, top_k=3))
pipe.connect("text_embedder.embedding", "retriever.query_embedding")

res = pipe.run({"text_embedder": {"text": "How many languages are there?"}})
assert "languages" in res["retriever"]["documents"][0].content
```

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: OracleDocumentStore,
             filters: Optional[dict[str, Any]] = None,
             top_k: int = 10,
             distance_strategy: Optional[Literal["dot", "euclidean",
                                                 "cosine"]] = "cosine",
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

Initialize the OracleEmbeddingRetriever.

:param document_store: OracleDocumentStore instance used to execute vector similarity queries.
:param filters: Optional base filters applied to every retrieval. Runtime filters provided to run/run_async
    are merged with these according to filter_policy.
:param top_k: Maximum number of Documents to return.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
:param filter_policy: Policy determining how runtime filters are merged with base filters.
:raises ValueError: If document_store is not an OracleDocumentStore or if distance_strategy is invalid.

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

:returns:
    Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OracleEmbeddingRetriever"
```

Deserializes the component from a dictionary.

:param data:
    Dictionary to deserialize from.
:returns:
    Deserialized component.

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever.run"></a>

#### run

```python
@component.output_types(documents=list[Document])
def run(
    query_embedding: list[float],
    filters: Optional[dict[str, Any]] = None,
    top_k: Optional[int] = None,
    distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None
) -> dict[str, list[Document]]
```

Retrieve documents from the OracleDocumentStore based on a query embedding.

Runtime filters are merged with the retriever's base filters using the configured filter_policy.

:param query_embedding: Embedding vector representing the query.
:param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
:param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
    Defaults to the value set at initialization.
:returns: A dictionary with:
    - documents: list of Documents similar to query_embedding.
:raises ValueError: If distance_strategy is invalid.

<a id="haystack_integrations.components.retrievers.oracle.embedding_retriever.OracleEmbeddingRetriever.run_async"></a>

#### run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
    query_embedding: list[float],
    filters: Optional[dict[str, Any]] = None,
    top_k: Optional[int] = None,
    distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the OracleDocumentStore based on a query embedding.

Runtime filters are merged with the retriever's base filters using the configured filter_policy.

:param query_embedding: Embedding vector representing the query.
:param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
:param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
    Defaults to the value set at initialization.
:returns: A dictionary with:
    - documents: list of Documents similar to query_embedding.
:raises ValueError: If distance_strategy is invalid.

<a id="haystack_integrations.components.retrievers.oracle.hybrid_retriever"></a>

# haystack\_integrations.components.retrievers.oracle.hybrid\_retriever

Oracle hybrid retriever component.

Executes DBMS_HYBRID_VECTOR.SEARCH against a hybrid vector index and returns
Haystack Documents from OracleDocumentStore. Supports keyword, semantic, and
hybrid modes plus Haystack-style metadata filters translated to Oracle
`filter_by` expressions.

<a id="haystack_integrations.components.retrievers.oracle.hybrid_retriever.OracleHybridRetriever"></a>

## OracleHybridRetriever Objects

```python
@component
class OracleHybridRetriever()
```

Retrieve documents from Oracle using DBMS_HYBRID_VECTOR.SEARCH.

The retriever requires an existing hybrid vector index and can run in
keyword-only, semantic-only, or combined hybrid mode.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever"></a>

# haystack\_integrations.components.retrievers.oracle.sparse\_embedding\_retriever

Oracle Sparse Embedding Retriever component.

Retrieves Documents from OracleDocumentStore using vector distance functions on sparse embeddings.
Provides synchronous and asynchronous interfaces, supports metadata filtering with FilterPolicy,
and configurable distance strategies ("dot", "euclidean", "cosine").

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever"></a>

## OracleSparseEmbeddingRetriever Objects

```python
@component
class OracleSparseEmbeddingRetriever()
```

Retrieve documents from an OracleDocumentStore based on sparse embedding similarity.

This component delegates retrieval to OracleDocumentStore, which executes a vector
similarity query in Oracle using the configured distance strategy. Runtime filters
are merged with those defined at initialization using the selected FilterPolicy.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever.__init__"></a>

#### \_\_init\_\_

```python
def __init__(document_store: OracleDocumentStore,
             filters: Optional[dict[str, Any]] = None,
             top_k: int = 10,
             distance_strategy: Optional[Literal["dot", "euclidean",
                                                 "cosine"]] = "cosine",
             filter_policy: Union[str, FilterPolicy] = FilterPolicy.REPLACE)
```

Initialize the OracleSparseEmbeddingRetriever.

:param document_store: OracleDocumentStore instance used to execute vector similarity queries.
:param filters: Optional base filters applied to every retrieval. Runtime filters provided to run/run_async
    are merged with these according to filter_policy.
:param top_k: Maximum number of Documents to return.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
:param filter_policy: Policy determining how runtime filters are merged with base filters.
:raises ValueError: If document_store is not an OracleDocumentStore or if distance_strategy is invalid.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever.to_dict"></a>

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Serializes the component to a dictionary.

:returns:
    Dictionary with serialized data.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever.from_dict"></a>

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "OracleSparseEmbeddingRetriever"
```

Deserializes the component from a dictionary.

:param data:
    Dictionary to deserialize from.
:returns:
    Deserialized component.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever.run"></a>

#### run

```python
@component.output_types(documents=list[Document])
def run(
    query_sparse_embedding: SparseEmbedding,
    filters: Optional[dict[str, Any]] = None,
    top_k: Optional[int] = None,
    distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None
) -> dict[str, list[Document]]
```

Retrieve documents from the OracleDocumentStore based on a sparse query embedding.

:param query_sparse_embedding: SparseEmbedding representing the query.
:param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
:param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
    Defaults to the value set at initialization.
:returns: A dictionary with:
    - documents: list of Documents similar to the given sparse embedding.
:raises ValueError: If distance_strategy is invalid.

<a id="haystack_integrations.components.retrievers.oracle.sparse_embedding_retriever.OracleSparseEmbeddingRetriever.run_async"></a>

#### run\_async

```python
@component.output_types(documents=list[Document])
async def run_async(
    query_sparse_embedding: SparseEmbedding,
    filters: Optional[dict[str, Any]] = None,
    top_k: Optional[int] = None,
    distance_strategy: Optional[Literal["dot", "euclidean", "cosine"]] = None
) -> dict[str, list[Document]]
```

Asynchronously retrieve documents from the OracleDocumentStore based on a sparse query embedding.

:param query_sparse_embedding: SparseEmbedding representing the query.
:param filters: Optional runtime filters to apply. Combined with base filters according to filter_policy.
:param top_k: Maximum number of Documents to return. Defaults to the value set at initialization.
:param distance_strategy: Vector distance metric to use. One of "dot", "euclidean", or "cosine".
    Defaults to the value set at initialization.
:returns: A dictionary with:
    - documents: list of Documents similar to the given sparse embedding.
:raises ValueError: If distance_strategy is invalid.

