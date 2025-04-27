import asyncio

import pytest
from haystack import Document

from haystack_integrations.document_stores.opensearch.document_store import OpenSearchDocumentStore


@pytest.fixture
def document_store(request):
    """
    We use this document store for basic tests and for testing filters.
    `return_embedding` is set to True because in filters tests we compare embeddings.
    """
    hosts = ["https://localhost:9200"]
    # Use a different index for each test so we can run them in parallel
    index = f"{request.node.name}"

    store = OpenSearchDocumentStore(
        hosts=hosts,
        index=index,
        http_auth=("admin", "admin"),
        verify_certs=False,
        embedding_dim=768,
        return_embedding=True,
        method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
    )
    yield store

    store._ensure_initialized()
    assert store._client
    store._client.indices.delete(index=index, params={"ignore": [400, 404]})
    asyncio.run(store._async_client.close())


@pytest.fixture
def document_store_readonly(request):
    """
    A document store that does not automatically create the underlying index.
    """
    hosts = ["https://localhost:9200"]
    # Use a different index for each test so we can run them in parallel
    index = f"{request.node.name}"

    store = OpenSearchDocumentStore(
        hosts=hosts,
        index=index,
        http_auth=("admin", "admin"),
        verify_certs=False,
        embedding_dim=768,
        method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
        create_index=False,
    )
    store._ensure_initialized()
    assert store._client
    store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": False}})
    yield store

    store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": True}})
    store._client.indices.delete(index=index, params={"ignore": [400, 404]})
    asyncio.run(store._async_client.close())


@pytest.fixture
def document_store_embedding_dim_4_no_emb_returned(request):
    """
    A document store with embedding dimension 4 that does not return embeddings.
    """
    hosts = ["https://localhost:9200"]
    # Use a different index for each test so we can run them in parallel
    index = f"{request.node.name}"

    store = OpenSearchDocumentStore(
        hosts=hosts,
        index=index,
        http_auth=("admin", "admin"),
        verify_certs=False,
        embedding_dim=4,
        return_embedding=False,
        method={"space_type": "cosinesimil", "engine": "nmslib", "name": "hnsw"},
    )
    yield store

    store._client.indices.delete(index=index, params={"ignore": [400, 404]})
    asyncio.run(store._async_client.close())


@pytest.fixture
def document_store_embedding_dim_4_no_emb_returned_faiss(request):
    """
    A document store with embedding dimension 4 that uses a FAISS engine with HNSW algorithm for vector search.
    We use this document store for testing efficient k-NN filtering according to
    https://opensearch.org/docs/latest/vector-search/filter-search-knn/efficient-knn-filtering/.
    """
    hosts = ["https://localhost:9200"]
    # Use a different index for each test so we can run them in parallel
    index = f"{request.node.name}"

    store = OpenSearchDocumentStore(
        hosts=hosts,
        index=index,
        http_auth=("admin", "admin"),
        verify_certs=False,
        embedding_dim=4,
        method={"space_type": "innerproduct", "engine": "faiss", "name": "hnsw"},
    )
    yield store

    store._client.indices.delete(index=index, params={"ignore": [400, 404]})
    asyncio.run(store._async_client.close())


@pytest.fixture
def test_documents():
    return [
        Document(
            content="Haskell is a functional programming language",
            meta={"likes": 100000, "language_type": "functional"},
            id="1",
        ),
        Document(
            content="Lisp is a functional programming language",
            meta={"likes": 10000, "language_type": "functional"},
            id="2",
        ),
        Document(
            content="Exilir is a functional programming language",
            meta={"likes": 1000, "language_type": "functional"},
            id="3",
        ),
        Document(
            content="F# is a functional programming language",
            meta={"likes": 100, "language_type": "functional"},
            id="4",
        ),
        Document(
            content="C# is a functional programming language",
            meta={"likes": 10, "language_type": "functional"},
            id="5",
        ),
        Document(
            content="C++ is an object oriented programming language",
            meta={"likes": 100000, "language_type": "object_oriented"},
            id="6",
        ),
        Document(
            content="Dart is an object oriented programming language",
            meta={"likes": 10000, "language_type": "object_oriented"},
            id="7",
        ),
        Document(
            content="Go is an object oriented programming language",
            meta={"likes": 1000, "language_type": "object_oriented"},
            id="8",
        ),
        Document(
            content="Python is a object oriented programming language",
            meta={"likes": 100, "language_type": "object_oriented"},
            id="9",
        ),
        Document(
            content="Ruby is a object oriented programming language",
            meta={"likes": 10, "language_type": "object_oriented"},
            id="10",
        ),
        Document(
            content="PHP is a object oriented programming language",
            meta={"likes": 1, "language_type": "object_oriented"},
            id="11",
        ),
    ]
