import asyncio
import uuid

import pytest
from haystack import Document

from haystack_integrations.document_stores.opensearch.document_store import OpenSearchDocumentStore

COMMON_KWARGS = {
    "hosts": ["https://localhost:9200"],
    "http_auth": ("admin", "SecureHaystack!2026"),
    "verify_certs": False,
}
DEFAULT_METHOD = {"space_type": "cosinesimil", "engine": "lucene", "name": "hnsw"}


def _get_unique_index_name() -> str:
    """
    Generate a unique, valid OpenSearch index name for test isolation.

    Each test gets its own index to enable parallel test execution without conflicts.
    """
    return f"test_{uuid.uuid4().hex}"


@pytest.fixture
def opensearch_store():
    created: list[OpenSearchDocumentStore] = []

    def _make(**overrides) -> OpenSearchDocumentStore:
        kwargs = {
            **COMMON_KWARGS,
            "index": _get_unique_index_name(),
            "embedding_dim": 768,
            "return_embedding": True,
            "method": DEFAULT_METHOD,
            **overrides,
        }
        store = OpenSearchDocumentStore(**kwargs)
        store._ensure_initialized()
        created.append(store)
        return store

    yield _make

    for store in created:
        asyncio.run(store._ensure_initialized_async())
        store._client.indices.delete(index=store._index, params={"ignore": [400, 404]})
        asyncio.run(store._async_client.close())


@pytest.fixture
def document_store(opensearch_store):
    return opensearch_store()


@pytest.fixture
def document_store_2(opensearch_store):
    return opensearch_store(return_embedding=False)


@pytest.fixture
def document_store_readonly(opensearch_store):
    store = opensearch_store(create_index=False)
    store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": False}})
    yield store
    store._client.cluster.put_settings(body={"transient": {"action.auto_create_index": True}})


@pytest.fixture
def document_store_embedding_dim_4_no_emb_returned(opensearch_store):
    return opensearch_store(embedding_dim=4, return_embedding=False)


@pytest.fixture
def document_store_embedding_dim_4_no_emb_returned_faiss(opensearch_store):
    """
    A document store with embedding dimension 4 that uses a FAISS engine with HNSW algorithm for vector search.
    We use this document store for testing efficient k-NN filtering according to
    https://opensearch.org/docs/latest/vector-search/filter-search-knn/efficient-knn-filtering/.
    """
    return opensearch_store(
        embedding_dim=4,
        method={"space_type": "innerproduct", "engine": "faiss", "name": "hnsw"},
    )


@pytest.fixture
def document_store_nested(opensearch_store):
    return opensearch_store(return_embedding=False, nested_fields=["refs", "tags"])


@pytest.fixture
def document_store_wildcard_nested(opensearch_store):
    return opensearch_store(return_embedding=False, nested_fields="*")


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
            content="C# is a functional programming language", meta={"likes": 10, "language_type": "functional"}, id="5"
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


@pytest.fixture
def test_documents_with_embeddings_2():
    return [
        Document(
            content="OpenSearch provides powerful search and analytics capabilities for large datasets",
            meta={"category": "search", "popularity": "high"},
            id="opensearch_doc",
            embedding=[0.9, 0.8, 0.7] + [0.0] * 765,
        ),
        Document(
            content="Elasticsearch is a distributed search and analytics engine built on Apache Lucene",
            meta={"category": "search", "popularity": "high"},
            id="elasticsearch_doc",
            embedding=[0.8, 0.7, 0.6] + [0.0] * 765,
        ),
        Document(
            content="Vector databases enable semantic search and similarity matching",
            meta={"category": "database", "popularity": "medium"},
            id="vector_db_doc",
            embedding=[0.7, 0.6, 0.7] + [0.0] * 765,
        ),
    ]


@pytest.fixture
def test_documents_with_embeddings_1():
    return [
        Document(
            content="Haskell is a functional programming language",
            meta={"likes": 100000, "language_type": "functional"},
            id="1",
            embedding=[0.1, 0.2, 0.3] + [0.0] * 765,
        ),
        Document(
            content="Lisp is a functional programming language",
            meta={"likes": 10000, "language_type": "functional"},
            id="2",
            embedding=[0.2, 0.3, 0.4] + [0.0] * 765,
        ),
        Document(
            content="Exilir is a functional programming language",
            meta={"likes": 1000, "language_type": "functional"},
            id="3",
            embedding=[0.3, 0.2, 0.1] + [0.0] * 765,
        ),
        Document(
            content="F# is a functional programming language",
            meta={"likes": 100, "language_type": "functional"},
            id="4",
            embedding=[0.3, 0.1, 0.1] + [0.0] * 765,
        ),
        Document(
            content="C# is a functional programming language",
            meta={"likes": 10, "language_type": "functional"},
            id="5",
            embedding=[0.1, 0.2, 0.2] + [0.0] * 765,
        ),
        Document(
            content="C++ is an object oriented programming language",
            meta={"likes": 100000, "language_type": "object_oriented"},
            id="6",
            embedding=[0.2, 0.3, 0.1] + [0.0] * 765,
        ),
        Document(
            content="Dart is an object oriented programming language",
            meta={"likes": 10000, "language_type": "object_oriented"},
            id="7",
            embedding=[0.3, 0.1, 0.1] + [0.0] * 765,
        ),
        Document(
            content="Go is an object oriented programming language",
            meta={"likes": 1000, "language_type": "object_oriented"},
            id="8",
            embedding=[0.3, 0.4, 0.5] + [0.0] * 765,
        ),
        Document(
            content="Python is a object oriented programming language",
            meta={"likes": 100, "language_type": "object_oriented"},
            id="9",
            embedding=[0.4, 0.4, 0.4] + [0.0] * 765,
        ),
        Document(
            content="Ruby is a object oriented programming language",
            meta={"likes": 10, "language_type": "object_oriented"},
            id="10",
            embedding=[0.1, 0.1, 0.2] + [0.0] * 765,
        ),
        Document(
            content="PHP is a object oriented programming language",
            meta={"likes": 1, "language_type": "object_oriented"},
            id="11",
            embedding=[0.1, 0.2, 0.3] + [0.1] * 765,
        ),
    ]
