# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

import pytest
from elasticsearch import Elasticsearch

from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@pytest.fixture(scope="session")
def supports_sparse_vector_query() -> bool:
    try:
        version = Elasticsearch(["http://localhost:9200"]).info()["version"]["number"]
        major, minor, patch = (int(x) for x in version.split(".")[:3])
    except Exception:
        return False
    return (major, minor, patch) >= (8, 15, 0)


def _get_unique_index_name() -> str:
    """
    Generate a unique, valid Elasticsearch index name for test isolation.

    Each test gets its own index to enable parallel test execution without conflicts.
    """
    return f"test_sql_{uuid.uuid4().hex}"


@pytest.fixture
def sparse_document_store(supports_sparse_vector_query):
    """
    Document store fixture with sparse_vector_field configured.
    Automatically skips if the running Elasticsearch instance is < 8.15.0.
    """
    if not supports_sparse_vector_query:
        pytest.skip("Requires Elasticsearch >= 8.15.0")
    index = f"test_sparse_{uuid.uuid4().hex}"
    store = ElasticsearchDocumentStore(
        hosts=["http://localhost:9200"],
        index=index,
        sparse_vector_field="sparse_vec",
    )
    yield store

    store._ensure_initialized()
    store.client.options(ignore_status=[400, 404]).indices.delete(index=index)
    store.client.close()
    if store._async_client is not None:
        asyncio.run(store._async_client.close())


@pytest.fixture
def document_store():
    """
    Document store fixture for SQL retriever integration tests.
    """
    hosts = ["http://localhost:9200"]
    index = _get_unique_index_name()
    embedding_similarity_function = "max_inner_product"

    store = ElasticsearchDocumentStore(
        hosts=hosts,
        index=index,
        embedding_similarity_function=embedding_similarity_function,
    )
    yield store

    store._ensure_initialized()
    store.client.options(ignore_status=[400, 404]).indices.delete(index=index)
    store.client.close()
    if store._async_client is not None:
        asyncio.run(store._async_client.close())


@pytest.fixture
def document_store_2():
    """
    Second document store fixture for runtime document store switching tests.
    """
    hosts = ["http://localhost:9200"]
    index = f"test_sql_2_{uuid.uuid4().hex}"
    embedding_similarity_function = "max_inner_product"

    store = ElasticsearchDocumentStore(
        hosts=hosts,
        index=index,
        embedding_similarity_function=embedding_similarity_function,
    )
    yield store

    store._ensure_initialized()
    store.client.options(ignore_status=[400, 404]).indices.delete(index=index)
    store.client.close()
    if store._async_client is not None:
        asyncio.run(store._async_client.close())
