# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.mariadb import (
    MariaDBEmbeddingRetriever,
    MariaDBKeywordRetriever,
)
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore


def _make_store(**kwargs) -> MariaDBDocumentStore:
    os.environ.setdefault("MARIADB_USER", "test_user")
    os.environ.setdefault("MARIADB_PASSWORD", "test_pass")
    store = MariaDBDocumentStore(**kwargs)
    return store


# ---------------------------------------------------------------------------
# MariaDBEmbeddingRetriever unit tests
# ---------------------------------------------------------------------------

class TestEmbeddingRetrieverInit:
    def test_requires_mariadb_store(self):
        with pytest.raises(ValueError, match="MariaDBDocumentStore"):
            MariaDBEmbeddingRetriever(document_store=MagicMock())

    def test_invalid_vector_function_raises(self):
        store = _make_store()
        with pytest.raises(ValueError, match="vector_function"):
            MariaDBEmbeddingRetriever(document_store=store, vector_function="bad")

    def test_defaults(self):
        store = _make_store()
        r = MariaDBEmbeddingRetriever(document_store=store)
        assert r.top_k == 10
        assert r.filter_policy == FilterPolicy.REPLACE
        assert r.vector_function == "cosine_similarity"

    def test_custom_vector_function(self):
        store = _make_store()
        r = MariaDBEmbeddingRetriever(document_store=store, vector_function="l2_distance")
        assert r.vector_function == "l2_distance"


class TestEmbeddingRetrieverSerialization:
    def test_to_dict(self):
        store = _make_store()
        r = MariaDBEmbeddingRetriever(document_store=store, top_k=3)
        d = r.to_dict()
        assert d["init_parameters"]["top_k"] == 3
        assert "document_store" in d["init_parameters"]

    def test_from_dict_roundtrip(self):
        store = _make_store()
        r = MariaDBEmbeddingRetriever(document_store=store, top_k=5)
        d = r.to_dict()
        restored = MariaDBEmbeddingRetriever.from_dict(d)
        assert restored.top_k == 5

    def test_from_dict_with_filter_policy(self):
        store = _make_store()
        r = MariaDBEmbeddingRetriever(document_store=store, filter_policy=FilterPolicy.MERGE)
        d = r.to_dict()
        restored = MariaDBEmbeddingRetriever.from_dict(d)
        assert restored.filter_policy == FilterPolicy.MERGE


class TestEmbeddingRetrieverRun:
    def test_run_calls_embedding_retrieval(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[Document(content="hit")])
        r = MariaDBEmbeddingRetriever(document_store=store, top_k=3)
        result = r.run(query_embedding=[0.1, 0.2])
        assert "documents" in result
        assert len(result["documents"]) == 1
        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2],
            filters={},
            top_k=3,
            vector_function="cosine_similarity",
        )

    def test_run_with_runtime_top_k(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])
        r = MariaDBEmbeddingRetriever(document_store=store, top_k=5)
        r.run(query_embedding=[0.1], top_k=2)
        call_kwargs = store._embedding_retrieval.call_args[1]
        assert call_kwargs["top_k"] == 2

    def test_run_with_runtime_vector_function(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])
        r = MariaDBEmbeddingRetriever(document_store=store)
        r.run(query_embedding=[0.1], vector_function="l2_distance")
        call_kwargs = store._embedding_retrieval.call_args[1]
        assert call_kwargs["vector_function"] == "l2_distance"

    def test_filter_policy_replace(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])
        r = MariaDBEmbeddingRetriever(
            document_store=store,
            filters={"field": "meta.x", "operator": "==", "value": 1},
            filter_policy=FilterPolicy.REPLACE,
        )
        runtime_filter = {"field": "meta.y", "operator": "==", "value": 2}
        r.run(query_embedding=[0.1], filters=runtime_filter)
        call_kwargs = store._embedding_retrieval.call_args[1]
        assert call_kwargs["filters"] == runtime_filter

    def test_filter_policy_merge(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])
        init_filter = {"field": "meta.x", "operator": "==", "value": 1}
        r = MariaDBEmbeddingRetriever(
            document_store=store,
            filters=init_filter,
            filter_policy=FilterPolicy.MERGE,
        )
        runtime_filter = {"field": "meta.y", "operator": "==", "value": 2}
        r.run(query_embedding=[0.1], filters=runtime_filter)
        call_kwargs = store._embedding_retrieval.call_args[1]
        # MERGE combines both filters into an AND condition
        assert call_kwargs["filters"]["operator"] == "AND"


# ---------------------------------------------------------------------------
# MariaDBKeywordRetriever unit tests
# ---------------------------------------------------------------------------

class TestKeywordRetrieverInit:
    def test_requires_mariadb_store(self):
        with pytest.raises(ValueError, match="MariaDBDocumentStore"):
            MariaDBKeywordRetriever(document_store=MagicMock())

    def test_defaults(self):
        store = _make_store()
        r = MariaDBKeywordRetriever(document_store=store)
        assert r.top_k == 10
        assert r.filter_policy == FilterPolicy.REPLACE


class TestKeywordRetrieverSerialization:
    def test_to_dict(self):
        store = _make_store()
        r = MariaDBKeywordRetriever(document_store=store, top_k=7)
        d = r.to_dict()
        assert d["init_parameters"]["top_k"] == 7

    def test_from_dict_roundtrip(self):
        store = _make_store()
        r = MariaDBKeywordRetriever(document_store=store, top_k=4)
        d = r.to_dict()
        restored = MariaDBKeywordRetriever.from_dict(d)
        assert restored.top_k == 4


class TestKeywordRetrieverRun:
    def test_run_calls_keyword_retrieval(self):
        store = _make_store()
        store._keyword_retrieval = MagicMock(return_value=[Document(content="result")])
        r = MariaDBKeywordRetriever(document_store=store, top_k=5)
        result = r.run(query="haystack")
        assert "documents" in result
        assert len(result["documents"]) == 1
        store._keyword_retrieval.assert_called_once_with(
            query="haystack",
            filters={},
            top_k=5,
        )

    def test_run_with_runtime_top_k(self):
        store = _make_store()
        store._keyword_retrieval = MagicMock(return_value=[])
        r = MariaDBKeywordRetriever(document_store=store, top_k=10)
        r.run(query="test", top_k=3)
        call_kwargs = store._keyword_retrieval.call_args[1]
        assert call_kwargs["top_k"] == 3

    def test_run_with_runtime_filters(self):
        store = _make_store()
        store._keyword_retrieval = MagicMock(return_value=[])
        r = MariaDBKeywordRetriever(document_store=store)
        runtime_filter = {"field": "meta.lang", "operator": "==", "value": "en"}
        r.run(query="test", filters=runtime_filter)
        call_kwargs = store._keyword_retrieval.call_args[1]
        assert call_kwargs["filters"] == runtime_filter


# ---------------------------------------------------------------------------
# Integration tests (require real MariaDB)
# ---------------------------------------------------------------------------

MARIADB_HOST = os.environ.get("MARIADB_HOST", "localhost")
MARIADB_PORT = int(os.environ.get("MARIADB_PORT", "3306"))
MARIADB_DB = os.environ.get("MARIADB_DATABASE", "haystack")
MARIADB_USER = os.environ.get("MARIADB_USER", "root")
MARIADB_PASSWORD = os.environ.get("MARIADB_PASSWORD", "password")


@pytest.fixture
def integration_store():
    store = MariaDBDocumentStore(
        host=MARIADB_HOST,
        port=MARIADB_PORT,
        database=MARIADB_DB,
        user=MARIADB_USER,
        password=MARIADB_PASSWORD,
        table_name="test_retrievers_docs",
        embedding_dimension=4,
        recreate_table=True,
    )
    yield store
    try:
        store._cursor.execute("DROP TABLE IF EXISTS `test_retrievers_docs`")
    except Exception:  # noqa: BLE001
        pass
    store.close()


@pytest.mark.integration
class TestIntegrationRetrievers:
    def test_embedding_retriever(self, integration_store):
        docs = [
            Document(content="cats are great", embedding=[1.0, 0.0, 0.0, 0.0]),
            Document(content="dogs are loyal", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        integration_store.write_documents(docs)
        retriever = MariaDBEmbeddingRetriever(document_store=integration_store, top_k=1)
        result = retriever.run(query_embedding=[1.0, 0.0, 0.0, 0.0])
        docs_out = result["documents"]
        assert len(docs_out) == 1
        assert docs_out[0].content == "cats are great"

    def test_keyword_retriever(self, integration_store):
        docs = [
            Document(content="natural language processing"),
            Document(content="computer vision research"),
        ]
        integration_store.write_documents(docs)
        retriever = MariaDBKeywordRetriever(document_store=integration_store, top_k=5)
        result = retriever.run(query="language")
        assert any("language" in d.content for d in result["documents"])

    def test_embedding_retriever_l2(self, integration_store):
        docs = [
            Document(content="near origin", embedding=[0.1, 0.1, 0.1, 0.1]),
            Document(content="far origin", embedding=[10.0, 10.0, 10.0, 10.0]),
        ]
        integration_store.write_documents(docs)
        retriever = MariaDBEmbeddingRetriever(
            document_store=integration_store, top_k=1, vector_function="l2_distance"
        )
        result = retriever.run(query_embedding=[0.0, 0.0, 0.0, 0.0])
        assert result["documents"][0].content == "near origin"
