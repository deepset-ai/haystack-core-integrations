# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret

from haystack_integrations.document_stores.arangodb import ArangoDocumentStore

_MODULE = "haystack_integrations.document_stores.arangodb.document_store"


def _make_store(**kwargs) -> ArangoDocumentStore:
    return ArangoDocumentStore(
        host="http://localhost:8529",
        database="haystack",
        username="root",
        password=Secret.from_token("test-password"),
        collection_name="test_docs",
        embedding_dimension=3,
        **kwargs,
    )


def _mock_db(store: ArangoDocumentStore, collection_docs: list[dict] | None = None) -> MagicMock:
    mock_col = MagicMock()
    mock_col.count.return_value = len(collection_docs or [])
    mock_col.has.return_value = False

    mock_db = MagicMock()
    mock_db.aql.execute.return_value = iter(collection_docs or [])

    store._db = mock_db
    store._col = mock_col
    return mock_db


class TestArangoDocumentStoreInit:
    def test_init_defaults(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        store = ArangoDocumentStore(
            host="http://localhost:8529",
            database="mydb",
            username="root",
            collection_name="docs",
            embedding_dimension=768,
        )
        assert store.host == "http://localhost:8529"
        assert store.database == "mydb"
        assert store.username == "root"
        assert store.collection_name == "docs"
        assert store.embedding_dimension == 768
        assert store.recreate_collection is False
        assert store._db is None
        assert store._col is None

    def test_init_custom(self):
        store = _make_store(recreate_collection=True)
        assert store.recreate_collection is True
        assert store.embedding_dimension == 3


class TestArangoDocumentStoreSerialization:
    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        store = ArangoDocumentStore(
            host="http://localhost:8529",
            database="haystack",
            username="root",
            collection_name="test_docs",
            embedding_dimension=3,
        )
        d = store.to_dict()
        assert d["type"].endswith("ArangoDocumentStore")
        params = d["init_parameters"]
        assert params["host"] == "http://localhost:8529"
        assert params["database"] == "haystack"
        assert params["collection_name"] == "test_docs"
        assert params["embedding_dimension"] == 3
        assert params["recreate_collection"] is False

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        data = {
            "type": "haystack_integrations.document_stores.arangodb.document_store.ArangoDocumentStore",
            "init_parameters": {
                "host": "http://localhost:8529",
                "database": "haystack",
                "username": "root",
                "password": {"env_vars": ["ARANGO_PASSWORD"], "strict": True, "type": "env_var"},
                "collection_name": "test_docs",
                "embedding_dimension": 3,
                "recreate_collection": False,
            },
        }
        store = ArangoDocumentStore.from_dict(data)
        assert store.host == "http://localhost:8529"
        assert store.collection_name == "test_docs"
        assert store.password.resolve_value() == "pw"

    def test_to_from_dict_roundtrip(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        store = ArangoDocumentStore(
            host="http://localhost:8529",
            database="haystack",
            username="root",
            collection_name="test_docs",
            embedding_dimension=3,
        )
        store2 = ArangoDocumentStore.from_dict(store.to_dict())
        assert store.host == store2.host
        assert store.database == store2.database
        assert store.collection_name == store2.collection_name
        assert store.embedding_dimension == store2.embedding_dimension


class TestArangoDocumentStoreCountDocuments:
    def test_count_documents(self):
        store = _make_store()
        _mock_db(store, [{"_key": "1"}, {"_key": "2"}])
        store._col.count.return_value = 2
        assert store.count_documents() == 2

    def test_count_documents_empty(self):
        store = _make_store()
        _mock_db(store, [])
        store._col.count.return_value = 0
        assert store.count_documents() == 0


class TestArangoDocumentStoreWriteDocuments:
    def test_write_documents(self):
        store = _make_store()
        _mock_db(store)
        docs = [Document(content="hello"), Document(content="world")]
        written = store.write_documents(docs)
        assert written == 2
        assert store._col.insert.call_count == 2

    def test_write_empty(self):
        store = _make_store()
        _mock_db(store)
        assert store.write_documents([]) == 0

    def test_write_invalid_type(self):
        store = _make_store()
        _mock_db(store)
        with pytest.raises(ValueError, match="Document"):
            store.write_documents(["not a document"])

    def test_write_duplicate_fail(self):
        store = _make_store()
        _mock_db(store)
        doc = Document(content="test")
        store._col.has.return_value = True
        with pytest.raises(DuplicateDocumentError):
            store.write_documents([doc], policy=DuplicatePolicy.FAIL)

    def test_write_duplicate_skip(self):
        store = _make_store()
        _mock_db(store)
        store._col.has.return_value = True
        doc = Document(content="test")
        written = store.write_documents([doc], policy=DuplicatePolicy.SKIP)
        assert written == 0
        store._col.insert.assert_not_called()

    def test_write_duplicate_overwrite(self):
        store = _make_store()
        _mock_db(store)
        store._col.has.return_value = True
        doc = Document(content="test")
        written = store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE)
        assert written == 1
        store._col.replace.assert_called_once()


class TestArangoDocumentStoreDeleteDocuments:
    def test_delete_documents(self):
        store = _make_store()
        _mock_db(store)
        store._col.has.return_value = True
        store.delete_documents(["id1", "id2"])
        assert store._col.delete.call_count == 2

    def test_delete_empty(self):
        store = _make_store()
        _mock_db(store)
        store.delete_documents([])
        store._col.delete.assert_not_called()

    def test_delete_missing(self):
        store = _make_store()
        _mock_db(store)
        store._col.has.return_value = False
        store.delete_documents(["missing"])
        store._col.delete.assert_not_called()


class TestArangoDocumentStoreFilterDocuments:
    def test_filter_documents_no_filter(self):
        store = _make_store()
        arango_docs = [
            {"_key": "1", "_id": "test_docs/1", "_rev": "abc", "content": "hello", "meta": {}},
            {"_key": "2", "_id": "test_docs/2", "_rev": "def", "content": "world", "meta": {}},
        ]
        _mock_db(store, arango_docs)
        docs = store.filter_documents()
        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[1].content == "world"

    def test_filter_documents_with_filter(self):
        store = _make_store()
        arango_docs = [{"_key": "1", "_id": "c/1", "_rev": "a", "content": "hello", "meta": {"topic": "ai"}}]
        mock_db = _mock_db(store, arango_docs)
        filters = {"field": "meta.topic", "operator": "==", "value": "ai"}
        docs = store.filter_documents(filters=filters)
        assert len(docs) == 1
        assert docs[0].meta["topic"] == "ai"
        mock_db.aql.execute.assert_called_once()
        call_args = mock_db.aql.execute.call_args
        assert "FILTER" in call_args[0][0]


class TestArangoDocumentStoreEmbeddingRetrieval:
    def test_embedding_retrieval(self):
        store = _make_store()
        arango_docs = [
            {
                "_key": "1",
                "_id": "c/1",
                "_rev": "a",
                "content": "doc1",
                "embedding": [0.1, 0.2, 0.3],
                "meta": {},
                "score": 0.99,
            },
        ]
        mock_db = _mock_db(store, arango_docs)
        docs = store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=5)
        assert len(docs) == 1
        assert docs[0].score == pytest.approx(0.99)
        mock_db.aql.execute.assert_called_once()

    def test_embedding_retrieval_empty_query(self):
        store = _make_store()
        _mock_db(store)
        with pytest.raises(ValueError, match="query_embedding"):
            store._embedding_retrieval(query_embedding=[])


@pytest.mark.skipif(
    not os.environ.get("ARANGO_PASSWORD") or not os.environ.get("ARANGO_HOST"),
    reason="Export ARANGO_HOST and ARANGO_PASSWORD to run integration tests.",
)
@pytest.mark.integration
class TestArangoDocumentStoreIntegration:
    def test_write_and_retrieve(self):
        store = ArangoDocumentStore(
            host=os.environ["ARANGO_HOST"],
            database="haystack_test",
            username="root",
            password=Secret.from_env_var("ARANGO_PASSWORD"),
            collection_name="integration_test",
            embedding_dimension=3,
            recreate_collection=True,
        )
        docs = [
            Document(content="Paris is the capital of France.", embedding=[0.1, 0.2, 0.3]),
            Document(content="London is the capital of the UK.", embedding=[0.4, 0.5, 0.6]),
        ]
        written = store.write_documents(docs)
        assert written == 2
        assert store.count_documents() == 2

        results = store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3], top_k=1)
        assert len(results) == 1
        assert "Paris" in results[0].content

        store.delete_documents([docs[0].id])
        assert store.count_documents() == 1
