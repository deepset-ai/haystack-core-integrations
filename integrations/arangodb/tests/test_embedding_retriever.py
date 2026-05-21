# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.retrievers.arangodb import ArangoEmbeddingRetriever
from haystack_integrations.document_stores.arangodb import ArangoDocumentStore


def _make_store() -> ArangoDocumentStore:
    return ArangoDocumentStore(
        host="http://localhost:8529",
        database="haystack",
        username="root",
        password=Secret.from_token("pw"),
        collection_name="docs",
        embedding_dimension=3,
    )


class TestArangoEmbeddingRetriever:
    def test_init_default(self):
        store = _make_store()
        retriever = ArangoEmbeddingRetriever(document_store=store)
        assert retriever.document_store is store
        assert retriever.top_k == 10
        assert retriever.filters is None

    def test_init_custom(self):
        store = _make_store()
        filters = {"field": "meta.lang", "operator": "==", "value": "en"}
        retriever = ArangoEmbeddingRetriever(document_store=store, top_k=5, filters=filters)
        assert retriever.top_k == 5
        assert retriever.filters == filters

    def test_run(self):
        store = _make_store()
        expected = [Document(content="result", score=0.9)]
        store._embedding_retrieval = MagicMock(return_value=expected)
        retriever = ArangoEmbeddingRetriever(document_store=store, top_k=3)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        assert result["documents"] == expected
        store._embedding_retrieval.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], top_k=3, filters=None)

    def test_run_overrides_top_k_and_filters(self):
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])
        retriever = ArangoEmbeddingRetriever(document_store=store, top_k=10)
        override_filter = {"field": "meta.x", "operator": "==", "value": 1}

        retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=2, filters=override_filter)

        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], top_k=2, filters=override_filter
        )

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        store = ArangoDocumentStore(
            host="http://localhost:8529",
            database="haystack",
            username="root",
            collection_name="docs",
            embedding_dimension=3,
        )
        retriever = ArangoEmbeddingRetriever(document_store=store, top_k=5)
        d = retriever.to_dict()
        assert d["type"].endswith("ArangoEmbeddingRetriever")
        assert d["init_parameters"]["top_k"] == 5
        assert d["init_parameters"]["document_store"]["type"].endswith("ArangoDocumentStore")

    def test_from_dict(self, monkeypatch):
        monkeypatch.setenv("ARANGO_PASSWORD", "pw")
        data = {
            "type": "haystack_integrations.components.retrievers.arangodb.embedding_retriever.ArangoEmbeddingRetriever",
            "init_parameters": {
                "top_k": 7,
                "filters": None,
                "document_store": {
                    "type": "haystack_integrations.document_stores.arangodb.document_store.ArangoDocumentStore",
                    "init_parameters": {
                        "host": "http://localhost:8529",
                        "database": "haystack",
                        "username": "root",
                        "password": {"env_vars": ["ARANGO_PASSWORD"], "strict": True, "type": "env_var"},
                        "collection_name": "docs",
                        "embedding_dimension": 3,
                        "recreate_collection": False,
                    },
                },
            },
        }
        retriever = ArangoEmbeddingRetriever.from_dict(data)
        assert retriever.top_k == 7
        assert retriever.document_store.collection_name == "docs"
