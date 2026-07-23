# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.arcadedb import ArcadeDBEmbeddingRetriever
from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore


class TestEmbeddingRetriever:
    def test_init_default(self):
        mock_store = Mock(spec=ArcadeDBDocumentStore)
        retriever = ArcadeDBEmbeddingRetriever(document_store=mock_store)
        assert retriever._document_store == mock_store
        assert retriever._filters is None
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE

    def test_init(self):
        mock_store = Mock(spec=ArcadeDBDocumentStore)
        retriever = ArcadeDBEmbeddingRetriever(
            document_store=mock_store, filters={"field": "value"}, top_k=5, filter_policy=FilterPolicy.MERGE
        )
        assert retriever._filters == {"field": "value"}
        assert retriever._top_k == 5
        assert retriever._filter_policy == FilterPolicy.MERGE

    def test_to_dict_from_dict(self):
        store = ArcadeDBDocumentStore(url="http://localhost:2480", database="test", create_database=False)
        retriever = ArcadeDBEmbeddingRetriever(document_store=store, filters={"field": "value"}, top_k=5)

        restored = ArcadeDBEmbeddingRetriever.from_dict(retriever.to_dict())

        assert isinstance(restored._document_store, ArcadeDBDocumentStore)
        assert restored._document_store._database == "test"
        assert restored._filters == {"field": "value"}
        assert restored._top_k == 5
        assert restored._filter_policy == FilterPolicy.REPLACE

    def test_close(self):
        mock_store = Mock(spec=ArcadeDBDocumentStore)
        retriever = ArcadeDBEmbeddingRetriever(document_store=mock_store)

        retriever.close()

        mock_store.close.assert_called_once()
        assert retriever._document_store is mock_store

    def test_run(self):
        mock_store = Mock(spec=ArcadeDBDocumentStore)
        doc = Document(content="Test doc", embedding=[0.1, 0.2])
        mock_store._embedding_retrieval.return_value = [doc]

        retriever = ArcadeDBEmbeddingRetriever(document_store=mock_store)
        res = retriever.run(query_embedding=[0.3, 0.5])

        mock_store._embedding_retrieval.assert_called_once_with(query_embedding=[0.3, 0.5], filters=None, top_k=10)
        assert res == {"documents": [doc]}
