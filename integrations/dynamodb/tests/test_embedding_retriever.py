# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.utils import Secret

from haystack_integrations.components.retrievers.dynamodb import DynamoDBEmbeddingRetriever
from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore


def _make_store() -> DynamoDBDocumentStore:
    return DynamoDBDocumentStore(
        table_name="test_docs",
        index_name="test_index",
        embedding_dimension=3,
        region_name="us-east-1",
        aws_access_key_id=Secret.from_token("test-key"),
        aws_secret_access_key=Secret.from_token("test-secret"),
    )


class TestDynamoDBEmbeddingRetriever:
    def test_init_rejects_wrong_store_type(self) -> None:
        with pytest.raises(ValueError, match="must be a DynamoDBDocumentStore"):
            DynamoDBEmbeddingRetriever(document_store="not a store")  # type: ignore[arg-type]

    def test_run_delegates_to_store_embedding_retrieval(self) -> None:
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[Document(id="1", content="hello")])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], top_k=5, filters=None
        )
        assert result["documents"][0].id == "1"

    def test_run_call_time_overrides_take_precedence(self) -> None:
        store = _make_store()
        store._embedding_retrieval = MagicMock(return_value=[])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5, filters={"a": "b"})

        retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=2, filters={"c": "d"})

        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], top_k=2, filters={"c": "d"}
        )

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        store = _make_store()
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=7)

        data = retriever.to_dict()
        rebuilt = DynamoDBEmbeddingRetriever.from_dict(data)

        assert rebuilt.top_k == 7
        assert isinstance(rebuilt.document_store, DynamoDBDocumentStore)
        assert rebuilt.document_store.table_name == store.table_name
