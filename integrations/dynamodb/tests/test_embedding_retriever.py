# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.dynamodb import DynamoDBEmbeddingRetriever
from haystack_integrations.document_stores.dynamodb import DynamoDBDocumentStore

from .conftest import make_store

_INIT_FILTER = {"field": "meta.topic", "operator": "==", "value": "ai"}
_RUNTIME_FILTER = {"field": "meta.year", "operator": ">=", "value": 2024}


def _serializable_store() -> DynamoDBDocumentStore:
    """Env-var secrets (the defaults) serialize; the token secrets in `make_store` deliberately do not."""
    return DynamoDBDocumentStore(table_name="test_docs", index_name="test_index", embedding_dimension=3)


class TestDynamoDBEmbeddingRetriever:
    def test_init_defaults(self) -> None:
        retriever = DynamoDBEmbeddingRetriever(document_store=make_store())
        assert retriever.top_k == 10
        assert retriever.filters is None
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_init_rejects_wrong_store_type(self) -> None:
        with pytest.raises(ValueError, match="must be a DynamoDBDocumentStore"):
            DynamoDBEmbeddingRetriever(document_store="not a store")  # type: ignore[arg-type]

    @pytest.mark.parametrize("top_k", [0, 101])
    def test_init_rejects_top_k_outside_dynamodb_limit(self, top_k: int) -> None:
        with pytest.raises(ValueError, match="top_k must be between 1 and 100"):
            DynamoDBEmbeddingRetriever(document_store=make_store(), top_k=top_k)

    def test_init_accepts_filter_policy_as_string(self) -> None:
        retriever = DynamoDBEmbeddingRetriever(document_store=make_store(), filter_policy="merge")
        assert retriever.filter_policy == FilterPolicy.MERGE

    def test_run_delegates_to_store_embedding_retrieval(self) -> None:
        store = make_store()
        store._embedding_retrieval = MagicMock(return_value=[Document(id="1", content="hello")])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5)

        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        store._embedding_retrieval.assert_called_once_with(query_embedding=[0.1, 0.2, 0.3], top_k=5, filters=None)
        assert result["documents"][0].id == "1"

    def test_run_replace_policy_uses_runtime_filters_and_top_k(self) -> None:
        store = make_store()
        store._embedding_retrieval = MagicMock(return_value=[])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5, filters=_INIT_FILTER)

        retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=2, filters=_RUNTIME_FILTER)

        store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3], top_k=2, filters=_RUNTIME_FILTER
        )

    def test_run_replace_policy_falls_back_to_init_filters(self) -> None:
        store = make_store()
        store._embedding_retrieval = MagicMock(return_value=[])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, filters=_INIT_FILTER)

        retriever.run(query_embedding=[0.1, 0.2, 0.3])

        assert store._embedding_retrieval.call_args.kwargs["filters"] == _INIT_FILTER

    def test_run_merge_policy_combines_init_and_runtime_filters(self) -> None:
        store = make_store()
        store._embedding_retrieval = MagicMock(return_value=[])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(
            document_store=store, filters=_INIT_FILTER, filter_policy=FilterPolicy.MERGE
        )

        retriever.run(query_embedding=[0.1, 0.2, 0.3], filters=_RUNTIME_FILTER)

        assert store._embedding_retrieval.call_args.kwargs["filters"] == {
            "operator": "AND",
            "conditions": [_INIT_FILTER, _RUNTIME_FILTER],
        }

    async def test_run_async_delegates_to_store_async_embedding_retrieval(self) -> None:
        store = make_store()
        store._embedding_retrieval_async = AsyncMock(return_value=[Document(id="1", content="hello")])  # type: ignore[method-assign]
        retriever = DynamoDBEmbeddingRetriever(document_store=store, top_k=5, filters=_INIT_FILTER)

        result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3], top_k=3)

        store._embedding_retrieval_async.assert_awaited_once_with(
            query_embedding=[0.1, 0.2, 0.3], top_k=3, filters=_INIT_FILTER
        )
        assert result["documents"][0].id == "1"

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        store = _serializable_store()
        retriever = DynamoDBEmbeddingRetriever(
            document_store=store, top_k=7, filters=_INIT_FILTER, filter_policy=FilterPolicy.MERGE
        )

        data = retriever.to_dict()
        assert data["init_parameters"]["filter_policy"] == "merge"
        assert data["init_parameters"]["filters"] == _INIT_FILTER
        assert data["init_parameters"]["document_store"]["type"].endswith("DynamoDBDocumentStore")

        rebuilt = DynamoDBEmbeddingRetriever.from_dict(data)
        assert rebuilt.top_k == 7
        assert rebuilt.filters == _INIT_FILTER
        assert rebuilt.filter_policy == FilterPolicy.MERGE
        assert isinstance(rebuilt.document_store, DynamoDBDocumentStore)
        assert rebuilt.document_store.table_name == store.table_name

    def test_from_dict_defaults_filter_policy_when_absent(self) -> None:
        data = DynamoDBEmbeddingRetriever(document_store=_serializable_store()).to_dict()
        del data["init_parameters"]["filter_policy"]
        assert DynamoDBEmbeddingRetriever.from_dict(data).filter_policy == FilterPolicy.REPLACE
