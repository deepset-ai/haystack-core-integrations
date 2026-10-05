# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.testing.document_store_async import (
    CountDocumentsAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    DeleteDocumentsAsyncTest,
    FilterDocumentsAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
    UpdateByFilterAsyncTest,
    WriteDocumentsAsyncTest,
)
from redis.exceptions import ResponseError

from haystack_integrations.document_stores.falkordb import FalkorDBDocumentStore
from haystack_integrations.document_stores.falkordb import document_store as document_store_module

from .test_document_store_common import FalkorDBDocumentStoreTestMixin

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_falkordb(monkeypatch):
    constructor = MagicMock()
    client = MagicMock()
    graph = MagicMock()
    client.select_graph.return_value = graph
    graph.query.return_value = MagicMock(result_set=[])
    constructor.return_value = client
    monkeypatch.setattr(document_store_module, "FalkorDB", constructor)
    return constructor, client, graph


@pytest.fixture
def mock_async_falkordb(monkeypatch):
    constructor = MagicMock()
    client = MagicMock()
    graph = MagicMock()
    client.select_graph.return_value = graph
    client.aclose = AsyncMock()
    graph.delete = AsyncMock()
    graph.query = AsyncMock(return_value=MagicMock(result_set=[]))
    constructor.return_value = client
    monkeypatch.setattr(document_store_module, "AsyncFalkorDB", constructor)
    return constructor, client, graph


def _result(rows):
    return MagicMock(result_set=rows)


@pytest.fixture
def warmed_async_store(mock_async_falkordb):
    _, client, graph = mock_async_falkordb
    store = FalkorDBDocumentStore(write_batch_size=2)
    store.async_client = client
    store.async_graph = graph
    store.async_initialized = True
    return store, graph


class TestFalkorDBDocumentStoreAsyncUnit:
    @pytest.mark.asyncio
    async def test_warm_up_async_initializes_client_and_schema(self, mock_async_falkordb) -> None:
        constructor, client, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()

        await store.warm_up_async()

        constructor.assert_called_once()
        assert store.async_client is client
        assert store.async_graph is graph
        assert store.async_initialized is True
        assert graph.query.await_count == 2

    @pytest.mark.asyncio
    async def test_warm_up_async_is_idempotent(self, mock_async_falkordb) -> None:
        constructor, _, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()

        await store.warm_up_async()
        await store.warm_up_async()

        constructor.assert_called_once()
        assert graph.query.await_count == 2

    @pytest.mark.asyncio
    async def test_close_then_warm_up_async_reopens(self, mock_async_falkordb) -> None:
        constructor, client, graph = mock_async_falkordb
        store = FalkorDBDocumentStore()
        await store.warm_up_async()

        await store.close_async()
        await store.warm_up_async()

        assert constructor.call_count == 2
        client.aclose.assert_awaited_once()
        assert store.async_client is client
        assert store.async_graph is graph
        assert store.async_initialized is True
        assert graph.query.await_count == 4

    @pytest.mark.asyncio
    async def test_warm_up_async_suppresses_missing_graph_error(self, mock_async_falkordb) -> None:
        _, _, graph = mock_async_falkordb
        graph.delete.side_effect = ResponseError("Invalid graph operation on empty key")
        store = FalkorDBDocumentStore(recreate_graph=True)

        await store.warm_up_async()

        assert store._recreate_graph_applied is True
        assert store.async_initialized is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [ResponseError("unexpected response"), ConnectionError("connection lost")],
        ids=["response-error", "connection-error"],
    )
    async def test_warm_up_async_retries_graph_deletion_after_error(self, mock_async_falkordb, error) -> None:
        _, _, graph = mock_async_falkordb
        graph.delete.side_effect = [error, None]
        store = FalkorDBDocumentStore(recreate_graph=True)

        with pytest.raises(type(error), match=str(error)):
            await store.warm_up_async()

        assert store._recreate_graph_applied is False

        await store.warm_up_async()

        assert graph.delete.await_count == 2
        assert store._recreate_graph_applied is True
        assert store.async_initialized is True

    @pytest.mark.asyncio
    async def test_sync_then_async_warm_up_recreates_graph_once(self, mock_falkordb, mock_async_falkordb) -> None:
        _, _, graph = mock_falkordb
        _, _, async_graph = mock_async_falkordb
        store = FalkorDBDocumentStore(recreate_graph=True)

        store.warm_up()
        await store.warm_up_async()

        graph.delete.assert_called_once_with()
        async_graph.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_async_then_sync_warm_up_recreates_graph_once(self, mock_falkordb, mock_async_falkordb) -> None:
        _, _, graph = mock_falkordb
        _, _, async_graph = mock_async_falkordb
        store = FalkorDBDocumentStore(recreate_graph=True)

        await store.warm_up_async()
        store.warm_up()

        async_graph.delete.assert_awaited_once_with()
        graph.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_warm_up_async_initializes_once(self, mock_async_falkordb) -> None:
        constructor, _, graph = mock_async_falkordb

        # Plain AsyncMocks never suspend; yield like real I/O so the warm-ups interleave.
        async def yield_to_other_tasks(*_args):
            await asyncio.sleep(0)
            return _result([])

        graph.delete.side_effect = yield_to_other_tasks
        graph.query.side_effect = yield_to_other_tasks
        store = FalkorDBDocumentStore(recreate_graph=True)

        await asyncio.gather(*(store.warm_up_async() for _ in range(5)))

        constructor.assert_called_once()
        graph.delete.assert_awaited_once_with()
        assert graph.query.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rows, expected", [([[7]], 7), ([], 0)])
    async def test_count_documents_async(self, warmed_async_store, rows, expected) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result(rows)

        assert await store.count_documents_async() == expected

    @pytest.mark.asyncio
    async def test_filter_documents_async_without_filters(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        node = SimpleNamespace(properties={"id": "doc-1", "content": "hello"})
        graph.query.return_value = _result([[node]])

        documents = await store.filter_documents_async()

        assert [document.content for document in documents] == ["hello"]
        assert "WHERE" not in graph.query.await_args.args[0]

    @pytest.mark.asyncio
    async def test_filter_documents_async_passes_filter_params(self, warmed_async_store) -> None:
        store, graph = warmed_async_store

        assert await store.filter_documents_async({"field": "year", "operator": "==", "value": 2024}) == []
        assert "WHERE" in graph.query.await_args.args[0]
        assert graph.query.await_args.args[1] == {"p0": 2024}

    @pytest.mark.asyncio
    async def test_filter_documents_async_rejects_malformed_filter(self, warmed_async_store) -> None:
        store, _ = warmed_async_store

        with pytest.raises(FilterError, match="Invalid filter syntax"):
            await store.filter_documents_async({"field": "year", "value": 2024})

    @pytest.mark.asyncio
    async def test_write_documents_async_rejects_non_documents(self, warmed_async_store) -> None:
        store, _ = warmed_async_store

        with pytest.raises(ValueError, match="expects a list of Documents"):
            await store.write_documents_async(["not a document"])

    @pytest.mark.asyncio
    async def test_write_documents_async_empty_is_noop(self, warmed_async_store, caplog) -> None:
        store, graph = warmed_async_store

        with caplog.at_level(logging.WARNING):
            assert await store.write_documents_async([]) == 0
        assert "empty list" in caplog.text
        graph.query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_write_documents_async_none_policy_fails_on_existing(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result([["a"]])

        with pytest.raises(DuplicateDocumentError, match="already exists"):
            await store.write_documents_async([Document(id="a", content="existing")])

    @pytest.mark.asyncio
    async def test_write_documents_async_skip_excludes_existing(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = [_result([["a"]]), _result([[1]])]

        written = await store.write_documents_async(
            [Document(id="a", content="old"), Document(id="b", content="new")], policy=DuplicatePolicy.SKIP
        )

        assert written == 1
        assert [record["id"] for record in graph.query.await_args_list[-1].args[1]["docs"]] == ["b"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("policy", [DuplicatePolicy.FAIL, DuplicatePolicy.OVERWRITE])
    async def test_write_documents_async_batches_normal_and_overwrite(self, warmed_async_store, policy) -> None:
        store, graph = warmed_async_store
        batch_results = [_result([[2]]), _result([[1]])]
        graph.query.side_effect = ([_result([])] if policy == DuplicatePolicy.FAIL else []) + batch_results
        documents = [Document(id=str(index), content="text") for index in range(3)]

        assert await store.write_documents_async(documents, policy=policy) == 3

        write_calls = graph.query.await_args_list[-2:]
        assert [len(call.args[1]["docs"]) for call in write_calls] == [2, 1]
        assert ("ON MATCH SET d = doc" in write_calls[0].args[0]) is (policy == DuplicatePolicy.OVERWRITE)

    @pytest.mark.asyncio
    async def test_write_documents_async_writes_embeddings_separately(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = [_result([[1]]), _result([])]

        assert (
            await store.write_documents_async(
                [Document(id="a", content="text", embedding=[0.1, 0.2])], policy=DuplicatePolicy.OVERWRITE
            )
            == 1
        )
        embedding_call = graph.query.await_args_list[-1]
        assert "vecf32" in embedding_call.args[0]
        assert embedding_call.args[1]["docs"] == [{"id": "a", "emb": [0.1, 0.2]}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "side_effect, document, message",
        [
            ([RuntimeError("write failed")], Document(id="a"), "Failed to write documents"),
            (
                [_result([[1]]), RuntimeError("embedding failed")],
                Document(id="a", embedding=[0.1]),
                "Failed to set embeddings",
            ),
        ],
    )
    async def test_write_documents_async_wraps_database_errors(
        self, warmed_async_store, side_effect, document, message
    ) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = side_effect

        with pytest.raises(DocumentStoreError, match=message):
            await store.write_documents_async([document], policy=DuplicatePolicy.OVERWRITE)

    @pytest.mark.asyncio
    async def test_delete_documents_async_empty_and_nonempty(self, warmed_async_store) -> None:
        store, graph = warmed_async_store

        await store.delete_documents_async([])
        graph.query.assert_not_awaited()
        await store.delete_documents_async(["a", "b"])

        assert "DETACH DELETE" in graph.query.await_args.args[0]
        assert graph.query.await_args.args[1] == {"ids": ["a", "b"]}

    @pytest.mark.asyncio
    async def test_delete_all_documents_async(self, warmed_async_store) -> None:
        store, graph = warmed_async_store

        await store.delete_all_documents_async()

        assert "DETACH DELETE" in graph.query.await_args.args[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rows, expected", [([[3]], 3), ([], 0)])
    async def test_delete_by_filter_async(self, warmed_async_store, rows, expected) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = [_result(rows), _result([])]

        count = await store.delete_by_filter_async({"field": "year", "operator": "==", "value": 2024})

        assert count == expected
        assert "DETACH DELETE" in graph.query.await_args_list[-1].args[0]
        assert graph.query.await_args_list[-1].args[1] == {"p0": 2024}

    @pytest.mark.asyncio
    async def test_update_by_filter_async_flattens_metadata(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result([[2]])

        count = await store.update_by_filter_async(
            {"field": "year", "operator": "==", "value": 2024},
            {"meta.status": "published", "owner": "team"},
        )

        assert count == 2
        assert graph.query.await_args.args[1]["meta_update"] == {"status": "published", "owner": "team"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rows, expected", [([[5]], 5), ([], 0)])
    async def test_count_documents_by_filter_async(self, warmed_async_store, rows, expected) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result(rows)

        assert (
            await store.count_documents_by_filter_async({"field": "year", "operator": ">", "value": 2020}) == expected
        )
        assert graph.query.await_args.args[1] == {"p0": 2020}

    @pytest.mark.asyncio
    async def test_count_unique_metadata_by_filter_async(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = [_result([[3]]), _result([])]

        counts = await store.count_unique_metadata_by_filter_async(
            {"field": "year", "operator": ">=", "value": 2020}, ["meta.category", "status"]
        )

        assert counts == {"category": 3, "status": 0}
        assert all(call.args[1] == {"p0": 2020} for call in graph.query.await_args_list)

    @pytest.mark.asyncio
    async def test_get_metadata_fields_info_async_type_branches(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = [
            _result([[["id", "active", "category", "missing", "rating", "year"]]]),
            _result([[True]]),
            _result([["news"]]),
            _result([]),
            _result([[4.5]]),
            _result([[2024]]),
        ]

        info = await store.get_metadata_fields_info_async()

        assert info == {
            "active": {"type": "bool"},
            "category": {"type": "str"},
            "rating": {"type": "float"},
            "year": {"type": "int"},
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rows, expected", [([[2020, 2024]], {"min": 2020, "max": 2024}), ([], {"min": None, "max": None})]
    )
    async def test_get_metadata_field_min_max_async(self, warmed_async_store, rows, expected) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result(rows)

        assert await store.get_metadata_field_min_max_async("meta.year") == expected
        assert "d.year" in graph.query.await_args.args[0]

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_with_filter_and_search(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result([[["Apple"], 1]])

        values = await store.get_metadata_field_unique_values_async(
            "meta.category",
            search_term="app",
            from_=2,
            size=3,
            filters={"field": "year", "operator": "==", "value": 2024},
        )

        assert values == (["Apple"], 1)
        query, params = graph.query.await_args.args
        assert "d.category" in query
        assert "CONTAINS" in query
        assert params == {"from_": 2, "size": 3, "p0": 2024, "search_term": "app"}

    @pytest.mark.asyncio
    async def test_get_metadata_field_unique_values_async_empty(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.return_value = _result([])

        assert await store.get_metadata_field_unique_values_async("category") == ([], 0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "similarity, raw_score, scale_score, filters, expected_score",
        [
            ("cosine", 0.4, True, None, 0.8),
            ("euclidean", 1.0, True, {"field": "year", "operator": "==", "value": 2024}, 0.5),
            ("cosine", 0.4, False, None, 0.4),
        ],
    )
    async def test_embedding_retrieval_async(
        self, warmed_async_store, similarity, raw_score, scale_score, filters, expected_score
    ) -> None:
        store, graph = warmed_async_store
        store.similarity = similarity
        node = SimpleNamespace(properties={"id": "doc-1", "content": "hello"})
        graph.query.return_value = _result([[node, raw_score]])

        documents = await store._embedding_retrieval_async(
            query_embedding=[0.1, 0.2], top_k=3, filters=filters, scale_score=scale_score
        )

        assert len(documents) == 1
        assert documents[0].content == "hello"
        assert documents[0].score == pytest.approx(expected_score)
        query, params = graph.query.await_args.args
        assert params["top_k"] == 3
        assert params["query_embedding"] == [0.1, 0.2]
        assert ("WHERE" in query) is (filters is not None)
        if filters is not None:
            assert params["p0"] == 2024

    @pytest.mark.asyncio
    async def test_cypher_retrieval_async(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        node = SimpleNamespace(properties={"id": "doc-1", "content": "hello"})
        graph.query.return_value = _result([[node]])

        documents = await store._cypher_retrieval_async("MATCH (d) WHERE d.id = $id RETURN d", {"id": "doc-1"})

        assert [document.content for document in documents] == ["hello"]
        graph.query.assert_awaited_once_with("MATCH (d) WHERE d.id = $id RETURN d", {"id": "doc-1"})

    @pytest.mark.asyncio
    async def test_cypher_retrieval_async_wraps_errors(self, warmed_async_store) -> None:
        store, graph = warmed_async_store
        graph.query.side_effect = RuntimeError("query failed")

        with pytest.raises(DocumentStoreError, match="Cypher query failed: query failed"):
            await store._cypher_retrieval_async("INVALID")


@pytest.mark.integration
@pytest.mark.asyncio
class TestFalkorDBDocumentStoreAsync(
    FalkorDBDocumentStoreTestMixin,
    CountDocumentsAsyncTest,
    WriteDocumentsAsyncTest,
    DeleteDocumentsAsyncTest,
    DeleteAllAsyncTest,
    DeleteByFilterAsyncTest,
    FilterDocumentsAsyncTest,
    UpdateByFilterAsyncTest,
    CountDocumentsByFilterAsyncTest,
    CountUniqueMetadataByFilterAsyncTest,
    GetMetadataFieldsInfoAsyncTest,
    GetMetadataFieldMinMaxAsyncTest,
    GetMetadataFieldUniqueValuesAsyncTest,
):
    @pytest_asyncio.fixture
    async def document_store(self) -> AsyncGenerator[FalkorDBDocumentStore, None]:
        graph_name = f"test_async_graph_{uuid.uuid4().hex}"
        store = FalkorDBDocumentStore(
            host=os.environ.get("FALKORDB_HOST", "localhost"),
            port=int(os.environ.get("FALKORDB_PORT", "6379")),
            graph_name=graph_name,
            embedding_dim=768,
            recreate_graph=True,
        )
        yield store

        try:
            if store.async_graph is not None:
                await store.async_graph.delete()
        except Exception:
            logger.debug("Could not delete graph %s during teardown", graph_name)
        finally:
            await store.close_async()

    async def test_write_documents_async(self, document_store: FalkorDBDocumentStore) -> None:
        """FalkorDB defaults to failing on duplicates; verify a normal default write."""
        document = Document(content="test doc")
        assert await document_store.write_documents_async([document]) == 1
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [document])

    async def test_get_metadata_field_unique_values_distinct_types_async(
        self, document_store: FalkorDBDocumentStore
    ) -> None:
        """Avoid Cypher DISTINCT collapsing a whole-number float and an equal integer."""
        documents = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        await document_store.write_documents_async(documents)

        int_values, int_count = await document_store.get_metadata_field_unique_values_async("priority_int")
        str_values, str_count = await document_store.get_metadata_field_unique_values_async("priority_str")
        float_values, float_count = await document_store.get_metadata_field_unique_values_async("priority_float")
        bool_values, bool_count = await document_store.get_metadata_field_unique_values_async("priority_bool")

        assert (int_count, str_count, float_count, bool_count) == (1, 1, 1, 1)
        assert int_values == [1] and type(int_values[0]) is int
        assert str_values == ["1"] and type(str_values[0]) is str
        assert float_values == [1.5] and type(float_values[0]) is float
        assert bool_values == [True] and type(bool_values[0]) is bool

    async def test_embedding_retrieval_async_integration(self, document_store: FalkorDBDocumentStore) -> None:
        document = Document(content="hello", embedding=[0.1] * 768)
        await document_store.write_documents_async([document])

        documents = await document_store._embedding_retrieval_async([0.1] * 768, top_k=1)

        assert len(documents) == 1
        assert documents[0].id == document.id
        assert documents[0].score == pytest.approx(1.0)

    async def test_cypher_retrieval_async_integration(self, document_store: FalkorDBDocumentStore) -> None:
        document = Document(content="hello")
        await document_store.write_documents_async([document])

        documents = await document_store._cypher_retrieval_async(
            f"MATCH (d:{document_store.node_label} {{id: $id}}) RETURN d", {"id": document.id}
        )

        self.assert_documents_are_equal(documents, [document])

    async def test_close_async_and_reopen(self, document_store: FalkorDBDocumentStore) -> None:
        document = Document(content="hello")
        await document_store.write_documents_async([document])

        await document_store.close_async()
        assert document_store.async_client is None
        await document_store.warm_up_async()
        assert document_store.async_client is not None
        self.assert_documents_are_equal(await document_store.filter_documents_async(), [document])
