# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy

from haystack_integrations.components.retrievers.solr import SolrEmbeddingRetriever
from haystack_integrations.document_stores.solr import SolrDocumentStore


@pytest.fixture
def store() -> SolrDocumentStore:
    return SolrDocumentStore(url="http://solr.test/solr", core="unit", auth=None)


class TestInit:
    def test_defaults(self, store):
        retriever = SolrEmbeddingRetriever(document_store=store)
        assert retriever._document_store is store
        assert retriever._filters == {}
        assert retriever._top_k == 10
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._raise_on_failure is True

    def test_rejects_wrong_document_store(self):
        with pytest.raises(ValueError, match="must be an instance of SolrDocumentStore"):
            SolrEmbeddingRetriever(document_store="not a store")

    def test_filter_policy_accepts_a_string(self, store):
        assert SolrEmbeddingRetriever(document_store=store, filter_policy="merge")._filter_policy == FilterPolicy.MERGE


class TestSerialization:
    def test_to_dict(self, store):
        data = SolrEmbeddingRetriever(document_store=store, top_k=3).to_dict()
        assert (
            data["type"]
            == "haystack_integrations.components.retrievers.solr.embedding_retriever.SolrEmbeddingRetriever"
        )
        assert data["init_parameters"]["top_k"] == 3
        assert data["init_parameters"]["filter_policy"] == "replace"

    def test_round_trip(self, store):
        retriever = SolrEmbeddingRetriever(
            document_store=store,
            filters={"field": "meta.page", "operator": "==", "value": "1"},
            top_k=5,
            filter_policy=FilterPolicy.MERGE,
            raise_on_failure=False,
        )
        restored = SolrEmbeddingRetriever.from_dict(retriever.to_dict())
        assert restored.to_dict() == retriever.to_dict()
        assert restored._filter_policy == FilterPolicy.MERGE
        assert restored._top_k == 5
        assert isinstance(restored._document_store, SolrDocumentStore)

    def test_from_dict_without_filter_policy_defaults_to_replace(self, store):
        data = SolrEmbeddingRetriever(document_store=store).to_dict()
        del data["init_parameters"]["filter_policy"]
        assert SolrEmbeddingRetriever.from_dict(data)._filter_policy == FilterPolicy.REPLACE


class TestRun:
    def test_forwards_init_values(self, store):
        store._embedding_retrieval = MagicMock(return_value=[])
        SolrEmbeddingRetriever(document_store=store, top_k=4).run(query_embedding=[0.1, 0.2])
        args, kwargs = store._embedding_retrieval.call_args
        assert args[0] == [0.1, 0.2]
        assert kwargs == {"filters": {}, "top_k": 4}

    def test_runtime_values_win(self, store):
        store._embedding_retrieval = MagicMock(return_value=[])
        SolrEmbeddingRetriever(document_store=store, top_k=4).run(query_embedding=[0.1], top_k=9)
        assert store._embedding_retrieval.call_args.kwargs["top_k"] == 9

    def test_merge_filter_policy(self, store):
        store._embedding_retrieval = MagicMock(return_value=[])
        init_filters = {"field": "meta.a", "operator": "==", "value": 1}
        run_filters = {"field": "meta.b", "operator": "==", "value": 2}
        SolrEmbeddingRetriever(document_store=store, filters=init_filters, filter_policy=FilterPolicy.MERGE).run(
            query_embedding=[0.1], filters=run_filters
        )
        merged = store._embedding_retrieval.call_args.kwargs["filters"]
        assert merged["operator"] == "AND"
        assert init_filters in merged["conditions"]
        assert run_filters in merged["conditions"]

    def test_raises_by_default(self, store):
        store._embedding_retrieval = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            SolrEmbeddingRetriever(document_store=store).run(query_embedding=[0.1])

    def test_swallows_failures_when_asked(self, store, caplog):
        store._embedding_retrieval = MagicMock(side_effect=RuntimeError("boom"))
        retriever = SolrEmbeddingRetriever(document_store=store, raise_on_failure=False)
        assert retriever.run(query_embedding=[0.1]) == {"documents": []}
        assert "boom" in caplog.text

    async def test_run_async(self, store):
        store._embedding_retrieval_async = AsyncMock(return_value=[Document(content="found")])
        result = await SolrEmbeddingRetriever(document_store=store).run_async(query_embedding=[0.1])
        assert result["documents"][0].content == "found"

    async def test_run_async_raises_by_default(self, store):
        store._embedding_retrieval_async = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            await SolrEmbeddingRetriever(document_store=store).run_async(query_embedding=[0.1])

    async def test_run_async_swallows_failures_when_asked(self, store):
        store._embedding_retrieval_async = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = SolrEmbeddingRetriever(document_store=store, raise_on_failure=False)
        assert await retriever.run_async(query_embedding=[0.1]) == {"documents": []}

    async def test_run_async_merges_filters(self, store):
        store._embedding_retrieval_async = AsyncMock(return_value=[])
        await SolrEmbeddingRetriever(document_store=store, top_k=2).run_async(query_embedding=[0.1], top_k=5)
        assert store._embedding_retrieval_async.call_args.kwargs["top_k"] == 5


class TestClose:
    def test_close_forwards_to_the_store(self, store):
        store.close = MagicMock()
        SolrEmbeddingRetriever(document_store=store).close()
        store.close.assert_called_once()

    async def test_close_async_forwards_to_the_store(self, store):
        store.close_async = AsyncMock()
        await SolrEmbeddingRetriever(document_store=store).close_async()
        store.close_async.assert_awaited_once()


@pytest.mark.integration
class TestEmbeddingRetrieverIntegration:
    @staticmethod
    def _documents() -> list[Document]:
        return [
            Document(id="1", content="near", meta={"group": "a"}, embedding=[1.0, 0.0] + [0.0] * 766),
            Document(id="2", content="far", meta={"group": "b"}, embedding=[0.0, 1.0] + [0.0] * 766),
        ]

    def test_retrieves_the_nearest_vector(self, document_store):
        document_store.write_documents(self._documents(), DuplicatePolicy.OVERWRITE)
        retriever = SolrEmbeddingRetriever(document_store=document_store)
        documents = retriever.run(query_embedding=[1.0, 0.0] + [0.0] * 766)["documents"]
        assert documents[0].id == "1"
        assert documents[0].score is not None

    def test_filters_act_as_a_prefilter(self, document_store):
        """
        A filter that excludes the nearest neighbour must still yield the next best match.

        This is why `{!knn}` is the main query rather than an `fq`: as an `fq` Solr applies no implicit
        graph pre-filter and the search comes back short.
        """
        document_store.write_documents(self._documents(), DuplicatePolicy.OVERWRITE)
        retriever = SolrEmbeddingRetriever(document_store=document_store, top_k=1)
        documents = retriever.run(
            query_embedding=[1.0, 0.0] + [0.0] * 766,
            filters={"field": "meta.group", "operator": "==", "value": "b"},
        )["documents"]
        assert [document.id for document in documents] == ["2"]

    def test_top_k(self, document_store):
        document_store.write_documents(
            [Document(id=str(index), content="x", embedding=[1.0, float(index)] + [0.0] * 766) for index in range(5)],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrEmbeddingRetriever(document_store=document_store, top_k=2)
        assert len(retriever.run(query_embedding=[1.0, 0.0] + [0.0] * 766)["documents"]) == 2

    async def test_run_async(self, document_store):
        await document_store.write_documents_async(self._documents(), DuplicatePolicy.OVERWRITE)
        retriever = SolrEmbeddingRetriever(document_store=document_store)
        documents = (await retriever.run_async(query_embedding=[1.0, 0.0] + [0.0] * 766))["documents"]
        assert documents[0].id == "1"
