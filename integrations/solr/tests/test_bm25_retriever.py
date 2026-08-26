# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy

from haystack_integrations.components.retrievers.solr import SolrBM25Retriever
from haystack_integrations.document_stores.solr import SolrDocumentStore


@pytest.fixture
def store() -> SolrDocumentStore:
    """A store that never talks to Solr; the client is only built on first request."""
    return SolrDocumentStore(url="http://solr.test/solr", core="unit", auth=None)


class TestInit:
    def test_defaults(self, store):
        retriever = SolrBM25Retriever(document_store=store)
        assert retriever._document_store is store
        assert retriever._filters == {}
        assert retriever._fuzziness == 0
        assert retriever._top_k == 10
        assert retriever._scale_score is False
        assert retriever._all_terms_must_match is False
        assert retriever._filter_policy == FilterPolicy.REPLACE
        assert retriever._raise_on_failure is True

    def test_rejects_wrong_document_store(self):
        with pytest.raises(ValueError, match="must be an instance of SolrDocumentStore"):
            SolrBM25Retriever(document_store="not a store")

    def test_filter_policy_accepts_a_string(self, store):
        retriever = SolrBM25Retriever(document_store=store, filter_policy="merge")
        assert retriever._filter_policy == FilterPolicy.MERGE


class TestSerialization:
    def test_to_dict(self, store):
        data = SolrBM25Retriever(document_store=store, top_k=3, fuzziness=2).to_dict()
        assert data["type"] == "haystack_integrations.components.retrievers.solr.bm25_retriever.SolrBM25Retriever"
        assert data["init_parameters"]["top_k"] == 3
        assert data["init_parameters"]["fuzziness"] == 2
        assert data["init_parameters"]["filter_policy"] == "replace"
        assert data["init_parameters"]["document_store"]["init_parameters"]["core"] == "unit"

    def test_round_trip(self, store):
        retriever = SolrBM25Retriever(
            document_store=store,
            filters={"field": "meta.page", "operator": "==", "value": "1"},
            top_k=7,
            scale_score=True,
            all_terms_must_match=True,
            filter_policy=FilterPolicy.MERGE,
            raise_on_failure=False,
        )
        restored = SolrBM25Retriever.from_dict(retriever.to_dict())
        assert restored.to_dict() == retriever.to_dict()
        assert restored._filter_policy == FilterPolicy.MERGE
        assert restored._top_k == 7
        assert restored._scale_score is True
        assert restored._all_terms_must_match is True
        assert restored._raise_on_failure is False
        assert isinstance(restored._document_store, SolrDocumentStore)

    def test_from_dict_without_filter_policy_defaults_to_replace(self, store):
        """Pipelines serialized before `filter_policy` existed must still load."""
        data = SolrBM25Retriever(document_store=store).to_dict()
        del data["init_parameters"]["filter_policy"]
        assert SolrBM25Retriever.from_dict(data)._filter_policy == FilterPolicy.REPLACE


class TestRun:
    def test_forwards_init_values(self, store):
        store._bm25_retrieval = MagicMock(return_value=[])
        SolrBM25Retriever(document_store=store, top_k=4, fuzziness=1, scale_score=True).run(query="hello")
        _, kwargs = store._bm25_retrieval.call_args
        assert kwargs == {
            "filters": {},
            "top_k": 4,
            "fuzziness": 1,
            "scale_score": True,
            "all_terms_must_match": False,
        }

    def test_runtime_values_win(self, store):
        store._bm25_retrieval = MagicMock(return_value=[])
        SolrBM25Retriever(document_store=store, top_k=4).run(query="hello", top_k=9)
        assert store._bm25_retrieval.call_args.kwargs["top_k"] == 9

    def test_zero_is_not_mistaken_for_unset(self, store):
        """`fuzziness=0` is meaningful, so the override check has to be `is not None`."""
        store._bm25_retrieval = MagicMock(return_value=[])
        SolrBM25Retriever(document_store=store, fuzziness=2).run(query="hello", fuzziness=0)
        assert store._bm25_retrieval.call_args.kwargs["fuzziness"] == 0

    def test_false_is_not_mistaken_for_unset(self, store):
        store._bm25_retrieval = MagicMock(return_value=[])
        SolrBM25Retriever(document_store=store, scale_score=True).run(query="hello", scale_score=False)
        assert store._bm25_retrieval.call_args.kwargs["scale_score"] is False

    def test_replace_filter_policy(self, store):
        store._bm25_retrieval = MagicMock(return_value=[])
        init_filters = {"field": "meta.a", "operator": "==", "value": 1}
        run_filters = {"field": "meta.b", "operator": "==", "value": 2}
        SolrBM25Retriever(document_store=store, filters=init_filters).run(query="x", filters=run_filters)
        assert store._bm25_retrieval.call_args.kwargs["filters"] == run_filters

    def test_merge_filter_policy(self, store):
        store._bm25_retrieval = MagicMock(return_value=[])
        init_filters = {"field": "meta.a", "operator": "==", "value": 1}
        run_filters = {"field": "meta.b", "operator": "==", "value": 2}
        SolrBM25Retriever(document_store=store, filters=init_filters, filter_policy=FilterPolicy.MERGE).run(
            query="x", filters=run_filters
        )
        merged = store._bm25_retrieval.call_args.kwargs["filters"]
        assert merged["operator"] == "AND"
        assert init_filters in merged["conditions"]
        assert run_filters in merged["conditions"]

    def test_returns_documents(self, store):
        store._bm25_retrieval = MagicMock(return_value=[Document(content="found")])
        assert SolrBM25Retriever(document_store=store).run(query="x")["documents"][0].content == "found"

    def test_raises_by_default(self, store):
        store._bm25_retrieval = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            SolrBM25Retriever(document_store=store).run(query="x")

    def test_swallows_failures_when_asked(self, store, caplog):
        store._bm25_retrieval = MagicMock(side_effect=RuntimeError("boom"))
        result = SolrBM25Retriever(document_store=store, raise_on_failure=False).run(query="x")
        assert result == {"documents": []}
        assert "boom" in caplog.text

    async def test_run_async(self, store):
        store._bm25_retrieval_async = AsyncMock(return_value=[Document(content="found")])
        result = await SolrBM25Retriever(document_store=store).run_async(query="x")
        assert result["documents"][0].content == "found"

    async def test_run_async_swallows_failures_when_asked(self, store):
        store._bm25_retrieval_async = AsyncMock(side_effect=RuntimeError("boom"))
        retriever = SolrBM25Retriever(document_store=store, raise_on_failure=False)
        assert await retriever.run_async(query="x") == {"documents": []}


class TestClose:
    def test_close_forwards_to_the_store(self, store):
        store.close = MagicMock()
        SolrBM25Retriever(document_store=store).close()
        store.close.assert_called_once()

    async def test_close_async_forwards_to_the_store(self, store):
        store.close_async = AsyncMock()
        await SolrBM25Retriever(document_store=store).close_async()
        store.close_async.assert_awaited_once()


@pytest.mark.integration
class TestBM25RetrieverIntegration:
    def test_retrieves_the_best_match(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="Apache Solr is an open source search platform"),
                Document(id="2", content="A recipe for chocolate cake"),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrBM25Retriever(document_store=document_store)
        documents = retriever.run(query="open source search")["documents"]
        assert documents[0].id == "1"
        assert documents[0].score is not None

    def test_filters_narrow_the_result(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="search platform", meta={"group": "a"}),
                Document(id="2", content="search platform", meta={"group": "b"}),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrBM25Retriever(document_store=document_store)
        documents = retriever.run(
            query="search platform", filters={"field": "meta.group", "operator": "==", "value": "a"}
        )["documents"]
        assert [document.id for document in documents] == ["1"]

    def test_top_k(self, document_store):
        document_store.write_documents(
            [Document(id=str(index), content="search platform") for index in range(5)],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrBM25Retriever(document_store=document_store, top_k=2)
        assert len(retriever.run(query="search platform")["documents"]) == 2

    def test_fuzziness_tolerates_a_typo(self, document_store):
        document_store.write_documents(
            [Document(id="1", content="Apache Solr search platform")], DuplicatePolicy.OVERWRITE
        )
        retriever = SolrBM25Retriever(document_store=document_store)
        # "Apach" is one edit away from the indexed term "apache".
        assert retriever.run(query="Apach")["documents"] == []
        assert retriever.run(query="Apach", fuzziness=1)["documents"][0].id == "1"

    async def test_run_async(self, document_store):
        await document_store.write_documents_async(
            [Document(id="1", content="Apache Solr search platform")], DuplicatePolicy.OVERWRITE
        )
        retriever = SolrBM25Retriever(document_store=document_store)
        documents = (await retriever.run_async(query="Apache Solr"))["documents"]
        assert documents[0].id == "1"
