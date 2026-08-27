# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest
from haystack import Pipeline, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.joiners.document_joiner import JoinMode
from haystack.dataclasses import Document
from haystack.document_stores.types import DuplicatePolicy, FilterPolicy

from haystack_integrations.components.retrievers.solr import SolrHybridRetriever
from haystack_integrations.document_stores.solr import SolrDocumentStore

EMBEDDING_DIM = 768


@component
class FakeTextEmbedder:
    """A deterministic stand-in for a real embedder, so tests need no model download."""

    def __init__(self, dimension: int = EMBEDDING_DIM) -> None:
        self.dimension = dimension

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        # A one-hot vector keyed on the text length keeps results predictable.
        embedding = [0.0] * self.dimension
        embedding[len(text) % self.dimension] = 1.0
        return {"embedding": embedding}


@pytest.fixture
def store() -> SolrDocumentStore:
    return SolrDocumentStore(url="http://solr.test/solr", core="unit", auth=None)


class TestInit:
    def test_document_store_may_be_positional(self, store):
        """Matches the Elasticsearch and OpenSearch hybrid retrievers, which take it positionally."""
        retriever = SolrHybridRetriever(store, embedder=FakeTextEmbedder())
        assert retriever.document_store is store

    def test_builds_the_wrapped_pipeline(self, store):
        retriever = SolrHybridRetriever(document_store=store, embedder=FakeTextEmbedder())
        assert isinstance(retriever.pipeline, Pipeline)
        assert set(retriever.pipeline.graph.nodes) == {
            "text_embedder",
            "embedding_retriever",
            "bm25_retriever",
            "document_joiner",
        }

    def test_query_feeds_both_branches(self, store):
        retriever = SolrHybridRetriever(document_store=store, embedder=FakeTextEmbedder())
        assert retriever.input_mapping["query"] == ["text_embedder.text", "bm25_retriever.query"]
        assert retriever.output_mapping == {"document_joiner.documents": "documents"}

    def test_rejects_unknown_extra_args(self, store):
        with pytest.raises(ValueError, match="valid extra args are only"):
            SolrHybridRetriever(document_store=store, embedder=FakeTextEmbedder(), document_joiner={"top_k": 1})

    def test_extra_args_reach_the_retrievers(self, store):
        retriever = SolrHybridRetriever(
            document_store=store,
            embedder=FakeTextEmbedder(),
            bm25_retriever={"raise_on_failure": False},
        )
        assert retriever.pipeline.get_component("bm25_retriever")._raise_on_failure is False

    def test_extra_args_cannot_swap_the_document_store(self, store):
        """Both branches must keep searching the same core."""
        retriever = SolrHybridRetriever(
            document_store=store,
            embedder=FakeTextEmbedder(),
            bm25_retriever={"document_store": SolrDocumentStore(core="somewhere-else", auth=None)},
        )
        assert retriever.pipeline.get_component("bm25_retriever")._document_store is store

    def test_per_branch_settings(self, store):
        retriever = SolrHybridRetriever(
            document_store=store,
            embedder=FakeTextEmbedder(),
            top_k_bm25=3,
            top_k_embedding=7,
            fuzziness=2,
        )
        assert retriever.pipeline.get_component("bm25_retriever")._top_k == 3
        assert retriever.pipeline.get_component("bm25_retriever")._fuzziness == 2
        assert retriever.pipeline.get_component("embedding_retriever")._top_k == 7


class TestSerialization:
    @pytest.fixture(autouse=True)
    def openai_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    def test_to_dict(self, store):
        data = SolrHybridRetriever(document_store=store, embedder=OpenAITextEmbedder()).to_dict()
        assert (
            data["type"] == "haystack_integrations.components.retrievers.solr.solr_hybrid_retriever.SolrHybridRetriever"
        )
        assert data["init_parameters"]["join_mode"] == "reciprocal_rank_fusion"
        assert data["init_parameters"]["filter_policy_bm25"] == "replace"
        assert data["init_parameters"]["embedder"]["type"].endswith("OpenAITextEmbedder")

    def test_round_trip(self, store):
        retriever = SolrHybridRetriever(
            document_store=store,
            embedder=OpenAITextEmbedder(),
            top_k_bm25=3,
            top_k_embedding=7,
            filter_policy_bm25=FilterPolicy.MERGE,
            join_mode=JoinMode.MERGE,
            top_k=5,
        )
        restored = SolrHybridRetriever.from_dict(retriever.to_dict())
        assert restored.to_dict() == retriever.to_dict()
        assert restored.top_k_bm25 == 3
        assert restored.top_k_embedding == 7
        assert restored.join_mode == JoinMode.MERGE
        assert isinstance(restored.document_store, SolrDocumentStore)

    def test_survives_a_pipeline_round_trip(self, store):
        """The whole point of serialisation: a pipeline YAML has to load back."""
        pipeline = Pipeline()
        pipeline.add_component("hybrid", SolrHybridRetriever(document_store=store, embedder=OpenAITextEmbedder()))
        restored = Pipeline.loads(pipeline.dumps())
        assert isinstance(restored.get_component("hybrid"), SolrHybridRetriever)


class TestClose:
    def test_close_forwards_to_the_store(self, store):
        store.close = MagicMock()
        SolrHybridRetriever(document_store=store, embedder=FakeTextEmbedder()).close()
        store.close.assert_called_once()

    async def test_close_async_forwards_to_the_store(self, store):
        store.close_async = AsyncMock()
        await SolrHybridRetriever(document_store=store, embedder=FakeTextEmbedder()).close_async()
        store.close_async.assert_awaited_once()


@pytest.mark.integration
class TestHybridRetrieverIntegration:
    def test_fuses_both_branches(self, document_store):
        """A document only the BM25 branch can find still shows up in the fused result."""
        keyword_only = Document(id="keyword", content="Apache Solr search platform", embedding=[0.0, 1.0] + [0.0] * 766)
        embedder = FakeTextEmbedder()
        query = "Apache Solr"
        vector_only = Document(id="vector", content="nothing in common", embedding=embedder.run(query)["embedding"])
        document_store.write_documents([keyword_only, vector_only], DuplicatePolicy.OVERWRITE)

        retriever = SolrHybridRetriever(document_store=document_store, embedder=embedder)
        documents = retriever.run(query=query)["documents"]
        assert {document.id for document in documents} == {"keyword", "vector"}

    def test_top_k_caps_the_fused_result(self, document_store):
        document_store.write_documents(
            [
                Document(id=str(index), content="Apache Solr search", embedding=[1.0, float(index)] + [0.0] * 766)
                for index in range(5)
            ],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrHybridRetriever(document_store=document_store, embedder=FakeTextEmbedder(), top_k=2)
        assert len(retriever.run(query="Apache Solr search")["documents"]) == 2

    def test_per_branch_filters(self, document_store):
        document_store.write_documents(
            [
                Document(id="1", content="Apache Solr", meta={"group": "a"}, embedding=[1.0] + [0.0] * 767),
                Document(id="2", content="Apache Solr", meta={"group": "b"}, embedding=[1.0] + [0.0] * 767),
            ],
            DuplicatePolicy.OVERWRITE,
        )
        retriever = SolrHybridRetriever(document_store=document_store, embedder=FakeTextEmbedder())
        documents = retriever.run(
            query="Apache Solr",
            filters_bm25={"field": "meta.group", "operator": "==", "value": "a"},
            filters_embedding={"field": "meta.group", "operator": "==", "value": "a"},
        )["documents"]
        assert [document.id for document in documents] == ["1"]

    def test_runs_inside_a_pipeline(self, document_store):
        document_store.write_documents(
            [Document(id="1", content="Apache Solr search platform")], DuplicatePolicy.OVERWRITE
        )
        pipeline = Pipeline()
        pipeline.add_component(
            "hybrid",
            SolrHybridRetriever(document_store=document_store, embedder=FakeTextEmbedder()),
        )
        result = pipeline.run({"hybrid": {"query": "Apache Solr"}})
        assert result["hybrid"]["documents"][0].id == "1"
