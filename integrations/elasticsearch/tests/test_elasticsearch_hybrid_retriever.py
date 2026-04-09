# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import Mock, patch

import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.core.component import component

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchHybridRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@component
class MockedTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str, param_a: str = "default", param_b: str = "another_default") -> dict[str, Any]:
        return {"embedding": [0.1, 0.2, 0.3], "metadata": {"text": text, "param_a": param_a, "param_b": param_b}}


class TestElasticsearchHybridRetriever:
    @pytest.fixture
    def mock_embedder(self):
        return MockedTextEmbedder()

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict(self, _mock_elasticsearch_client) -> None:
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()  # we use actual embedder here for the de/serialization
        hybrid_retriever = ElasticsearchHybridRetriever(document_store=doc_store, embedder=embedder)
        result = hybrid_retriever.to_dict()

        # Remove device info as it varies by system
        result["init_parameters"]["embedder"]["init_parameters"].pop("device", None)

        # Check the structure and key components
        assert (
            result["type"]
            == "haystack_integrations.components.retrievers.elasticsearch.elasticsearch_hybrid_retriever.ElasticsearchHybridRetriever"  # noqa: E501
        )
        assert "document_store" in result["init_parameters"]
        assert "embedder" in result["init_parameters"]
        assert result["init_parameters"]["fuzziness"] == "AUTO"
        assert result["init_parameters"]["top_k_bm25"] == 10
        assert result["init_parameters"]["top_k_embedding"] == 10
        assert result["init_parameters"]["scale_score"] is False
        assert result["init_parameters"]["join_mode"] == "reciprocal_rank_fusion"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_from_dict(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()
        hybrid_retriever = ElasticsearchHybridRetriever(document_store=doc_store, embedder=embedder)
        serialized = hybrid_retriever.to_dict()

        # Deserialize and verify
        deserialized = ElasticsearchHybridRetriever.from_dict(serialized)
        assert isinstance(deserialized, ElasticsearchHybridRetriever)
        roundtrip = deserialized.to_dict()
        assert roundtrip["type"] == serialized["type"]
        assert roundtrip["init_parameters"]["fuzziness"] == "AUTO"
        assert roundtrip["init_parameters"]["top_k_bm25"] == 10
        assert roundtrip["init_parameters"]["top_k_embedding"] == 10
        assert roundtrip["init_parameters"]["num_candidates"] is None
        assert roundtrip["init_parameters"]["join_mode"] == "reciprocal_rank_fusion"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_extra_args(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()
        hybrid_retriever = ElasticsearchHybridRetriever(
            document_store=doc_store,
            embedder=embedder,
            bm25_retriever={"filters": {"source": "bm25_init"}},
            embedding_retriever={"num_candidates": 42, "filters": {"source": "embedding_init"}},
        )
        result = hybrid_retriever.to_dict()

        # Check that extra args are preserved
        assert "bm25_retriever" in result["init_parameters"]
        assert "embedding_retriever" in result["init_parameters"]
        assert result["init_parameters"]["bm25_retriever"] == {"filters": {"source": "bm25_init"}}
        assert result["init_parameters"]["embedding_retriever"] == {
            "num_candidates": 42,
            "filters": {"source": "embedding_init"},
        }

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_from_dict_with_extra_args(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()
        hybrid_retriever = ElasticsearchHybridRetriever(
            document_store=doc_store,
            embedder=embedder,
            bm25_retriever={"filters": {"source": "bm25_init"}},
            embedding_retriever={"num_candidates": 42, "filters": {"source": "embedding_init"}},
        )
        serialized = hybrid_retriever.to_dict()

        # Deserialize and verify extra args are preserved
        deserialized = ElasticsearchHybridRetriever.from_dict(serialized)
        assert isinstance(deserialized, ElasticsearchHybridRetriever)
        roundtrip = deserialized.to_dict()
        assert roundtrip["init_parameters"]["bm25_retriever"] == {"filters": {"source": "bm25_init"}}
        assert roundtrip["init_parameters"]["embedding_retriever"] == {
            "num_candidates": 42,
            "filters": {"source": "embedding_init"},
        }

    def test_run(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = ElasticsearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
        result = retriever.run(query="test query")

        assert len(result) == 1
        assert len(result["documents"]) == 2
        assert any(doc.content == "Test doc BM25" for doc in result["documents"])
        assert any(doc.content == "Test doc Embedding" for doc in result["documents"])

    def test_run_with_extra_arg(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = ElasticsearchHybridRetriever(
            document_store=mock_store,
            embedder=mock_embedder,
            bm25_retriever={"filters": {"source": "bm25_init"}},
            embedding_retriever={"num_candidates": 42, "filters": {"source": "embedding_init"}},
        )
        result = retriever.run(query="test query")

        # Verify the retrievers were called with propagated init-time extra args
        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"source": "bm25_init"},
            top_k=10,
            fuzziness="AUTO",
            scale_score=False,
        )
        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"source": "embedding_init"},
            top_k=10,
            num_candidates=42,
        )

        # Verify the results
        assert len(result) == 1
        assert len(result["documents"]) == 2
        assert any(doc.content == "Test doc BM25" for doc in result["documents"])
        assert any(doc.content == "Test doc Embedding" for doc in result["documents"])

    def test_run_with_extra_arg_invalid_param(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        with pytest.raises(
            ValueError, match=r"valid extra args are only: 'bm25_retriever' and 'embedding_retriever'\."
        ):
            _ = ElasticsearchHybridRetriever(
                document_store=mock_store,
                embedder=mock_embedder,
                invalid_a={},
                invalid_b={},
            )

    def test_run_with_extra_runtime_params(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = ElasticsearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
        _ = retriever.run(
            query="test query",
            filters_bm25={"key": "value"},
            filters_embedding={"key": "value"},
            top_k_bm25=1,
            top_k_embedding=1,
        )

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"key": "value"},
            top_k=1,
            fuzziness="AUTO",
            scale_score=False,
        )
        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"key": "value"},
            top_k=1,
            num_candidates=None,
        )

    def test_run_in_pipeline(self, mock_embedder):
        # mocked document store
        pipeline = Pipeline()
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = ElasticsearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
        pipeline.add_component("retriever", retriever)

        # Should not fail
        _ = pipeline.run(data={"retriever": {"query": "test query", "filters_bm25": {"param_a": "default"}}})

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"param_a": "default"},
            top_k=10,
            fuzziness="AUTO",
            scale_score=False,
        )
        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={},
            top_k=10,
            num_candidates=None,
        )
