# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any
from unittest.mock import Mock, patch

import pytest
from haystack import Document, Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.joiners.document_joiner import JoinMode
from haystack.core.component import component
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchHybridRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


@component
class MockedTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str, param_a: str = "default", param_b: str = "another_default") -> dict[str, Any]:
        return {"embedding": [0.1, 0.2, 0.3], "metadata": {"text": text, "param_a": param_a, "param_b": param_b}}


class TestElasticsearchHybridRetriever:
    serialised = {  # noqa: RUF012
        "type": "haystack_integrations.components.retrievers.elasticsearch.elasticsearch_hybrid_retriever.ElasticsearchHybridRetriever",  # noqa: E501
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
                "init_parameters": {
                    "hosts": None,
                    "custom_mapping": None,
                    "index": "default",
                    "api_key": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False},
                    "api_key_id": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY_ID"], "strict": False},
                    "embedding_similarity_function": "cosine",
                    "sparse_vector_field": None,
                    "ingest_pipeline": None,
                },
            },
            "embedder": {
                "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
                "init_parameters": {
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                    "prefix": "",
                    "suffix": "",
                    "local_files_only": False,
                    "batch_size": 32,
                    "progress_bar": True,
                    "normalize_embeddings": False,
                    "trust_remote_code": False,
                    "truncate_dim": None,
                    "model_kwargs": None,
                    "tokenizer_kwargs": None,
                    "config_kwargs": None,
                    "precision": "float32",
                    "encode_kwargs": None,
                    "backend": "torch",
                },
            },
            "filters_bm25": None,
            "fuzziness": "AUTO",
            "top_k_bm25": 10,
            "scale_score": False,
            "filter_policy_bm25": "replace",
            "filters_embedding": None,
            "top_k_embedding": 10,
            "num_candidates": None,
            "filter_policy_embedding": "replace",
            "join_mode": "reciprocal_rank_fusion",
            "weights": None,
            "top_k": None,
            "sort_by_score": True,
        },
    }

    @pytest.fixture
    def mock_embedder(self):
        return MockedTextEmbedder()

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict(self, _mock_elasticsearch_client) -> None:
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()  # we use actual embedder here for the de/serialization
        hybrid_retriever = ElasticsearchHybridRetriever(document_store=doc_store, embedder=embedder)
        result = hybrid_retriever.to_dict()

        result["init_parameters"]["embedder"]["init_parameters"].pop("device", None)

        expected = deepcopy(self.serialised)
        # revision was added in Haystack 2.20.0; include it in expected if present in result
        if "revision" in result["init_parameters"]["embedder"]["init_parameters"]:
            expected["init_parameters"]["embedder"]["init_parameters"]["revision"] = None

        assert result == expected

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_from_dict(self, _mock_elasticsearch_client):
        data = deepcopy(self.serialised)
        deserialized = ElasticsearchHybridRetriever.from_dict(data)
        assert isinstance(deserialized, ElasticsearchHybridRetriever)
        result = deserialized.to_dict()
        result["init_parameters"]["embedder"]["init_parameters"].pop("device", None)
        result["init_parameters"]["embedder"]["init_parameters"].pop("revision", None)
        assert result == self.serialised

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

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_enum_filter_policies(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()
        hybrid_retriever = ElasticsearchHybridRetriever(
            document_store=doc_store,
            embedder=embedder,
            filter_policy_bm25=FilterPolicy.MERGE,
            filter_policy_embedding=FilterPolicy.MERGE,
        )
        result = hybrid_retriever.to_dict()
        # FilterPolicy enum values must be serialized to plain strings
        assert result["init_parameters"]["filter_policy_bm25"] == "merge"
        assert result["init_parameters"]["filter_policy_embedding"] == "merge"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_enum_join_mode(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()
        hybrid_retriever = ElasticsearchHybridRetriever(
            document_store=doc_store,
            embedder=embedder,
            join_mode=JoinMode.CONCATENATE,
        )
        result = hybrid_retriever.to_dict()
        # JoinMode enum value must be serialized to a plain string
        assert result["init_parameters"]["join_mode"] == "concatenate"

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
