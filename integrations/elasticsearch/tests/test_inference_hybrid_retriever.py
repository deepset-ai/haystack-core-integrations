# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import Mock, patch

import pytest
from haystack import Document, Pipeline
from haystack.components.joiners.document_joiner import JoinMode
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceHybridRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


class TestElasticsearchInferenceHybridRetriever:
    serialised = {  # noqa: RUF012
        "type": "haystack_integrations.components.retrievers.elasticsearch.inference_hybrid_retriever.ElasticsearchInferenceHybridRetriever",  # noqa: E501
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
                },
            },
            "inference_id": ".elser_model_2",
            "filters_bm25": None,
            "fuzziness": "AUTO",
            "top_k_bm25": 10,
            "scale_score": False,
            "filter_policy_bm25": "replace",
            "filters_sparse": None,
            "top_k_sparse": 10,
            "filter_policy_sparse": "replace",
            "join_mode": "reciprocal_rank_fusion",
            "weights": None,
            "top_k": None,
            "sort_by_score": True,
        },
    }

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict(self, _mock_elasticsearch_client) -> None:
        doc_store = ElasticsearchDocumentStore()
        retriever = ElasticsearchInferenceHybridRetriever(document_store=doc_store, inference_id=".elser_model_2")
        assert retriever.to_dict() == self.serialised

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_from_dict(self, _mock_elasticsearch_client):
        data = deepcopy(self.serialised)
        deserialized = ElasticsearchInferenceHybridRetriever.from_dict(data)
        assert isinstance(deserialized, ElasticsearchInferenceHybridRetriever)
        assert deserialized.to_dict() == self.serialised

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_extra_args(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        retriever = ElasticsearchInferenceHybridRetriever(
            document_store=doc_store,
            inference_id=".elser_model_2",
            bm25_retriever={"filters": {"source": "bm25_init"}},
            sparse_retriever={"filters": {"source": "sparse_init"}},
        )
        result = retriever.to_dict()
        assert result["init_parameters"]["bm25_retriever"] == {"filters": {"source": "bm25_init"}}
        assert result["init_parameters"]["sparse_retriever"] == {"filters": {"source": "sparse_init"}}

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_from_dict_with_extra_args(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        retriever = ElasticsearchInferenceHybridRetriever(
            document_store=doc_store,
            inference_id=".elser_model_2",
            bm25_retriever={"filters": {"source": "bm25_init"}},
            sparse_retriever={"filters": {"source": "sparse_init"}},
        )
        serialized = retriever.to_dict()
        deserialized = ElasticsearchInferenceHybridRetriever.from_dict(serialized)
        assert isinstance(deserialized, ElasticsearchInferenceHybridRetriever)
        roundtrip = deserialized.to_dict()
        assert roundtrip["init_parameters"]["bm25_retriever"] == {"filters": {"source": "bm25_init"}}
        assert roundtrip["init_parameters"]["sparse_retriever"] == {"filters": {"source": "sparse_init"}}

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_enum_filter_policies(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        retriever = ElasticsearchInferenceHybridRetriever(
            document_store=doc_store,
            inference_id=".elser_model_2",
            filter_policy_bm25=FilterPolicy.MERGE,
            filter_policy_sparse=FilterPolicy.MERGE,
        )
        result = retriever.to_dict()
        assert result["init_parameters"]["filter_policy_bm25"] == "merge"
        assert result["init_parameters"]["filter_policy_sparse"] == "merge"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_to_dict_with_enum_join_mode(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        retriever = ElasticsearchInferenceHybridRetriever(
            document_store=doc_store,
            inference_id=".elser_model_2",
            join_mode=JoinMode.CONCATENATE,
        )
        result = retriever.to_dict()
        assert result["init_parameters"]["join_mode"] == "concatenate"

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_invalid_document_store(self, _mock_elasticsearch_client):
        with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
            ElasticsearchInferenceHybridRetriever(document_store=Mock(), inference_id=".elser_model_2")

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_empty_inference_id(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        with pytest.raises(ValueError, match="inference_id must be provided"):
            ElasticsearchInferenceHybridRetriever(document_store=doc_store, inference_id="")

    @patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
    def test_invalid_extra_kwarg(self, _mock_elasticsearch_client):
        doc_store = ElasticsearchDocumentStore()
        with pytest.raises(ValueError, match=r"valid extra args are only: 'bm25_retriever' and 'sparse_retriever'"):
            ElasticsearchInferenceHybridRetriever(
                document_store=doc_store,
                inference_id=".elser_model_2",
                invalid_key={},
            )

    def test_run(self):
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="BM25 result")]
        mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Sparse result")]

        retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
        result = retriever.run(query="test query")

        assert len(result["documents"]) == 2
        assert any(doc.content == "BM25 result" for doc in result["documents"])
        assert any(doc.content == "Sparse result" for doc in result["documents"])

    def test_run_with_extra_args(self):
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="BM25 result")]
        mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Sparse result")]

        retriever = ElasticsearchInferenceHybridRetriever(
            document_store=mock_store,
            inference_id=".elser_model_2",
            bm25_retriever={"filters": {"source": "bm25"}},
            sparse_retriever={"filters": {"source": "sparse"}},
        )
        result = retriever.run(query="test query")

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"source": "bm25"},
            top_k=10,
            fuzziness="AUTO",
            scale_score=False,
        )
        mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
            query="test query",
            inference_id=".elser_model_2",
            filters={"source": "sparse"},
            top_k=10,
        )

        assert len(result["documents"]) == 2

    def test_run_with_runtime_params(self):
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="BM25 result")]
        mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Sparse result")]

        retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
        retriever.run(
            query="test query",
            filters_bm25={"field": "value"},
            filters_sparse={"field": "value"},
            top_k_bm25=3,
            top_k_sparse=3,
        )

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"field": "value"},
            top_k=3,
            fuzziness="AUTO",
            scale_score=False,
        )
        mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
            query="test query",
            inference_id=".elser_model_2",
            filters={"field": "value"},
            top_k=3,
        )

    def test_run_in_pipeline(self):
        mock_store = Mock(spec=ElasticsearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="BM25 result")]
        mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Sparse result")]

        retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)

        result = pipeline.run(data={"retriever": {"query": "test query", "filters_bm25": {"tag": "news"}}})

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"tag": "news"},
            top_k=10,
            fuzziness="AUTO",
            scale_score=False,
        )
        assert len(result["retriever"]["documents"]) == 2
