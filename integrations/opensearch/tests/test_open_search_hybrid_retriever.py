from copy import deepcopy
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from haystack import Document
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.core.component import component

from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


@component
class MockedTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str, param_a: str = "default", param_b: str = "another_default") -> Dict[str, Any]:
        return {"embedding": [0.1, 0.2, 0.3], "metadata": {"text": text, "param_a": param_a, "param_b": param_b}}


class TestOpenSearchHybridRetriever:

    serialised = {  # noqa: RUF012
        "type": "haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever",  # noqa: E501
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                "init_parameters": {
                    "hosts": None,
                    "index": "default",
                    "max_chunk_bytes": 104857600,
                    "embedding_dim": 768,
                    "method": None,
                    "mappings": {
                        "properties": {
                            "embedding": {"type": "knn_vector", "index": True, "dimension": 768},
                            "content": {"type": "text"},
                        },
                        "dynamic_templates": [
                            {"strings": {"match_mapping_type": "string", "mapping": {"type": "keyword"}}}
                        ],
                    },
                    "settings": {"index.knn": True},
                    "create_index": True,
                    "return_embedding": False,
                    "http_auth": None,
                    "use_ssl": None,
                    "verify_certs": None,
                    "timeout": None,
                },
            },
            "embedder": {
                "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
                "init_parameters": {
                    "model": "sentence-transformers/all-mpnet-base-v2",
                    "device": {"type": "single", "device": "mps"},
                    "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
                    "prefix": "",
                    "suffix": "",
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
            "all_terms_must_match": False,
            "filter_policy_bm25": "replace",
            "custom_query_bm25": None,
            "filters_embedding": None,
            "top_k_embedding": 10,
            "filter_policy_embedding": "replace",
            "custom_query_embedding": None,
            "join_mode": "reciprocal_rank_fusion",
            "weights": None,
            "top_k": None,
            "sort_by_score": True,
        },
    }

    @pytest.fixture
    def mock_embedder(self):
        return MockedTextEmbedder()

    def test_to_dict(self) -> None:
        doc_store = OpenSearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()  # we use actual embedder here for the de/serialization
        hybrid_retriever = OpenSearchHybridRetriever(document_store=doc_store, embedder=embedder)
        result = hybrid_retriever.to_dict()
        assert result == self.serialised

    def test_from_dict(self):
        data = deepcopy(self.serialised)
        super_component = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(super_component, OpenSearchHybridRetriever)
        assert super_component.to_dict()

    def test_to_dict_with_extra_args(self):
        doc_store = OpenSearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()  # an actual embedder here for the de/serialization
        hybrid_retriever = OpenSearchHybridRetriever(
            document_store=doc_store, embedder=embedder, extra_arg={"embedding_retriever": {"raise_on_failure": True}}
        )
        result = hybrid_retriever.to_dict()
        expected = deepcopy(self.serialised)
        expected["init_parameters"]["extra_arg"] = {"embedding_retriever": {"raise_on_failure": True}}
        assert result == expected

    def test_from_dict_with_extra_args(self):
        data = deepcopy(self.serialised)
        data["init_parameters"]["extra_arg"] = {"embedding_retriever": {"raise_on_failure": True}}
        hybrid = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(hybrid, OpenSearchHybridRetriever)
        assert hybrid.to_dict()

    def test_run(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = OpenSearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
        result = retriever.run(query="test query")

        assert len(result) == 1
        assert len(result["documents"]) == 2
        assert any(doc.content == "Test doc BM25" for doc in result["documents"])
        assert any(doc.content == "Test doc Embedding" for doc in result["documents"])
