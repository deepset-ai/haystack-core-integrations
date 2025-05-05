import pytest

from haystack_integrations.components.retrievers.open_search_hybrid_retriever import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


class TestOpenSearchHybridRetriever:

    @pytest.fixture
    def hybrid_retriever(self, monkeypatch) -> OpenSearchHybridRetriever:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-api-key")
        doc_store = OpenSearchDocumentStore()
        return OpenSearchHybridRetriever(document_store=doc_store)

    def test_to_dict(self, hybrid_retriever: OpenSearchHybridRetriever) -> None:
        result = hybrid_retriever.to_dict()
        expected = {
            "type": "haystack_integrations.components.retrievers.open_search_hybrid_retriever.OpenSearchHybridRetriever",  # noqa: E501
            "init_parameters": {
                "document_store": {
                    "init_parameters": {
                        "create_index": True,
                        "embedding_dim": 768,
                        "hosts": None,
                        "http_auth": None,
                        "index": "default",
                        "mappings": {
                            "dynamic_templates": [
                                {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                            ],
                            "properties": {
                                "content": {"type": "text"},
                                "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                            },
                        },
                        "max_chunk_bytes": 104857600,
                        "method": None,
                        "return_embedding": False,
                        "settings": {"index.knn": True},
                        "timeout": None,
                        "use_ssl": None,
                        "verify_certs": None,
                    },
                    "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                },
                "text_embedder_model": "sentence-transformers/all-mpnet-base-v2",
                "device": None,
                "normalize_embeddings": False,
                "model_kwargs": {},
                "tokenizer_kwargs": {},
                "config_kwargs": {},
                "encode_kwargs": {},
                "backend": "torch",
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
                "join_mode": "concatenate",
                "weights": None,
                "top_k": None,
                "sort_by_score": True,
                "template": None,
                "generator_model": "gpt-4o-mini",
                "streaming_callback": None,
                "api_base_url": None,
                "organization": None,
                "generation_kwargs": {},
                "http_client_kwargs": {},
                "pattern": None,
                "reference_pattern": None,
            },
        }
        assert expected == result

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.open_search_hybrid_retriever.OpenSearchHybridRetriever",  # noqa: E501
            "init_parameters": {
                "document_store": {
                    "init_parameters": {
                        "create_index": True,
                        "embedding_dim": 768,
                        "hosts": None,
                        "http_auth": None,
                        "index": "default",
                        "mappings": {
                            "dynamic_templates": [
                                {"strings": {"mapping": {"type": "keyword"}, "match_mapping_type": "string"}}
                            ],
                            "properties": {
                                "content": {"type": "text"},
                                "embedding": {"dimension": 768, "index": True, "type": "knn_vector"},
                            },
                        },
                        "max_chunk_bytes": 104857600,
                        "method": None,
                        "return_embedding": False,
                        "settings": {"index.knn": True},
                        "timeout": None,
                        "use_ssl": None,
                        "verify_certs": None,
                    },
                    "type": "haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore",
                },
                "text_embedder_model": "sentence-transformers/all-mpnet-base-v2",
                "device": None,
                "normalize_embeddings": False,
                "model_kwargs": {},
                "tokenizer_kwargs": {},
                "config_kwargs": {},
                "encode_kwargs": {},
                "backend": "torch",
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
                "join_mode": "concatenate",
                "weights": None,
                "top_k": None,
                "sort_by_score": True,
                "template": None,
                "generator_model": "gpt-4o-mini",
                "streaming_callback": None,
                "api_base_url": None,
                "organization": None,
                "generation_kwargs": {},
                "http_client_kwargs": {},
                "pattern": None,
                "reference_pattern": None,
            },
        }
        super_component = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(super_component, OpenSearchHybridRetriever)
