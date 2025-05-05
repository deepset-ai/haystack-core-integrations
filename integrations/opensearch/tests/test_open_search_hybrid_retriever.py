from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Document

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
            },
        }
        super_component = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(super_component, OpenSearchHybridRetriever)

    def test_run(self):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # mock the LazyImport check for sentence_transformers
        with patch("haystack.lazy_imports.LazyImport.check") as mock_lazy_check:
            mock_lazy_check.return_value = None

            # mock the sentence_transformers module and SentenceTransformer class
            with patch.dict("sys.modules", {"sentence_transformers": MagicMock()}):
                with patch("sentence_transformers.SentenceTransformer") as mock_sentence_transformer:

                    # mocked SentenceTransformer
                    mock_transformer_instance = mock_sentence_transformer.return_value
                    mock_transformer_instance.encode.return_value = [0.1, 0.2, 0.3]

                    # mock the _SentenceTransformersEmbeddingBackend
                    with patch(
                        "haystack.components.embedders.backends.sentence_transformers_backend._SentenceTransformersEmbeddingBackend"
                    ) as mock_backend:

                        # mocked backend
                        mock_backend_instance = mock_backend.return_value
                        mock_backend_instance.embed.return_value = [0.1, 0.2, 0.3]

                        # mock the SentenceTransformersTextEmbedder
                        with patch(
                            "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder"
                        ) as mock_embedder:

                            # mocked embedder
                            mock_embedder_instance = mock_embedder.return_value
                            mock_embedder_instance.run.return_value = {"embedding": [0.1, 0.2, 0.3]}

                            retriever = OpenSearchHybridRetriever(document_store=mock_store)
                            result = retriever.run(query="test query")

                            assert len(result) == 1
                            assert len(result["documents"]) == 2
                            assert any(doc.content == "Test doc BM25" for doc in result["documents"])
                            assert any(doc.content == "Test doc Embedding" for doc in result["documents"])
