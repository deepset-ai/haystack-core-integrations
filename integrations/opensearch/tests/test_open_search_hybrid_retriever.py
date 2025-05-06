from unittest.mock import MagicMock, Mock, patch

import pytest
from haystack import Document
from haystack.utils import DeviceType

from haystack_integrations.components.retrievers.opensearch import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore


class TestOpenSearchHybridRetriever:

    expected = {
        "type": "haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever",  # noqa: E501
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
            "model": "sentence-transformers/all-mpnet-base-v2",
            "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
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
            "join_mode": "reciprocal_rank_fusion",
            "weights": None,
            "top_k": None,
            "sort_by_score": True,
        },
    }

    @pytest.fixture
    def hybrid_retriever(self, monkeypatch) -> OpenSearchHybridRetriever:
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-api-key")
        doc_store = OpenSearchDocumentStore()
        return OpenSearchHybridRetriever(document_store=doc_store)

    def test_to_dict(self, hybrid_retriever: OpenSearchHybridRetriever) -> None:
        result = hybrid_retriever.to_dict()
        assert self.expected == result

    def test_from_dict(self):
        data = {
            "type": "haystack_integrations.components.retrievers.opensearch.open_search_hybrid_retriever.OpenSearchHybridRetriever",  # noqa: E501
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

    def test_to_dict_with_extra_args(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-api-key")
        doc_store = OpenSearchDocumentStore()
        hybrid = OpenSearchHybridRetriever(
            document_store=doc_store, extra_arg={"text_embedder": {"progress_bar": False}}
        )
        result = hybrid.to_dict()
        added_extra_args = {"extra_arg": {"text_embedder": {"progress_bar": False}}}
        self.expected['init_parameters'].update(added_extra_args)
        assert result == self.expected

    def test_from_dict_with_extra_args(self):
        added_extra_args = {"extra_arg": {"text_embedder": {"progress_bar": False}}}
        hybrid = OpenSearchHybridRetriever.from_dict(self.expected['init_parameters'].update(added_extra_args))
        print(hybrid.to_dict())


    def test_run(self):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # mock the LazyImport check for sentence_transformers
        with patch("haystack.lazy_imports.LazyImport.check") as mock_lazy_check:
            mock_lazy_check.return_value = None

            # mock the device detection
            with patch("haystack.utils.device._get_default_device") as mock_get_device:
                # Create a mock Device object
                mock_device = MagicMock()
                mock_device.type = DeviceType.CPU
                mock_get_device.return_value = mock_device

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
