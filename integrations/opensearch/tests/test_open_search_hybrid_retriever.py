import pytest
from unittest.mock import patch, MagicMock
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.open_search_hybrid_retriever import OpenSearchHybridRetriever
from haystack_integrations.document_stores.opensearch import OpenSearchDocumentStore

class TestOpenSearchHybridRetriever:

    @pytest.fixture
    def hybrid_retriever(self) -> OpenSearchHybridRetriever:
        doc_store = OpenSearchDocumentStore()
        return OpenSearchHybridRetriever(document_store=doc_store)

    def test_to_dict(self, hybrid_retriever: OpenSearchHybridRetriever) -> None:
        result = hybrid_retriever.to_dict()
        expected = {
            'type': 'haystack_integrations.components.retrievers.open_search_hybrid_retriever.OpenSearchHybridRetriever',
            'init_parameters': {
                'document_store': {
                    'init_parameters': {
                        'create_index': True,
                        'embedding_dim': 768,
                        'hosts': None,
                        'http_auth': None,
                        'index': 'default',
                        'mappings': {
                            'dynamic_templates': [
                                {'strings': {'mapping': {'type': 'keyword'}, 'match_mapping_type': 'string'}}],
                            'properties': {
                                'content': {'type': 'text'},
                                'embedding': {'dimension': 768, 'index': True, 'type': 'knn_vector'}
                            },
                        },
                        'max_chunk_bytes': 104857600,
                        'method': None,
                        'return_embedding': False,
                        'settings': {'index.knn': True},
                        'timeout': None,
                        'use_ssl': None,
                        'verify_certs': None,
                        },
                    'type': 'haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore'
                },
                'text_embedder_model': 'sentence-transformers/all-mpnet-base-v2',
                'device': None,
                'normalize_embeddings': False,
                'model_kwargs': {},
                'tokenizer_kwargs': {},
                'config_kwargs': {},
                'encode_kwargs': {},
                'backend': 'torch',
                'filters_bm25': None,
                'fuzziness': 'AUTO',
                'top_k_bm25': 10,
                'scale_score': False,
                'all_terms_must_match': False,
                'filter_policy_bm25': 'replace',
                'custom_query_bm25': None,
                'filters_embedding': None,
                'top_k_embedding': 10,
                'filter_policy_embedding': 'replace',
                'custom_query_embedding': None,
                'join_mode': 'concatenate',
                'weights': None,
                'top_k': None,
                'sort_by_score': True,
                'template': None,
                'generator_model': 'gpt-4o-mini',
                'streaming_callback': None,
                'api_base_url': None,
                'organization': None,
                'generation_kwargs': {},
                'http_client_kwargs': {},
                'pattern': None,
                'reference_pattern': None}
        }
        assert expected == result

    def test_from_dict(self):
        data = {
            'type': 'haystack_integrations.components.retrievers.open_search_hybrid_retriever.OpenSearchHybridRetriever',
            'init_parameters': {
                'document_store': {
                    'init_parameters': {
                        'create_index': True,
                        'embedding_dim': 768,
                        'hosts': None,
                        'http_auth': None,
                        'index': 'default',
                        'mappings': {
                            'dynamic_templates': [
                                {'strings': {'mapping': {'type': 'keyword'}, 'match_mapping_type': 'string'}}],
                            'properties': {
                                'content': {'type': 'text'},
                                'embedding': {'dimension': 768, 'index': True, 'type': 'knn_vector'}
                            },
                        },
                        'max_chunk_bytes': 104857600,
                        'method': None,
                        'return_embedding': False,
                        'settings': {'index.knn': True},
                        'timeout': None,
                        'use_ssl': None,
                        'verify_certs': None,
                        },
                    'type': 'haystack_integrations.document_stores.opensearch.document_store.OpenSearchDocumentStore'
                },
                'text_embedder_model': 'sentence-transformers/all-mpnet-base-v2',
                'device': None,
                'normalize_embeddings': False,
                'model_kwargs': {},
                'tokenizer_kwargs': {},
                'config_kwargs': {},
                'encode_kwargs': {},
                'backend': 'torch',
                'filters_bm25': None,
                'fuzziness': 'AUTO',
                'top_k_bm25': 10,
                'scale_score': False,
                'all_terms_must_match': False,
                'filter_policy_bm25': 'replace',
                'custom_query_bm25': None,
                'filters_embedding': None,
                'top_k_embedding': 10,
                'filter_policy_embedding': 'replace',
                'custom_query_embedding': None,
                'join_mode': 'concatenate',
                'weights': None,
                'top_k': None,
                'sort_by_score': True,
                'template': None,
                'generator_model': 'gpt-4o-mini',
                'streaming_callback': None,
                'api_base_url': None,
                'organization': None,
                'generation_kwargs': {},
                'http_client_kwargs': {},
                'pattern': None,
                'reference_pattern': None
            }
        }
        super_component = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(super_component, OpenSearchHybridRetriever)

    @patch("haystack_integrations.document_stores.opensearch.document_store.OpenSearch")
    @patch("haystack.components.embedders.SentenceTransformersTextEmbedder")
    def test_run(self, mock_embedder, mock_opensearch, hybrid_retriever: OpenSearchHybridRetriever):
        # mock document store
        mock_client = MagicMock()
        mock_opensearch.return_value = mock_client
        
        # mock text embedder
        mock_embedder_instance = MagicMock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.run.return_value = {"embedding": [0.1, 0.2, 0.3]}  # mock embedding
        
        # test documents
        test_documents = [
            Document(content="Haskell is a functional programming language", meta={"language_type": "functional"}),
            Document(content="Lisp is a functional programming language", meta={"language_type": "functional"}),
            Document(content="Python is an object-oriented programming language", meta={"language_type": "object_oriented"})
        ]
        
        # mock search response
        mock_search_response = {
            "hits": {
                "hits": [
                    {"_source": {"content": doc.content, "meta": doc.meta}, "_score": 1.0}
                    for doc in test_documents
                ]
            }
        }
        mock_client.search.return_value = mock_search_response

        query = "What are functional programming languages?"
        result = hybrid_retriever.run(query=query)
        
        assert "answers" in result
        assert isinstance(result["answers"], list)
        assert len(result["answers"]) > 0
        answer = result["answers"][0]
        assert answer != "None"

    def test_run_custom_prompt(self):
        # ToDo: Implement a test
        pass