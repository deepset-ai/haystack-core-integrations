# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from typing import Any, Dict
from unittest.mock import Mock

import pytest
from haystack import Document, Pipeline
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
        result["init_parameters"]["embedder"]["init_parameters"].pop("device")  # remove device info for comparison
        data = deepcopy(self.serialised)
        # We add revision to the expected dict if it exists in the result for comparison
        # This was added in PR https://github.com/deepset-ai/haystack/pull/10003 and released in Haystack 2.20.0
        if "revision" in result["init_parameters"]["embedder"]["init_parameters"]:
            data["init_parameters"]["embedder"]["init_parameters"]["revision"] = None
        assert result == data

    def test_from_dict(self):
        data = deepcopy(self.serialised)
        super_component = OpenSearchHybridRetriever.from_dict(data)
        assert isinstance(super_component, OpenSearchHybridRetriever)
        assert super_component.to_dict()

    def test_to_dict_with_extra_args(self):
        doc_store = OpenSearchDocumentStore()
        embedder = SentenceTransformersTextEmbedder()  # an actual embedder here for the de/serialization
        hybrid_retriever = OpenSearchHybridRetriever(
            document_store=doc_store, embedder=embedder, embedding_retriever={"raise_on_failure": True}
        )
        result = hybrid_retriever.to_dict()
        expected = deepcopy(self.serialised)
        expected["init_parameters"]["embedding_retriever"] = {"raise_on_failure": True}
        # We add revision to the expected dict if it exists in the result for comparison
        # This was added in PR https://github.com/deepset-ai/haystack/pull/10003 and released in Haystack 2.20.0
        if "revision" in result["init_parameters"]["embedder"]["init_parameters"]:
            expected["init_parameters"]["embedder"]["init_parameters"]["revision"] = None
        result["init_parameters"]["embedder"]["init_parameters"].pop("device")  # remove device info for comparison
        assert result == expected

    def test_from_dict_with_extra_args(self):
        data = deepcopy(self.serialised)
        data["init_parameters"]["embedding_retriever"] = {"raise_on_failure": True}
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

    def test_run_with_extra_arg(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = OpenSearchHybridRetriever(
            document_store=mock_store,
            embedder=mock_embedder,
            bm25_retriever={"raise_on_failure": True},
            embedding_retriever={"raise_on_failure": False},
        )
        result = retriever.run(query="test query")

        # Verify the retrievers were called with the extra arguments
        mock_store._bm25_retrieval.assert_called_once()
        mock_store._embedding_retrieval.assert_called_once()

        # Verify the results
        assert len(result) == 1
        assert len(result["documents"]) == 2
        assert any(doc.content == "Test doc BM25" for doc in result["documents"])
        assert any(doc.content == "Test doc Embedding" for doc in result["documents"])

    def test_run_with_extra_arg_invalid_param(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        with pytest.raises(
            ValueError, match=r"valid extra args are only: 'bm25_retriever' and 'embedding_retriever'\."
        ):
            _ = OpenSearchHybridRetriever(
                document_store=mock_store,
                embedder=mock_embedder,
                invalid_a={"raise_on_failure": True},
                invalid_b={"raise_on_failure": False},
            )

    def test_run_with_extra_runtime_params(self, mock_embedder):
        # mocked document store
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = OpenSearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
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
            all_terms_must_match=False,
            fuzziness="AUTO",
            scale_score=False,
            custom_query=None,
        )
        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={"key": "value"},
            top_k=1,
            custom_query=None,
            efficient_filtering=False,
        )

    def test_run_in_pipeline(self, mock_embedder):
        # mocked document store
        pipeline = Pipeline()
        mock_store = Mock(spec=OpenSearchDocumentStore)
        mock_store._bm25_retrieval.return_value = [Document(content="Test doc BM25")]
        mock_store._embedding_retrieval.return_value = [Document(content="Test doc Embedding")]

        # use the mocked embedder
        retriever = OpenSearchHybridRetriever(document_store=mock_store, embedder=mock_embedder)
        # result = retriever.run(query="test query")
        pipeline.add_component("retriever", retriever)

        # Should not fail
        _ = pipeline.run(data={"retriever": {"query": "test query", "filters_bm25": {"param_a": "default"}}})

        mock_store._bm25_retrieval.assert_called_once_with(
            query="test query",
            filters={"param_a": "default"},
            top_k=10,
            all_terms_must_match=False,
            fuzziness="AUTO",
            scale_score=False,
            custom_query=None,
        )
        mock_store._embedding_retrieval.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            filters={},
            top_k=10,
            custom_query=None,
            efficient_filtering=False,
        )
