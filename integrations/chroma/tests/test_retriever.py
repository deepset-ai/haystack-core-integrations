# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever, ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


class TestChromaQueryTextRetriever:
    def test_init(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        retriever = ChromaQueryTextRetriever(ds, filters={"foo": "bar"}, top_k=99, filter_policy="replace")
        assert retriever.filter_policy == FilterPolicy.REPLACE

        with pytest.raises(ValueError):
            ChromaQueryTextRetriever(ds, filters={"foo": "bar"}, top_k=99, filter_policy="unknown")

    def test_to_dict(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        retriever = ChromaQueryTextRetriever(ds, filters={"foo": "bar"}, top_k=99)
        assert retriever.to_dict() == {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever",
            "init_parameters": {
                "filters": {"foo": "bar"},
                "top_k": 99,
                "filter_policy": "replace",
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_to_dict",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": None,
                        "host": None,
                        "port": None,
                        "api_key": "1234567890",
                        "distance_function": "l2",
                        "client_settings": None,
                    },
                },
            },
        }

    def test_from_dict(self, request):
        data = {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever",
            "init_parameters": {
                "filters": {"bar": "baz"},
                "top_k": 42,
                "filter_policy": "replace",
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_from_dict",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": ".",
                        "api_key": "1234567890",
                        "distance_function": "l2",
                    },
                },
            },
        }
        retriever = ChromaQueryTextRetriever.from_dict(data)
        assert retriever.document_store._collection_name == request.node.name
        assert retriever.document_store._embedding_function == "HuggingFaceEmbeddingFunction"
        assert retriever.document_store._embedding_function_params == {"api_key": "1234567890"}
        assert retriever.document_store._persist_path == "."
        assert retriever.filters == {"bar": "baz"}
        assert retriever.top_k == 42
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_from_dict_no_filter_policy(self, request):
        data = {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever",
            "init_parameters": {
                "filters": {"bar": "baz"},
                "top_k": 42,
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_from_dict_no_filter_policy",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": ".",
                        "api_key": "1234567890",
                        "distance_function": "l2",
                    },
                },
            },
        }
        retriever = ChromaQueryTextRetriever.from_dict(data)
        assert retriever.document_store._collection_name == request.node.name
        assert retriever.document_store._embedding_function == "HuggingFaceEmbeddingFunction"
        assert retriever.document_store._embedding_function_params == {"api_key": "1234567890"}
        assert retriever.document_store._persist_path == "."
        assert retriever.filters == {"bar": "baz"}
        assert retriever.top_k == 42
        assert retriever.filter_policy == FilterPolicy.REPLACE  # default even if not specified

    def test_run_delegates_to_document_store_search(self):
        ds = mock.Mock(spec=ChromaDocumentStore)
        expected = [Document(content="hit")]
        ds.search.return_value = [expected]
        retriever = ChromaQueryTextRetriever(ds, top_k=5)

        result = retriever.run(query="q")

        ds.search.assert_called_once_with(["q"], 5, {})
        assert result == {"documents": expected}

    @pytest.mark.asyncio
    async def test_run_async_delegates_to_document_store_search_async(self):
        ds = mock.Mock(spec=ChromaDocumentStore)
        expected = [Document(content="hit")]
        ds.search_async = mock.AsyncMock(return_value=[expected])
        retriever = ChromaQueryTextRetriever(ds, top_k=3)

        result = await retriever.run_async(query="q")

        ds.search_async.assert_awaited_once_with(["q"], 3, {})
        assert result == {"documents": expected}


class TestChromaEmbeddingRetriever:
    def test_init(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        retriever = ChromaEmbeddingRetriever(ds, filters={"foo": "bar"}, top_k=99, filter_policy="replace")
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_to_dict(self, request):
        ds = ChromaDocumentStore(
            collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
        )
        retriever = ChromaEmbeddingRetriever(ds, filters={"foo": "bar"}, top_k=99)
        assert retriever.to_dict() == {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever",
            "init_parameters": {
                "filters": {"foo": "bar"},
                "top_k": 99,
                "filter_policy": "replace",
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_to_dict",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": None,
                        "host": None,
                        "port": None,
                        "api_key": "1234567890",
                        "distance_function": "l2",
                        "client_settings": None,
                    },
                },
            },
        }

    def test_from_dict(self, request):
        data = {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever",
            "init_parameters": {
                "filters": {"bar": "baz"},
                "top_k": 42,
                "filter_policy": "replace",
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_from_dict",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": ".",
                        "api_key": "1234567890",
                        "distance_function": "l2",
                    },
                },
            },
        }
        retriever = ChromaEmbeddingRetriever.from_dict(data)
        assert retriever.document_store._collection_name == request.node.name
        assert retriever.document_store._embedding_function == "HuggingFaceEmbeddingFunction"
        assert retriever.document_store._embedding_function_params == {"api_key": "1234567890"}
        assert retriever.document_store._persist_path == "."
        assert retriever.filters == {"bar": "baz"}
        assert retriever.top_k == 42
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_from_dict_no_filter_policy(self, request):
        data = {
            "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaEmbeddingRetriever",
            "init_parameters": {
                "filters": {"bar": "baz"},
                "top_k": 42,
                "document_store": {
                    "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                    "init_parameters": {
                        "collection_name": "test_from_dict_no_filter_policy",
                        "embedding_function": "HuggingFaceEmbeddingFunction",
                        "persist_path": ".",
                        "api_key": "1234567890",
                        "distance_function": "l2",
                    },
                },
            },
        }
        retriever = ChromaEmbeddingRetriever.from_dict(data)
        assert retriever.filter_policy == FilterPolicy.REPLACE

    def test_run_delegates_to_document_store_search_embeddings(self):
        ds = mock.Mock(spec=ChromaDocumentStore)
        expected = [Document(content="hit")]
        ds.search_embeddings.return_value = [expected]
        retriever = ChromaEmbeddingRetriever(ds, top_k=7)

        result = retriever.run(query_embedding=[0.1, 0.2])

        ds.search_embeddings.assert_called_once_with([[0.1, 0.2]], 7, {})
        assert result == {"documents": expected}

    @pytest.mark.asyncio
    async def test_run_async_delegates_to_document_store_search_embeddings_async(self):
        ds = mock.Mock(spec=ChromaDocumentStore)
        expected = [Document(content="hit")]
        ds.search_embeddings_async = mock.AsyncMock(return_value=[expected])
        retriever = ChromaEmbeddingRetriever(ds, top_k=4)

        result = await retriever.run_async(query_embedding=[0.5])

        ds.search_embeddings_async.assert_awaited_once_with([[0.5]], 4, {})
        assert result == {"documents": expected}
