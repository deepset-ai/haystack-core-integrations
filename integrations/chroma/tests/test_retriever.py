# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack_integrations.components.retrievers.chroma import ChromaQueryTextRetriever
from haystack_integrations.document_stores.chroma import ChromaDocumentStore


@pytest.mark.integration
def test_retriever_to_json(request):
    ds = ChromaDocumentStore(
        collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
    )
    retriever = ChromaQueryTextRetriever(ds, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "document_store": {
                "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                "init_parameters": {
                    "collection_name": "test_retriever_to_json",
                    "embedding_function": "HuggingFaceEmbeddingFunction",
                    "persist_path": None,
                    "api_key": "1234567890",
                },
            },
        },
    }


@pytest.mark.integration
def test_retriever_from_json(request):
    data = {
        "type": "haystack_integrations.components.retrievers.chroma.retriever.ChromaQueryTextRetriever",
        "init_parameters": {
            "filters": {"bar": "baz"},
            "top_k": 42,
            "document_store": {
                "type": "haystack_integrations.document_stores.chroma.document_store.ChromaDocumentStore",
                "init_parameters": {
                    "collection_name": "test_retriever_from_json",
                    "embedding_function": "HuggingFaceEmbeddingFunction",
                    "persist_path": ".",
                    "api_key": "1234567890",
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
