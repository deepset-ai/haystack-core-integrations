import pytest

from chroma_haystack.document_store import ChromaDocumentStore
from chroma_haystack.retriever import ChromaQueryRetriever


@pytest.mark.integration
def test_retriever_to_json(request):
    ds = ChromaDocumentStore(
        collection_name=request.node.name, embedding_function="HuggingFaceEmbeddingFunction", api_key="1234567890"
    )
    retriever = ChromaQueryRetriever(ds, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "ChromaQueryRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "document_store": {
                "collection_name": request.node.name,
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "api_key": "1234567890",
            },
        },
    }


@pytest.mark.integration
def test_retriever_from_json(request):
    data = {
        "type": "ChromaQueryRetriever",
        "init_parameters": {
            "filters": {"bar": "baz"},
            "top_k": 42,
            "document_store": {
                "collection_name": request.node.name,
                "embedding_function": "HuggingFaceEmbeddingFunction",
                "api_key": "1234567890",
            },
        },
    }
    retriever = ChromaQueryRetriever.from_dict(data)
    assert retriever.document_store._collection_name == request.node.name
    assert retriever.document_store._embedding_function == "HuggingFaceEmbeddingFunction"
    assert retriever.document_store._embedding_function_params == {"api_key": "1234567890"}
    assert retriever.filters == {"bar": "baz"}
    assert retriever.top_k == 42
