import pytest

from chroma_haystack.document_store import ChromaDocumentStore
from chroma_haystack.retriever import ChromaDenseRetriever


@pytest.mark.integration
def test_retriever_to_json(request):
    ds = ChromaDocumentStore(collection_name=request.node.name, embedding_function="OpenAIEmbeddingFunction")
    retriever = ChromaDenseRetriever(ds, filters={"foo": "bar"}, top_k=99)
    assert retriever.to_dict() == {
        "type": "ChromaDenseRetriever",
        "init_parameters": {
            "filters": {"foo": "bar"},
            "top_k": 99,
            "document_store": {"collection_name": request.node.name, "embedding_function": "OpenAIEmbeddingFunction"},
        },
    }


@pytest.mark.integration
def test_retriever_from_json(request):
    data = {
        "type": "ChromaDenseRetriever",
        "init_parameters": {
            "filters": {"bar": "baz"},
            "top_k": 42,
            "document_store": {"collection_name": request.node.name, "embedding_function": "OpenAIEmbeddingFunction"},
        },
    }
    retriever = ChromaDenseRetriever.from_dict(data)
    assert retriever.document_store._collection_name == request.node.name
    assert retriever.document_store._embedding_function == "OpenAIEmbeddingFunction"
    assert retriever.filters == {"bar": "baz"}
    assert retriever.top_k == 42
