from unittest.mock import MagicMock, patch

from haystack_integrations.components.retrievers.upstash import UpstashEmbeddingRetriever
from haystack_integrations.document_stores.upstash import UpstashDocumentStore


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_init(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    store = UpstashDocumentStore()
    retriever = UpstashEmbeddingRetriever(
        document_store=store, filters={"field": "meta.genre", "operator": "==", "value": "tech"}, top_k=5
    )
    assert retriever.filters == {"field": "meta.genre", "operator": "==", "value": "tech"}
    assert retriever.top_k == 5


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_to_dict_from_dict(mock_index, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")
    store = UpstashDocumentStore()
    retriever = UpstashEmbeddingRetriever(
        document_store=store, filters={"field": "meta.genre", "operator": "==", "value": "tech"}, top_k=5
    )

    data = retriever.to_dict()
    assert data == {
        "type": "haystack_integrations.components.retrievers.upstash.embedding_retriever.UpstashEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "type": "haystack_integrations.document_stores.upstash.document_store.UpstashDocumentStore",
                "init_parameters": {
                    "url": {"env_vars": ["UPSTASH_VECTOR_REST_URL"], "strict": True, "type": "env_var"},
                    "token": {"env_vars": ["UPSTASH_VECTOR_REST_TOKEN"], "strict": True, "type": "env_var"},
                },
            },
            "filters": {"field": "meta.genre", "operator": "==", "value": "tech"},
            "top_k": 5,
        },
    }

    restored = UpstashEmbeddingRetriever.from_dict(data)
    assert restored.filters == {"field": "meta.genre", "operator": "==", "value": "tech"}
    assert restored.top_k == 5
    assert restored.document_store.url.resolve_value() == "http://test"


@patch("haystack_integrations.document_stores.upstash.document_store.Index")
def test_run(mock_index, monkeypatch):
    monkeypatch.setenv("UPSTASH_VECTOR_REST_URL", "http://test")
    monkeypatch.setenv("UPSTASH_VECTOR_REST_TOKEN", "test-token")

    mock_instance = MagicMock()
    mock_index.return_value = mock_instance
    mock_res = MagicMock()
    mock_res.id = "1"
    mock_res.data = "test content"
    mock_res.vector = [0.1, 0.2, 0.3]
    mock_res.metadata = {"genre": "tech"}
    mock_res.score = 0.95
    mock_instance.query.return_value = [mock_res]

    store = UpstashDocumentStore()
    retriever = UpstashEmbeddingRetriever(document_store=store)

    result = retriever.run(query_embedding=[0.1, 0.2, 0.3], top_k=2)
    docs = result["documents"]

    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "test content"
    assert docs[0].meta["genre"] == "tech"
    assert docs[0].score == 0.95

    _args, kwargs = mock_instance.query.call_args
    assert kwargs["vector"] == [0.1, 0.2, 0.3]
    assert kwargs["top_k"] == 2
    assert kwargs["include_data"] is True
