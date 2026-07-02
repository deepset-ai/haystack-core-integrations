# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# Integration tests in TestElasticsearchInferenceDenseRetriever connect to a managed
# Elastic Cloud cluster. They are skipped automatically when the required environment
# variables are absent.
#
# --- Account & project setup -------------------------------------------------
#
#   1. Sign up for a free trial at https://cloud.elastic.co/signup (no credit card needed).
#   2. Go to cloud.elastic.co -> Create project -> Elasticsearch (Serverless).
#   3. Once the project is ready, collect:
#        Endpoint URL  -> listed as "Elasticsearch endpoint" in the project Overview
#        API key       -> Project Settings -> API Keys -> Create API key
#
# --- Inference endpoint -------------------------------------------------------
#
#   These tests use a text-embedding inference endpoint to generate dense query vectors
#   server-side (no local embedding model needed).
#
#   The inference endpoint to use depends on your cluster deployment type:
#
#     Serverless project   -> .multilingual-e5-small-elasticsearch  (built-in, no deployment needed)
#     Stateful ESS cluster -> deploy a text_embedding endpoint via the ML nodes UI
#
#   To list all inference endpoints available on your cluster:
#
#     curl -s \
#       -H "Authorization: ApiKey <your-key>" \
#       "https://<your-cluster-endpoint>/_inference" | jq '.[].inference_id'
#
# --- Environment variables ----------------------------------------------------
#
#   Required:
#     ELASTICSEARCH_URL                 - cluster endpoint
#     ELASTIC_API_KEY                   - API key
#     ELASTICSEARCH_DENSE_INFERENCE_ID  - dense text-embedding inference endpoint ID
#
# --- Running the tests --------------------------------------------------------
#
#   Serverless project:
#     ELASTICSEARCH_URL="https://my-project.es.<region>.aws.elastic.cloud" \
#     ELASTIC_API_KEY="<your-key>" \
#     ELASTICSEARCH_DENSE_INFERENCE_ID=".multilingual-e5-small-elasticsearch" \
#     pytest -m integration tests/test_cloud_dense_retriever.py


import os
from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceDenseRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

serialised = {
    "type": "haystack_integrations.components.retrievers.elasticsearch.inference_dense_retriever.ElasticsearchInferenceDenseRetriever",  # noqa: E501
    "init_parameters": {
        "document_store": {
            "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            "init_parameters": {
                "hosts": None,
                "custom_mapping": None,
                "index": "default",
                "api_key": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY"], "strict": False},
                "api_key_id": {"type": "env_var", "env_vars": ["ELASTIC_API_KEY_ID"], "strict": False},
                "embedding_similarity_function": "cosine",
                "sparse_vector_field": None,
                "ingest_pipeline": None,
            },
        },
        "inference_id": ".multilingual-e5-small-elasticsearch",
        "filters": {},
        "top_k": 10,
        "num_candidates": None,
        "filter_policy": "replace",
    },
}


# --- Unit tests ---


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_default(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=doc_store, inference_id=".multilingual-e5-small-elasticsearch"
    )
    assert retriever._document_store is doc_store
    assert retriever._inference_id == ".multilingual-e5-small-elasticsearch"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._num_candidates is None
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_custom(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=doc_store,
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={"field": "lang", "operator": "==", "value": "en"},
        top_k=5,
        num_candidates=50,
        filter_policy=FilterPolicy.MERGE,
    )
    assert retriever._filters == {"field": "lang", "operator": "==", "value": "en"}
    assert retriever._top_k == 5
    assert retriever._num_candidates == 50
    assert retriever._filter_policy == FilterPolicy.MERGE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_requires_inference_id(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    with pytest.raises(ValueError, match="inference_id must be provided"):
        ElasticsearchInferenceDenseRetriever(document_store=doc_store, inference_id="")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_wrong_document_store_type(_mock_es):
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchInferenceDenseRetriever(document_store=Mock(), inference_id=".multilingual-e5-small-elasticsearch")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=doc_store, inference_id=".multilingual-e5-small-elasticsearch"
    )
    assert retriever.to_dict() == serialised


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_es):
    data = deepcopy(serialised)
    deserialized = ElasticsearchInferenceDenseRetriever.from_dict(data)
    assert isinstance(deserialized, ElasticsearchInferenceDenseRetriever)
    assert deserialized.to_dict() == serialised


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_es):
    data = deepcopy(serialised)
    del data["init_parameters"]["filter_policy"]
    deserialized = ElasticsearchInferenceDenseRetriever.from_dict(data)
    assert isinstance(deserialized, ElasticsearchInferenceDenseRetriever)


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._dense_retrieval_inference.return_value = [
        Document(content="Dense result 1"),
        Document(content="Dense result 2"),
    ]
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=mock_store, inference_id=".multilingual-e5-small-elasticsearch"
    )
    result = retriever.run(query="test query")

    mock_store._dense_retrieval_inference.assert_called_once_with(
        query="test query",
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={},
        top_k=10,
        num_candidates=None,
    )
    assert len(result["documents"]) == 2


def test_run_runtime_top_k_overrides():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._dense_retrieval_inference.return_value = [Document(content="Result")]
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=mock_store, inference_id=".multilingual-e5-small-elasticsearch"
    )
    retriever.run(query="test query", top_k=3)

    mock_store._dense_retrieval_inference.assert_called_once_with(
        query="test query",
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={},
        top_k=3,
        num_candidates=None,
    )


def test_run_replace_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._dense_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=mock_store,
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={"field": "lang", "operator": "==", "value": "en"},
        filter_policy=FilterPolicy.REPLACE,
    )
    retriever.run(query="test", filters={"field": "category", "operator": "==", "value": "news"})

    call_filters = mock_store._dense_retrieval_inference.call_args.kwargs["filters"]
    assert call_filters == {"field": "category", "operator": "==", "value": "news"}


def test_run_merge_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._dense_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=mock_store,
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={"field": "lang", "operator": "==", "value": "en"},
        filter_policy=FilterPolicy.MERGE,
    )
    retriever.run(query="test", filters={"field": "category", "operator": "==", "value": "news"})

    call_filters = mock_store._dense_retrieval_inference.call_args.kwargs["filters"]
    assert call_filters is not None


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._dense_retrieval_inference_async = AsyncMock(return_value=[Document(content="Async result")])
    retriever = ElasticsearchInferenceDenseRetriever(
        document_store=mock_store, inference_id=".multilingual-e5-small-elasticsearch"
    )
    result = await retriever.run_async(query="test query")

    mock_store._dense_retrieval_inference_async.assert_awaited_once_with(
        query="test query",
        inference_id=".multilingual-e5-small-elasticsearch",
        filters={},
        top_k=10,
        num_candidates=None,
    )
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "Async result"


def test_body_query_vector_builder_shape():
    """_create_dense_retrieval_inference_body must produce knn.query_vector_builder, not query_vector."""
    doc_store = ElasticsearchDocumentStore.__new__(ElasticsearchDocumentStore)
    doc_store._sparse_vector_field = None
    body = doc_store._create_dense_retrieval_inference_body(
        query="test query",
        inference_id=".multilingual-e5-small-elasticsearch",
        top_k=5,
    )
    assert "knn" in body
    knn = body["knn"]
    assert knn["field"] == "embedding"
    assert knn["k"] == 5
    assert knn["num_candidates"] == 50
    assert "query_vector_builder" in knn
    assert knn["query_vector_builder"]["text_embedding"]["model_id"] == ".multilingual-e5-small-elasticsearch"
    assert knn["query_vector_builder"]["text_embedding"]["model_text"] == "test query"
    assert "query_vector" not in knn


def test_body_includes_filter():
    doc_store = ElasticsearchDocumentStore.__new__(ElasticsearchDocumentStore)
    doc_store._sparse_vector_field = None
    body = doc_store._create_dense_retrieval_inference_body(
        query="test",
        inference_id="my-endpoint",
        filters={"field": "lang", "operator": "==", "value": "en"},
        top_k=3,
    )
    assert "filter" in body["knn"]


def test_body_empty_query_raises():
    doc_store = ElasticsearchDocumentStore.__new__(ElasticsearchDocumentStore)
    doc_store._sparse_vector_field = None
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        doc_store._create_dense_retrieval_inference_body(query="", inference_id="my-endpoint")


def test_body_custom_num_candidates():
    doc_store = ElasticsearchDocumentStore.__new__(ElasticsearchDocumentStore)
    doc_store._sparse_vector_field = None
    body = doc_store._create_dense_retrieval_inference_body(
        query="test",
        inference_id="my-endpoint",
        top_k=5,
        num_candidates=200,
    )
    assert body["knn"]["num_candidates"] == 200


# --- Integration tests ---


@pytest.mark.integration
class TestElasticsearchInferenceDenseRetriever:
    """
    End-to-end tests against a real Elastic Cloud cluster with a deployed text-embedding endpoint.
    Run with: pytest -m integration
    """

    @pytest.fixture()
    def dense_inference_document_store(self):
        url = os.environ.get("ELASTICSEARCH_URL")
        api_key = os.environ.get("ELASTIC_API_KEY")
        inference_id = os.environ.get("ELASTICSEARCH_DENSE_INFERENCE_ID")
        if not all([url, api_key, inference_id]):
            pytest.skip(
                "Set ELASTICSEARCH_URL, ELASTIC_API_KEY, and ELASTICSEARCH_DENSE_INFERENCE_ID to run integration tests"
            )
        store = ElasticsearchDocumentStore(
            hosts=url,
            api_key=api_key,
            index=f"test-dense-inference-{inference_id.strip('.').replace('-', '_')}",
        )
        yield store, inference_id
        store.client.indices.delete(index=store._index, ignore_unavailable=True)

    def _index_documents_with_inference(self, client, index, inference_id, docs):
        response = client.inference.inference(
            inference_id=inference_id,
            input=[doc["content"] for doc in docs],
        )
        embeddings = [item["embedding"] for item in response["dense_embedding"]]
        for doc, embedding in zip(docs, embeddings, strict=False):
            body = {"id": doc["id"], "content": doc["content"], "embedding": embedding}
            body.update(doc.get("meta", {}))
            client.index(index=index, id=doc["id"], body=body)
        client.indices.refresh(index=index)

    def test_retrieval_returns_documents(self, dense_inference_document_store):
        store, inference_id = dense_inference_document_store
        retriever = ElasticsearchInferenceDenseRetriever(document_store=store, inference_id=inference_id, top_k=2)
        self._index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The Eiffel Tower is a famous landmark in Paris, France."},
                {"id": "2", "content": "The Amazon rainforest covers most of the Amazon basin in South America."},
                {"id": "3", "content": "Mount Fuji is the highest mountain in Japan."},
            ],
        )

        result = retriever.run(query="famous tower in France")

        assert 0 < len(result["documents"]) <= 2
        assert all(isinstance(doc, Document) for doc in result["documents"])

    def test_most_relevant_document_ranks_first(self, dense_inference_document_store):
        store, inference_id = dense_inference_document_store
        retriever = ElasticsearchInferenceDenseRetriever(document_store=store, inference_id=inference_id, top_k=3)
        self._index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The Eiffel Tower is a famous landmark in Paris, France."},
                {"id": "2", "content": "The Amazon rainforest covers most of the Amazon basin in South America."},
                {"id": "3", "content": "Mount Fuji is the highest mountain in Japan."},
            ],
        )

        result = retriever.run(query="famous tower in France")

        assert len(result["documents"]) > 0
        assert "Eiffel" in result["documents"][0].content

    def test_respects_top_k(self, dense_inference_document_store):
        store, inference_id = dense_inference_document_store
        retriever = ElasticsearchInferenceDenseRetriever(document_store=store, inference_id=inference_id, top_k=1)
        self._index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The Eiffel Tower is a famous landmark in Paris, France."},
                {"id": "2", "content": "The Amazon rainforest covers most of the Amazon basin in South America."},
            ],
        )

        result = retriever.run(query="famous landmark")

        assert len(result["documents"]) == 1
