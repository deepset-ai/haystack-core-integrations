# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# Integration tests in TestElasticsearchInferenceHybridRetriever connect to a managed
# Elastic Cloud cluster. They are skipped automatically when the required environment
# variables are absent.
#
# --- Account & project setup -------------------------------------------------
#
#   1. Sign up for a free trial at https://cloud.elastic.co/signup (no credit card needed).
#   2. Go to cloud.elastic.co -> Create project -> Elasticsearch (Serverless).
#   3. Choose a region close to you, give the project a name, and click Create.
#   4. Once the project is ready (~1-2 min), collect:
#        Endpoint URL  -> listed as "Elasticsearch endpoint" in the project Overview
#        API key       -> Project Settings -> API Keys -> Create API key
#
# --- Inference endpoint -------------------------------------------------------
#
#   These tests combine ELSER sparse retrieval with BM25 full-text search via
#   server-side Reciprocal Rank Fusion (single Elasticsearch request).
#   ELASTICSEARCH_INFERENCE_ID is required - there is no default.
#
#   The inference endpoint to use depends on your cluster deployment type:
#
#     Serverless project   -> .elser-2-elasticsearch  (built-in, no deployment needed)
#     Stateful ESS cluster -> .elser-2-elastic         (Elastic-hosted ELSER, no ML node capacity consumed)
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
#     ELASTICSEARCH_URL          - cluster endpoint
#                                  e.g. https://my-project.es.<region>.aws.elastic.cloud
#     ELASTIC_API_KEY            - API key created in the project settings
#     ELASTICSEARCH_INFERENCE_ID - sparse inference endpoint (no default, must be set explicitly)
#
# --- Running the tests --------------------------------------------------------
#
#   Serverless project:
#     ELASTICSEARCH_URL="https://my-project.es.<region>.aws.elastic.cloud" \
#     ELASTIC_API_KEY="<your-key>" \
#     ELASTICSEARCH_INFERENCE_ID=".elser-2-elasticsearch" \
#     pytest -m integration tests/test_cloud_hybrid_retriever.py
#
#   Stateful ESS cluster:
#     ELASTICSEARCH_URL="https://my-cluster.es.io:443" \
#     ELASTIC_API_KEY="<your-key>" \
#     ELASTICSEARCH_INFERENCE_ID=".elser-2-elastic" \
#     pytest -m integration tests/test_cloud_hybrid_retriever.py


from copy import deepcopy
from unittest.mock import AsyncMock, Mock, patch

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceHybridRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore

serialised = {
    "type": "haystack_integrations.components.retrievers.elasticsearch.inference_hybrid_retriever.ElasticsearchInferenceHybridRetriever",  # noqa: E501
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
        "inference_id": ".elser_model_2",
        "filters": {},
        "fuzziness": "AUTO",
        "top_k": 10,
        "filter_policy": "replace",
        "rank_window_size": 100,
        "rank_constant": 60,
    },
}


# --- Unit tests ---


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_default(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceHybridRetriever(document_store=doc_store, inference_id=".elser_model_2")
    assert retriever._document_store is doc_store
    assert retriever._inference_id == ".elser_model_2"
    assert retriever._filters == {}
    assert retriever._fuzziness == "AUTO"
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE
    assert retriever._rank_window_size == 100
    assert retriever._rank_constant == 60


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_custom(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceHybridRetriever(
        document_store=doc_store,
        inference_id=".elser_model_2",
        filters={"field": "value"},
        fuzziness="1",
        top_k=5,
        filter_policy=FilterPolicy.MERGE,
        rank_window_size=50,
        rank_constant=30,
    )
    assert retriever._filters == {"field": "value"}
    assert retriever._fuzziness == "1"
    assert retriever._top_k == 5
    assert retriever._filter_policy == FilterPolicy.MERGE
    assert retriever._rank_window_size == 50
    assert retriever._rank_constant == 30


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_requires_inference_id(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    with pytest.raises(ValueError, match="inference_id must be provided"):
        ElasticsearchInferenceHybridRetriever(document_store=doc_store, inference_id="")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_init_wrong_document_store_type(_mock_es):
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchInferenceHybridRetriever(document_store=Mock(), inference_id=".elser_model_2")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_es):
    doc_store = ElasticsearchDocumentStore()
    retriever = ElasticsearchInferenceHybridRetriever(document_store=doc_store, inference_id=".elser_model_2")
    assert retriever.to_dict() == serialised


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_es):
    data = deepcopy(serialised)
    deserialized = ElasticsearchInferenceHybridRetriever.from_dict(data)
    assert isinstance(deserialized, ElasticsearchInferenceHybridRetriever)
    assert deserialized.to_dict() == serialised


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_es):
    data = deepcopy(serialised)
    del data["init_parameters"]["filter_policy"]
    deserialized = ElasticsearchInferenceHybridRetriever.from_dict(data)
    assert isinstance(deserialized, ElasticsearchInferenceHybridRetriever)


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._hybrid_retrieval_inference.return_value = [
        Document(content="BM25 result"),
        Document(content="Sparse result"),
    ]
    retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
    result = retriever.run(query="test query")

    mock_store._hybrid_retrieval_inference.assert_called_once_with(
        query="test query",
        inference_id=".elser_model_2",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        rank_window_size=100,
        rank_constant=60,
    )
    assert len(result["documents"]) == 2


def test_run_runtime_top_k_overrides():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._hybrid_retrieval_inference.return_value = [Document(content="Result")]
    retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
    retriever.run(query="test query", top_k=3)

    mock_store._hybrid_retrieval_inference.assert_called_once_with(
        query="test query",
        inference_id=".elser_model_2",
        filters={},
        fuzziness="AUTO",
        top_k=3,
        rank_window_size=100,
        rank_constant=60,
    )


def test_run_replace_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._hybrid_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceHybridRetriever(
        document_store=mock_store,
        inference_id=".elser_model_2",
        filters={"field": "init", "operator": "==", "value": "init"},
        filter_policy=FilterPolicy.REPLACE,
    )
    retriever.run(query="test", filters={"field": "runtime", "operator": "==", "value": "runtime"})

    call_filters = mock_store._hybrid_retrieval_inference.call_args.kwargs["filters"]
    assert call_filters == {"field": "runtime", "operator": "==", "value": "runtime"}


def test_run_merge_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._hybrid_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceHybridRetriever(
        document_store=mock_store,
        inference_id=".elser_model_2",
        filters={"field": "category", "operator": "==", "value": "news"},
        filter_policy=FilterPolicy.MERGE,
    )
    retriever.run(query="test", filters={"field": "lang", "operator": "==", "value": "en"})

    call_filters = mock_store._hybrid_retrieval_inference.call_args.kwargs["filters"]
    assert call_filters is not None


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._hybrid_retrieval_inference_async = AsyncMock(return_value=[Document(content="Async result")])
    retriever = ElasticsearchInferenceHybridRetriever(document_store=mock_store, inference_id=".elser_model_2")
    result = await retriever.run_async(query="test query")

    mock_store._hybrid_retrieval_inference_async.assert_awaited_once_with(
        query="test query",
        inference_id=".elser_model_2",
        filters={},
        fuzziness="AUTO",
        top_k=10,
        rank_window_size=100,
        rank_constant=60,
    )
    assert len(result["documents"]) == 1
    assert result["documents"][0].content == "Async result"


# --- Integration tests ---


def _index_documents_with_inference(client, index: str, inference_id: str, docs: list[dict]) -> None:
    response = client.inference.inference(
        inference_id=inference_id,
        input=[doc["content"] for doc in docs],
    )
    embeddings = [item["embedding"] for item in response["sparse_embedding"]]
    for doc, sparse_embedding in zip(docs, embeddings, strict=False):
        body: dict = {"id": doc["id"], "content": doc["content"], "sparse_vec": sparse_embedding}
        body.update(doc.get("meta", {}))
        client.index(index=index, id=doc["id"], body=body)
    client.indices.refresh(index=index)


@pytest.mark.integration
class TestElasticsearchInferenceHybridRetriever:
    """
    End-to-end tests against a real Elastic Cloud cluster with a deployed ELSER endpoint.
    Run with: pytest -m integration
    """

    def test_retrieval_returns_documents(self, hybrid_inference_document_store):
        store, inference_id = hybrid_inference_document_store
        retriever = ElasticsearchInferenceHybridRetriever(document_store=store, inference_id=inference_id, top_k=2)
        _index_documents_with_inference(
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

    def test_most_relevant_document_ranks_first(self, hybrid_inference_document_store):
        store, inference_id = hybrid_inference_document_store
        retriever = ElasticsearchInferenceHybridRetriever(document_store=store, inference_id=inference_id, top_k=3)
        _index_documents_with_inference(
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

    def test_respects_top_k(self, hybrid_inference_document_store):
        store, inference_id = hybrid_inference_document_store
        retriever = ElasticsearchInferenceHybridRetriever(document_store=store, inference_id=inference_id, top_k=1)
        _index_documents_with_inference(
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

    def test_filter_applied_to_both_retrievers(self, hybrid_inference_document_store):
        store, inference_id = hybrid_inference_document_store
        retriever = ElasticsearchInferenceHybridRetriever(document_store=store, inference_id=inference_id, top_k=5)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The Eiffel Tower is a famous landmark in Paris, France.", "category": "europe"},
                {"id": "2", "content": "The Amazon rainforest covers most of the Amazon basin.", "category": "america"},
            ],
        )

        result = retriever.run(
            query="famous landmark",
            filters={"field": "category", "operator": "==", "value": "europe"},
        )

        assert all(doc.meta.get("category") == "europe" for doc in result["documents"])

    def test_single_elasticsearch_request(self, hybrid_inference_document_store):
        """Only one search call should be made — confirms server-side RRF, not client-side fusion."""
        store, inference_id = hybrid_inference_document_store
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [{"id": "1", "content": "The Eiffel Tower is in Paris."}],
        )
        retriever = ElasticsearchInferenceHybridRetriever(document_store=store, inference_id=inference_id, top_k=1)

        original_search = store.client.search
        call_count = 0

        def counting_search(**kwargs):
            nonlocal call_count
            call_count += 1
            return original_search(**kwargs)

        with patch.object(store.client, "search", side_effect=counting_search):
            retriever.run(query="tower in France")

        assert call_count == 1, f"Expected 1 search call (server-side RRF), got {call_count}"
