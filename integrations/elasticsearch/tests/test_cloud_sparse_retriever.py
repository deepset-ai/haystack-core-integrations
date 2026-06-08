# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# Integration tests in TestElasticsearchInferenceSparseRetriever connect to a managed
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
#   These tests use ELSER (Elastic Learned Sparse Encoder) to generate sparse
#   embeddings at query time.  The inference endpoint to use depends on your
#   cluster deployment type:
#
#     Serverless project   -> .elser-2-elasticsearch  (built-in, no deployment needed)
#     Stateful ESS cluster -> .elser-2-elastic         (Elastic-hosted ELSER, no ML node capacity consumed)
#
#   The default is ".elser-2-elastic". Override with ELASTICSEARCH_INFERENCE_ID.
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
#
#   Optional:
#     ELASTICSEARCH_INFERENCE_ID - sparse inference endpoint to use
#                                  default: ".elser-2-elastic"  (stateful ESS)
#                                  Serverless: set to ".elser-2-elasticsearch"
#
# --- Running the tests --------------------------------------------------------
#
#   Serverless project:
#     ELASTICSEARCH_URL="https://my-project.es.<region>.aws.elastic.cloud" \
#     ELASTIC_API_KEY="<your-key>" \
#     ELASTICSEARCH_INFERENCE_ID=".elser-2-elasticsearch" \
#     pytest -m integration tests/test_cloud_sparse_retriever.py
#
#   Stateful ESS cluster (uses the default inference endpoint):
#     ELASTICSEARCH_URL="https://my-cluster.es.io:443" \
#     ELASTIC_API_KEY="<your-key>" \
#     pytest -m integration tests/test_cloud_sparse_retriever.py

import uuid
from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchInferenceSparseRetriever
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore


def test_init_default():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")

    assert retriever._document_store == mock_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_init_requires_inference_id():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    with pytest.raises(ValueError, match="inference_id must be provided"):
        ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="")


def test_init_wrong_document_store_type():
    with pytest.raises(ValueError, match="document_store must be an instance of ElasticsearchDocumentStore"):
        ElasticsearchInferenceSparseRetriever(document_store=Mock(), inference_id="ELSER")


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_to_dict(_mock_elasticsearch_client):
    document_store = ElasticsearchDocumentStore(hosts="some fake host", sparse_vector_field="sparse_vec")
    retriever = ElasticsearchInferenceSparseRetriever(document_store=document_store, inference_id="ELSER")
    retriever_type = (
        "haystack_integrations.components.retrievers.elasticsearch."
        "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
    )

    assert retriever.to_dict() == {
        "type": retriever_type,
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "api_key": {
                        "env_vars": [
                            "ELASTIC_API_KEY",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "api_key_id": {
                        "env_vars": [
                            "ELASTIC_API_KEY_ID",
                        ],
                        "strict": False,
                        "type": "env_var",
                    },
                    "hosts": "some fake host",
                    "custom_mapping": None,
                    "index": "default",
                    "embedding_similarity_function": "cosine",
                    "sparse_vector_field": "sparse_vec",
                    "ingest_pipeline": None,
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "inference_id": "ELSER",
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "inference_id": "ELSER",
            "filters": {},
            "top_k": 10,
            "filter_policy": "replace",
        },
    }
    retriever = ElasticsearchInferenceSparseRetriever.from_dict(data)

    assert retriever._document_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


@patch("haystack_integrations.document_stores.elasticsearch.document_store.Elasticsearch")
def test_from_dict_no_filter_policy(_mock_elasticsearch_client):
    data = {
        "type": (
            "haystack_integrations.components.retrievers.elasticsearch."
            "inference_sparse_retriever.ElasticsearchInferenceSparseRetriever"
        ),
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "hosts": "some fake host",
                    "index": "default",
                    "sparse_vector_field": "sparse_vec",
                },
                "type": "haystack_integrations.document_stores.elasticsearch.document_store.ElasticsearchDocumentStore",
            },
            "inference_id": "ELSER",
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = ElasticsearchInferenceSparseRetriever.from_dict(data)

    assert retriever._document_store
    assert retriever._inference_id == "ELSER"
    assert retriever._filters == {}
    assert retriever._top_k == 10
    assert retriever._filter_policy == FilterPolicy.REPLACE


def test_run():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Test doc")]
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")
    res = retriever.run(query="Find docs")

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"


def test_run_init_params():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = [Document(content="Test doc")]
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters=init_filters,
        top_k=3,
        filter_policy=FilterPolicy.MERGE,
    )

    res = retriever.run(query="Find docs")

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters=init_filters,
        top_k=3,
    )
    assert len(res["documents"]) == 1


def test_run_replace_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters={"field": "meta.source", "operator": "==", "value": "wiki"},
        top_k=5,
        filter_policy=FilterPolicy.REPLACE,
    )
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}

    retriever.run(query="Find docs", filters=runtime_filters)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters=runtime_filters,
        top_k=5,
    )


def test_run_merge_filter_policy():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    init_filters = {"field": "meta.source", "operator": "==", "value": "wiki"}
    runtime_filters = {"field": "meta.lang", "operator": "==", "value": "en"}
    retriever = ElasticsearchInferenceSparseRetriever(
        document_store=mock_store,
        inference_id="ELSER",
        filters=init_filters,
        top_k=5,
        filter_policy=FilterPolicy.MERGE,
    )

    retriever.run(query="Find docs", filters=runtime_filters)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={"operator": "AND", "conditions": [init_filters, runtime_filters]},
        top_k=5,
    )


def test_run_runtime_top_k_overrides():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference.return_value = []
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER", top_k=10)

    retriever.run(query="Find docs", top_k=2)

    mock_store._sparse_vector_retrieval_inference.assert_called_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=2,
    )


@pytest.mark.asyncio
async def test_run_async():
    mock_store = Mock(spec=ElasticsearchDocumentStore)
    mock_store._sparse_vector_retrieval_inference_async.return_value = [Document(content="test document")]
    retriever = ElasticsearchInferenceSparseRetriever(document_store=mock_store, inference_id="ELSER")
    res = await retriever.run_async(query="Find docs")

    mock_store._sparse_vector_retrieval_inference_async.assert_awaited_once_with(
        query="Find docs",
        inference_id="ELSER",
        filters={},
        top_k=10,
    )
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "test document"


def _index_documents_with_inference(client, index: str, inference_id: str, documents: list[dict]) -> None:
    """
    Encode each document's content via the ES inference API (ELSER) and index the result.

    Documents are indexed directly via the ES client so that the sparse_vec field contains
    real ELSER token weights (string → float map) rather than Haystack's integer-indexed
    SparseEmbedding format.  This is required for the sparse_vector query with inference_id
    to return semantically meaningful results.

    Each dict in `documents` must have a "content" key and may have "id" and "meta" keys.
    """
    response = client.inference.inference(
        inference_id=inference_id,
        input=[doc["content"] for doc in documents],
    )
    embeddings = [item["embedding"] for item in response["sparse_embedding"]]
    for doc, sparse_embedding in zip(documents, embeddings, strict=False):
        doc_id = doc.get("id", uuid.uuid4().hex)
        # Haystack's Document.to_dict() flattens meta keys to the top level, so _normalize_filters
        # strips the "meta." prefix when building ES queries. Mirror that here.
        body: dict = {"id": doc_id, "content": doc["content"], "sparse_vec": sparse_embedding}
        body.update(doc.get("meta", {}))
        client.index(index=index, id=doc_id, body=body)
    client.indices.refresh(index=index)


@pytest.mark.integration
class TestElasticsearchInferenceSparseRetriever:
    """
    End-to-end integration tests for ElasticsearchInferenceSparseRetriever.

    These tests connect to a real Elastic Cloud cluster and call a deployed inference
    endpoint (e.g. ELSER v2) server-side, so they are slower and require credentials.
    Run them with: pytest -m integration
    """

    def test_retrieval_returns_most_relevant_document(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=1)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The Eiffel Tower is a famous landmark in Paris, France."},
                {"id": "2", "content": "The Amazon rainforest covers most of the Amazon basin in South America."},
            ],
        )

        result = retriever.run(query="famous tower in France")

        assert len(result["documents"]) == 1
        assert "Eiffel" in result["documents"][0].content

    def test_retrieval_respects_top_k(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=2)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "Python is a popular programming language."},
                {"id": "2", "content": "Java is widely used in enterprise software."},
                {"id": "3", "content": "Rust is a systems programming language focused on memory safety."},
            ],
        )

        result = retriever.run(query="programming language")

        assert 0 < len(result["documents"]) <= 2

    def test_retrieval_top_k_runtime_override(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=10)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The sun is a star at the center of our solar system."},
                {"id": "2", "content": "Jupiter is the largest planet in the solar system."},
                {"id": "3", "content": "The Moon orbits the Earth roughly every 27 days."},
            ],
        )

        result = retriever.run(query="solar system planets", top_k=1)

        assert len(result["documents"]) == 1

    def test_retrieval_with_filter(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=5)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "Berlin is the capital of Germany.", "meta": {"lang": "en"}},
                {"id": "2", "content": "Berlin ist die Hauptstadt von Deutschland.", "meta": {"lang": "de"}},
            ],
        )

        result = retriever.run(
            query="capital of Germany",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Berlin is the capital of Germany."

    def test_retrieval_replace_filter_policy(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(
            document_store=store,
            inference_id=inference_id,
            top_k=5,
            filters={"field": "meta.lang", "operator": "==", "value": "de"},
            filter_policy=FilterPolicy.REPLACE,
        )
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "The cat sat on the mat.", "meta": {"lang": "en"}},
                {"id": "2", "content": "Die Katze saß auf der Matte.", "meta": {"lang": "de"}},
            ],
        )

        # REPLACE: the runtime filter overwrites the init filter
        result = retriever.run(
            query="cat on mat",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["lang"] == "en"

    def test_retrieval_merge_filter_policy(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(
            document_store=store,
            inference_id=inference_id,
            top_k=10,
            filters={"field": "meta.category", "operator": "==", "value": "science"},
            filter_policy=FilterPolicy.MERGE,
        )
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {
                    "id": "1",
                    "content": "Quantum mechanics governs the behaviour of subatomic particles.",
                    "meta": {"category": "science", "lang": "en"},
                },
                {
                    "id": "2",
                    "content": "La mécanique quantique décrit la nature à l'échelle atomique.",
                    "meta": {"category": "science", "lang": "fr"},
                },
                {
                    "id": "3",
                    "content": "Shakespeare wrote Hamlet around 1600.",
                    "meta": {"category": "literature", "lang": "en"},
                },
            ],
        )

        # MERGE: AND(category==science, lang==en) — only doc 1 should match
        result = retriever.run(
            query="quantum physics",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["category"] == "science"
        assert result["documents"][0].meta["lang"] == "en"

    def test_returned_documents_have_content(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=3)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "Photosynthesis converts sunlight into chemical energy in plants."},
                {"id": "2", "content": "The water cycle describes the continuous movement of water on Earth."},
            ],
        )

        result = retriever.run(query="how plants produce energy")

        assert len(result["documents"]) > 0
        for doc in result["documents"]:
            assert doc.content is not None
            assert doc.score is not None
            # ELSER uses non-integer token keys; sparse_embedding will be None after deserialization
            assert doc.sparse_embedding is None

    @pytest.mark.asyncio
    async def test_async_retrieval_returns_most_relevant_document(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=1)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "Mount Everest is the highest mountain on Earth."},
                {"id": "2", "content": "The Pacific Ocean is the largest and deepest ocean on Earth."},
            ],
        )

        result = await retriever.run_async(query="tallest mountain in the world")

        assert len(result["documents"]) == 1
        assert "Everest" in result["documents"][0].content

    @pytest.mark.asyncio
    async def test_async_retrieval_with_filter(self, inference_sparse_document_store):
        store, inference_id = inference_sparse_document_store
        retriever = ElasticsearchInferenceSparseRetriever(document_store=store, inference_id=inference_id, top_k=5)
        _index_documents_with_inference(
            store.client,
            store._index,
            inference_id,
            [
                {"id": "1", "content": "Rome is the capital of Italy.", "meta": {"lang": "en"}},
                {"id": "2", "content": "Roma è la capitale d'Italia.", "meta": {"lang": "it"}},
            ],
        )

        result = await retriever.run_async(
            query="capital of Italy",
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Rome is the capital of Italy."
