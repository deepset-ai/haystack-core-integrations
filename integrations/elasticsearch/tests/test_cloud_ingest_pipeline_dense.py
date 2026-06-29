# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
#
# Integration tests in TestElasticSearchIngestPipelineDense connect to a managed
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
#   These tests use a dense embedding model to generate float vectors at index time
#   via an Elasticsearch ingest pipeline.  Haystack writes raw text; the pipeline
#   fills the embedding field before the document is committed to the index.
#
#   Two service types exist for inference endpoints:
#
#     service: elastic       - hosted externally by Elastic, no ML node capacity needed,
#                              works on Serverless free trial out of the box
#     service: elasticsearch - runs on the cluster's own ML nodes; requires available
#                              ML memory (may fail on a free trial with limited capacity)
#
#   The default ELASTICSEARCH_DENSE_INFERENCE_ID is ".multilingual-e5-small-elasticsearch"
#   (service: elasticsearch, 384 dims). It requires ML node memory and will fail on a
#   free-trial Serverless cluster with insufficient capacity.
#
#   To list all inference endpoints available on your cluster:
#
#     curl -s \
#       -H "Authorization: ApiKey <your-key>" \
#       "https://<your-cluster-endpoint>/_inference" | jq '.[].inference_id'
#
#   The following "service: elastic" text_embedding endpoints are available out of the
#   box on Serverless free trial (no ML node capacity needed):
#
#     .jina-embeddings-v3               (1024 dims)  - GA, multilingual
#     .jina-embeddings-v5-text-nano     (768 dims)   - GA, multilingual
#     .jina-embeddings-v5-text-small    (1024 dims)  - GA, multilingual
#     .google-gemini-embedding-001      (3072 dims)  - GA, multilingual
#     .openai-text-embedding-3-small    (1536 dims)  - GA, multilingual
#     .openai-text-embedding-3-large    (3072 dims)  - GA, multilingual
#
# --- Environment variables ----------------------------------------------------
#
#   Required:
#     ELASTICSEARCH_URL                  - cluster endpoint
#                                          e.g. https://my-project.es.<region>.aws.elastic.cloud
#     ELASTIC_API_KEY                    - API key created in the project settings
#
#   Optional:
#     ELASTICSEARCH_DENSE_INFERENCE_ID   - dense inference endpoint to use
#                                          default: ".multilingual-e5-small-elasticsearch"
#                                          Serverless free trial: use a "service: elastic" endpoint,
#                                          e.g. ".jina-embeddings-v3"
#     ELASTICSEARCH_DENSE_EMBEDDING_DIMS - output dimensions of the chosen model
#                                          default: "384" (matches .multilingual-e5-small-elasticsearch)
#                                          e.g. "1024" for .jina-embeddings-v3
#
# --- Running the tests --------------------------------------------------------
#
#   Serverless free trial (Jina embeddings, no ML nodes needed):
#     ELASTICSEARCH_URL="https://my-project.es.<region>.aws.elastic.cloud" \
#     ELASTIC_API_KEY="<your-key>" \
#     ELASTICSEARCH_DENSE_INFERENCE_ID=".jina-embeddings-v3" \
#     ELASTICSEARCH_DENSE_EMBEDDING_DIMS="1024" \
#     pytest -m integration tests/test_cloud_ingest_pipeline_dense.py
#
#   Stateful ESS cluster (ML nodes available, default multilingual-e5-small):
#     ELASTICSEARCH_URL="https://my-cluster.es.io:443" \
#     ELASTIC_API_KEY="<your-key>" \
#     pytest -m integration tests/test_cloud_ingest_pipeline_dense.py

import pytest
from haystack.dataclasses import Document

from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever


def _get_dense_query_embedding(client, inference_id: str, text: str) -> list[float]:
    """Call the ES inference API to embed a query string using the same model as the ingest pipeline."""
    response = client.inference.inference(inference_id=inference_id, input=[text])
    return response["text_embedding"][0]["embedding"]


@pytest.mark.integration
class TestElasticSearchIngestPipelineDense:
    """
    End-to-end integration tests for ElasticsearchDocumentStore with an ingest pipeline
    that generates dense embeddings at index time.

    The fixture creates a real ES ingest pipeline (inference processor → embedding field)
    on Elastic Cloud. Documents are written without pre-computed embeddings; the pipeline
    fills the embedding field before the document is committed to the index.
    """

    def test_indexed_document_has_embedding_filled_by_pipeline(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        store.write_documents([Document(id="doc-1", content="The Eiffel Tower is located in Paris.")])

        store.client.indices.refresh(index=store._index)
        raw = store.client.get(index=store._index, id="doc-1")

        # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
        assert raw["_source"].get("embedding") is None

        # The vector IS in the index: a KNN search using the same model finds the document.
        query_embedding = _get_dense_query_embedding(store.client, inference_id, "Eiffel Tower Paris")
        result = store.client.search(
            index=store._index,
            knn={"field": "embedding", "query_vector": query_embedding, "k": 1, "num_candidates": 10},
        )
        assert result["hits"]["total"]["value"] == 1, "pipeline did not populate the 'embedding' field"

    def test_embedding_retriever_finds_most_relevant_document(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=1)

        store.write_documents(
            [
                Document(id="1", content="The Eiffel Tower is a famous landmark in Paris, France."),
                Document(id="2", content="The Amazon River flows through the South American rainforest."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "famous tower in France")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) == 1
        assert "Eiffel" in result["documents"][0].content

    def test_embedding_retriever_respects_top_k(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Python is a popular high-level programming language."),
                Document(id="2", content="Java is widely used in enterprise software development."),
                Document(id="3", content="Rust focuses on memory safety and systems programming."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "programming language")
        result = retriever.run(query_embedding=query_embedding)

        assert 0 < len(result["documents"]) <= 2

    def test_embedding_retriever_with_metadata_filter(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=5)

        store.write_documents(
            [
                Document(id="1", content="Berlin is the capital of Germany.", meta={"lang": "en"}),
                Document(id="2", content="Berlin ist die Hauptstadt von Deutschland.", meta={"lang": "de"}),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "capital of Germany")
        result = retriever.run(
            query_embedding=query_embedding,
            filters={"field": "meta.lang", "operator": "==", "value": "en"},
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "Berlin is the capital of Germany."

    def test_multiple_documents_are_all_indexed_with_embeddings(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        docs = [Document(id=f"doc-{i}", content=f"Document number {i} about various topics.") for i in range(5)]
        store.write_documents(docs)

        store.client.indices.refresh(index=store._index)
        assert store.count_documents() == 5

        for doc in docs:
            raw = store.client.get(index=store._index, id=doc.id)
            # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
            assert raw["_source"].get("embedding") is None

        # Verify all documents have embeddings via KNN search — finds docs only if vectors exist.
        query_embedding = _get_dense_query_embedding(store.client, inference_id, "document about topics")
        result = store.client.search(
            index=store._index,
            knn={"field": "embedding", "query_vector": query_embedding, "k": 5, "num_candidates": 50},
        )
        assert result["hits"]["total"]["value"] == 5, "not all documents were indexed with embeddings"

    def test_retrieved_documents_carry_score(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=2)

        store.write_documents(
            [
                Document(id="1", content="Mount Everest is the highest mountain on Earth."),
                Document(id="2", content="The Pacific Ocean is the largest ocean on Earth."),
            ]
        )

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "tallest mountain")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) > 0
        for doc in result["documents"]:
            assert doc.score is not None
            assert doc.content is not None

    @pytest.mark.asyncio
    async def test_async_write_documents_via_pipeline(self, ingest_pipeline_dense_document_store):
        store, inference_id = ingest_pipeline_dense_document_store
        retriever = ElasticsearchEmbeddingRetriever(document_store=store, top_k=1)

        await store.write_documents_async(
            [
                Document(id="async-1", content="Rome is the capital of Italy."),
                Document(id="async-2", content="Tokyo is the capital of Japan."),
            ]
        )

        store.client.indices.refresh(index=store._index)
        raw = store.client.get(index=store._index, id="async-1")
        # ES inference pipelines do NOT write dense_vector data to _source; only to the vector index.
        assert raw["_source"].get("embedding") is None

        query_embedding = _get_dense_query_embedding(store.client, inference_id, "capital of Italy")
        result = retriever.run(query_embedding=query_embedding)

        assert len(result["documents"]) == 1
        assert "Rome" in result["documents"][0].content
