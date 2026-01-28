# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document
from haystack.document_stores.types import FilterPolicy

from haystack_integrations.components.retrievers.valkey import ValkeyEmbeddingRetriever
from haystack_integrations.document_stores.valkey import ValkeyDocumentStore


@pytest.fixture
def document_store():
    store = ValkeyDocumentStore(
        nodes_list=[("localhost", 6379)],
        index_name="test_retriever",
        embedding_dim=3,
        distance_metric="cosine",
        metadata_fields={"category": str, "priority": int},
    )
    yield store
    store.delete_all_documents()
    store.close()


@pytest.fixture
def sample_documents():
    return [
        Document(content="Document 1", embedding=[0.1, 0.2, 0.3], meta={"category": "A", "priority": 1}),
        Document(content="Document 2", embedding=[0.2, 0.3, 0.4], meta={"category": "B", "priority": 2}),
        Document(content="Document 3", embedding=[0.3, 0.4, 0.5], meta={"category": "A", "priority": 3}),
    ]


class TestValkeyEmbeddingRetriever:
    @pytest.mark.integration
    def test_init(self, document_store):
        retriever = ValkeyEmbeddingRetriever(document_store=document_store)
        assert retriever.document_store == document_store
        assert retriever.filters == {}
        assert retriever.top_k == 10
        assert retriever.filter_policy == FilterPolicy.REPLACE

    @pytest.mark.integration
    def test_init_with_parameters(self, document_store):
        filters = {"field": "meta.category", "operator": "==", "value": "A"}
        retriever = ValkeyEmbeddingRetriever(
            document_store=document_store, filters=filters, top_k=5, filter_policy=FilterPolicy.MERGE
        )
        assert retriever.filters == filters
        assert retriever.top_k == 5
        assert retriever.filter_policy == FilterPolicy.MERGE

    @pytest.mark.integration
    def test_init_invalid_document_store(self):
        with pytest.raises(ValueError, match="document_store must be an instance of ValkeyDocumentStore"):
            ValkeyEmbeddingRetriever(document_store="not_a_store")

    @pytest.mark.integration
    def test_to_dict(self, document_store):
        retriever = ValkeyEmbeddingRetriever(
            document_store=document_store,
            filters={"field": "meta.category", "operator": "==", "value": "A"},
            top_k=5,
        )
        result = retriever.to_dict()
        assert (
            result["type"]
            == "haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever"
        )
        assert result["init_parameters"]["filters"] == {"field": "meta.category", "operator": "==", "value": "A"}
        assert result["init_parameters"]["top_k"] == 5

    @pytest.mark.integration
    def test_from_dict(self, document_store):
        data = {
            "type": "haystack_integrations.components.retrievers.valkey.embedding_retriever.ValkeyEmbeddingRetriever",
            "init_parameters": {
                "document_store": document_store.to_dict(),
                "filters": {"field": "meta.category", "operator": "==", "value": "A"},
                "top_k": 5,
                "filter_policy": "replace",
            },
        }
        retriever = ValkeyEmbeddingRetriever.from_dict(data)
        assert isinstance(retriever.document_store, ValkeyDocumentStore)
        assert retriever.filters == {"field": "meta.category", "operator": "==", "value": "A"}
        assert retriever.top_k == 5

    @pytest.mark.integration
    def test_run(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store, top_k=2)
        query_embedding = [0.15, 0.25, 0.35]

        result = retriever.run(query_embedding=query_embedding)

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(doc.score is not None for doc in result["documents"])

    @pytest.mark.integration
    def test_run_with_filters(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store)
        filters = {"field": "meta.category", "operator": "==", "value": "A"}

        result = retriever.run(query_embedding=[0.15, 0.25, 0.35], filters=filters)

        assert len(result["documents"]) == 2
        assert all(doc.meta["category"] == "A" for doc in result["documents"])

    @pytest.mark.integration
    def test_run_with_top_k_override(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store, top_k=10)
        result = retriever.run(query_embedding=[0.15, 0.25, 0.35], top_k=1)

        assert len(result["documents"]) == 1

    @pytest.mark.integration
    def test_run_empty_store(self, document_store):
        retriever = ValkeyEmbeddingRetriever(document_store=document_store)
        result = retriever.run(query_embedding=[0.1, 0.2, 0.3])

        assert result["documents"] == []

    @pytest.mark.integration
    def test_run_with_filter_policy_merge(self, document_store, sample_documents):
        document_store.write_documents(sample_documents)

        init_filters = {"field": "meta.category", "operator": "==", "value": "A"}
        retriever = ValkeyEmbeddingRetriever(
            document_store=document_store, filters=init_filters, filter_policy=FilterPolicy.MERGE
        )

        runtime_filters = {"field": "meta.priority", "operator": ">=", "value": 2}
        result = retriever.run(query_embedding=[0.15, 0.25, 0.35], filters=runtime_filters)

        # Should merge both filters (category A AND priority >= 2)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["category"] == "A"
        assert result["documents"][0].meta["priority"] >= 2


class TestValkeyEmbeddingRetrieverAsync:
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async(self, document_store, sample_documents):
        await document_store.write_documents_async(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store, top_k=2)
        query_embedding = [0.15, 0.25, 0.35]

        result = await retriever.run_async(query_embedding=query_embedding)

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert all(isinstance(doc, Document) for doc in result["documents"])
        assert all(doc.score is not None for doc in result["documents"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_with_filters(self, document_store, sample_documents):
        await document_store.write_documents_async(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store)
        filters = {"field": "meta.category", "operator": "==", "value": "A"}

        result = await retriever.run_async(query_embedding=[0.15, 0.25, 0.35], filters=filters)

        assert len(result["documents"]) == 2
        assert all(doc.meta["category"] == "A" for doc in result["documents"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_with_top_k_override(self, document_store, sample_documents):
        await document_store.write_documents_async(sample_documents)

        retriever = ValkeyEmbeddingRetriever(document_store=document_store, top_k=10)
        result = await retriever.run_async(query_embedding=[0.15, 0.25, 0.35], top_k=1)

        assert len(result["documents"]) == 1

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_empty_store(self, document_store):
        retriever = ValkeyEmbeddingRetriever(document_store=document_store)
        result = await retriever.run_async(query_embedding=[0.1, 0.2, 0.3])

        assert result["documents"] == []

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_run_async_with_filter_policy_merge(self, document_store, sample_documents):
        await document_store.write_documents_async(sample_documents)

        init_filters = {"field": "meta.category", "operator": "==", "value": "A"}
        retriever = ValkeyEmbeddingRetriever(
            document_store=document_store, filters=init_filters, filter_policy=FilterPolicy.MERGE
        )

        runtime_filters = {"field": "meta.priority", "operator": ">=", "value": 2}
        result = await retriever.run_async(query_embedding=[0.15, 0.25, 0.35], filters=runtime_filters)

        # Should merge both filters (category A AND priority >= 2)
        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["category"] == "A"
        assert result["documents"][0].meta["priority"] >= 2
