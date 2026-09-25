# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.components.retrievers.astra import AstraEmbeddingRetriever
from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN env var not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_API_ENDPOINT", "") == "", reason="ASTRA_DB_API_ENDPOINT env var not set")
class TestEmbeddingRetrieval:
    @pytest.fixture(scope="class")
    def document_store(self):
        store = AstraDocumentStore(
            collection_name="haystack_test_embedding_retrieval",
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dimension=768,
        )
        try:
            yield store
        finally:
            store.index._astra_db.drop_collection(store.collection_name)

    @pytest.fixture(autouse=True)
    def run_before_tests(self, document_store: AstraDocumentStore):
        """
        Cleaning up document store
        """
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_search_with_top_k(self, document_store):
        query_embedding = [0.1] * 768
        common_embedding = [0.8] * 768

        documents = [Document(content=f"This is document number {i}", embedding=common_embedding) for i in range(0, 3)]

        document_store.write_documents(documents)

        top_k = 2

        result = document_store.search(query_embedding, top_k)

        assert top_k == len(result)

        for document in result:
            assert document.score is not None

        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    async def test_native_async_retrieval(self, document_store):
        documents = [
            Document(id="1", content="included", embedding=[0.12345678901234568] * 768, meta={"category": "news"}),
            Document(id="2", content="excluded", embedding=[0.2] * 768, meta={"category": "other"}),
        ]
        assert document_store.write_documents(documents) == 2
        filters = {"field": "meta.category", "operator": "==", "value": "news"}
        retriever = AstraEmbeddingRetriever(document_store, filters=filters, top_k=1)
        expected = retriever.run(query_embedding=[0.1] * 768)
        assert expected["documents"][0].embedding == documents[0].embedding
        assert [doc.id for doc in expected["documents"]] == ["1"]
        try:
            assert await retriever.run_async(query_embedding=[0.1] * 768) == expected
            assert await retriever.run_async(query_embedding=[0.1] * 768) == expected
            await document_store.close_async()
            assert await retriever.run_async(query_embedding=[0.1] * 768) == expected
            assert sorted(document_store.get_documents_by_id(["1", "2"]), key=lambda doc: doc.id) == documents
        finally:
            await document_store.close_async()
