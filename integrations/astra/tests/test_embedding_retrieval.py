import os

import pytest
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN env var not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_API_ENDPOINT", "") == "", reason="ASTRA_DB_API_ENDPOINT env var not set")
class TestEmbeddingRetrieval:
    @pytest.fixture(scope="class")
    def document_store(self) -> AstraDocumentStore:
        return AstraDocumentStore(
            collection_name="haystack_integration",
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dimension=768,
        )

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
