from typing import List

import pytest
from haystack import Document
from haystack.dataclasses import SparseEmbedding
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteDocumentsTest,
    WriteDocumentsTest,
    _random_embeddings,
)
from haystack_integrations.document_stores.qdrant.document_store import QdrantDocumentStore, QdrantStoreError


class TestQdrantDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
            use_sparse_embeddings=False,
        )

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test.
        """

        # Check that the lengths of the lists are the same
        assert len(received) == len(expected)

        # Check that the sets are equal, meaning the content and IDs match regardless of order
        assert {doc.id for doc in received} == {doc.id for doc in expected}

    def test_write_documents(self, document_store: QdrantDocumentStore):
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_query_hybrid(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)

        docs = []
        for i in range(20):
            docs.append(
                Document(
                    content=f"doc {i}", sparse_embedding=generate_sparse_embedding(), embedding=_random_embeddings(768)
                )
            )

        document_store.write_documents(docs)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: List[Document] = document_store._query_hybrid(
            query_sparse_embedding=sparse_embedding, query_embedding=embedding, top_k=10, return_embedding=True
        )
        assert len(results) == 10

        for document in results:
            assert document.sparse_embedding
            assert document.embedding

    def test_query_hybrid_fail_without_sparse_embedding(self, document_store):
        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with pytest.raises(QdrantStoreError):

            document_store._query_hybrid(
                query_sparse_embedding=sparse_embedding,
                query_embedding=embedding,
            )
