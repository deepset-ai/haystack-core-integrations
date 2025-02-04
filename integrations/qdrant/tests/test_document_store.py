from typing import List
from unittest.mock import patch

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
from qdrant_client.http import models as rest

from haystack_integrations.document_stores.qdrant.document_store import (
    SPARSE_VECTORS_NAME,
    QdrantDocumentStore,
    QdrantStoreError,
)


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

    def test_init_is_lazy(self):
        with patch("haystack_integrations.document_stores.qdrant.document_store.qdrant_client") as mocked_qdrant:
            QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)
            mocked_qdrant.assert_not_called()

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

    def test_sparse_configuration(self):
        document_store = QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            use_sparse_embeddings=True,
            sparse_idf=True,
        )

        client = document_store.client
        sparse_config = client.get_collection("Document").config.params.sparse_vectors

        assert SPARSE_VECTORS_NAME in sparse_config

        # check that the `sparse_idf` parameter takes effect
        assert hasattr(sparse_config[SPARSE_VECTORS_NAME], "modifier")
        assert sparse_config[SPARSE_VECTORS_NAME].modifier == rest.Modifier.IDF

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

    def test_query_hybrid_with_group_by(self, generate_sparse_embedding):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)

        docs = []
        for i in range(20):
            docs.append(
                Document(
                    content=f"doc {i}",
                    sparse_embedding=generate_sparse_embedding(),
                    embedding=_random_embeddings(768),
                    meta={"group_field": i // 2},
                )
            )

        document_store.write_documents(docs)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        results: List[Document] = document_store._query_hybrid(
            query_sparse_embedding=sparse_embedding,
            query_embedding=embedding,
            top_k=3,
            return_embedding=True,
            group_by="meta.group_field",
            group_size=2,
        )
        assert len(results) == 6

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

    def test_query_hybrid_search_batch_failure(self):
        document_store = QdrantDocumentStore(location=":memory:", use_sparse_embeddings=True)

        sparse_embedding = SparseEmbedding(indices=[0, 1, 2, 3], values=[0.1, 0.8, 0.05, 0.33])
        embedding = [0.1] * 768

        with patch.object(document_store.client, "query_points", side_effect=Exception("query_points")):

            with pytest.raises(QdrantStoreError):
                document_store._query_hybrid(query_sparse_embedding=sparse_embedding, query_embedding=embedding)

    def test_docx_metadata(self, document_store):
        from haystack.components.converters.docx import DOCXMetadata

        docx_metadata = DOCXMetadata(
            author="an author",
            category="a category",
            comments="some comments",
            content_status="a status",
            created="2025-01-29T12:00:00Z",
            identifier="an identifier",
            keywords="some keywords",
            language="en",
            last_modified_by="a last modified by",
            last_printed="2025-01-29T12:00:00Z",
            modified="2025-01-29T12:00:00Z",
            revision="a revision",
            subject="a subject",
            title="a title",
            version="a version",
        )

        doc = Document(
            id="mydocwithdocxmetadata",
            content="A Foo Document",
            meta={"page": "100", "chapter": "intro", "docx": docx_metadata},
        )
        document_store.write_documents([doc])

        retrieved_docs = document_store.filter_documents(
            filters={"field": "id", "operator": "==", "value": "mydocwithdocxmetadata"}
        )
        assert len(retrieved_docs) == 1
        retrieved_docs[0].score = None
        retrieved_docs[0].embedding = None
        assert retrieved_docs[0] == doc
