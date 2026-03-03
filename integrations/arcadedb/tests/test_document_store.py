# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

ARCADEDB_URL = os.getenv("ARCADEDB_URL", "http://localhost:2480")


def _sample_docs(n: int = 3, dim: int = 4) -> list[Document]:
    docs = []
    for i in range(n):
        docs.append(
            Document(
                content=f"Document number {i}",
                embedding=[float(i)] * dim,
                meta={"category": "test", "priority": i},
            )
        )
    return docs


class TestSerialization:
    def test_to_dict_from_dict(self):
        store = ArcadeDBDocumentStore(
            url="http://localhost:2480",
            database="test_db",
            embedding_dimension=4,
        )
        data = store.to_dict()
        restored = ArcadeDBDocumentStore.from_dict(data)
        assert restored._database == store._database
        assert restored._embedding_dimension == store._embedding_dimension
        assert restored._url == store._url


@pytest.mark.skipif(
    not os.environ.get("ARCADEDB_PASSWORD"),
    reason="Set ARCADEDB_PASSWORD (e.g. via repo secret in CI) to run integration tests.",
)
@pytest.mark.integration
class TestArcadeDBDocumentStore(DocumentStoreBaseTests):
    """
    Run Haystack DocumentStore mixin tests against ArcadeDBDocumentStore.

    Base tests cover: count_documents, delete_documents, filter_documents, write_documents.
    ArcadeDB does not implement delete_all_documents, delete_by_filter, or update_by_filter,
    so DocumentStoreBaseTests (not Extended) is used.
    """

    @pytest.fixture
    def document_store(self, document_store: ArcadeDBDocumentStore) -> ArcadeDBDocumentStore:
        """Override to provide ArcadeDB document store from conftest."""
        yield document_store

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        """
        Compare document lists for tests. Clear score (filter_documents does not set it;
        embedding_retrieval does). Compare embeddings approximately for float round-trip.
        Documents written without embeddings get zero-padded in the store; treat as None for comparison.
        """
        assert len(received) == len(expected)
        received = sorted(received, key=lambda x: x.id)
        expected = sorted(expected, key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
            received_doc.score = None
            if expected_doc.embedding is None:
                received_doc.embedding = None
            elif received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc

    def test_write_documents(self, document_store: ArcadeDBDocumentStore):
        """Override mixin: test default write_documents and duplicate fail behaviour."""
        docs = [Document(id="1")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, policy=DuplicatePolicy.FAIL)

    def test_write_overwrite(self, document_store: ArcadeDBDocumentStore):
        """ArcadeDB-specific: overwrite updates content."""
        docs = _sample_docs(1)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        updated = dataclasses.replace(docs[0], content="Updated content")
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 1
        assert all_docs[0].content == "Updated content"

    def test_embedding_retrieval(self, document_store: ArcadeDBDocumentStore):
        """ArcadeDB-specific: vector search via _embedding_retrieval."""
        # Use store's embedding_dimension (768 from conftest); create small test docs
        dim = document_store._embedding_dimension
        docs = [
            Document(
                content=f"Document number {i}",
                embedding=[float(i)] * dim,
                meta={"category": "test", "priority": i},
            )
            for i in range(5)
        ]
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._embedding_retrieval(
            query_embedding=[4.0] * dim,
            top_k=3,
        )
        assert len(results) <= 3
        assert results[0].score is not None
