# SPDX-FileCopyrightText: 2025-present ArcadeData Ltd <info@arcadedb.com>
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os

import pytest
from haystack import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy

from haystack_integrations.document_stores.arcadedb import ArcadeDBDocumentStore

ARCADEDB_URL = os.getenv("ARCADEDB_URL", "http://localhost:2480")


@pytest.fixture()
def document_store():
    store = ArcadeDBDocumentStore(
        url=ARCADEDB_URL,
        database="haystack_test",
        embedding_dimension=4,
        recreate_type=True,
    )
    return store


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


# ---------------------------------------------------------------------------
# Unit tests (no ArcadeDB required)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Integration tests (require a running ArcadeDB instance)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestArcadeDBDocumentStoreIntegration:
    def test_count_empty(self, document_store):
        assert document_store.count_documents() == 0

    def test_count_after_write(self, document_store):
        docs = _sample_docs(5)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert document_store.count_documents() == 5

    def test_write_and_read(self, document_store):
        docs = _sample_docs(2)
        written = document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)
        assert written == 2

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 2

    def test_write_overwrite(self, document_store):
        docs = _sample_docs(1)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        updated = dataclasses.replace(docs[0], content="Updated content")
        document_store.write_documents([updated], policy=DuplicatePolicy.OVERWRITE)

        all_docs = document_store.filter_documents()
        assert len(all_docs) == 1
        assert all_docs[0].content == "Updated content"

    def test_write_skip(self, document_store):
        docs = _sample_docs(1)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        written = document_store.write_documents(docs, policy=DuplicatePolicy.SKIP)
        assert written == 0
        assert document_store.count_documents() == 1

    def test_write_duplicate_raises(self, document_store):
        docs = _sample_docs(1)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, policy=DuplicatePolicy.NONE)

    def test_delete(self, document_store):
        docs = _sample_docs(3)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        ids_to_delete = [docs[0].id, docs[1].id]
        document_store.delete_documents(ids_to_delete)

        assert document_store.count_documents() == 1

    def test_filter_equality(self, document_store):
        docs = _sample_docs(3)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        result = document_store.filter_documents(filters={"field": "meta.category", "operator": "==", "value": "test"})
        assert len(result) == 3

    def test_filter_comparison(self, document_store):
        docs = _sample_docs(5)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        result = document_store.filter_documents(filters={"field": "meta.priority", "operator": ">", "value": 2})
        assert len(result) == 2  # priority 3 and 4

    def test_filter_and(self, document_store):
        docs = _sample_docs(5)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        result = document_store.filter_documents(
            filters={
                "operator": "AND",
                "conditions": [
                    {"field": "meta.category", "operator": "==", "value": "test"},
                    {"field": "meta.priority", "operator": ">=", "value": 3},
                ],
            }
        )
        assert len(result) == 2

    def test_embedding_retrieval(self, document_store):
        docs = _sample_docs(5, dim=4)
        document_store.write_documents(docs, policy=DuplicatePolicy.OVERWRITE)

        results = document_store._embedding_retrieval(query_embedding=[4.0, 4.0, 4.0, 4.0], top_k=3)
        assert len(results) <= 3
        assert results[0].score is not None
