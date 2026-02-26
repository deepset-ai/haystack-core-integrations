# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    UpdateByFilterTest,
)

from haystack_integrations.document_stores.faiss import FAISSDocumentStore


class TestFAISSDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
):
    @pytest.fixture
    def document_store(self, tmp_path):
        return FAISSDocumentStore(index_path=str(tmp_path / "test_index"))

    def test_write_documents(self, document_store):

        doc = Document(content="test")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1
        assert document_store.filter_documents()[0].id == doc.id

    def test_persistence(self, tmp_path):

        path = tmp_path / "persistent_index"
        ds = FAISSDocumentStore(index_path=str(path), embedding_dim=3)

        doc = Document(content="test persistence", embedding=[0.1, 0.2, 0.3])
        ds.write_documents([doc])
        ds.save(path)

        # Load in new instance
        ds_loaded = FAISSDocumentStore(index_path=str(path), embedding_dim=3)
        assert ds_loaded.count_documents() == 1
        assert ds_loaded.filter_documents()[0].content == "test persistence"
        assert ds_loaded.filter_documents()[0].embedding == [0.1, 0.2, 0.3]

    def test_persistence_no_embeddings(self, tmp_path):

        path = tmp_path / "persistent_index_no_embed"
        ds = FAISSDocumentStore(index_path=str(path), embedding_dim=3)

        doc = Document(content="test no embedding")
        ds.write_documents([doc])
        ds.save(path)

        # Load in new instance
        ds_loaded = FAISSDocumentStore(index_path=str(path), embedding_dim=3)
        assert ds_loaded.count_documents() == 1
        assert ds_loaded.filter_documents()[0].content == "test no embedding"
        assert ds_loaded.filter_documents()[0].embedding is None

    def test_load_missing_files(self, tmp_path):
        path = tmp_path / "missing_index"
        ds = FAISSDocumentStore(index_path=str(path), embedding_dim=3)
        with pytest.raises(ValueError, match="File not found"):
            ds.load(path)

    def test_search_with_and_without_filters(self, document_store):

        # Setup documents with missing/varied embeddings to test edge cases
        doc1 = Document(content="test1", embedding=[0.1, 0.2, 0.3], meta={"category": "A"})
        doc2 = Document(content="test2", embedding=[0.4, 0.5, 0.6], meta={"category": "B"})
        doc3 = Document(content="test3", meta={"category": "A"})  # No embedding

        # document_store from fixture uses default embedding_dim=768, so we must recreate
        ds = FAISSDocumentStore(index_path=document_store.index_path, embedding_dim=3)
        ds.write_documents([doc1, doc2, doc3])

        # Test search based on query embedding
        results = ds.search(query_embedding=[0.1, 0.2, 0.3], top_k=2)
        assert len(results) == 2
        assert results[0].content == "test1"  # Closest match

        # Test search with filter
        results_filtered = ds.search(
            query_embedding=[0.1, 0.2, 0.3], top_k=2, filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert len(results_filtered) == 1
        assert results_filtered[0].content == "test2"

    def test_to_dict_from_dict(self):
        ds = FAISSDocumentStore(index_path="test_index", index_string="Flat", embedding_dim=128)

        data = ds.to_dict()
        assert data["type"] == "haystack_integrations.document_stores.faiss.document_store.FAISSDocumentStore"
        assert data["init_parameters"]["index_path"] == "test_index"
        assert data["init_parameters"]["index_string"] == "Flat"
        assert data["init_parameters"]["embedding_dim"] == 128

        ds_loaded = FAISSDocumentStore.from_dict(data)
        assert ds_loaded.index_path == "test_index"
        assert ds_loaded.index_string == "Flat"
        assert ds_loaded.embedding_dim == 128

    def test_count_documents_by_filter(self, document_store):

        docs = [
            Document(content="test1", meta={"category": "A"}),
            Document(content="test2", meta={"category": "B"}),
            Document(content="test3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)

        count = document_store.count_documents_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert count == 2

    def test_get_metadata_fields_info(self, document_store):

        docs = [Document(content="test1", meta={"category": "A", "count": 1, "is_active": True})]
        document_store.write_documents(docs)

        info = document_store.get_metadata_fields_info()
        assert "category" in info
        assert info["category"]["type"] == "keyword"
        assert "count" in info
        assert info["count"]["type"] == "long"
        assert "is_active" in info
        assert info["is_active"]["type"] == "boolean"

    def test_count_unique_metadata_by_filter(self, document_store):

        docs = [
            Document(content="test1", meta={"category": "A", "status": "active"}),
            Document(content="test2", meta={"category": "B", "status": "inactive"}),
            Document(content="test3", meta={"category": "A", "status": "active"}),
        ]
        document_store.write_documents(docs)

        counts = document_store.count_unique_metadata_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, fields=["meta.status"]
        )
        assert "meta.status" in counts
        assert counts["meta.status"] == 1  # Only "active" status for category A

    def test_not_filter_with_empty_conditions_raises_filter_error(self, document_store):
        document_store.write_documents([Document(content="test", meta={"category": "A"})])

        with pytest.raises(FilterError, match="NOT operator expects at least one condition"):
            document_store.filter_documents(filters={"operator": "NOT", "conditions": []})
