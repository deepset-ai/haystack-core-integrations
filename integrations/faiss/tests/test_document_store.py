# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.errors import FilterError
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountDocumentsTest,
    CountUniqueMetadataByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
    UpdateByFilterTest,
)

from haystack_integrations.document_stores.faiss import FAISSDocumentStore


class TestFAISSDocumentStoreUnit:
    def test_create_new_index_with_invalid_factory_raises(self):
        with pytest.raises(DocumentStoreError, match="Could not create FAISS index"):
            FAISSDocumentStore(index_string="NotARealIndex", embedding_dim=3)

    def test_get_index_or_raise_when_index_is_none(self):
        ds = FAISSDocumentStore(embedding_dim=3)
        ds.index = None
        with pytest.raises(DocumentStoreError, match="FAISS index has not been initialized"):
            ds._get_index_or_raise()

    @pytest.mark.parametrize(
        "bad_input",
        [
            "a string",
            b"some bytes",
            123,
            [Document(content="ok"), "not a document"],
        ],
    )
    def test_write_documents_rejects_invalid_input(self, bad_input):
        ds = FAISSDocumentStore(embedding_dim=3)
        with pytest.raises(ValueError, match="iterable of objects of type Document"):
            ds.write_documents(bad_input)

    def test_write_documents_fail_policy_raises_on_duplicate(self):
        ds = FAISSDocumentStore(embedding_dim=3)
        ds.write_documents([Document(id="dup", content="first")])
        with pytest.raises(DuplicateDocumentError, match="already exists"):
            ds.write_documents([Document(id="dup", content="second")], policy=DuplicatePolicy.FAIL)

    @pytest.mark.parametrize(
        ("filters", "error_match"),
        [
            ({"operator": "OR"}, "Missing 'conditions' for OR operator"),
            ({"operator": "AND"}, "Missing 'conditions' for AND operator"),
            ({"operator": "NOT"}, "Missing 'conditions' for NOT operator"),
            ({"operator": "==", "field": 42, "value": "x"}, "'field' in filter condition must be a string"),
        ],
    )
    def test_check_condition_invalid_structure_raises_filter_error(self, filters, error_match):
        ds = FAISSDocumentStore(embedding_dim=3)
        ds.write_documents([Document(content="test", meta={"category": "A"})])
        with pytest.raises(FilterError, match=error_match):
            ds.filter_documents(filters=filters)


@pytest.mark.integration
class TestFAISSDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
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

    def test_not_filter_with_empty_conditions_raises_filter_error(self, document_store):
        document_store.write_documents([Document(content="test", meta={"category": "A"})])
        with pytest.raises(FilterError, match="NOT operator expects at least one condition"):
            document_store.filter_documents(filters={"operator": "NOT", "conditions": []})
