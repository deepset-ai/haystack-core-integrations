import pytest
from haystack.testing.document_store import (
    CountDocumentsTest,
    DeleteAllTest,
    DeleteByFilterTest,
    DeleteDocumentsTest,
    FilterableDocsFixtureMixin,
    FilterDocumentsTest,
    UpdateByFilterTest,
)

from haystack_integrations.document_stores.faiss import FAISSDocumentStore


class TestFAISSDocumentStore(
    CountDocumentsTest,
    DeleteDocumentsTest,
    FilterDocumentsTest,
    FilterableDocsFixtureMixin,
    UpdateByFilterTest,
    DeleteAllTest,
    DeleteByFilterTest,
):
    @pytest.fixture
    def document_store(self, tmp_path):
        return FAISSDocumentStore(index_path=str(tmp_path / "test_index"))

    def test_write_documents(self, document_store):
        from haystack.dataclasses import Document

        doc = Document(content="test")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1
        assert document_store.filter_documents()[0].id == doc.id

    def test_persistence(self, tmp_path):
        from haystack.dataclasses import Document

        path = tmp_path / "persistent_index"
        ds = FAISSDocumentStore(index_path=str(path))

        doc = Document(content="test persistence")
        ds.write_documents([doc])
        ds.save(path)

        # Load in new instance
        ds_loaded = FAISSDocumentStore(index_path=str(path))
        assert ds_loaded.count_documents() == 1
        assert ds_loaded.filter_documents()[0].content == "test persistence"
