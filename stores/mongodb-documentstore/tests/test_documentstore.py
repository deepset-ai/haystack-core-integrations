import pytest

from haystack.testing import DocumentStoreBaseTestAbstract

from mongodb_documentstore import MongoDBDocumentStore


class TestMongoDBDocumentStore(DocumentStoreBaseTestAbstract):
    @pytest.fixture
    def ds(self, tmp_path):
        return MongoDBDocumentStore()

    def test_init(self):
        MongoDBDocumentStore()
