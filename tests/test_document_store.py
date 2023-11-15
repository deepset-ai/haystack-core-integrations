# SPDX-FileCopyrightText: 2023-present John Doe <jd@example.com>
#
# SPDX-License-Identifier: Apache-2.0
import os
import pytest
from haystack.preview import Document
from haystack.preview.testing.document_store import DocumentStoreBaseTests

from src.astra_store.document_store import AstraDocumentStore


@pytest.mark.skip("This is an example Document Store")
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def doc_store(self) -> AstraDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        # from src.astra_store.astra_client import AstraClient
        # astra_id = os.getenv("ASTRA_DB_ID", "8b400d8c-9cd5-436c-9e3a-59c1223fc993")
        # astra_region = os.getenv("ASTRA_DB_REGION", "us-east1")
        # astra_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN",
        #                                     "AstraCS:QtcpGfRxYillIMbEuvfypEJk:76f79aa49c8776c0be0a8ca31b5f8e7976d599db6b72f7723423840916ca7a28")
        # collection_name = os.getenv("COLLECTION_NAME", "superagent_vector_json")
        # keyspace_name = os.getenv("KEYSPACE_NAME", "recommender_demo")
        #
        # client = AstraClient(astra_id=astra_id,
        #                  region=astra_region,
        #                  token=astra_application_token,
        #                  keyspace_name=keyspace_name,
        #                  collection_name=collection_name)
        # return AstraDocumentStore()
        return AstraDocumentStore()

    @pytest.mark.unit
    def test_delete_not_empty_non_existing(self, doc_store: AstraDocumentStore):
        """
        Deleting a non-existing document should not raise with Marqo
        """
        doc = Document(text="test doc")
        doc_store.write_documents([doc])
        doc_store.delete_documents(["non_existing"])
        assert doc_store.get_documents_by_id(ids=[doc.id])[0].id == doc.id
        doc_store.delete_documents([doc.id])

