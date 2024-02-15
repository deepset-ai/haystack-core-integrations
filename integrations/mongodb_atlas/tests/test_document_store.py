# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from haystack.utils import Secret
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import CountDocumentsTest, DeleteDocumentsTest, WriteDocumentsTest
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from pandas import DataFrame


@pytest.fixture
def document_store(request):
    store = MongoDBAtlasDocumentStore(
        database_name="haystack_integration_test",
        collection_name=request.node.name + str(uuid4()),
    )
    yield store
    store.collection.drop()


@pytest.mark.skipif(
    "MONGO_CONNECTION_STRING" not in os.environ,
    reason="No MongoDB Atlas connection string provided",
)
class TestDocumentStore(CountDocumentsTest, WriteDocumentsTest, DeleteDocumentsTest):

    def test_write_documents(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(content="some text")]
        assert document_store.write_documents(docs) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(docs, DuplicatePolicy.FAIL)

    def test_write_blob(self, document_store: MongoDBAtlasDocumentStore):
        bytestream = ByteStream(b"test", meta={"meta_key": "meta_value"}, mime_type="mime_type")
        docs = [Document(blob=bytestream)]
        document_store.write_documents(docs)
        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    def test_write_dataframe(self, document_store: MongoDBAtlasDocumentStore):
        dataframe = DataFrame({"col1": [1, 2], "col2": [3, 4]})
        docs = [Document(dataframe=dataframe)]
        document_store.write_documents(docs)
        retrieved_docs = document_store.filter_documents()
        assert retrieved_docs == docs

    @patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient")
    def test_to_dict(self, _):
        document_store = MongoDBAtlasDocumentStore(
            database_name="database_name",
            collection_name="collection_name",
        )
        assert document_store.to_dict() == {
            "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
            "init_parameters": {
                "mongo_connection_string": {
                    "env_vars": [
                        "MONGO_CONNECTION_STRING",
                    ],
                    "strict": True,
                    "type": "env_var",
                },
                "database_name": "database_name",
                "collection_name": "collection_name",
                "recreate_collection": False,
            },
        }

    @patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient")
    def test_from_dict(self, _):
        docstore = MongoDBAtlasDocumentStore.from_dict({
            "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
            "init_parameters": {
                "mongo_connection_string": {
                    "env_vars": [
                        "MONGO_CONNECTION_STRING",
                    ],
                    "strict": True,
                    "type": "env_var",
                },
                "database_name": "database_name",
                "collection_name": "collection_name",
                "recreate_collection": True,
            }})
        assert docstore.mongo_connection_string == Secret.from_env_var("MONGO_CONNECTION_STRING")
        assert docstore.database_name == "database_name"
        assert docstore.collection_name == "collection_name"
        assert docstore.recreate_collection == True
