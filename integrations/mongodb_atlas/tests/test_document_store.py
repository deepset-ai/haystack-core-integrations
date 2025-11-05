# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import patch
from uuid import uuid4

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests
from haystack.utils import Secret
from pymongo import MongoClient
from pymongo.driver_info import DriverInfo

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


@patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient")
def test_init_is_lazy(_mock_client):
    MongoDBAtlasDocumentStore(
        mongo_connection_string=Secret.from_token("test"),
        database_name="database_name",
        collection_name="collection_name",
        vector_search_index="cosine_index",
        full_text_search_index="full_text_index",
        embedding_field="embedding",
    )
    _mock_client.assert_not_called()


@patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient")
class TestMongoDBDocumentStoreConversion:
    def test_haystack_doc_to_mongo_doc_with_unsupported_fields(self, _mock_client):
        """Test the document conversion with unsupported fields like sparse_embedding and dataframe."""
        docstore = MongoDBAtlasDocumentStore(
            database_name="haystack_integration_test",
            collection_name="test_collection",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
        )

        doc_dict = {
            "id": "test_id",
            "content": "test content",
            "embedding": [0.1, 0.2, 0.3],
            "sparse_embedding": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
        }
        doc = Document.from_dict(doc_dict)

        mongo_doc = docstore._haystack_doc_to_mongo_doc(doc)

        assert "sparse_embedding" not in mongo_doc

        doc_dict = {
            "id": "test_id2",
            "content": "test content",
            "embedding": [0.1, 0.2, 0.3],
            "dataframe": {"some": "dataframe"},
        }
        doc = Document.from_dict(doc_dict)

        mongo_doc = docstore._haystack_doc_to_mongo_doc(doc)
        assert "dataframe" not in mongo_doc

    def test_document_conversion_methods_with_custom_field_names(self, _mock_client):
        """Test the document conversion helper methods with custom field mappings."""
        custom_store = MongoDBAtlasDocumentStore(
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="test_index",
            full_text_search_index="test_index",
            embedding_field="custom_vector",
            content_field="custom_text",
        )

        haystack_doc = Document(content="test content", embedding=[0.1, 0.2, 0.3], meta={"test_meta": "test_value"})

        mongo_doc = custom_store._haystack_doc_to_mongo_doc(haystack_doc)

        # Check field mapping
        assert "custom_text" in mongo_doc
        assert mongo_doc["custom_text"] == "test content"
        assert "content" not in mongo_doc

        assert "custom_vector" in mongo_doc
        assert mongo_doc["custom_vector"] == [0.1, 0.2, 0.3]
        assert "embedding" not in mongo_doc

        assert "meta" in mongo_doc
        assert mongo_doc["meta"] == {"test_meta": "test_value"}

        # Test mongo_doc_to_haystack_doc
        converted_doc = {
            "id": "test_id",
            "custom_text": "test content from mongo",
            "custom_vector": [0.4, 0.5, 0.6],
            "meta": {"mongo_meta": "mongo_value"},
            "_id": "mongodb_internal_id",  # This should be removed
        }

        haystack_doc = custom_store._mongo_doc_to_haystack_doc(converted_doc)

        assert haystack_doc.content == "test content from mongo"
        assert haystack_doc.embedding == [0.4, 0.5, 0.6]
        assert haystack_doc.meta == {"mongo_meta": "mongo_value"}
        assert haystack_doc.id == "test_id"

        assert not hasattr(haystack_doc, "_id")


@pytest.mark.skipif(
    not os.environ.get("MONGO_CONNECTION_STRING"),
    reason="No MongoDB Atlas connection string provided",
)
@pytest.mark.integration
class TestDocumentStore(DocumentStoreBaseTests):
    @pytest.fixture
    def document_store(self):
        database_name = "haystack_integration_test"
        collection_name = "test_collection_" + str(uuid4())

        connection: MongoClient = MongoClient(
            os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        )
        database = connection[database_name]
        if collection_name in database.list_collection_names():
            database[collection_name].drop()
        database.create_collection(collection_name)
        database[collection_name].create_index("id", unique=True)

        store = MongoDBAtlasDocumentStore(
            database_name=database_name,
            collection_name=collection_name,
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            embedding_field="embedding",
        )
        yield store
        database[collection_name].drop()

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

    def test_to_dict(self, document_store):
        serialized_store = document_store.to_dict()
        assert serialized_store["init_parameters"].pop("collection_name").startswith("test_collection_")
        assert serialized_store == {
            "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
            "init_parameters": {
                "mongo_connection_string": {
                    "env_vars": [
                        "MONGO_CONNECTION_STRING",
                    ],
                    "strict": True,
                    "type": "env_var",
                },
                "database_name": "haystack_integration_test",
                "vector_search_index": "cosine_index",
                "full_text_search_index": "full_text_index",
            },
        }

    def test_from_dict(self):
        docstore = MongoDBAtlasDocumentStore.from_dict(
            {
                "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
                "init_parameters": {
                    "mongo_connection_string": {
                        "env_vars": [
                            "MONGO_CONNECTION_STRING",
                        ],
                        "strict": True,
                        "type": "env_var",
                    },
                    "database_name": "haystack_integration_test",
                    "collection_name": "test_embeddings_collection",
                    "vector_search_index": "cosine_index",
                    "full_text_search_index": "full_text_index",
                    "embedding_field": "custom_embedding",
                },
            }
        )
        assert docstore.mongo_connection_string == Secret.from_env_var("MONGO_CONNECTION_STRING")
        assert docstore.database_name == "haystack_integration_test"
        assert docstore.collection_name == "test_embeddings_collection"
        assert docstore.vector_search_index == "cosine_index"
        assert docstore.full_text_search_index == "full_text_index"
        assert docstore.embedding_field == "custom_embedding"

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": "90"},
                        {"field": "meta.chapter", "operator": "==", "value": "conclusion"},
                    ],
                },
            ],
        }

        result = document_store.filter_documents(filters=filters)

        self.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if (d.meta.get("number") == 100 and d.meta.get("chapter") == "intro")
                or (d.meta.get("page") == "90" and d.meta.get("chapter") == "conclusion")
            ],
        )

    @pytest.mark.integration
    def test_custom_embedding_field(self):
        """Test that the custom embedding field is correctly used in the document store."""
        # Create a document store with a custom embedding field
        database_name = "haystack_integration_test"
        collection_name = "test_custom_embeddings_" + str(uuid4())

        connection: MongoClient = MongoClient(
            os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        )
        database = connection[database_name]
        if collection_name in database.list_collection_names():
            database[collection_name].drop()
        database.create_collection(collection_name)
        database[collection_name].create_index("id", unique=True)

        try:
            custom_field_store = MongoDBAtlasDocumentStore(
                database_name=database_name,
                collection_name=collection_name,
                vector_search_index="cosine_index",
                full_text_search_index="full_text_index",
                embedding_field="custom_vector",
            )

            # Check that the embedding field is correctly set
            assert custom_field_store.embedding_field == "custom_vector"

            # This is a mock test since we can't execute vector search without a real vector index
            with patch.object(custom_field_store, "_collection") as mock_collection:
                # Setup the mock
                mock_collection.aggregate.return_value = []

                # Execute the method
                custom_field_store._embedding_retrieval(query_embedding=[0.1, 0.2, 0.3])

                # Verify that the correct embedding field was used in the pipeline
                args = mock_collection.aggregate.call_args[0][0]
                assert args[0]["$vectorSearch"]["path"] == "custom_vector"

        finally:
            database[collection_name].drop()

    @pytest.mark.integration
    def test_custom_content_field(self):
        """Test that the custom content field is correctly used in the document store."""
        # Create a document store with a custom content field
        database_name = "haystack_integration_test"
        collection_name = "test_custom_content_" + str(uuid4())

        connection: MongoClient = MongoClient(
            os.environ["MONGO_CONNECTION_STRING"], driver=DriverInfo(name="MongoDBAtlasHaystackIntegration")
        )
        database = connection[database_name]
        if collection_name in database.list_collection_names():
            database[collection_name].drop()
        database.create_collection(collection_name)
        database[collection_name].create_index("id", unique=True)

        try:
            custom_field_store = MongoDBAtlasDocumentStore(
                database_name=database_name,
                collection_name=collection_name,
                vector_search_index="cosine_index",
                full_text_search_index="full_text_index",
                content_field="custom_text",
            )

            # Check that the content field is correctly set
            assert custom_field_store.content_field == "custom_text"

            # Write a document with standard content field
            doc = Document(content="test content")
            custom_field_store.write_documents([doc])

            # Verify it's stored with the custom field name in MongoDB
            database_doc = database[collection_name].find_one({"id": doc.id})
            assert "custom_text" in database_doc
            assert database_doc["custom_text"] == "test content"
            assert "content" not in database_doc

            # Retrieve the document and verify it has the standard content field
            retrieved_docs = custom_field_store.filter_documents()
            assert len(retrieved_docs) == 1
            assert retrieved_docs[0].content == "test content"

            # This is a mock test for text search
            with patch.object(custom_field_store, "_collection") as mock_collection:
                # Setup the mock
                mock_collection.aggregate.return_value = []

                # Execute the method
                custom_field_store._fulltext_retrieval(query="test query")

                # Verify that the text search is using the standard content path
                args = mock_collection.aggregate.call_args[0][0]
                assert args[0]["$search"]["compound"]["must"][0]["text"]["path"] == "custom_text"

        finally:
            database[collection_name].drop()

    def test_delete_by_filter(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Delete documents with category="A"
        deleted_count = document_store.delete_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}
        )
        assert deleted_count == 2
        assert document_store.count_documents() == 1

        # Verify the remaining document is the one with category="B"
        remaining_docs = document_store.filter_documents()
        assert len(remaining_docs) == 1
        assert remaining_docs[0].meta["category"] == "B"

    def test_update_by_filter(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"category": "A"}),
            Document(content="Doc 2", meta={"category": "B"}),
            Document(content="Doc 3", meta={"category": "A"}),
        ]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 3

        # Update documents with category="A" to have status="published"
        updated_count = document_store.update_by_filter(
            filters={"field": "meta.category", "operator": "==", "value": "A"}, meta={"status": "published"}
        )
        assert updated_count == 2

        # Verify the updated documents have the new metadata
        published_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "published"}
        )
        assert len(published_docs) == 2
        for doc in published_docs:
            assert doc.meta["status"] == "published"
            assert doc.meta["category"] == "A"

        # Verify documents with category="B" were not updated
        unpublished_docs = document_store.filter_documents(
            filters={"field": "meta.category", "operator": "==", "value": "B"}
        )
        assert len(unpublished_docs) == 1
        assert "status" not in unpublished_docs[0].meta

    def test_delete_all_documents(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="first doc"), Document(id="2", content="second doc")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_delete_all_documents_empty_collection(self, document_store: MongoDBAtlasDocumentStore):
        assert document_store.count_documents() == 0
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_delete_all_documents_with_recreate_collection(self, document_store: MongoDBAtlasDocumentStore):
        docs = [Document(id="1", content="first doc"), Document(id="2", content="second doc")]
        document_store.write_documents(docs)
        assert document_store.count_documents() == 2

        # Delete all documents with collection recreation
        document_store.delete_all_documents(recreate_collection=True)
        assert document_store.count_documents() == 0

        # Verify collection still exists and we can write to it
        new_docs = [Document(id="3", content="third doc")]
        document_store.write_documents(new_docs)
        assert document_store.count_documents() == 1
