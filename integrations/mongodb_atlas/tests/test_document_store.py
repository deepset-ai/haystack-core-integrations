# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest
from haystack.dataclasses.document import ByteStream, Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import (
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    DocumentStoreBaseExtendedTests,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldUniqueValuesTest,
)
from haystack.utils import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestMongoDBDocumentStoreInit:
    @pytest.mark.parametrize("client_cls", ["MongoClient", "AsyncMongoClient"])
    def test_init_is_lazy(self, client_cls):
        with patch(f"haystack_integrations.document_stores.mongodb_atlas.document_store.{client_cls}") as mock_client:
            MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("test"),
                database_name="database_name",
                collection_name="collection_name",
                vector_search_index="cosine_index",
                full_text_search_index="full_text_index",
            )
            mock_client.assert_not_called()

    def test_invalid_collection_name_raises(self):
        with pytest.raises(ValueError, match="Invalid collection name"):
            MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("test"),
                database_name="test_db",
                collection_name="bad name!",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

    @pytest.mark.parametrize("attr", ["connection", "collection"])
    def test_property_raises_when_not_setup(self, local_store, attr):
        with pytest.raises(DocumentStoreError, match="not established"):
            getattr(local_store, attr)


class TestEnsureConnectionSetup:
    def test_raises_when_ping_fails(self, local_store):
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_cls:
            mock_cls.return_value.admin.command.side_effect = RuntimeError("nope")
            with pytest.raises(DocumentStoreError, match="Connection to MongoDB Atlas failed"):
                local_store._ensure_connection_setup()

    def test_raises_when_collection_missing(self, local_store):
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_cls:
            client = mock_cls.return_value
            client.admin.command.return_value = {"ok": 1}
            db = MagicMock()
            db.list_collection_names.return_value = ["other_collection"]
            client.__getitem__.return_value = db
            with pytest.raises(DocumentStoreError, match="does not exist"):
                local_store._ensure_connection_setup()


class TestMongoDBDocumentStoreUnit:
    def test_count_documents_by_filter(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.count_documents.return_value = 5

        count = store.count_documents_by_filter({"field": "meta.type", "operator": "==", "value": "article"})

        assert count == 5
        assert collection.count_documents.call_args[0][0] == {"meta.type": {"$eq": "article"}}

    def test_count_unique_metadata_by_filter(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.return_value = [{"category": [{"count": 3}], "author": [{"count": 5}]}]

        counts = store.count_unique_metadata_by_filter(
            {"field": "meta.type", "operator": "==", "value": "article"}, ["category", "author"]
        )

        assert counts == {"category": 3, "author": 5}
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0] == {"$match": {"meta.type": {"$eq": "article"}}}
        assert "category" in pipeline[1]["$facet"]
        assert "author" in pipeline[1]["$facet"]

    def test_get_metadata_fields_info(self, mocked_store_collection):
        store, collection = mocked_store_collection
        cursor = MagicMock()
        collection.find.return_value = cursor
        cursor.sort.return_value = cursor
        cursor.limit.return_value = [
            {"meta": {"category": "A", "number": 1, "ratio": 0.5}},
            {"meta": {"category": "B", "is_valid": True}},
        ]

        fields_info = store.get_metadata_fields_info()

        assert fields_info["content"] == {"type": "text"}
        assert fields_info["category"] == {"type": "keyword"}
        assert fields_info["number"] == {"type": "long"}
        assert fields_info["ratio"] == {"type": "float"}
        assert fields_info["is_valid"] == {"type": "boolean"}

    def test_get_metadata_field_min_max(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.return_value = [{"min": 10, "max": 100}]

        result = store.get_metadata_field_min_max("number")

        assert result == {"min": 10, "max": 100}
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"]["min"] == {"$min": "$meta.number"}

    def test_get_metadata_field_unique_values(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.return_value = [{"count": [{"count": 5}], "values": [{"_id": "val1"}, {"_id": "val2"}]}]

        values, count = store.get_metadata_field_unique_values("category", search_term="val", size=2)

        assert values == ["val1", "val2"]
        assert count == 5
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"] == {"_id": "$meta.category"}
        assert pipeline[1]["$match"] == {"_id": {"$regex": "val", "$options": "i"}}
        assert pipeline[2]["$facet"]["values"][2]["$limit"] == 2


class TestMongoDBDocumentStoreConversion:
    def test_haystack_doc_to_mongo_doc_with_unsupported_fields(self, local_store):
        doc = Document.from_dict(
            {
                "id": "test_id",
                "content": "test content",
                "embedding": [0.1, 0.2, 0.3],
                "sparse_embedding": {"indices": [1, 2, 3], "values": [0.1, 0.2, 0.3]},
            }
        )
        mongo_doc = local_store._haystack_doc_to_mongo_doc(doc)
        assert "sparse_embedding" not in mongo_doc

        doc = Document.from_dict(
            {
                "id": "test_id2",
                "content": "test content",
                "embedding": [0.1, 0.2, 0.3],
                "dataframe": {"some": "dataframe"},
            }
        )
        mongo_doc = local_store._haystack_doc_to_mongo_doc(doc)
        assert "dataframe" not in mongo_doc

    def test_document_conversion_methods_with_custom_field_names(self):
        custom_store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
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
        assert mongo_doc["custom_text"] == "test content"
        assert "content" not in mongo_doc
        assert mongo_doc["custom_vector"] == [0.1, 0.2, 0.3]
        assert "embedding" not in mongo_doc
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
class TestDocumentStore(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    @pytest.fixture
    def document_store(self, real_collection):
        database_name, collection_name, _ = real_collection
        return MongoDBAtlasDocumentStore(
            database_name=database_name,
            collection_name=collection_name,
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            embedding_field="embedding",
        )

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
                    "env_vars": ["MONGO_CONNECTION_STRING"],
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
                        "env_vars": ["MONGO_CONNECTION_STRING"],
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

    def test_get_metadata_field_unique_values(self, document_store: MongoDBAtlasDocumentStore):
        docs = [
            Document(content="Doc 1", meta={"tag": "alpha"}),
            Document(content="Doc 2", meta={"tag": "beta"}),
            Document(content="Doc 3", meta={"tag": "gamma"}),
            Document(content="Doc 4", meta={"tag": "alpha"}),
        ]
        document_store.write_documents(docs)

        values, total_count = document_store.get_metadata_field_unique_values("tag")
        assert total_count == 3
        assert sorted(values) == ["alpha", "beta", "gamma"]

        values_subset, count_subset = document_store.get_metadata_field_unique_values(
            "tag", search_term="b", from_=0, size=10
        )
        assert count_subset == 1
        assert sorted(values_subset) == ["beta"]

        values_page, count_page = document_store.get_metadata_field_unique_values("tag", from_=1, size=1)
        assert count_page == 3
        assert len(values_page) == 1
        assert values_page[0] in ["alpha", "beta", "gamma"]

    def test_custom_content_field(self, real_collection):
        database_name, collection_name, client = real_collection
        custom_store = MongoDBAtlasDocumentStore(
            database_name=database_name,
            collection_name=collection_name,
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            content_field="custom_text",
        )
        assert custom_store.content_field == "custom_text"

        doc = Document(content="test content")
        custom_store.write_documents([doc])

        database_doc = client[database_name][collection_name].find_one({"id": doc.id})
        assert database_doc["custom_text"] == "test content"
        assert "content" not in database_doc

        retrieved_docs = custom_store.filter_documents()
        assert len(retrieved_docs) == 1
        assert retrieved_docs[0].content == "test content"

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
