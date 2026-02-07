# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from haystack.utils.auth import Secret

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore


class TestMongoDBDocumentStoreAdditionsMock:
    def test_count_documents_by_filter_mock(self):
        # Mock the collection and its count_documents method
        mock_collection = MagicMock()
        mock_collection.count_documents.return_value = 5

        # Create store with mocked connection/collection
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            # Mock the database and collection access
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection

            # Mock the database list_collection_names method for _collection_exists check
            mock_db.list_collection_names.return_value = ["test_collection"]

            # Mock admin command for connection check
            mock_client.admin.command.return_value = {"ok": 1}

            store = MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("mongodb://localhost:27017"),
                database_name="test_db",
                collection_name="test_collection",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

            # Call the method
            filter_dict = {"field": "meta.type", "operator": "==", "value": "article"}
            count = store.count_documents_by_filter(filter_dict)

            # Assertions
            assert count == 5
            mock_collection.count_documents.assert_called_once()
            # The filter should be normalized to MongoDB syntax
            call_args = mock_collection.count_documents.call_args
            # normalized filter: {'meta.type': {'$eq': 'article'}}
            assert call_args[0][0] == {"meta.type": {"$eq": "article"}}

    def test_count_unique_metadata_by_filter_mock(self):
        # Mock the collection and its aggregate method
        mock_collection = MagicMock()
        # count_unique_metadata_by_filter uses aggregation
        # It expects a list with one item which is a dict of counts, or empty list
        mock_collection.aggregate.return_value = [{"category": [{"count": 3}], "author": [{"count": 5}]}]

        # Create store with mocked connection/collection
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = ["test_collection"]
            mock_client.admin.command.return_value = {"ok": 1}

            store = MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("mongodb://localhost:27017"),
                database_name="test_db",
                collection_name="test_collection",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

            filter_dict = {"field": "meta.type", "operator": "==", "value": "article"}
            counts = store.count_unique_metadata_by_filter(filter_dict, ["category", "author"])

            assert counts == {"category": 3, "author": 5}

            # Verify aggregation pipeline
            mock_collection.aggregate.assert_called_once()
            pipeline = mock_collection.aggregate.call_args[0][0]
            assert pipeline[0] == {"$match": {"meta.type": {"$eq": "article"}}}
            assert "$facet" in pipeline[1]
            assert "category" in pipeline[1]["$facet"]
            assert "author" in pipeline[1]["$facet"]

    def test_get_metadata_fields_info_mock(self):
        # Mock connection/collection
        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_collection = MagicMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = ["test_collection"]
            mock_client.admin.command.return_value = {"ok": 1}

            # Mock cursor for finding docs
            mock_cursor = MagicMock()
            mock_collection.find.return_value = mock_cursor
            mock_cursor.sort.return_value = mock_cursor
            mock_cursor.limit.return_value = [
                {"meta": {"category": "A", "number": 1, "ratio": 0.5}},
                {"meta": {"category": "B", "is_valid": True}},
            ]

            store = MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("mongodb://localhost:27017"),
                database_name="test_db",
                collection_name="test_collection",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

            fields_info = store.get_metadata_fields_info()

            assert fields_info["content"] == {"type": "text"}
            assert fields_info["category"] == {"type": "keyword"}
            assert fields_info["number"] == {"type": "long"}
            assert fields_info["ratio"] == {"type": "float"}
            assert fields_info["is_valid"] == {"type": "boolean"}

    def test_get_metadata_field_min_max_mock(self):
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = [{"min": 10, "max": 100}]

        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = ["test_collection"]
            mock_client.admin.command.return_value = {"ok": 1}

            store = MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("mongodb://localhost:27017"),
                database_name="test_db",
                collection_name="test_collection",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

            result = store.get_metadata_field_min_max("number")
            assert result == {"min": 10, "max": 100}

            # Verify pipeline
            pipeline = mock_collection.aggregate.call_args[0][0]
            assert pipeline[0]["$group"]["min"] == {"$min": "$meta.number"}

    def test_get_metadata_field_unique_values_mock(self):
        # Mock collection
        mock_collection = MagicMock()
        mock_collection.aggregate.return_value = [
            {"count": [{"count": 5}], "values": [{"_id": "val1"}, {"_id": "val2"}]}
        ]

        with patch("haystack_integrations.document_stores.mongodb_atlas.document_store.MongoClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_db.__getitem__.return_value = mock_collection
            mock_db.list_collection_names.return_value = ["test_collection"]
            mock_client.admin.command.return_value = {"ok": 1}

            store = MongoDBAtlasDocumentStore(
                mongo_connection_string=Secret.from_token("mongodb://localhost:27017"),
                database_name="test_db",
                collection_name="test_collection",
                vector_search_index="idx",
                full_text_search_index="idx",
            )

            values, count = store.get_metadata_field_unique_values("category", search_term="val", size=2)

            assert values == ["val1", "val2"]
            assert count == 5

            # Verify pipeline
            pipeline = mock_collection.aggregate.call_args[0][0]
            assert pipeline[0]["$group"] == {"_id": "$meta.category"}
            assert pipeline[1]["$match"] == {"_id": {"$regex": "val", "$options": "i"}}
            assert pipeline[2]["$facet"]["values"][2]["$limit"] == 2
