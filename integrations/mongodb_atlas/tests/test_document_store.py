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

    def test_to_dict_and_from_dict(self):
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING"),
            database_name="database_name",
            collection_name="collection_name",
            vector_search_index="cosine_index",
            full_text_search_index="full_text_index",
            embedding_field="custom_embedding",
            content_field="custom_content",
            meta_project_mapping={"source": "source_field"},
        )
        serialized = store.to_dict()
        assert serialized == {
            "type": "haystack_integrations.document_stores.mongodb_atlas.document_store.MongoDBAtlasDocumentStore",
            "init_parameters": {
                "mongo_connection_string": {
                    "env_vars": ["MONGO_CONNECTION_STRING"],
                    "strict": True,
                    "type": "env_var",
                },
                "database_name": "database_name",
                "collection_name": "collection_name",
                "vector_search_index": "cosine_index",
                "full_text_search_index": "full_text_index",
                "embedding_field": "custom_embedding",
                "content_field": "custom_content",
                "meta_project_mapping": {"source": "source_field"},
            },
        }

        deserialized = MongoDBAtlasDocumentStore.from_dict(serialized)
        assert deserialized.mongo_connection_string == Secret.from_env_var("MONGO_CONNECTION_STRING")
        assert deserialized.database_name == "database_name"
        assert deserialized.collection_name == "collection_name"
        assert deserialized.vector_search_index == "cosine_index"
        assert deserialized.full_text_search_index == "full_text_index"
        assert deserialized.embedding_field == "custom_embedding"
        assert deserialized.content_field == "custom_content"
        assert deserialized.meta_project_mapping == {"source": "source_field"}

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
        assert pipeline[1]["$match"] == {"_id": {"$ne": None}}
        assert pipeline[2]["$match"] == {"_id": {"$regex": "val", "$options": "i"}}
        assert pipeline[3]["$facet"]["values"][2]["$limit"] == 2

    def test_close(self, local_store):
        connection = MagicMock()
        local_store._connection = connection
        local_store.close()
        connection.close.assert_called_once()
        assert local_store._connection is None
        local_store.close()
        connection.close.assert_called_once()

    def test_close_is_exception_safe(self, local_store):
        connection = MagicMock()
        connection.close.side_effect = RuntimeError("boom")
        local_store._connection = connection
        local_store.close()
        assert local_store._connection is None

    def test_get_metadata_field_unique_values_with_meta_prefix(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.return_value = [{"count": [{"count": 2}], "values": [{"_id": "val1"}, {"_id": "val2"}]}]

        values, count = store.get_metadata_field_unique_values("meta.category")

        assert values == ["val1", "val2"]
        assert count == 2
        pipeline = collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"] == {"_id": "$meta.category"}

    def test_get_metadata_field_unique_values_preserves_non_string_types(self, mocked_store_collection):
        store, collection = mocked_store_collection
        collection.aggregate.return_value = [{"count": [{"count": 2}], "values": [{"_id": 1}, {"_id": 2}]}]

        values, count = store.get_metadata_field_unique_values("priority")

        assert values == [1, 2]
        assert count == 2


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

    def test_document_conversion_methods_with_meta_project_mapping(self):
        custom_store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="test_index",
            full_text_search_index="test_index",
            meta_project_mapping={
                "source": "source",
                "author": "metadata.author",
            },
        )

        # 1. Haystack -> MongoDB conversion
        haystack_doc = Document(
            content="test content",
            meta={"source": "url", "author": "john", "other_meta": "keep_me"},
        )
        mongo_doc = custom_store._haystack_doc_to_mongo_doc(haystack_doc)

        assert mongo_doc["source"] == "url"
        assert mongo_doc["metadata"]["author"] == "john"
        assert mongo_doc["meta"] == {"other_meta": "keep_me"}
        assert "content" in mongo_doc
        assert mongo_doc["embedding"] is None

        # Test when all meta is mapped (meta dict should be removed)
        haystack_doc_all_mapped = Document(
            content="test content",
            meta={"source": "url", "author": "john"},
        )
        mongo_doc_all_mapped = custom_store._haystack_doc_to_mongo_doc(haystack_doc_all_mapped)
        assert "meta" not in mongo_doc_all_mapped
        assert mongo_doc_all_mapped["source"] == "url"
        assert mongo_doc_all_mapped["metadata"]["author"] == "john"

        # 2. MongoDB -> Haystack conversion
        converted_doc = {
            "id": "test_id",
            "content": "test content from mongo",
            "source": "url2",
            "metadata": {
                "author": "jane",
            },
            "meta": {"other_meta": "keep_me_too"},
            "_id": "mongodb_internal_id",
        }
        haystack_doc = custom_store._mongo_doc_to_haystack_doc(converted_doc)

        assert haystack_doc.content == "test content from mongo"
        assert haystack_doc.meta == {
            "other_meta": "keep_me_too",
            "source": "url2",
            "author": "jane",
        }
        assert haystack_doc.id == "test_id"
        assert not hasattr(haystack_doc, "_id")


class TestMongoDBDocumentStoreHelpers:
    def test_get_nested_value(self):
        doc = {"a": {"b": {"c": 123}}, "x": "y"}
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "a.b.c") == 123
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "x") == "y"
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "a.b.d") is None
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "non_existent") is None

    def test_set_nested_value(self):
        doc: dict = {}
        MongoDBAtlasDocumentStore._set_nested_value(doc, "a.b.c", 123)
        assert doc == {"a": {"b": {"c": 123}}}

        MongoDBAtlasDocumentStore._set_nested_value(doc, "x", "y")
        assert doc == {"a": {"b": {"c": 123}}, "x": "y"}

        # overwrite existing value or modify intermediate type
        MongoDBAtlasDocumentStore._set_nested_value(doc, "a.b", "new_val")
        assert doc == {"a": {"b": "new_val"}, "x": "y"}

    def test_pop_nested_value(self):
        doc = {"a": {"b": {"c": 123, "d": 456}}, "x": "y"}

        # Pop sibling leaves
        val = MongoDBAtlasDocumentStore._pop_nested_value(doc, "a.b.c")
        assert val == 123
        assert doc == {"a": {"b": {"d": 456}}, "x": "y"}

        # Pop remaining sibling, which should recursively delete parent dicts
        val = MongoDBAtlasDocumentStore._pop_nested_value(doc, "a.b.d")
        assert val == 456
        assert doc == {"x": "y"}

        # Pop root field
        val = MongoDBAtlasDocumentStore._pop_nested_value(doc, "x")
        assert val == "y"
        assert doc == {}

    def test_translate_filters(self):
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="test_index",
            full_text_search_index="test_index",
            meta_project_mapping={
                "source": "source",
                "author": "metadata.author",
            },
        )

        # Simple condition
        filters = {"field": "meta.source", "operator": "==", "value": "url"}
        translated = store._translate_filters(filters)
        assert translated == {"field": "source", "operator": "==", "value": "url"}

        # With leading $ in mapping values - normalization happens at __init__ so the
        # store with "$source" / "$metadata.author" behaves identically.
        store_dollar = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="test_index",
            full_text_search_index="test_index",
            meta_project_mapping={"source": "$source", "author": "$metadata.author"},
        )
        translated = store_dollar._translate_filters(filters)
        assert translated == {"field": "source", "operator": "==", "value": "url"}

        # Nested condition
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.source", "operator": "==", "value": "url"},
                {"field": "meta.author", "operator": "==", "value": "john"},
                {"field": "meta.unmapped", "operator": "==", "value": "val"},
            ],
        }
        translated = store._translate_filters(filters)
        assert translated == {
            "operator": "AND",
            "conditions": [
                {"field": "source", "operator": "==", "value": "url"},
                {"field": "metadata.author", "operator": "==", "value": "john"},
                {"field": "meta.unmapped", "operator": "==", "value": "val"},
            ],
        }

        # None / empty filters
        assert store._translate_filters(None) is None
        assert store._translate_filters({}) == {}

    def test_metadata_methods_and_filters_with_mapping(self):
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="test_db",
            collection_name="test_collection",
            vector_search_index="test_index",
            full_text_search_index="test_index",
            meta_project_mapping={
                "source": "source",
                "author": "metadata.author",
            },
        )
        # Mock connection and collection
        mock_collection = MagicMock()
        store._collection = mock_collection
        # Mock _ensure_connection_setup to do nothing
        store._ensure_connection_setup = lambda: None

        # 1. count_documents_by_filter
        filters = {"field": "meta.source", "operator": "==", "value": "url"}
        store.count_documents_by_filter(filters)
        mock_collection.count_documents.assert_called_with({"source": {"$eq": "url"}})

        # 2. delete_by_filter
        store.delete_by_filter(filters)
        mock_collection.delete_many.assert_called_with(filter={"source": {"$eq": "url"}})

        # 3. update_by_filter
        meta_to_update = {"source": "new_url", "author": "john", "unmapped": "val"}
        store.update_by_filter(filters, meta_to_update)
        mock_collection.update_many.assert_called_with(
            filter={"source": {"$eq": "url"}},
            update={"$set": {"source": "new_url", "metadata.author": "john", "meta.unmapped": "val"}},
        )

        # 4. get_metadata_field_min_max
        mock_collection.aggregate.return_value = [{"min": 1, "max": 10}]
        store.get_metadata_field_min_max("author")
        # should construct group stage on $metadata.author
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"]["min"] == {"$min": "$metadata.author"}

        # 5. get_metadata_field_unique_values
        mock_collection.aggregate.return_value = [{"count": [{"count": 0}], "values": []}]
        store.get_metadata_field_unique_values("author")
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0]["$group"] == {"_id": "$metadata.author"}

        # 6. count_unique_metadata_by_filter
        mock_collection.aggregate.return_value = [{"source": [{"count": 1}]}]
        store.count_unique_metadata_by_filter(filters, ["source"])
        pipeline = mock_collection.aggregate.call_args[0][0]
        assert pipeline[0] == {"$match": {"source": {"$eq": "url"}}}
        assert pipeline[1]["$facet"]["source"][0]["$group"] == {"_id": "$source"}

        # 7. get_metadata_fields_info
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = [
            {"source": "url", "metadata": {"author": "john"}, "meta": {"unmapped": "val"}}
        ]
        mock_collection.find.return_value = mock_cursor

        info = store.get_metadata_fields_info()
        # Verify find was called with expected projection
        mock_collection.find.assert_called_with({}, {"meta": 1, "source": 1, "metadata.author": 1})
        # Verify computed info contains mapped fields
        assert info["source"] == {"type": "keyword"}
        assert info["author"] == {"type": "keyword"}
        assert info["unmapped"] == {"type": "keyword"}


class TestNestedHelperStaticMethods:
    """Requirement 1 - helpers are static and callable without an instance."""

    def test_helpers_do_not_strip_dollar(self):
        """Helpers receive clean bare paths; they do not strip a leading '$' from path strings."""
        doc = {"a": {"b": 10}, "field": "plain_val"}
        # Clean paths work as expected
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "field") == "plain_val"
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "a.b") == 10
        # A path starting with '$' is NOT stripped; it is treated as a literal key lookup,
        # so a non-existent key returns None.
        assert MongoDBAtlasDocumentStore._get_nested_value(doc, "$field") is None


class TestMetaProjectMappingNormalization:
    """Requirement 2 - meta_project_mapping values are normalized once at init."""

    def _make_store(self, mapping):
        return MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="db",
            collection_name="col",
            vector_search_index="idx",
            full_text_search_index="idx",
            meta_project_mapping=mapping,
        )

    def test_bare_field_path_unchanged(self):
        store = self._make_store({"source": "source_field", "author": "meta.author"})
        assert store.meta_project_mapping == {"source": "source_field", "author": "meta.author"}

    def test_dollar_prefixed_values_are_stripped(self):
        store = self._make_store({"source": "$source_field", "author": "$meta.author"})
        assert store.meta_project_mapping == {"source": "source_field", "author": "meta.author"}

    def test_dollar_and_bare_produce_identical_mapping(self):
        bare_store = self._make_store({"source": "source_field", "author": "nested.author"})
        dollar_store = self._make_store({"source": "$source_field", "author": "$nested.author"})
        assert bare_store.meta_project_mapping == dollar_store.meta_project_mapping

    def test_none_mapping_stays_none(self):
        store = self._make_store(None)
        assert store.meta_project_mapping is None

    def test_to_dict_serializes_normalized_values(self):
        """Serialized mapping should contain the already-normalized (bare) values."""
        store = MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_env_var("MONGO_CONNECTION_STRING"),
            database_name="db",
            collection_name="col",
            vector_search_index="idx",
            full_text_search_index="idx",
            meta_project_mapping={"source": "$source_field"},
        )
        d = store.to_dict()
        assert d["init_parameters"]["meta_project_mapping"] == {"source": "source_field"}


class TestMongoDocToHaystackDocReconstructionFix:
    """Requirement 3 - unmapped root-level MongoDB fields do not cause ValueError."""

    def _make_store(self, mapping):
        return MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="db",
            collection_name="col",
            vector_search_index="idx",
            full_text_search_index="idx",
            meta_project_mapping=mapping,
        )

    def test_mapped_field_plus_unmapped_root_field_does_not_raise(self):
        """Mapped meta + leftover root-level field must not raise ValueError."""
        store = self._make_store({"source": "source_field"})
        mongo_doc = {
            "id": "doc-1",
            "content": "hello",
            "source_field": "url",  # consumed by mapping → goes into meta
            "stale_root_key": "ignored",  # NOT a Document field; should be dropped
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.id == "doc-1"
        assert doc.content == "hello"
        assert doc.meta["source"] == "url"

    def test_no_mapped_value_populated_preserves_unmapped_root_fields(self):
        """When the mapped field is absent, meta stays empty.
        Unmapped root-level fields are discarded regardless of whether mapped fields were found.
        This mongo_doc has no extra root-level fields so meta remains empty."""
        store = self._make_store({"source": "source_field"})
        mongo_doc = {
            "id": "doc-2",
            "content": "world",
            # source_field is absent - mapping produces nothing
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.id == "doc-2"
        assert doc.content == "world"
        assert doc.meta == {}

    def test_nested_mapped_field_still_works(self):
        """Nested path mapping (e.g. 'meta.author') continues to work correctly."""
        store = self._make_store({"author": "nested.author"})
        mongo_doc = {
            "id": "doc-3",
            "content": "text",
            "nested": {"author": "Alice"},
            "extra_root": "drop_me",
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.meta["author"] == "Alice"

    def test_legitimate_document_fields_preserved(self):
        """id, content, embedding, score, blob, meta, sparse_embedding are all kept."""
        store = self._make_store({"tag": "tag_field"})
        mongo_doc = {
            "id": "doc-4",
            "content": "c",
            "embedding": [0.1, 0.2],
            "score": 0.99,
            "meta": {"existing": "val"},
            "tag_field": "sports",
            "junk_field": "remove_me",
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.id == "doc-4"
        assert doc.content == "c"
        assert doc.embedding == [0.1, 0.2]
        assert doc.score == 0.99
        assert doc.meta["existing"] == "val"
        assert doc.meta["tag"] == "sports"


class TestMongoDocToHaystackDocMaintainerFixes:
    """Regression tests for the three _mongo_doc_to_haystack_doc issues raised in maintainer review."""

    def _make_store(self, mapping):
        return MongoDBAtlasDocumentStore(
            mongo_connection_string=Secret.from_token("test"),
            database_name="db",
            collection_name="col",
            vector_search_index="idx",
            full_text_search_index="idx",
            meta_project_mapping=mapping,
        )

    # ── 2a) Unmapped root-level fields must be discarded when mapping is configured ──

    def test_unmapped_root_field_discarded_when_mapped_field_absent(self):
        """mapping configured + mapped field absent + extra root-level field → extra field NOT in meta."""
        store = self._make_store({"source": "source_field"})
        mongo_doc = {
            "id": "doc-x",
            "content": "hello",
            # source_field is absent, so nothing is mapped into meta.
            "category": "sports",  # unmapped root-level field - must be discarded
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.id == "doc-x"
        assert doc.content == "hello"
        # Unmapped root-level field must NOT appear in meta
        assert "category" not in doc.meta
        # Absent mapped field must NOT create a meta entry either
        assert "source" not in doc.meta
        assert doc.meta == {}

    def test_unmapped_root_field_discarded_even_when_mapped_field_present(self):
        """When a mapped field IS present, other root-level fields are still discarded."""
        store = self._make_store({"source": "source_field"})
        mongo_doc = {
            "id": "doc-y",
            "content": "hello",
            "source_field": "http://example.com",  # mapped
            "category": "sports",  # unmapped root-level field - must be discarded
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.meta["source"] == "http://example.com"
        assert "category" not in doc.meta

    # ── 2b) Mapping from a path inside 'meta' works without losing the value ──

    def test_mapping_from_meta_path_preserves_value(self):
        """meta_project_mapping={"author": "meta.author"} must produce meta["author"] correctly."""
        store = self._make_store({"author": "meta.author"})
        mongo_doc = {
            "id": "1",
            "content": "c",
            "meta": {"author": "jane"},
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.meta == {"author": "jane"}

    def test_mapping_from_meta_path_with_sibling_meta_keys(self):
        """When mapping pulls from meta.X but other meta keys exist, they are preserved."""
        store = self._make_store({"author": "meta.author"})
        mongo_doc = {
            "id": "2",
            "content": "c",
            "meta": {"author": "jane", "title": "My Doc"},
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.meta["author"] == "jane"
        assert doc.meta["title"] == "My Doc"

    # ── 2c) Explicitly mapped None values must be preserved ──

    def test_mapped_none_value_preserved(self):
        """A field that exists in the document with value None must appear in meta as None."""
        store = self._make_store({"score": "score_field"})
        mongo_doc = {
            "id": "3",
            "content": "c",
            "score_field": None,  # explicit None - must be preserved in meta
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert "score" in doc.meta
        assert doc.meta["score"] is None

    def test_absent_mapped_field_does_not_create_meta_entry(self):
        """A field that does not exist in the document must NOT create a meta key."""
        store = self._make_store({"score": "score_field"})
        mongo_doc = {
            "id": "4",
            "content": "c",
            # score_field is absent
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert "score" not in doc.meta

    # ── 2d) Existing metadata is preserved alongside mapped metadata ──

    def test_existing_meta_preserved_alongside_mapped_meta(self):
        """Existing meta dict contents and mapped values must all be present in the result."""
        store = self._make_store({"source": "source_field"})
        mongo_doc = {
            "id": "5",
            "content": "c",
            "source_field": "http://example.com",
            "meta": {"author": "bob", "tags": ["a", "b"]},
        }
        doc = store._mongo_doc_to_haystack_doc(mongo_doc)
        assert doc.meta["source"] == "http://example.com"
        assert doc.meta["author"] == "bob"
        assert doc.meta["tags"] == ["a", "b"]


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

    def test_get_metadata_field_unique_values_distinct_types(self, document_store: MongoDBAtlasDocumentStore):
        """
        Override: the base mixin test stores int, float, str and bool under the *same* metadata field
        name and expects all four back as distinct values. MongoDB's aggregation `$group` compares
        numeric values across BSON subtypes (int vs double), so an int and a numerically equal float
        (e.g. 1 and 1.0) collapse into a single group regardless of which other values share that field.

        This adapts the same intent - int, float, str and bool must come back as distinct, unmangled
        types via get_metadata_field_unique_values() - using one field per type instead of one shared
        field, which is what MongoDB can actually support.

        The float value is a non-whole number (1.5, not 1.0): a whole-number float would still collapse
        with an int under MongoDB's numeric comparison even in its own field, so a fractional value is
        used to sidestep that ambiguity entirely.
        """
        docs = [
            Document(content="Doc 1", meta={"priority_int": 1}),
            Document(content="Doc 2", meta={"priority_str": "1"}),
            Document(content="Doc 3", meta={"priority_float": 1.5}),
            Document(content="Doc 4", meta={"priority_bool": True}),
        ]
        document_store.write_documents(docs)

        int_values, int_count = document_store.get_metadata_field_unique_values(metadata_field="priority_int")
        str_values, str_count = document_store.get_metadata_field_unique_values(metadata_field="priority_str")
        float_values, float_count = document_store.get_metadata_field_unique_values(metadata_field="priority_float")
        bool_values, bool_count = document_store.get_metadata_field_unique_values(metadata_field="priority_bool")

        assert (int_count, str_count, float_count, bool_count) == (1, 1, 1, 1)
        assert int_values == [1] and type(int_values[0]) is int
        assert str_values == ["1"] and type(str_values[0]) is str
        assert float_values == [1.5] and type(float_values[0]) is float
        assert bool_values == [True] and type(bool_values[0]) is bool

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
                "embedding_field": "embedding",
                "content_field": "content",
                "meta_project_mapping": None,
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

    def test_close_and_reopen(self, document_store: MongoDBAtlasDocumentStore):
        document_store.count_documents()
        assert document_store._connection is not None
        document_store.close()
        assert document_store._connection is None
        assert document_store.count_documents() == 0
