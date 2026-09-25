# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import operator
import os
from copy import deepcopy
from unittest import mock

import pytest
from astrapy import Collection, DataAPIClient
from astrapy.exceptions import DataAPIResponseException
from astrapy.info import CollectionDefinition, CollectionDescriptor
from haystack import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError, MissingDocumentError
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

from haystack_integrations.document_stores.astra import AstraDocumentStore
from haystack_integrations.document_stores.astra.errors import AstraDocumentStoreFilterError

CLIENT_PATH = "haystack_integrations.document_stores.astra.astra_client.DataAPIClient"


@pytest.fixture
def mocked_store(mock_auth):  # noqa: ARG001
    """Returns (store, collection) with the astrapy collection mocked out."""
    collection = mock.MagicMock(spec=Collection)
    with mock.patch.object(AstraDocumentStore, "_get_collection", return_value=collection):
        yield AstraDocumentStore(), collection


@mock.patch(CLIENT_PATH)
def test_init_is_lazy(_mock_client, mock_auth):  # noqa
    _ = AstraDocumentStore()
    _mock_client.assert_not_called()


def test_to_dict(mock_auth):  # noqa
    with mock.patch(CLIENT_PATH):
        ds = AstraDocumentStore()
        result = ds.to_dict()
        assert result["type"] == "haystack_integrations.document_stores.astra.document_store.AstraDocumentStore"
        assert set(result["init_parameters"]) == {
            "api_endpoint",
            "token",
            "collection_name",
            "embedding_dimension",
            "duplicates_policy",
            "similarity",
            "namespace",
        }


@pytest.mark.parametrize("policy", [DuplicatePolicy.SKIP, DuplicatePolicy.OVERWRITE, DuplicatePolicy.FAIL])
def test_configuration_round_trip(mock_auth, policy):  # noqa: ARG001
    store = AstraDocumentStore(
        collection_name="custom_collection",
        embedding_dimension=4,
        duplicates_policy=policy,
        similarity="dot_product",
        namespace="custom_keyspace",
    )
    serialized = store.to_dict()
    restored = AstraDocumentStore.from_dict(deepcopy(serialized))
    assert restored.to_dict() == serialized
    assert restored.duplicates_policy is policy
    assert restored.api_endpoint.resolve_value() == "http://example.com"
    assert restored.token.resolve_value() == "test_token"


def test_from_dict_invalid_duplicates_policy(mock_auth):  # noqa: ARG001
    serialized = AstraDocumentStore().to_dict()
    serialized["init_parameters"]["duplicates_policy"] = "INVALID"
    with pytest.raises(ValueError, match=r"Invalid duplicates_policy 'INVALID'\. Expected one of"):
        AstraDocumentStore.from_dict(serialized)


@pytest.fixture
def native_sync_store(mock_auth):  # noqa: ARG001
    with mock.patch(CLIENT_PATH, autospec=DataAPIClient) as client:
        database = client.return_value.get_database.return_value
        database.list_collections.return_value = []
        collection = mock.MagicMock(spec=Collection)
        database.create_collection.return_value = collection
        store = AstraDocumentStore(
            collection_name="custom", embedding_dimension=4, similarity="dot_product", namespace="keyspace"
        )
        yield store, client, database, collection


def test_native_sync_configuration(native_sync_store):
    store, client, database, collection = native_sync_store
    assert store._get_collection() is collection
    serdes = client.call_args.kwargs["api_options"].serdes_options
    assert serdes.binary_encode_vectors is False
    assert serdes.custom_datatypes_in_reading is False
    client.return_value.get_database.assert_called_once_with(
        api_endpoint="http://example.com", token="test_token", keyspace="keyspace"
    )
    database.create_collection.assert_called_once_with(
        name="custom",
        definition={
            "vector": {"dimension": 4, "metric": "dot_product"},
            "indexing": {"deny": ["metadata._node_content", "content"]},
        },
    )


@pytest.mark.parametrize(
    "indexing,warning_match",
    [(None, "having indexing turned on"), ({"deny": ["something_else"]}, "unexpected 'indexing' settings")],
)
def test_existing_collection_with_unexpected_indexing_warns(native_sync_store, indexing, warning_match):
    store, _, database, _ = native_sync_store
    database.list_collections.return_value = [
        CollectionDescriptor(name="custom", definition=CollectionDefinition(indexing=indexing), raw_descriptor={})
    ]
    with pytest.warns(UserWarning, match=warning_match):
        assert store._get_collection() is database.get_collection.return_value
    database.get_collection.assert_called_once_with("custom")
    database.create_collection.assert_not_called()


def test_existing_collection_with_expected_indexing_is_reused_silently(native_sync_store, recwarn):
    store, _, database, _ = native_sync_store
    database.list_collections.return_value = [
        CollectionDescriptor(
            name="custom",
            definition=CollectionDefinition(indexing={"deny": ["metadata._node_content", "content"]}),
            raw_descriptor={},
        )
    ]
    store._get_collection()
    store._get_collection()
    assert not recwarn
    database.list_collections.assert_called_once_with()
    database.create_collection.assert_not_called()


def test_collection_creation_error_propagates(native_sync_store):
    store, _, database, _ = native_sync_store
    database.create_collection.side_effect = DataAPIResponseException.from_response(
        command=None,
        raw_response={
            "errors": [
                {
                    "message": "Collection already exists with different settings",
                    "errorCode": "EXISTING_COLLECTION_DIFFERENT_SETTINGS",
                }
            ]
        },
    )
    with pytest.raises(DataAPIResponseException):
        store._get_collection()
    assert store._collection is None


def test_native_sync_write_read(native_sync_store):
    store, _, _, collection = native_sync_store
    collection.find_one.return_value = None
    collection.find.return_value = [{"_id": "1", "content": "text", "$vector": [0.1] * 4, "meta": {}}]
    collection.insert_many.return_value.inserted_ids = ["1"]
    doc = Document(id="1", content="text", embedding=[0.1] * 4)
    assert store.write_documents([doc]) == 1
    assert store.get_documents_by_id(["1"]) == [doc]


@pytest.mark.parametrize("filters", [None, {"field": "meta.category", "operator": "==", "value": "news"}])
def test_search_uses_native_api(native_sync_store, filters):
    store, _, _, collection = native_sync_store
    collection.find.return_value = [
        {"_id": "1", "content": "text", "$vector": [0.1] * 4, "meta": {"category": "news"}, "$similarity": 0.9}
    ]
    result = store.search([0.2] * 4, 2, filters)
    assert result == [Document(id="1", content="text", embedding=[0.1] * 4, meta={"category": "news"}, score=0.9)]
    collection.find.assert_called_once_with(
        filter={"meta.category": {"$eq": "news"}} if filters else None,
        sort={"$vector": [0.2] * 4},
        limit=2,
        include_similarity=True,
        projection={"*": 1},
    )


def test_search_empty_results(native_sync_store, caplog):
    store, _, _, collection = native_sync_store
    collection.find.return_value = []
    assert store.search([0.1] * 4, 2) == []
    assert "No documents found" in caplog.text


def test_close_drops_sync_collection_and_reopens(native_sync_store):
    store, client, _, collection = native_sync_store
    assert store._get_collection() is collection
    store.close()
    store.close()
    assert store._collection is None
    assert store._get_collection() is collection
    assert client.call_count == 2


def test_count_documents_by_filter(mocked_store):
    store, collection = mocked_store
    collection.count_documents.return_value = 2

    count = store.count_documents_by_filter({"field": "meta.status", "operator": "==", "value": "draft"})

    assert count == 2
    collection.count_documents.assert_called_once_with({"meta.status": {"$eq": "draft"}}, upper_bound=1_000_000_000)


def test_count_documents(mocked_store):
    store, collection = mocked_store
    collection.count_documents.return_value = 7
    assert store.count_documents() == 7
    collection.count_documents.assert_called_once_with({}, upper_bound=10_000)


def test_count_unique_metadata_by_filter(mocked_store):
    store, collection = mocked_store
    collection.distinct.side_effect = [["news", "docs", ["docs", "faq"], None], [1, 2, 2]]

    counts = store.count_unique_metadata_by_filter(
        {"field": "meta.status", "operator": "==", "value": "published"}, ["category", "priority"]
    )

    assert counts == {"category": 3, "priority": 2}
    assert collection.distinct.call_args_list == [
        mock.call("meta.category", filter={"meta.status": {"$eq": "published"}}),
        mock.call("meta.priority", filter={"meta.status": {"$eq": "published"}}),
    ]


def test_get_metadata_fields_info(mocked_store):
    store, collection = mocked_store
    collection.find.return_value = [
        {"content": "Doc 1", "meta": {"category": "news", "priority": 1, "active": True}},
        {"content": "Doc 2", "meta": {"category": "docs", "priority": 2.5, "tags": ["a", "b"]}},
    ]

    fields_info = store.get_metadata_fields_info()

    assert fields_info == {
        "content": {"type": "text"},
        "category": {"type": "keyword"},
        "priority": {"type": "long"},
        "active": {"type": "boolean"},
        "tags": {"type": "keyword"},
    }
    collection.find.assert_called_once_with(projection={"content": 1, "meta": 1})


def test_get_metadata_field_min_max(mocked_store):
    store, collection = mocked_store
    collection.distinct.return_value = [10, 3, 7]

    assert store.get_metadata_field_min_max("priority") == {"min": 3, "max": 10}
    collection.distinct.assert_called_once_with("meta.priority")


def test_get_metadata_field_unique_values(mocked_store):
    store, collection = mocked_store
    collection.distinct.return_value = ["Beta", "alpha", ["gamma", "alphabet"], None]

    values, total_count = store.get_metadata_field_unique_values("category", search_term="alp", from_=0, size=5)

    assert values == ["alpha", "alphabet"]
    assert total_count == 2
    collection.distinct.assert_called_once_with("meta.category", filter=None)


def test_get_metadata_field_unique_values_preserves_non_string_types(mocked_store):
    store, collection = mocked_store
    collection.distinct.return_value = [1, 2, 1, 3]

    values, total_count = store.get_metadata_field_unique_values("priority")

    assert values == [1, 2, 3]
    assert total_count == 3
    collection.distinct.assert_called_once_with("meta.priority", filter=None)


def test_get_documents_by_id_batches_ids(mocked_store):
    store, collection = mocked_store
    collection.find.side_effect = [
        [{"_id": str(i), "content": "a"} for i in range(20)],
        [{"_id": "20", "content": "a"}],
    ]
    assert len(store.get_documents_by_id([str(i) for i in range(21)])) == 21
    assert [c.kwargs["filter"] for c in collection.find.call_args_list] == [
        {"_id": {"$in": [str(i) for i in range(20)]}},
        {"_id": {"$in": ["20"]}},
    ]


def test_get_document_by_id_missing_raises(mocked_store):
    store, collection = mocked_store
    collection.find.return_value = []
    with pytest.raises(MissingDocumentError, match="does not exist"):
        store.get_document_by_id("missing")


@pytest.mark.parametrize(
    "filters,expected_kwargs",
    [
        (None, {"filter": None, "limit": 1000}),
        (
            {"field": "meta.k", "operator": "==", "value": "v"},
            {"filter": {"meta.k": {"$eq": "v"}}, "limit": 1000},
        ),
    ],
)
def test_filter_documents_forwards_filters(mocked_store, filters, expected_kwargs):
    store, collection = mocked_store
    collection.find.return_value = [{"_id": "1", "content": "a", "meta": {"k": "v"}}]
    assert store.filter_documents(filters) == [Document(id="1", content="a", meta={"k": "v"})]
    collection.find.assert_called_once_with(**expected_kwargs, projection={"*": 1})


@pytest.mark.parametrize(
    "api_endpoint,token,match",
    [
        (
            Secret.from_env_var("ASTRA_DB_API_ENDPOINT", strict=False),
            Secret.from_token("tok"),
            "API endpoint",
        ),
        (
            Secret.from_token("http://example.com"),
            Secret.from_env_var("ASTRA_DB_APPLICATION_TOKEN", strict=False),
            "authentication token",
        ),
    ],
)
def test_init_raises_when_secret_resolves_to_none(monkeypatch, api_endpoint, token, match):
    monkeypatch.delenv("ASTRA_DB_API_ENDPOINT", raising=False)
    monkeypatch.delenv("ASTRA_DB_APPLICATION_TOKEN", raising=False)
    with pytest.raises(ValueError, match=match):
        AstraDocumentStore(api_endpoint=api_endpoint, token=token)


@pytest.mark.parametrize(
    "doc,expected_exc,match",
    [
        ({"id": "1", "_id": "1", "content": "x"}, Exception, "Duplicate id definitions"),
        ({"_id": 42, "content": "x"}, Exception, "is not a string"),
        ("not-a-doc", ValueError, "Unsupported type"),
    ],
)
def test_write_documents_input_validation_errors(mocked_store, doc, expected_exc, match):
    store, _ = mocked_store
    with pytest.raises(expected_exc, match=match):
        store.write_documents([doc])


def test_write_documents_fail_policy_raises_on_duplicate(mocked_store):
    store, collection = mocked_store
    collection.find_one.return_value = {"_id": "1"}
    with pytest.raises(DuplicateDocumentError, match="already exists"):
        store.write_documents([Document(id="1", content="a")], policy=DuplicatePolicy.FAIL)
    collection.find_one.assert_called_once_with({"_id": "1"}, projection={"_id": True})


def test_write_documents_sparse_embedding_is_dropped_with_warning(mocked_store, caplog):
    store, collection = mocked_store
    collection.find_one.return_value = None
    collection.insert_many.return_value.inserted_ids = ["1"]
    store.write_documents([{"_id": "1", "content": "x", "sparse_embedding": {"indices": [0], "values": [1.0]}}])
    inserted = collection.insert_many.call_args.kwargs["documents"][0]
    assert "sparse_embedding" not in inserted
    assert "sparse embeddings in Astra" in caplog.text


@pytest.mark.parametrize("updated,expected_count", [({"_id": "1"}, 1), (None, 0)])
def test_write_documents_overwrite_updates_existing(mocked_store, caplog, updated, expected_count):
    store, collection = mocked_store
    collection.find_one.return_value = {"_id": "1"}
    collection.find_one_and_update.return_value = updated
    doc = {"_id": "1", "content": "new", "meta": {"k": "v"}}
    assert store.write_documents([doc], policy=DuplicatePolicy.OVERWRITE) == expected_count
    collection.find_one_and_update.assert_called_once_with(
        {"_id": "1"}, {"$set": {"content": "new", "meta": {"k": "v"}}}, projection={"_id": True}
    )
    collection.insert_many.assert_not_called()
    assert doc["_id"] == "1"
    assert ("not updated" in caplog.text) is (updated is None)


def test_delete_documents_batches_ids(mocked_store):
    store, collection = mocked_store
    collection.find_one.return_value = {"_id": "x"}
    collection.delete_many.return_value.deleted_count = 1
    store.delete_documents([str(i) for i in range(21)])
    assert [c.args[0] for c in collection.delete_many.call_args_list] == [
        {"_id": {"$in": [str(i) for i in range(20)]}},
        {"_id": {"$in": ["20"]}},
    ]


def test_delete_documents_missing_raises(mocked_store):
    store, collection = mocked_store
    collection.find_one.return_value = {"_id": "x"}
    collection.delete_many.return_value.deleted_count = 0
    with pytest.raises(MissingDocumentError, match="does not exist"):
        store.delete_documents(["missing"])


def test_delete_documents_empty_store_is_noop(mocked_store):
    store, collection = mocked_store
    collection.find_one.return_value = None
    store.delete_documents(["1"])
    collection.delete_many.assert_not_called()


def test_delete_by_filter(mocked_store):
    store, collection = mocked_store
    collection.delete_many.return_value.deleted_count = 3
    assert store.delete_by_filter({"field": "meta.k", "operator": "==", "value": "v"}) == 3
    collection.delete_many.assert_called_once_with({"meta.k": {"$eq": "v"}})


def test_delete_all_documents_wraps_exception(mocked_store):
    store, collection = mocked_store
    collection.delete_many.side_effect = RuntimeError("boom")
    with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
        store.delete_all_documents()
    collection.delete_many.assert_called_once_with({})


def test_delete_all_documents_recreate_index(mocked_store):
    store, collection = mocked_store
    store.delete_all_documents(recreate_index=True)
    collection.drop.assert_called_once_with()
    collection.database.create_collection.assert_called_once_with(
        "documents", definition=collection.options.return_value
    )
    collection.delete_many.assert_not_called()
    assert store._collection is collection.database.create_collection.return_value


def test_delete_all_documents_recreate_index_failure(mocked_store):
    store, collection = mocked_store
    store._collection = collection
    collection.database.create_collection.side_effect = RuntimeError("boom")
    with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
        store.delete_all_documents(recreate_index=True)
    # The next operation creates the collection again from the store settings.
    assert store._collection is None


@pytest.mark.parametrize(
    "filters,meta,match",
    [
        ("bad", {}, "Filters must be a dictionary"),
        ({}, "bad", "Meta must be a dictionary"),
    ],
)
def test_update_by_filter_validation_errors(mocked_store, filters, meta, match):
    store, _ = mocked_store
    with pytest.raises(AstraDocumentStoreFilterError, match=match):
        store.update_by_filter(filters=filters, meta=meta)


def test_update_by_filter_applies_meta_with_dot_notation(mocked_store):
    store, collection = mocked_store
    collection.update_many.return_value.update_info = {"nModified": 4}
    count = store.update_by_filter(
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        meta={"reviewed": True, "priority": 1},
    )
    assert count == 4
    collection.update_many.assert_called_once_with(
        {"meta.category": {"$eq": "news"}}, {"$set": {"meta.reviewed": True, "meta.priority": 1}}
    )


def test_infer_metadata_field_type_mixed_types_warn_and_default_to_keyword(caplog):
    assert AstraDocumentStore._infer_metadata_field_type([1, "a"]) == "keyword"
    assert "mixed metadata types" in caplog.text


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN env var not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_API_ENDPOINT", "") == "", reason="ASTRA_DB_API_ENDPOINT env var not set")
class TestDocumentStore(
    DocumentStoreBaseExtendedTests,
    CountDocumentsByFilterTest,
    CountUniqueMetadataByFilterTest,
    GetMetadataFieldsInfoTest,
    GetMetadataFieldMinMaxTest,
    GetMetadataFieldUniqueValuesTest,
):
    """
    Common test cases will be provided by `DocumentStoreBaseExtendedTests` but
    you can add more to this class.
    """

    @pytest.fixture(scope="class")
    def document_store(self):
        store = AstraDocumentStore(
            collection_name="haystack_test_document_store",
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dimension=768,
        )
        try:
            yield store
        finally:
            store._get_collection().drop()

    @pytest.fixture(autouse=True)
    def run_before_tests(self, document_store: AstraDocumentStore):
        """
        Cleaning up document store
        """
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    @staticmethod
    def assert_documents_are_equal(received: list[Document], expected: list[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        received.sort(key=operator.attrgetter("id"))
        expected.sort(key=operator.attrgetter("id"))
        assert received == expected

    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})
        # Astra does not support filtering on None, it returns empty list
        TestDocumentStore.assert_documents_are_equal(result, [])

    def test_get_metadata_field_unique_values_distinct_types(self, document_store: AstraDocumentStore):
        """
        Override: the base mixin test stores int, float, str and bool under the *same* metadata field
        name and expects all four back as distinct values.

        This adapts the same intent - int, float, str and bool must come back as distinct, unmangled
        types via get_metadata_field_unique_values() - using one field per type instead of one shared
        field, which is what AstraDB can actually support.

        The float value is a non-whole number (1.5, not 1.0): AstraDB's Data API canonicalizes any
        whole-number float to an int on storage - unconditionally, not just when it shares a field with
        an int - so a whole-number float could never come back as a float here regardless of field
        separation. A fractional value has no such ambiguity and round-trips as a float.
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

    def test_write_documents(self, document_store: AstraDocumentStore):
        """
        Test write_documents() overwrites stored Document when trying to write one with same id
        using DuplicatePolicy.OVERWRITE.
        """
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        TestDocumentStore.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        TestDocumentStore.assert_documents_are_equal(document_store.filter_documents(), [doc1])

    def test_write_documents_skip_duplicates(self, document_store: AstraDocumentStore):
        docs = [
            Document(id="1", content="test doc 1"),
            Document(id="1", content="test doc 2"),
        ]
        assert document_store.write_documents(docs, policy=DuplicatePolicy.SKIP) == 1

    def test_delete_documents_non_existing_document(self, document_store: AstraDocumentStore):
        """
        Test delete_documents() doesn't delete any Document when called with non existing id.
        """
        doc = Document(content="test doc")
        document_store.write_documents([doc])
        assert document_store.count_documents() == 1

        with pytest.raises(MissingDocumentError):
            document_store.delete_documents(["non_existing_id"])

        # No Document has been deleted
        assert document_store.count_documents() == 1

    def test_delete_documents_more_than_twenty_delete_all(self, document_store: AstraDocumentStore):
        """
        Test delete_documents() deletes all documents when called on an Astra DB with
        more than 20 documents. Twenty documents is the maximum number of deleted
        documents in one call for Astra.
        """
        docs = []
        for i in range(1, 26):
            doc = Document(content=f"test doc {i}", id=str(i))
            docs.append(doc)
        document_store.write_documents(docs)
        assert document_store.count_documents() == 25

        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_delete_documents_more_than_twenty_delete_ids(self, document_store: AstraDocumentStore):
        """
        Test delete_documents() deletes all documents when called on an Astra DB with
        more than 20 documents. Twenty documents is the maximum number of deleted
        documents in one call for Astra.
        """
        docs = []
        document_ids = []
        for i in range(1, 26):
            doc = Document(content=f"test doc {i}", id=str(i))
            docs.append(doc)
            document_ids.append(str(i))
        document_store.write_documents(docs)
        assert document_store.count_documents() == 25

        document_store.delete_documents(document_ids=document_ids)

        # No Document has been deleted
        assert document_store.count_documents() == 0

    def test_filter_documents_nested_filters(self, document_store, filterable_docs):
        filter_criteria = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.page", "operator": "==", "value": "100"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.chapter", "operator": "==", "value": "abstract"},
                        {"field": "meta.chapter", "operator": "==", "value": "intro"},
                    ],
                },
            ],
        }

        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters=filter_criteria)

        TestDocumentStore.assert_documents_are_equal(
            result,
            [
                d
                for d in filterable_docs
                if d.meta.get("page") == "100"
                and (d.meta.get("chapter") == "abstract" or d.meta.get("chapter") == "intro")
            ],
        )

    def test_filter_documents_by_id(self, document_store):
        docs = [Document(id="1", content="test doc 1"), Document(id="2", content="test doc 2")]
        document_store.write_documents(docs)
        result = document_store.filter_documents(filters={"field": "id", "operator": "==", "value": "1"})
        TestDocumentStore.assert_documents_are_equal(result, [docs[0]])

    def test_filter_documents_by_in_operator(self, document_store):
        docs = [Document(id="3", content="test doc 3"), Document(id="4", content="test doc 4")]
        document_store.write_documents(docs)
        result = document_store.filter_documents(filters={"field": "id", "operator": "in", "value": ["3", "4"]})

        # Sort the result in place by the id field
        result.sort(key=lambda x: x.id)

        TestDocumentStore.assert_documents_are_equal([result[0]], [docs[0]])
        TestDocumentStore.assert_documents_are_equal([result[1]], [docs[1]])

    def test_not_operator_over_not_equal_none(self, document_store, filterable_docs):
        # `!= None` produces a compound `{$exists: true, $ne: null}` clause; wrapping
        # it in NOT exercises the disjunction-based negation in `_negate`.
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [{"field": "meta.number", "operator": "!=", "value": None}],
            }
        )
        TestDocumentStore.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") is None]
        )
