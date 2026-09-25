# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from types import SimpleNamespace
from unittest import mock

import pytest
from astrapy.data.utils.collection_converters import postprocess_collection_response
from astrapy.exceptions import DataAPIResponseException
from astrapy.info import CollectionDefinition, CollectionDescriptor
from astrapy.utils.api_options import defaultAPIOptions

from haystack_integrations.document_stores.astra.astra_client import (
    _API_OPTIONS,
    AstraClient,
    QueryResponse,
    Response,
    _format_query_response,
)

CLIENT_PATH = "haystack_integrations.document_stores.astra.astra_client.AstraDBClient"


CLIENT_KWARGS = {
    "api_endpoint": "http://example.com",
    "token": "test_token",
    "collection_name": "my_collection",
    "embedding_dimension": 4,
    "similarity_function": "cosine",
}


@pytest.fixture
def mock_db():
    with mock.patch(CLIENT_PATH) as patched_client:
        patched_client.return_value.get_database.return_value.list_collections.return_value = []
        yield patched_client.return_value.get_database.return_value


@pytest.fixture
def client(mock_db) -> AstraClient:  # noqa: ARG001
    return AstraClient(**CLIENT_KWARGS)


def test_query_response_get_returns_value():
    match = Response("id1", "text", [0.1], {"k": "v"}, 0.9)
    assert QueryResponse(matches=[match]).get("matches") == [match]


def test_api_options_read_plain_python_types():
    serdes = defaultAPIOptions(environment="prod").with_override(_API_OPTIONS).serdes_options
    doc = postprocess_collection_response(
        {"_id": "1", "$vector": [0.12345678901234568, 0.2], "meta": {"d": {"$date": 0}}}, options=serdes
    )
    assert isinstance(doc["$vector"], list)
    assert doc["$vector"] == [0.12345678901234568, 0.2]
    assert isinstance(doc["meta"]["d"], datetime)


class TestAstraClientInit:
    @pytest.mark.parametrize("similarity", ["cosine", "dot_product", "euclidean"])
    def test_creates_collection(self, mock_db, similarity):
        client = AstraClient(**{**CLIENT_KWARGS, "similarity_function": similarity})
        mock_db.create_collection.assert_called_once_with(
            name="my_collection",
            definition={
                "vector": {"dimension": 4, "metric": similarity},
                "indexing": {"deny": ["metadata._node_content", "content"]},
            },
        )
        assert client._astra_db_collection is client._astra_db.create_collection.return_value

    @pytest.mark.parametrize(
        "pre_indexing,warning_match",
        [
            (None, "having indexing turned on"),
            ({"deny": ["something_else"]}, "unexpected 'indexing' settings"),
        ],
    )
    def test_preexisting_collection_with_mismatched_indexing_warns(self, mock_db, pre_indexing, warning_match):
        mock_db.list_collections.return_value = [
            CollectionDescriptor(
                name="my_collection", definition=CollectionDefinition(indexing=pre_indexing), raw_descriptor={}
            )
        ]
        with pytest.warns(UserWarning, match=warning_match):
            client = AstraClient(**CLIENT_KWARGS)
        mock_db.get_collection.assert_called_once_with("my_collection")
        assert client._astra_db_collection is mock_db.get_collection.return_value
        mock_db.create_collection.assert_not_called()

    def test_preexisting_collection_with_matching_indexing_reuses_silently(self, mock_db):
        mock_db.list_collections.return_value = [
            CollectionDescriptor(
                name="my_collection",
                definition=CollectionDefinition(indexing={"deny": ["metadata._node_content", "content"]}),
                raw_descriptor={},
            )
        ]
        AstraClient(**CLIENT_KWARGS)
        mock_db.get_collection.assert_called_once_with("my_collection")
        mock_db.create_collection.assert_not_called()

    def test_collection_creation_error_propagates(self, mock_db):
        mock_db.create_collection.side_effect = DataAPIResponseException.from_response(
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
        mock_db.list_collections.return_value = []
        with pytest.raises(DataAPIResponseException):
            AstraClient(**CLIENT_KWARGS)


@pytest.mark.parametrize(
    "include_metadata,include_values,expected_meta,expected_values",
    [
        (True, True, {"meta": {"k": "v"}}, [0.1]),
        (False, False, {}, []),
    ],
)
def test_format_query_response(include_metadata, include_values, expected_meta, expected_values):
    responses = [{"_id": "1", "$similarity": 0.5, "content": "hi", "$vector": [0.1], "meta": {"k": "v"}}]
    result = _format_query_response(responses, include_metadata=include_metadata, include_values=include_values)
    match = result.matches[0]
    assert (match.document_id, match.score, match.text) == ("1", 0.5, "hi")
    assert match.values == expected_values
    assert match.metadata == expected_meta


def test_format_query_response_with_none_returns_empty_matches():
    assert _format_query_response(None, include_metadata=True, include_values=True).matches == []


class TestAstraClientMethods:
    @pytest.mark.parametrize(
        "query_kwargs,expected_find_kwargs",
        [
            (
                {"vector": [0.1, 0.2, 0.3, 0.4], "top_k": 5},
                {"sort": {"$vector": [0.1, 0.2, 0.3, 0.4]}, "limit": 5, "include_similarity": True},
            ),
            (
                {"vector": [0.1] * 4, "top_k": 3, "query_filter": {"meta.k": {"$eq": "v"}}},
                {"filter": {"meta.k": {"$eq": "v"}}},
            ),
            (
                {"query_filter": {"meta.k": {"$eq": "v"}}, "top_k": 2},
                {"filter": {"meta.k": {"$eq": "v"}}, "limit": 2},
            ),
        ],
    )
    def test_query_forwards_args_to_find(self, client, query_kwargs, expected_find_kwargs):
        client._astra_db_collection.find.return_value = iter([])
        client.query(**query_kwargs)
        actual = client._astra_db_collection.find.call_args.kwargs
        for key, value in expected_find_kwargs.items():
            assert actual[key] == value

    def test_find_documents_warns_when_empty(self, client, caplog):
        client._astra_db_collection.find.return_value = iter([])
        assert client.find_documents({"filter": {"x": 1}}) == []
        assert "No documents found" in caplog.text

    @pytest.mark.parametrize(
        "return_value,expected,log_substring",
        [
            ({"_id": "x"}, {"_id": "x"}, ""),
            (None, None, "No document found"),
        ],
    )
    def test_find_one_document(self, client, caplog, return_value, expected, log_substring):
        client._astra_db_collection.find_one.return_value = return_value
        assert client.find_one_document({"filter": {}}) == expected
        if log_substring:
            assert log_substring in caplog.text

    def test_get_documents_batches_ids(self, client):
        client._astra_db_collection.find.side_effect = [
            iter([{"_id": str(i), "$similarity": None, "content": "a", "$vector": [0.0] * 4} for i in range(20)]),
            iter([{"_id": "20", "$similarity": None, "content": "a", "$vector": [0.0] * 4}]),
        ]
        result = client.get_documents([str(i) for i in range(21)], batch_size=20)
        assert len(result.matches) == 21
        assert client._astra_db_collection.find.call_count == 2

    def test_insert_returns_ids(self, client):
        client._astra_db_collection.insert_many.return_value = SimpleNamespace(inserted_ids=[1, 2])
        assert client.insert([{"_id": "1"}, {"_id": "2"}]) == ["1", "2"]

    @pytest.mark.parametrize(
        "returned,expected,log_substring",
        [
            ({"_id": "1", "content": "x"}, True, ""),
            (None, False, "not updated"),
        ],
    )
    def test_update_document(self, client, caplog, returned, expected, log_substring):
        client._astra_db_collection.find_one_and_update.return_value = returned
        assert client.update_document({"_id": "1", "content": "x"}, "_id") is expected
        if log_substring:
            assert log_substring in caplog.text

    @pytest.mark.parametrize(
        "delete_kwargs,expected_filter",
        [
            ({"ids": ["a", "b"]}, {"_id": {"$in": ["a", "b"]}}),
            ({"filters": {"meta.k": {"$eq": "v"}}}, {"meta.k": {"$eq": "v"}}),
            ({}, {}),
        ],
    )
    def test_delete_builds_filter(self, client, delete_kwargs, expected_filter):
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=1)
        client.delete(**delete_kwargs)
        assert client._astra_db_collection.delete_many.call_args.kwargs["filter"] == expected_filter

    def test_delete_all_documents_returns_deleted_count(self, client):
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=5)
        assert client.delete_all_documents() == 5

    def test_count_documents_passes_defaults(self, client):
        client._astra_db_collection.count_documents.return_value = 7
        assert client.count_documents() == 7
        client._astra_db_collection.count_documents.assert_called_once_with({}, upper_bound=10000)

    def test_distinct_forwards_filter(self, client):
        client._astra_db_collection.distinct.return_value = ["a", "b"]
        assert client.distinct("meta.k") == ["a", "b"]
        client._astra_db_collection.distinct.assert_called_once_with("meta.k", filter=None)

    def test_update_returns_modified_count(self, client):
        client._astra_db_collection.update_many.return_value = SimpleNamespace(update_info={"nModified": 4})
        assert client.update(filters={"meta.k": "v"}, update={"$set": {"meta.k": "v2"}}) == 4
