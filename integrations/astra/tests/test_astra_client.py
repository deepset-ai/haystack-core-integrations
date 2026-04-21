# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for AstraClient (no Astra DB instance required)."""

from types import SimpleNamespace
from unittest import mock

import pytest
from astrapy.exceptions import CollectionAlreadyExistsException

from haystack_integrations.document_stores.astra.astra_client import (
    AstraClient,
    QueryResponse,
    Response,
)

CLIENT_PATH = "haystack_integrations.document_stores.astra.astra_client.AstraDBClient"


def _existing_exc() -> CollectionAlreadyExistsException:
    return CollectionAlreadyExistsException(text="exists", keyspace="default", collection_name="my_collection")


@pytest.fixture
def mock_astra_client():
    with mock.patch(CLIENT_PATH) as patched_client:
        db = patched_client.return_value.get_database.return_value
        db.create_collection.return_value = mock.MagicMock(name="collection")
        yield patched_client


def _make_client(**overrides) -> AstraClient:
    kwargs = {
        "api_endpoint": "http://example.com",
        "token": "test_token",
        "collection_name": "my_collection",
        "embedding_dimension": 4,
        "similarity_function": "cosine",
        "namespace": None,
    }
    kwargs.update(overrides)
    return AstraClient(**kwargs)


class TestQueryResponse:
    def test_get_returns_value(self):
        match = Response("id1", "text", [0.1], {"k": "v"}, 0.9)
        response = QueryResponse(matches=[match])
        assert response.get("matches") == [match]


class TestAstraClientInit:
    def test_creates_collection(self, mock_astra_client):
        client = _make_client()
        db = mock_astra_client.return_value.get_database.return_value
        db.create_collection.assert_called_once_with(
            name="my_collection",
            dimension=4,
            indexing={"deny": ["metadata._node_content", "content"]},
        )
        assert client._astra_db_collection is db.create_collection.return_value

    def test_legacy_collection_warns_and_reuses(self, mock_astra_client):
        # No 'indexing' on the existing collection -> legacy warning path
        db = mock_astra_client.return_value.get_database.return_value
        db.create_collection.side_effect = _existing_exc()
        legacy = SimpleNamespace(name="my_collection", options=SimpleNamespace(indexing=None))
        db.list_collections.return_value = [legacy]

        with pytest.warns(UserWarning, match="having indexing turned on"):
            client = _make_client()

        db.get_collection.assert_called_once_with("my_collection")
        assert client._astra_db_collection is db.get_collection.return_value

    def test_existing_collection_with_matching_indexing_reuses(self, mock_astra_client):
        db = mock_astra_client.return_value.get_database.return_value
        db.create_collection.side_effect = _existing_exc()
        existing = SimpleNamespace(
            name="my_collection",
            options=SimpleNamespace(indexing={"deny": ["metadata._node_content", "content"]}),
        )
        db.list_collections.return_value = [existing]

        client = _make_client()

        db.get_collection.assert_called_once_with("my_collection")
        assert client._astra_db_collection is db.get_collection.return_value

    def test_existing_collection_with_unexpected_indexing_warns(self, mock_astra_client):
        db = mock_astra_client.return_value.get_database.return_value
        db.create_collection.side_effect = _existing_exc()
        existing = SimpleNamespace(
            name="my_collection",
            options=SimpleNamespace(indexing={"deny": ["something_else"]}),
        )
        db.list_collections.return_value = [existing]

        with pytest.warns(UserWarning, match="unexpected 'indexing' settings"):
            _make_client()

        db.get_collection.assert_called_once_with("my_collection")

    def test_already_exists_for_unrelated_collection_reraises(self, mock_astra_client):
        db = mock_astra_client.return_value.get_database.return_value
        db.create_collection.side_effect = _existing_exc()
        db.list_collections.return_value = []  # no preexisting collection by that name

        with pytest.raises(CollectionAlreadyExistsException):
            _make_client()


class TestFormatQueryResponse:
    @pytest.mark.parametrize(
        "responses,include_metadata,include_values,expected_meta,expected_values",
        [
            (None, True, True, {}, []),
            (
                [{"_id": "1", "$similarity": 0.5, "content": "hi", "$vector": [0.1], "meta": {"k": "v"}}],
                True,
                True,
                {"meta": {"k": "v"}},
                [0.1],
            ),
            (
                [{"_id": "1", "$similarity": 0.5, "content": "hi", "$vector": [0.1], "meta": {"k": "v"}}],
                False,
                False,
                {},
                [],
            ),
        ],
    )
    def test_format(self, responses, include_metadata, include_values, expected_meta, expected_values):
        result = AstraClient._format_query_response(
            responses, include_metadata=include_metadata, include_values=include_values
        )
        if responses is None:
            assert result.matches == []
            return
        assert len(result.matches) == 1
        match = result.matches[0]
        assert match.document_id == "1"
        assert match.score == 0.5
        assert match.text == "hi"
        assert match.values == expected_values
        assert match.metadata == expected_meta


@pytest.mark.usefixtures("mock_astra_client")
class TestAstraClientMethods:
    def test_query_with_vector_calls_find_with_sort(self):
        client = _make_client()
        client._astra_db_collection.find.return_value = iter(
            [{"_id": "1", "$similarity": 0.9, "content": "x", "$vector": [0.1, 0.2, 0.3, 0.4]}]
        )
        response = client.query(vector=[0.1, 0.2, 0.3, 0.4], top_k=5, include_metadata=True, include_values=True)
        assert isinstance(response, QueryResponse)
        assert response.matches[0].document_id == "1"
        kwargs = client._astra_db_collection.find.call_args.kwargs
        assert kwargs["sort"] == {"$vector": [0.1, 0.2, 0.3, 0.4]}
        assert kwargs["limit"] == 5
        assert kwargs["include_similarity"] is True

    def test_query_with_vector_and_filter(self):
        client = _make_client()
        client._astra_db_collection.find.return_value = iter([])
        client.query(vector=[0.1] * 4, top_k=3, query_filter={"meta.k": {"$eq": "v"}})
        kwargs = client._astra_db_collection.find.call_args.kwargs
        assert kwargs["filter"] == {"meta.k": {"$eq": "v"}}

    def test_query_without_vector_uses_filter(self):
        client = _make_client()
        client._astra_db_collection.find.return_value = iter([])
        client.query(query_filter={"meta.k": {"$eq": "v"}}, top_k=2)
        kwargs = client._astra_db_collection.find.call_args.kwargs
        assert kwargs["filter"] == {"meta.k": {"$eq": "v"}}
        assert kwargs["limit"] == 2

    def test_find_documents_warns_when_empty(self, caplog):
        client = _make_client()
        client._astra_db_collection.find.return_value = iter([])
        result = client.find_documents({"filter": {"x": 1}})
        assert result == []
        assert "No documents found" in caplog.text

    def test_find_one_document_returns_result(self):
        client = _make_client()
        client._astra_db_collection.find_one.return_value = {"_id": "x"}
        assert client.find_one_document({"filter": {"_id": "x"}}) == {"_id": "x"}

    def test_find_one_document_returns_none_and_warns(self, caplog):
        client = _make_client()
        client._astra_db_collection.find_one.return_value = None
        assert client.find_one_document({"filter": {}}) is None
        assert "No document found" in caplog.text

    def test_get_documents_batches_ids(self):
        client = _make_client()
        client._astra_db_collection.find.side_effect = [
            iter([{"_id": str(i), "$similarity": None, "content": "a", "$vector": [0.0] * 4} for i in range(20)]),
            iter([{"_id": "20", "$similarity": None, "content": "a", "$vector": [0.0] * 4}]),
        ]
        result = client.get_documents([str(i) for i in range(21)], batch_size=20)
        assert len(result.matches) == 21
        assert client._astra_db_collection.find.call_count == 2

    def test_insert_returns_ids(self):
        client = _make_client()
        client._astra_db_collection.insert_many.return_value = SimpleNamespace(inserted_ids=[1, 2])
        assert client.insert([{"_id": "1"}, {"_id": "2"}]) == ["1", "2"]

    def test_update_document_success(self):
        client = _make_client()
        client._astra_db_collection.find_one_and_update.return_value = {"_id": "1", "content": "x"}
        assert client.update_document({"_id": "1", "content": "x"}, "_id") is True

    def test_update_document_no_match_returns_false(self, caplog):
        client = _make_client()
        client._astra_db_collection.find_one_and_update.return_value = None
        assert client.update_document({"_id": "1", "content": "x"}, "_id") is False
        assert "not updated" in caplog.text

    def test_delete_by_ids_passes_filter(self):
        client = _make_client()
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=2)
        assert client.delete(ids=["a", "b"]) == 2
        kwargs = client._astra_db_collection.delete_many.call_args.kwargs
        assert kwargs["filter"] == {"_id": {"$in": ["a", "b"]}}

    def test_delete_by_filter_passes_filter(self):
        client = _make_client()
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=3)
        assert client.delete(filters={"meta.k": {"$eq": "v"}}) == 3
        kwargs = client._astra_db_collection.delete_many.call_args.kwargs
        assert kwargs["filter"] == {"meta.k": {"$eq": "v"}}

    def test_delete_with_no_args_uses_empty_filter(self):
        client = _make_client()
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=0)
        assert client.delete() == 0
        kwargs = client._astra_db_collection.delete_many.call_args.kwargs
        assert kwargs["filter"] == {}

    def test_delete_all_documents(self):
        client = _make_client()
        client._astra_db_collection.delete_many.return_value = SimpleNamespace(deleted_count=5)
        assert client.delete_all_documents() == 5

    def test_count_documents_passes_defaults(self):
        client = _make_client()
        client._astra_db_collection.count_documents.return_value = 7
        assert client.count_documents() == 7
        client._astra_db_collection.count_documents.assert_called_once_with({}, upper_bound=10000)

    def test_distinct(self):
        client = _make_client()
        client._astra_db_collection.distinct.return_value = ["a", "b"]
        assert client.distinct("meta.k") == ["a", "b"]
        client._astra_db_collection.distinct.assert_called_once_with("meta.k", filter=None)

    def test_update_returns_modified_count(self):
        client = _make_client()
        client._astra_db_collection.update_many.return_value = SimpleNamespace(update_info={"nModified": 4})
        assert client.update(filters={"meta.k": "v"}, update={"$set": {"meta.k": "v2"}}) == 4
