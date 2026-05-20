# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import operator
import os
from unittest import mock

import pytest
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


@pytest.fixture
def mock_auth(monkeypatch):
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "http://example.com")
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "test_token")


@pytest.fixture
def mocked_store(mock_auth):  # noqa: ARG001
    """Returns (store, mock_index) with AstraClient fully mocked out."""
    with mock.patch("haystack_integrations.document_stores.astra.document_store.AstraClient") as mock_client:
        mock_index = mock_client.return_value
        store = AstraDocumentStore()
        yield store, mock_index


@mock.patch("haystack_integrations.document_stores.astra.astra_client.AstraDBClient")
def test_init_is_lazy(_mock_client, mock_auth):  # noqa
    _ = AstraDocumentStore()
    _mock_client.assert_not_called()


def test_to_dict(mock_auth):  # noqa
    with mock.patch("haystack_integrations.document_stores.astra.astra_client.AstraDBClient"):
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


def test_count_documents_by_filter(mocked_store):
    store, mock_index = mocked_store
    mock_index.count_documents.return_value = 2

    count = store.count_documents_by_filter({"field": "meta.status", "operator": "==", "value": "draft"})

    assert count == 2
    mock_index.count_documents.assert_called_once_with(
        filters={"meta.status": {"$eq": "draft"}}, upper_bound=1_000_000_000
    )


def test_count_unique_metadata_by_filter(mocked_store):
    store, mock_index = mocked_store
    mock_index.distinct.side_effect = [["news", "docs", ["docs", "faq"], None], [1, 2, 2]]

    counts = store.count_unique_metadata_by_filter(
        {"field": "meta.status", "operator": "==", "value": "published"}, ["category", "priority"]
    )

    assert counts == {"category": 3, "priority": 2}
    assert mock_index.distinct.call_args_list == [
        mock.call("meta.category", filters={"meta.status": {"$eq": "published"}}),
        mock.call("meta.priority", filters={"meta.status": {"$eq": "published"}}),
    ]


def test_get_metadata_fields_info(mocked_store):
    store, mock_index = mocked_store
    mock_index.find_documents.return_value = [
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
    mock_index.find_documents.assert_called_once_with({}, projection={"content": 1, "meta": 1})


def test_get_metadata_field_min_max(mocked_store):
    store, mock_index = mocked_store
    mock_index.distinct.return_value = [10, 3, 7]

    assert store.get_metadata_field_min_max("priority") == {"min": 3, "max": 10}
    mock_index.distinct.assert_called_once_with("meta.priority")


def test_get_metadata_field_unique_values(mocked_store):
    store, mock_index = mocked_store
    mock_index.distinct.return_value = ["Beta", "alpha", ["gamma", "alphabet"], None]

    values, total_count = store.get_metadata_field_unique_values("category", search_term="alp", from_=0, size=5)

    assert values == ["alpha", "alphabet"]
    assert total_count == 2
    mock_index.distinct.assert_called_once_with("meta.category")


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
    store, mock_index = mocked_store
    mock_index.find_documents.return_value = [{"_id": "1"}]
    with pytest.raises(DuplicateDocumentError, match="already exists"):
        store.write_documents([Document(id="1", content="a")], policy=DuplicatePolicy.FAIL)


def test_write_documents_sparse_embedding_is_dropped_with_warning(mocked_store, caplog):
    store, mock_index = mocked_store
    mock_index.find_documents.return_value = []
    mock_index.insert.return_value = ["1"]
    store.write_documents([{"_id": "1", "content": "x", "sparse_embedding": {"indices": [0], "values": [1.0]}}])
    inserted = mock_index.insert.call_args.args[0][0]
    assert "sparse_embedding" not in inserted
    assert "sparse embeddings in Astra" in caplog.text


def test_delete_all_documents_wraps_exception(mocked_store):
    store, mock_index = mocked_store
    mock_index.delete_all_documents.side_effect = RuntimeError("boom")
    with pytest.raises(DocumentStoreError, match="Failed to delete all documents"):
        store.delete_all_documents()


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
    store, mock_index = mocked_store
    mock_index.update.return_value = 4
    count = store.update_by_filter(
        filters={"field": "meta.category", "operator": "==", "value": "news"},
        meta={"reviewed": True, "priority": 1},
    )
    assert count == 4
    kwargs = mock_index.update.call_args.kwargs
    assert kwargs["filters"] == {"meta.category": {"$eq": "news"}}
    assert kwargs["update"] == {"$set": {"meta.reviewed": True, "meta.priority": 1}}


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
    def document_store(self) -> AstraDocumentStore:
        return AstraDocumentStore(
            collection_name="haystack_integration",
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dimension=768,
        )

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
