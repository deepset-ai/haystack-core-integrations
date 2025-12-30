# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

import operator
import os
from unittest import mock

import pytest
from haystack import Document
from haystack.document_stores.errors import MissingDocumentError
from haystack.document_stores.types import DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from haystack_integrations.document_stores.astra import AstraDocumentStore


@pytest.fixture
def mock_auth(monkeypatch):
    monkeypatch.setenv("ASTRA_DB_API_ENDPOINT", "http://example.com")
    monkeypatch.setenv("ASTRA_DB_APPLICATION_TOKEN", "test_token")


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


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("ASTRA_DB_APPLICATION_TOKEN", "") == "", reason="ASTRA_DB_APPLICATION_TOKEN env var not set"
)
@pytest.mark.skipif(os.environ.get("ASTRA_DB_API_ENDPOINT", "") == "", reason="ASTRA_DB_API_ENDPOINT env var not set")
class TestDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
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

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
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
        self.assert_documents_are_equal(result, [])

    def test_write_documents(self, document_store: AstraDocumentStore):
        """
        Test write_documents() overwrites stored Document when trying to write one with same id
        using DuplicatePolicy.OVERWRITE.
        """
        doc1 = Document(id="1", content="test doc 1")
        doc2 = Document(id="1", content="test doc 2")

        assert document_store.write_documents([doc2], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc2])
        assert document_store.write_documents(documents=[doc1], policy=DuplicatePolicy.OVERWRITE) == 1
        self.assert_documents_are_equal(document_store.filter_documents(), [doc1])

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

        self.assert_documents_are_equal(
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
        self.assert_documents_are_equal(result, [docs[0]])

    def test_filter_documents_by_in_operator(self, document_store):
        docs = [Document(id="3", content="test doc 3"), Document(id="4", content="test doc 4")]
        document_store.write_documents(docs)
        result = document_store.filter_documents(filters={"field": "id", "operator": "in", "value": ["3", "4"]})

        # Sort the result in place by the id field
        result.sort(key=lambda x: x.id)

        self.assert_documents_are_equal([result[0]], [docs[0]])
        self.assert_documents_are_equal([result[1]], [docs[1]])

    def test_delete_all_documents(self, document_store: AstraDocumentStore):
        """
        Test delete_all_documents() on an Astra.
        """
        document_store.delete_all_documents()
        assert document_store.count_documents() == 0

    def test_delete_by_filter(self, document_store: AstraDocumentStore, filterable_docs):
        document_store.write_documents(filterable_docs)
        initial_count = document_store.count_documents()
        assert initial_count > 0

        # count documents that match the filter before deletion
        matching_docs = [d for d in filterable_docs if d.meta.get("chapter") == "intro"]
        expected_deleted_count = len(matching_docs)

        # delete all documents with chapter="intro"
        deleted_count = document_store.delete_by_filter(
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"}
        )

        assert deleted_count == expected_deleted_count
        assert document_store.count_documents() == initial_count - deleted_count

        # remaining documents don't have chapter="intro"
        remaining_docs = document_store.filter_documents()
        for doc in remaining_docs:
            assert doc.meta.get("chapter") != "intro"

        # all documents with chapter="intro" were deleted
        intro_docs = document_store.filter_documents(
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"}
        )
        assert len(intro_docs) == 0

    def test_update_by_filter(self, document_store: AstraDocumentStore, filterable_docs):
        document_store.write_documents(filterable_docs)
        initial_count = document_store.count_documents()
        assert initial_count > 0

        # count documents that match the filter before update
        matching_docs = [d for d in filterable_docs if d.meta.get("chapter") == "intro"]
        expected_updated_count = len(matching_docs)

        # update all documents with chapter="intro" to have status="updated"
        updated_count = document_store.update_by_filter(
            filters={"field": "meta.chapter", "operator": "==", "value": "intro"},
            meta={"status": "updated"},
        )

        assert updated_count == expected_updated_count
        assert document_store.count_documents() == initial_count

        # verify the updated documents have the new metadata
        updated_docs = document_store.filter_documents(
            filters={"field": "meta.status", "operator": "==", "value": "updated"}
        )
        assert len(updated_docs) == expected_updated_count
        for doc in updated_docs:
            assert doc.meta.get("chapter") == "intro"
            assert doc.meta.get("status") == "updated"

        # verify other documents weren't affected
        all_docs = document_store.filter_documents()
        for doc in all_docs:
            if doc.meta.get("chapter") != "intro":
                assert doc.meta.get("status") != "updated"

    @pytest.mark.skip(reason="Unsupported filter operator not.")
    def test_not_operator(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $neq.")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $neq.")
    def test_comparison_not_equal(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    def test_comparison_not_in(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    def test_comparison_not_in_with_with_non_list(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $nin.")
    def test_comparison_not_in_with_with_non_list_iterable(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    def test_comparison_greater_than_with_string(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    def test_comparison_greater_than_with_list(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gt.")
    def test_comparison_greater_than(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    def test_comparison_greater_than_equal(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    def test_comparison_greater_than_equal_with_list(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    def test_comparison_greater_than_equal_with_string(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $gte.")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    def test_comparison_less_than_equal(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    def test_comparison_less_than_equal_with_string(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    def test_comparison_less_than_equal_with_list(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lte.")
    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    def test_comparison_less_than_with_list(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    def test_comparison_less_than_with_string(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $lt.")
    def test_comparison_less_than(self, document_store, filterable_docs):
        pass
