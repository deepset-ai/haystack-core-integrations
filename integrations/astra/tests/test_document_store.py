# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0
import os
from typing import List
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


@mock.patch("haystack_integrations.document_stores.astra.astra_client.AstraDB")
def test_init_is_lazy(_mock_client, mock_auth):  # noqa
    _ = AstraDocumentStore()
    _mock_client.assert_not_called()


def test_namespace_init(mock_auth):  # noqa
    with mock.patch("haystack_integrations.document_stores.astra.astra_client.AstraDB") as client:
        _ = AstraDocumentStore().index
        assert "namespace" in client.call_args.kwargs
        assert client.call_args.kwargs["namespace"] is None

        _ = AstraDocumentStore(namespace="foo").index
        assert "namespace" in client.call_args.kwargs
        assert client.call_args.kwargs["namespace"] == "foo"


def test_to_dict(mock_auth):  # noqa
    with mock.patch("haystack_integrations.document_stores.astra.astra_client.AstraDB"):
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

    @pytest.fixture
    def document_store(self) -> AstraDocumentStore:
        return AstraDocumentStore(
            collection_name="haystack_integration",
            duplicates_policy=DuplicatePolicy.OVERWRITE,
            embedding_dimension=768,
        )

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self, document_store: AstraDocumentStore):
        """
        Cleaning up document store
        """
        document_store.delete_documents(delete_all=True)
        assert document_store.count_documents() == 0

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test, if a Document Store implementation has a different behaviour
        it should override this method.

        This can happen for example when the Document Store sets a score to returned Documents.
        Since we can't know what the score will be, we can't compare the Documents reliably.
        """
        import operator

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

        document_store.delete_documents(delete_all=True)

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

    @pytest.mark.skip(reason="Unsupported filter operator not.")
    def test_not_operator(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $neq.")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $neq.")
    def test_comparison_not_equal(self, document_store, filterable_docs):
        pass

    @pytest.mark.skip(reason="Unsupported filter operator $neq.")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs):
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
    def test_comparison_greater_than_with_dataframe(self, document_store, filterable_docs):
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
    def test_comparison_greater_than_equal_with_dataframe(self, document_store, filterable_docs):
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
    def test_comparison_less_than_equal_with_dataframe(self, document_store, filterable_docs):
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
    def test_comparison_less_than_with_dataframe(self, document_store, filterable_docs):
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
