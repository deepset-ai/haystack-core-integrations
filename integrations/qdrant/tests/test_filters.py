from typing import List

import pytest
from haystack import Document
from haystack.testing.document_store import FilterDocumentsTest
from haystack.utils.filters import FilterError

from qdrant_haystack import QdrantDocumentStore


class TestQdrantStoreBaseTests(FilterDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
        )

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.
        This is used in every test.
        """

        # Check that the lengths of the lists are the same
        assert len(received) == len(expected)

        # Check that the sets are equal, meaning the content and IDs match regardless of order
        assert {doc.id for doc in received} == {doc.id for doc in expected}

    def test_not_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )
        self.assert_documents_are_equal(
            result,
            [d for d in filterable_docs if (d.meta.get("number") != 100 and d.meta.get("name") != "name_0")],
        )

    # ======== OVERRIDES FOR NONE VALUED FILTERS ========

    def test_comparison_equal_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": "==", "value": None})
            self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") is None])

    def test_comparison_not_equal_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": "!=", "value": None})
            self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") is not None])

    def test_comparison_greater_than_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": ">", "value": None})
            self.assert_documents_are_equal(result, [])

    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": ">=", "value": None})
            self.assert_documents_are_equal(result, [])

    def test_comparison_less_than_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": "<", "value": None})
            self.assert_documents_are_equal(result, [])

    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            result = document_store.filter_documents(filters={"field": "meta.number", "operator": "<=", "value": None})
            self.assert_documents_are_equal(result, [])

    # ======== ========================== ========

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with dataframe")
    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with dataframe")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with Dates")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with Dates")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with Dates")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with Dates")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs):
        ...

    @pytest.mark.skip(reason="Cannot distinguish errors yet")
    def test_missing_top_level_operator_key(self, document_store, filterable_docs):
        ...
