from typing import List

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import (
    FilterDocumentsTest,
)


class TestFilters(FilterDocumentsTest):

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        for doc in received:
            # Supabase seems to convert integers to floats
            # We convert them back to integers to compare them
            if "number" in doc.meta:
                doc.meta["number"] = int(doc.meta["number"])

        # Lists comparison
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected):
            assert received_doc.meta == expected_doc.meta
            assert received_doc.content == expected_doc.content
            if received_doc.dataframe is None:
                assert expected_doc.dataframe is None
            else:
                assert received_doc.dataframe.equals(expected_doc.dataframe)
            # unfortunately, Supabase returns a slightly different embedding
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

    @pytest.mark.skip(reason="Supabase does not support not_in comparison")
    def test_comparison_not_in(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support not_in comparison")
    def test_comparison_not_in_with_with_non_list_iterable(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support not_in comparison")
    def test_comparison_not_in_with_with_non_list(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with null values")
    def test_comparison_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support not_in comparison")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with null values")
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with null values")
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with null values")
    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with null values")
    def test_comparison_less_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support the 'not' operator")
    def test_not_operator(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with dates")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with dates")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with dates")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase does not support comparison with dates")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase doesn't support comparision with dataframe")
    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Supabase doesn't support comparision with dataframe")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs): ...

    def test_or_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "OR",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )

        self.assert_documents_are_equal(
            result, [d for d in filterable_docs if d.meta.get("number") == 100 or d.meta.get("name") == "name_0"]
        )

    def test_comparison_not_equal(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents({"field": "meta.number", "operator": "!=", "value": 100})
        self.assert_documents_are_equal(result, [d for d in filterable_docs if d.meta.get("number") != 100])

    def test_no_filters(self, document_store):
        self.assert_documents_are_equal(document_store.filter_documents(), [])
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), [])
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        self.assert_documents_are_equal(document_store.filter_documents(), docs)
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), docs)
