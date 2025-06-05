import os
from typing import List

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import (
    FilterDocumentsTest,
)


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        for doc in received:
            # Pinecone seems to convert integers to floats (undocumented behavior)
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
            # unfortunately, Pinecone returns a slightly different embedding
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_not_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with dates")
    def test_comparison_greater_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_greater_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with dates")
    def test_comparison_greater_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_greater_than_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with dates")
    def test_comparison_less_than_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_less_than_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with dates")
    def test_comparison_less_than_equal_with_iso_date(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support comparison with null values")
    def test_comparison_less_than_equal_with_none(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Pinecone does not support the 'not' operator")
    def test_not_operator(self, document_store, filterable_docs): ...
