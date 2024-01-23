from typing import List

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import FilterDocumentsTest


class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected):
            # we first compare the embeddings approximately
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)

            received_doc.embedding, expected_doc.embedding = None, None
            assert received_doc == expected_doc

    def test_complex_filter(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        filters = {
            "operator": "OR",
            "conditions": [
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.number", "operator": "==", "value": 100},
                        {"field": "meta.name", "operator": "==", "value": "name_0"},
                    ],
                },
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.page", "operator": "==", "value": 90},
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
                if (d.meta.get("number") == 100 and d.meta.get("name") == "name_0")
                or (d.meta.get("page") == 90 and d.meta.get("chapter") == "conclusion")
            ],
        )
