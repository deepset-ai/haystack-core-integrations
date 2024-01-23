from typing import List

import pytest
from haystack.dataclasses.document import Document
from haystack.testing.document_store import (
    FilterDocumentsTest
)


class TestFilters(FilterDocumentsTest):

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        # for doc in received:
        #     # Pinecone seems to convert strings to datetime objects (undocumented behavior)
        #     # We convert them back to strings to compare them
        #     if "date" in doc.meta:
        #         doc.meta["date"] = doc.meta["date"].isoformat()
        #     # Pinecone seems to convert integers to floats (undocumented behavior)
        #     # We convert them back to integers to compare them
        #     if "number" in doc.meta:
        #         doc.meta["number"] = int(doc.meta["number"])

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
            # unfortunately, Pinecone returns a slightly different embedding
            if received_doc.embedding is None:
                assert expected_doc.embedding is None
            else:
                assert received_doc.embedding == pytest.approx(expected_doc.embedding)
  

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
            result, [d for d in filterable_docs if 
                     (d.meta.get("number") == 100 and d.meta.get("name") == "name_0")
                        or (d.meta.get("page") == 90 and d.meta.get("chapter") == "conclusion")
                     ]
        )