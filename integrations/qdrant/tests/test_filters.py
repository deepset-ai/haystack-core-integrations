from typing import List

import pytest
from haystack import Document
from haystack.testing.document_store import FilterDocumentsTest
from haystack.utils.filters import FilterError
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from qdrant_client.http import models


class TestQdrantStoreBaseTests(FilterDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
        )

    def test_filter_documents_with_qdrant_filters(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters=models.Filter(
                must_not=[
                    models.FieldCondition(key="meta.number", match=models.MatchValue(value=100)),
                    models.FieldCondition(key="meta.name", match=models.MatchValue(value="name_0")),
                ]
            )
        )
        self.assert_documents_are_equal(
            result,
            [d for d in filterable_docs if (d.meta.get("number") != 100 and d.meta.get("name") != "name_0")],
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

    def test_filter_criteria(self, document_store):
        documents = [
            Document(
                content="This is test document 1.",
                meta={"file_name": "file1", "classification": {"details": {"category1": 0.9, "category2": 0.3}}},
            ),
            Document(
                content="This is test document 2.",
                meta={"file_name": "file2", "classification": {"details": {"category1": 0.1, "category2": 0.7}}},
            ),
            Document(
                content="This is test document 3.",
                meta={"file_name": "file3", "classification": {"details": {"category1": 0.7, "category2": 0.9}}},
            ),
        ]

        document_store.write_documents(documents)
        filter_criteria = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.file_name", "operator": "in", "value": ["file1", "file2"]},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.classification.details.category1", "operator": ">=", "value": 0.85},
                        {"field": "meta.classification.details.category2", "operator": ">=", "value": 0.85},
                    ],
                },
            ],
        }
        result = document_store.filter_documents(filter_criteria)
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in documents
                if (d.meta.get("file_name") in ["file1", "file2"])
                and (
                    (d.meta.get("classification").get("details").get("category1") >= 0.85)
                    or (d.meta.get("classification").get("details").get("category2") >= 0.85)
                )
            ],
        )

    def test_complex_filter_criteria(self, document_store):
        documents = [
            Document(
                content="This is test document 1.",
                meta={
                    "file_name": "file1",
                    "classification": {"details": {"category1": 0.45, "category2": 0.5, "category3": 0.2}},
                },
            ),
            Document(
                content="This is test document 2.",
                meta={
                    "file_name": "file2",
                    "classification": {"details": {"category1": 0.95, "category2": 0.85, "category3": 0.4}},
                },
            ),
            Document(
                content="This is test document 3.",
                meta={
                    "file_name": "file3",
                    "classification": {"details": {"category1": 0.85, "category2": 0.7, "category3": 0.95}},
                },
            ),
        ]

        document_store.write_documents(documents)
        filter_criteria = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.file_name", "operator": "in", "value": ["file1", "file2", "file3"]},
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.classification.details.category1", "operator": ">=", "value": 0.85},
                        {
                            "operator": "OR",
                            "conditions": [
                                {"field": "meta.classification.details.category2", "operator": ">=", "value": 0.8},
                                {"field": "meta.classification.details.category3", "operator": ">=", "value": 0.9},
                            ],
                        },
                    ],
                },
            ],
        }
        result = document_store.filter_documents(filter_criteria)
        self.assert_documents_are_equal(
            result,
            [
                d
                for d in documents
                if (d.meta.get("file_name") in ["file1", "file2", "file3"])
                and (
                    (d.meta.get("classification").get("details").get("category1") >= 0.85)
                    and (
                        (d.meta.get("classification").get("details").get("category2") >= 0.8)
                        or (d.meta.get("classification").get("details").get("category3") >= 0.9)
                    )
                )
            ],
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
    def test_comparison_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Qdrant doesn't support comparision with dataframe")
    def test_comparison_not_equal_with_dataframe(self, document_store, filterable_docs): ...

    @pytest.mark.skip(reason="Cannot distinguish errors yet")
    def test_missing_top_level_operator_key(self, document_store, filterable_docs): ...
