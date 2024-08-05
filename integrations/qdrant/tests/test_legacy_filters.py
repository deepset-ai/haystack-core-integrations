from typing import List

import pytest
from haystack import Document
from haystack.document_stores.types import DocumentStore
from haystack.testing.document_store import LegacyFilterDocumentsTest
from haystack.utils.filters import FilterError
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

# The tests below are from haystack.testing.document_store.LegacyFilterDocumentsTest
# Updated to include `meta` prefix for filter keys wherever necessary
# And skip tests that are not supported in Qdrant(Dataframes, embeddings)


class TestQdrantLegacyFilterDocuments(LegacyFilterDocumentsTest):
    """
    Utility class to test a Document Store `filter_documents` method using different types of legacy filters
    """

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

    def test_filter_simple_metadata_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": "100"})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_filter_document_dataframe(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    def test_eq_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$eq": "100"}})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_eq_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": "100"})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_eq_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_eq_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsNotEqualTest

    def test_ne_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$ne": "100"}})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") != "100"])

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_ne_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_ne_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsInTest

    def test_filter_simple_list_single_element(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100"]})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") == "100"])

    def test_filter_simple_list_one_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100"]})
        self.assert_documents_are_equal(result, [doc for doc in filterable_docs if doc.meta.get("page") in ["100"]])

    def test_filter_simple_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100", "123"]})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    def test_incorrect_filter_value(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["nope"]})
        self.assert_documents_are_equal(result, [])

    def test_in_filter_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$in": ["100", "123", "n.a."]}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    def test_in_filter_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": ["100", "123", "n.a."]})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") in ["100", "123"]],
        )

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_in_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_in_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsNotInTest

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_nin_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_nin_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    def test_nin_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.page": {"$nin": ["100", "123", "n.a."]}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("page") not in ["100", "123"]],
        )

    # LegacyFilterDocumentsGreaterThanTest

    def test_gt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$gt": 0.0}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] > 0],
        )

    def test_gt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$gt": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_gt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_gt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsGreaterThanEqualTest

    def test_gte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$gte": -2}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if "number" in doc.meta and doc.meta["number"] >= -2],
        )

    def test_gte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$gte": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_gte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_gte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsLessThanTest

    def test_lt_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lt": 0.0}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("number") is not None and doc.meta["number"] < 0],
        )

    def test_lt_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$lt": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_lt_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_lt_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsLessThanEqualTest

    def test_lte_filter(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lte": 2.0}})
        self.assert_documents_are_equal(
            result,
            [doc for doc in filterable_docs if doc.meta.get("number") is not None and doc.meta["number"] <= 2.0],
        )

    def test_lte_filter_non_numeric(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        with pytest.raises(FilterError):
            document_store.filter_documents(filters={"meta.page": {"$lte": "100"}})

    @pytest.mark.skip(reason="Dataframe filtering is not supported in Qdrant")
    def test_lte_filter_table(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    @pytest.mark.skip(reason="Embedding filtering is not supported in Qdrant")
    def test_lte_filter_embedding(self, document_store: DocumentStore, filterable_docs: List[Document]): ...

    # LegacyFilterDocumentsSimpleLogicalTest

    def test_filter_simple_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": {
                "meta.name": {"$in": ["name_0", "name_1"]},
                "meta.number": {"$lt": 1.0},
            }
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                or doc.meta.get("name") in ["name_0", "name_1"]
            ],
        )

    def test_filter_simple_implicit_and_with_multi_key_dict(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lte": 2.0, "$gte": 0.0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] >= 0.0 and doc.meta["number"] <= 2.0
            ],
        )

    def test_filter_simple_explicit_and_with_list(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$and": [{"$lte": 2}, {"$gte": 0}]}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )

    def test_filter_simple_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(filters={"meta.number": {"$lte": 2.0, "$gte": 0}})
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if "number" in doc.meta and doc.meta["number"] <= 2.0 and doc.meta["number"] >= 0.0
            ],
        )

    # LegacyFilterDocumentsNestedLogicalTest(

    def test_filter_nested_implicit_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "meta.number": {"$lte": 2, "$gte": 0},
            "meta.name": ["name_0", "name_1"],
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    "number" in doc.meta
                    and doc.meta["number"] <= 2
                    and doc.meta["number"] >= 0
                    and doc.meta.get("name") in ["name_0", "name_1"]
                )
            ],
        )

    def test_filter_nested_or(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": {
                "meta.name": {"$in": ["name_0", "name_1"]},
                "meta.number": {"$lt": 1.0},
            }
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("name") in ["name_0", "name_1"]
                    or (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                )
            ],
        )

    def test_filter_nested_and_or_explicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$and": {
                "meta.page": {"$eq": "123"},
                "$or": {
                    "meta.name": {"$in": ["name_0", "name_1"]},
                    "meta.number": {"$lt": 1.0},
                },
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_and_or_implicit(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "meta.page": {"$eq": "123"},
            "$or": {
                "meta.name": {"$in": ["name_0", "name_1"]},
                "meta.number": {"$lt": 1.0},
            },
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    doc.meta.get("page") in ["123"]
                    and (
                        doc.meta.get("name") in ["name_0", "name_1"]
                        or ("number" in doc.meta and doc.meta["number"] < 1)
                    )
                )
            ],
        )

    def test_filter_nested_or_and(self, document_store: DocumentStore, filterable_docs: List[Document]):
        document_store.write_documents(filterable_docs)
        filters_simplified = {
            "$or": {
                "meta.number": {"$lt": 1},
                "$and": {
                    "meta.name": {"$in": ["name_0", "name_1"]},
                    "$not": {"meta.chapter": {"$eq": "intro"}},
                },
            }
        }
        result = document_store.filter_documents(filters=filters_simplified)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("number") is not None and doc.meta["number"] < 1)
                    or (doc.meta.get("name") in ["name_0", "name_1"] and (doc.meta.get("chapter") != "intro"))
                )
            ],
        )

    def test_filter_nested_multiple_identical_operators_same_level(
        self, document_store: DocumentStore, filterable_docs: List[Document]
    ):
        document_store.write_documents(filterable_docs)
        filters = {
            "$or": [
                {
                    "$and": {
                        "meta.name": {"$in": ["name_0", "name_1"]},
                        "meta.page": "100",
                    }
                },
                {
                    "$and": {
                        "meta.chapter": {"$in": ["intro", "abstract"]},
                        "meta.page": "123",
                    }
                },
            ]
        }
        result = document_store.filter_documents(filters=filters)
        self.assert_documents_are_equal(
            result,
            [
                doc
                for doc in filterable_docs
                if (
                    (doc.meta.get("name") in ["name_0", "name_1"] and doc.meta.get("page") == "100")
                    or (doc.meta.get("chapter") in ["intro", "abstract"] and doc.meta.get("page") == "123")
                )
            ],
        )

    def test_no_filter_not_empty(self, document_store: DocumentStore):
        docs = [Document(content="test doc")]
        document_store.write_documents(docs)
        self.assert_documents_are_equal(document_store.filter_documents(), docs)
        self.assert_documents_are_equal(document_store.filter_documents(filters={}), docs)
