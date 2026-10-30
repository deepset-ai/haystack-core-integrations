# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack import Document
from haystack.testing.document_store import FilterDocumentsTest
from haystack.utils.filters import FilterError
from qdrant_client.http import models

from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.document_stores.qdrant.filters import (
    convert_filters_to_qdrant,
    is_datetime_string,
)


class TestConvertFiltersToQdrantUnit:
    def test_native_filter_passthrough(self):
        native = models.Filter(must=[])
        assert convert_filters_to_qdrant(native) is native

    def test_none_or_empty_returns_none(self):
        assert convert_filters_to_qdrant(None) is None
        assert convert_filters_to_qdrant({}) is None
        assert convert_filters_to_qdrant([]) is None

    @pytest.mark.parametrize(
        ("filter_input", "match"),
        [
            ({"field": "meta.x", "value": 1}, "Operator not found"),
            ({"operator": "~=", "field": "meta.x", "value": 1}, "Unknown operator"),
            ({"operator": "AND"}, "'conditions' not found"),
            ({"operator": "==", "value": 1}, "'field' or 'value' not found"),
            ({"operator": "==", "field": "meta.x"}, "'field' or 'value' not found"),
            ({"operator": "in", "field": "meta.x", "value": "not-a-list"}, "is not a list"),
            ({"operator": "not in", "field": "meta.x", "value": "not-a-list"}, "is not a list"),
        ],
    )
    def test_invalid_filter_raises(self, filter_input, match):
        with pytest.raises(FilterError, match=match):
            convert_filters_to_qdrant(filter_input)

    @pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
    def test_range_operators_reject_non_numeric_non_datetime(self, operator):
        with pytest.raises(FilterError, match="not an int or float or datetime string"):
            convert_filters_to_qdrant({"operator": operator, "field": "meta.x", "value": "not-a-date"})

    @pytest.mark.parametrize("operator", ["<", "<=", ">", ">="])
    def test_range_operators_accept_datetime_strings(self, operator):
        qdrant_filter = convert_filters_to_qdrant(
            {"operator": operator, "field": "meta.created_at", "value": "2024-01-01T00:00:00"}
        )
        assert isinstance(qdrant_filter, models.Filter)

    def test_single_logical_condition_unwrapped(self):
        qdrant_filter = convert_filters_to_qdrant(
            {
                "operator": "AND",
                "conditions": [{"operator": "==", "field": "meta.x", "value": 1}],
            }
        )
        # AND of single condition gets returned directly as the inner Filter
        assert isinstance(qdrant_filter, models.Filter)

    def test_multiple_top_level_conditions_combined_with_and(self):
        qdrant_filter = convert_filters_to_qdrant(
            [
                {"operator": "==", "field": "meta.a", "value": 1},
                {"operator": "==", "field": "meta.b", "value": 2},
            ]
        )
        assert isinstance(qdrant_filter, models.Filter)
        assert len(qdrant_filter.must) == 2

    @pytest.mark.parametrize("value", [4, 4.5, -3, 2**53])
    def test_eq_numeric_uses_closed_range(self, value):
        qdrant_filter = convert_filters_to_qdrant({"operator": "==", "field": "meta.x", "value": value})
        condition = qdrant_filter.must[0]
        assert condition.match is None
        assert condition.range == models.Range(gte=value, lte=value)

    @pytest.mark.parametrize("value", [True, False, "a", 2**53 + 1, -(2**60)])
    def test_eq_non_numeric_or_unsafe_int_keeps_match_value(self, value):
        qdrant_filter = convert_filters_to_qdrant({"operator": "==", "field": "meta.x", "value": value})
        condition = qdrant_filter.must[0]
        assert condition.range is None
        assert condition.match == models.MatchValue(value=value)
        assert type(condition.match.value) is type(value)

    def test_in_and_not_in_mixed_numerics(self):
        values = [1, 2.5, "a", True]
        in_filter = convert_filters_to_qdrant({"operator": "in", "field": "meta.x", "value": values})
        nin_filter = convert_filters_to_qdrant({"operator": "not in", "field": "meta.x", "value": values})
        for conditions in (in_filter.should, nin_filter.must_not):
            assert [(c.range, c.match) for c in conditions] == [
                (models.Range(gte=1, lte=1), None),
                (models.Range(gte=2.5, lte=2.5), None),
                (None, models.MatchValue(value="a")),
                (None, models.MatchValue(value=True)),
            ]

    def test_ne_float(self):
        qdrant_filter = convert_filters_to_qdrant({"operator": "!=", "field": "meta.x", "value": 4.5})
        assert qdrant_filter.must_not[0].range == models.Range(gte=4.5, lte=4.5)


def test_is_datetime_string():
    assert is_datetime_string("2024-01-01T00:00:00") is True
    assert is_datetime_string("not a date") is False


@pytest.mark.integration
class TestQdrantFilters(FilterDocumentsTest):
    @pytest.fixture
    def document_store(self) -> QdrantDocumentStore:
        return QdrantDocumentStore(
            ":memory:",
            recreate_index=True,
            return_embedding=True,
            wait_result_from_api=True,
        )

    @pytest.mark.parametrize(("value", "longer_value"), [("news", "newsletter"), ("New York", "New York City")])
    def test_eq_matches_exact_string(self, document_store, value, longer_value):
        exact_document = Document(content=value, meta={"name": value})
        other_document = Document(content=longer_value, meta={"name": longer_value})
        document_store.write_documents([exact_document, other_document])

        result = document_store.filter_documents(filters={"field": "meta.name", "operator": "==", "value": value})

        self.assert_documents_are_equal(result, [exact_document])

    @pytest.mark.parametrize(("value", "longer_value"), [("news", "newsletter"), ("New York", "New York City")])
    def test_ne_excludes_exact_string(self, document_store, value, longer_value):
        exact_document = Document(content=value, meta={"name": value})
        other_document = Document(content=longer_value, meta={"name": longer_value})
        document_store.write_documents([exact_document, other_document])

        result = document_store.filter_documents(filters={"field": "meta.name", "operator": "!=", "value": value})

        self.assert_documents_are_equal(result, [other_document])

    def test_in_matches_exact_strings(self, document_store):
        document_store.write_documents(
            [
                Document(content=name, meta={"name": name})
                for name in ["news", "newsletter", "New York", "New York City"]
            ]
        )

        result = document_store.filter_documents(
            filters={"field": "meta.name", "operator": "in", "value": ["news", "New York"]}
        )

        assert sorted(doc.meta["name"] for doc in result) == ["New York", "news"]

    def test_not_in_excludes_exact_strings(self, document_store):
        document_store.write_documents(
            [
                Document(content=name, meta={"name": name})
                for name in ["news", "newsletter", "New York", "New York City"]
            ]
        )

        result = document_store.filter_documents(
            filters={"field": "meta.name", "operator": "not in", "value": ["news", "New York"]}
        )

        assert sorted(doc.meta["name"] for doc in result) == ["New York City", "newsletter"]

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

    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
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

    @pytest.mark.skip(reason="Cannot distinguish errors yet")
    def test_missing_top_level_operator_key(self, document_store, filterable_docs): ...

    @pytest.mark.parametrize(
        ("operator", "value", "expected"),
        [
            ("==", 4, [4, 4.0]),
            ("==", 4.5, [4.5]),
            ("==", 4.0, [4, 4.0]),
            ("!=", 4.5, [3, 4, 4.0]),
            ("!=", 4, [3, 4.5]),
            ("in", [3, 4.5], [3, 4.5]),
            ("in", [4.0, 9], [4, 4.0]),
            ("not in", [3, 4.5], [4, 4.0]),
            ("not in", [4, 9.5], [3, 4.5]),
        ],
    )
    def test_numeric_filters_match_int_and_float_payloads(self, document_store, operator, value, expected):
        ratings = [3, 4, 4.0, 4.5]
        document_store.write_documents(
            [Document(content=str(r), meta={"rating": r}, embedding=[0.1] * 768) for r in ratings]
        )

        result = document_store.filter_documents({"field": "meta.rating", "operator": operator, "value": value})

        assert sorted(d.meta["rating"] for d in result) == sorted(expected)

    def test_eq_bool_does_not_match_numeric_payload(self, document_store):
        document_store.write_documents(
            [
                Document(content="int", meta={"flag": 1}, embedding=[0.1] * 768),
                Document(content="bool", meta={"flag": True}, embedding=[0.1] * 768),
            ]
        )

        result = document_store.filter_documents({"field": "meta.flag", "operator": "==", "value": True})

        assert [d.content for d in result] == ["bool"]
