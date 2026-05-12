import os

import pytest
from haystack.dataclasses.document import Document
from haystack.errors import FilterError
from haystack.testing.document_store import (
    FilterDocumentsTest,
)

from haystack_integrations.document_stores.pinecone.filters import (
    _normalize_filters,
    _validate_filters,
)


def test_normalize_filters_rejects_non_dict():
    with pytest.raises(FilterError, match="Filters must be a dictionary"):
        _normalize_filters("not-a-dict")


@pytest.mark.parametrize(
    ("operator", "value", "expected"),
    [
        ("==", "foo", {"field": {"$eq": "foo"}}),
        ("!=", 5, {"field": {"$ne": 5}}),
        (">", 1.5, {"field": {"$gt": 1.5}}),
        (">=", 2, {"field": {"$gte": 2}}),
        ("<", 3, {"field": {"$lt": 3}}),
        ("<=", 4.2, {"field": {"$lte": 4.2}}),
        ("in", ["a", "b"], {"field": {"$in": ["a", "b"]}}),
        ("not in", [1, 2], {"field": {"$nin": [1, 2]}}),
    ],
)
def test_comparison_operators(operator, value, expected):
    condition = {"field": "field", "operator": operator, "value": value}
    assert _normalize_filters(condition) == expected


@pytest.mark.parametrize(
    ("operator", "value"),
    [
        (">", "not-a-number"),
        (">=", "not-a-number"),
        ("<", "not-a-number"),
        ("<=", "not-a-number"),
        ("==", [1, 2]),
        ("!=", [1, 2]),
    ],
)
def test_comparison_rejects_unsupported_value_types(operator, value):
    condition = {"field": "field", "operator": operator, "value": value}
    with pytest.raises(FilterError, match="Unsupported type"):
        _normalize_filters(condition)


@pytest.mark.parametrize(
    ("operator", "value", "match"),
    [
        ("in", "not-a-list", "must be a list"),
        ("not in", "not-a-list", "must be a list"),
        ("in", [{"nested": "dict"}], "Unsupported type"),
        ("not in", [{"nested": "dict"}], "Unsupported type"),
    ],
)
def test_in_and_not_in_errors(operator, value, match):
    with pytest.raises(FilterError, match=match):
        _normalize_filters({"field": "field", "operator": operator, "value": value})


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        ({"conditions": []}, "'operator' key missing"),
        ({"operator": "AND"}, "'conditions' key missing"),
        (
            {"operator": "XOR", "conditions": [{"field": "a", "operator": "==", "value": 1}]},
            "Unknown logical operator",
        ),
    ],
)
def test_logical_condition_errors(condition, match):
    with pytest.raises(FilterError, match=match):
        _normalize_filters(condition)


@pytest.mark.parametrize(
    ("condition", "match"),
    [
        ({"field": "a", "value": 1}, "'operator' key missing"),
        ({"field": "a", "operator": "=="}, "'value' key missing"),
    ],
)
def test_comparison_condition_errors(condition, match):
    with pytest.raises(FilterError, match=match):
        _normalize_filters(condition)


def test_meta_prefix_is_stripped():
    condition = {"field": "meta.category", "operator": "==", "value": "A"}
    assert _normalize_filters(condition) == {"category": {"$eq": "A"}}


def test_nested_logical_conditions_are_parsed():
    filters = {
        "operator": "AND",
        "conditions": [
            {
                "operator": "OR",
                "conditions": [
                    {"field": "a", "operator": "==", "value": 1},
                    {"field": "b", "operator": ">", "value": 2},
                ],
            },
        ],
    }
    assert _normalize_filters(filters) == {"$and": [{"$or": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}]}


def test_validate_filters_rejects_invalid_syntax():
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"foo": "bar"})


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set")
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]):
        for doc in received:
            # Pinecone seems to convert integers to floats (undocumented behavior)
            # We convert them back to integers to compare them
            if "number" in doc.meta:
                doc.meta["number"] = int(doc.meta["number"])

        # Lists comparison
        assert len(received) == len(expected)
        received.sort(key=lambda x: x.id)
        expected.sort(key=lambda x: x.id)
        for received_doc, expected_doc in zip(received, expected, strict=True):
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
