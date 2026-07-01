# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.dakera.filters import _normalize_filters, _validate_filters


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


def test_strips_meta_prefix():
    condition = {"field": "meta.category", "operator": "==", "value": "x"}
    assert _normalize_filters(condition) == {"category": {"$eq": "x"}}


def test_logical_and():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.category", "operator": "==", "value": "x"},
            {"field": "meta.year", "operator": ">=", "value": 2020},
        ],
    }
    assert _normalize_filters(filters) == {"$and": [{"category": {"$eq": "x"}}, {"year": {"$gte": 2020}}]}


def test_logical_or():
    filters = {
        "operator": "OR",
        "conditions": [
            {"field": "a", "operator": "==", "value": 1},
            {"field": "b", "operator": "==", "value": 2},
        ],
    }
    assert _normalize_filters(filters) == {"$or": [{"a": {"$eq": 1}}, {"b": {"$eq": 2}}]}


def test_nested_logical():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.category", "operator": "==", "value": "x"},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.year", "operator": "==", "value": 2020},
                    {"field": "meta.year", "operator": "==", "value": 2021},
                ],
            },
        ],
    }
    assert _normalize_filters(filters) == {
        "$and": [
            {"category": {"$eq": "x"}},
            {"$or": [{"year": {"$eq": 2020}}, {"year": {"$eq": 2021}}]},
        ]
    }


def test_unknown_logical_operator():
    with pytest.raises(FilterError, match="Unknown logical operator"):
        _normalize_filters({"operator": "XOR", "conditions": []})


def test_unknown_comparison_operator():
    with pytest.raises(FilterError, match="Unknown comparison operator"):
        _normalize_filters({"field": "a", "operator": "~=", "value": 1})


def test_greater_than_rejects_non_numeric():
    with pytest.raises(FilterError, match="greater than"):
        _normalize_filters({"field": "a", "operator": ">", "value": "not-a-number"})


def test_greater_than_rejects_bool():
    with pytest.raises(FilterError, match="greater than"):
        _normalize_filters({"field": "a", "operator": ">", "value": True})


def test_in_requires_list():
    with pytest.raises(FilterError, match="must be a list"):
        _normalize_filters({"field": "a", "operator": "in", "value": "x"})


def test_missing_operator_key():
    with pytest.raises(FilterError, match="'operator' key missing"):
        _normalize_filters({"field": "a", "value": 1})


def test_missing_value_key():
    with pytest.raises(FilterError, match="'value' key missing"):
        _normalize_filters({"field": "a", "operator": "=="})


def test_validate_filters_rejects_malformed():
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"category": "x"})


def test_validate_filters_accepts_none():
    _validate_filters(None)
