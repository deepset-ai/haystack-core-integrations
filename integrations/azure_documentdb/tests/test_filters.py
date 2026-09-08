# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.azure_documentdb.filters import _normalize_filters


def test_normalizes_comparison_filter():
    assert _normalize_filters({"field": "meta.year", "operator": ">=", "value": 2024}) == {"meta.year": {"$gte": 2024}}


def test_normalizes_nested_logical_filter():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "meta.kind", "operator": "==", "value": "guide"},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.year", "operator": ">", "value": 2024},
                    {"field": "meta.year", "operator": "==", "value": 2024},
                ],
            },
        ],
    }
    assert _normalize_filters(filters) == {
        "$and": [
            {"meta.kind": {"$eq": "guide"}},
            {"$or": [{"meta.year": {"$gt": 2024}}, {"meta.year": {"$eq": 2024}}]},
        ]
    }


def test_membership_requires_list():
    with pytest.raises(FilterError, match="must be a list"):
        _normalize_filters({"field": "meta.kind", "operator": "in", "value": "guide"})


def test_unknown_operator():
    with pytest.raises(FilterError, match="Unknown comparison operator"):
        _normalize_filters({"field": "meta.kind", "operator": "contains", "value": "guide"})


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        ("!=", {"meta.year": {"$ne": 2024}}),
        ("<", {"meta.year": {"$lt": 2024}}),
        ("<=", {"meta.year": {"$lte": 2024}}),
        ("in", {"meta.year": {"$in": [2024, 2025]}}),
        ("not in", {"meta.year": {"$nin": [2024, 2025]}}),
    ],
)
def test_normalizes_comparison_operators(operator, expected):
    value = [2024, 2025] if operator in {"in", "not in"} else 2024
    assert _normalize_filters({"field": "meta.year", "operator": operator, "value": value}) == expected


def test_normalizes_not_filter():
    assert _normalize_filters(
        {"operator": "NOT", "conditions": [{"field": "meta.kind", "operator": "==", "value": "guide"}]}
    ) == {"$nor": [{"$and": [{"meta.kind": {"$eq": "guide"}}]}]}


@pytest.mark.parametrize(
    "filters",
    [
        "not-a-dictionary",
        {"conditions": []},
        {"operator": "AND"},
        {"operator": "XOR", "conditions": []},
        {"field": "meta.kind", "value": "guide"},
        {"field": "meta.kind", "operator": "=="},
    ],
)
def test_rejects_malformed_filters(filters):
    with pytest.raises(FilterError):
        _normalize_filters(filters)


@pytest.mark.parametrize("value", [[2024], "not-a-date"])
def test_ordered_comparison_rejects_unsupported_values(value):
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.year", "operator": ">", "value": value})


@pytest.mark.parametrize(
    ("operator", "expected_operator"),
    [(">=", "$gt"), ("<=", "$lt")],
)
def test_null_inclusive_comparison_is_normalized(operator, expected_operator):
    assert _normalize_filters({"field": "meta.year", "operator": operator, "value": None}) == {
        "meta.year": {expected_operator: None}
    }
