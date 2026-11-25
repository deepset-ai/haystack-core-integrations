# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the Haystack → S3 Vectors filter conversion.

This is a pure function with real logic (operator mapping, type validation,
meta. prefix stripping, logical nesting) so unit testing is high signal.
"""

from dataclasses import replace

import pytest
from haystack.dataclasses import Document
from haystack.errors import FilterError
from haystack.testing.document_store import FilterDocumentsTest

from haystack_integrations.document_stores.amazon_s3_vectors.filters import _normalize_filters, _validate_filters


def test_comparison_operators():
    """All comparison operators produce the correct S3 Vectors filter."""
    cases = [
        ({"field": "meta.x", "operator": "==", "value": "a"}, {"x": {"$eq": "a"}}),
        ({"field": "meta.x", "operator": "!=", "value": "a"}, {"x": {"$ne": "a"}}),
        ({"field": "meta.x", "operator": ">", "value": 1}, {"x": {"$gt": 1}}),
        ({"field": "meta.x", "operator": ">=", "value": 1}, {"x": {"$gte": 1}}),
        ({"field": "meta.x", "operator": "<", "value": 1.5}, {"x": {"$lt": 1.5}}),
        ({"field": "meta.x", "operator": "<=", "value": 1.5}, {"x": {"$lte": 1.5}}),
        ({"field": "meta.x", "operator": "in", "value": [1, 2]}, {"x": {"$in": [1, 2]}}),
        ({"field": "meta.x", "operator": "not in", "value": ["a"]}, {"x": {"$nin": ["a"]}}),
    ]
    for haystack_filter, expected in cases:
        assert _normalize_filters(haystack_filter) == expected


@pytest.mark.parametrize(("operator", "expected_operator"), [("AND", "$and"), ("OR", "$or")])
def test_logical_operators(operator, expected_operator):
    filters = {
        "operator": operator,
        "conditions": [
            {"field": "meta.a", "operator": "==", "value": 1},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.b", "operator": ">", "value": 2},
                    {"field": "meta.c", "operator": "==", "value": "x"},
                ],
            },
        ],
    }
    assert _normalize_filters(filters) == {
        expected_operator: [{"a": {"$eq": 1}}, {"$or": [{"b": {"$gt": 2}}, {"c": {"$eq": "x"}}]}]
    }


def test_meta_prefix_stripped():
    """The meta. prefix is stripped since S3 Vectors stores metadata flat."""
    assert _normalize_filters({"field": "meta.category", "operator": "==", "value": "news"}) == {
        "category": {"$eq": "news"}
    }
    # Fields without meta. prefix also work
    assert _normalize_filters({"field": "category", "operator": "==", "value": "news"}) == {"category": {"$eq": "news"}}


@pytest.mark.parametrize(
    ("filters", "error"),
    [
        ("not a dict", "Filters must be a dictionary"),
        ({"field": "meta.x", "value": "a"}, "'operator' key missing"),
        ({"field": "meta.x", "operator": "=="}, "'value' key missing"),
        ({"conditions": []}, "'operator' key missing"),
        ({"operator": "AND"}, "'conditions' key missing"),
        ({"operator": "XOR", "conditions": []}, "Unknown logical operator"),
        ({"field": "meta.x", "operator": "unknown", "value": 1}, "Unknown comparison operator"),
        ({"field": "meta.x", "operator": "==", "value": [1]}, "Unsupported type"),
        ({"field": "meta.x", "operator": ">", "value": "text"}, "Unsupported type"),
        ({"field": "meta.x", "operator": "in", "value": "text"}, "must be a list"),
        ({"field": "meta.x", "operator": "not in", "value": "text"}, "must be a list"),
        ({"field": "meta.x", "operator": "in", "value": [{}]}, "Unsupported type"),
    ],
)
def test_invalid_filters(filters, error):
    with pytest.raises(FilterError, match=error):
        _normalize_filters(filters)


def test_validate_filters():
    _validate_filters(None)  # None is valid
    _validate_filters({"operator": "AND", "conditions": []})  # valid structure
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"field": "meta.x"})  # missing operator/conditions


@pytest.mark.integration
class TestFilters(FilterDocumentsTest):
    def assert_documents_are_equal(self, received: list[Document], expected: list[Document]) -> None:
        assert len(received) == len(expected)
        for r, e in zip(sorted(received, key=lambda d: d.id), sorted(expected, key=lambda d: d.id), strict=True):
            assert replace(r, embedding=None, score=None) == replace(e, embedding=None, score=None)
            assert r.embedding == pytest.approx(e.embedding, abs=1e-5)
