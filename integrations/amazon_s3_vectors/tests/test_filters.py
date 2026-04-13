# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the Haystack → S3 Vectors filter conversion.

This is a pure function with real logic (operator mapping, type validation,
meta. prefix stripping, logical nesting) so unit testing is high signal.
"""

import pytest
from haystack.errors import FilterError

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


def test_logical_operators():
    """AND/OR produce $and/$or with nested conditions."""
    assert _normalize_filters(
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": 1},
                {"field": "meta.b", "operator": ">", "value": 2},
            ],
        }
    ) == {"$and": [{"a": {"$eq": 1}}, {"b": {"$gt": 2}}]}

    assert _normalize_filters(
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": "x"},
                {"field": "meta.a", "operator": "==", "value": "y"},
            ],
        }
    ) == {"$or": [{"a": {"$eq": "x"}}, {"a": {"$eq": "y"}}]}


def test_meta_prefix_stripped():
    """The meta. prefix is stripped since S3 Vectors stores metadata flat."""
    assert _normalize_filters({"field": "meta.category", "operator": "==", "value": "news"}) == {
        "category": {"$eq": "news"}
    }
    # Fields without meta. prefix also work
    assert _normalize_filters({"field": "category", "operator": "==", "value": "news"}) == {"category": {"$eq": "news"}}


def test_unsupported_type_raises():
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.x", "operator": "==", "value": [1, 2, 3]})


def test_in_requires_list():
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.x", "operator": "in", "value": "not-a-list"})


def test_missing_keys_raise():
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.x", "value": "a"})  # missing operator
    with pytest.raises(FilterError):
        _normalize_filters({"field": "meta.x", "operator": "=="})  # missing value
    with pytest.raises(FilterError):
        _normalize_filters("not a dict")


def test_validate_filters():
    _validate_filters(None)  # None is valid
    _validate_filters({"operator": "AND", "conditions": []})  # valid structure
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"field": "meta.x"})  # missing operator/conditions
