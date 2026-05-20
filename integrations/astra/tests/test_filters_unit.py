# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for filter conversion

Integration tests for filters are included in test_document_store.py
"""

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.astra.filters import (
    _convert_filters,
    _parse_comparison_condition,
    _parse_logical_condition,
)


class TestConvertFilters:
    @pytest.mark.parametrize("empty", [None, {}])
    def test_empty_returns_none(self, empty):
        assert _convert_filters(empty) is None

    @pytest.mark.parametrize(
        "filters,expected",
        [
            (
                {"field": "meta.year", "operator": "==", "value": 2024},
                {"meta.year": {"$eq": 2024}},
            ),
            (
                {"field": "meta.year", "operator": "!=", "value": 2024},
                {"meta.year": {"$ne": 2024}},
            ),
            (
                {"field": "meta.score", "operator": ">", "value": 1},
                {"meta.score": {"$gt": 1}},
            ),
            (
                {"field": "meta.score", "operator": ">=", "value": 1},
                {"meta.score": {"$gte": 1}},
            ),
            (
                {"field": "meta.score", "operator": "<", "value": 1},
                {"meta.score": {"$lt": 1}},
            ),
            (
                {"field": "meta.score", "operator": "<=", "value": 1},
                {"meta.score": {"$lte": 1}},
            ),
            (
                {"field": "meta.tag", "operator": "in", "value": ["a", "b"]},
                {"meta.tag": {"$in": ["a", "b"]}},
            ),
            (
                {"field": "meta.tag", "operator": "not in", "value": ["a", "b"]},
                {"meta.tag": {"$nin": ["a", "b"]}},
            ),
        ],
    )
    def test_comparison_operators(self, filters, expected):
        assert _convert_filters(filters) == expected

    def test_id_field_is_renamed_to_underscore_id(self):
        filters = {"field": "id", "operator": "==", "value": "abc"}
        assert _convert_filters(filters) == {"_id": {"$eq": "abc"}}

    def test_in_operator_with_non_list_raises(self):
        filters = {"field": "meta.tag", "operator": "in", "value": "not-a-list"}
        with pytest.raises(FilterError, match=r"\$in operator must have `ARRAY`"):
            _convert_filters(filters)

    def test_logical_and(self):
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": 1},
                {"field": "meta.b", "operator": "==", "value": 2},
            ],
        }
        result = _convert_filters(filters)
        assert result == {
            "$and": [
                {"meta.a": {"$eq": 1}},
                {"meta.b": {"$eq": 2}},
            ]
        }

    def test_logical_or(self):
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.a", "operator": "==", "value": 1},
                {"field": "meta.b", "operator": "==", "value": 2},
            ],
        }
        result = _convert_filters(filters)
        assert result == {
            "$or": [
                {"meta.a": {"$eq": 1}},
                {"meta.b": {"$eq": 2}},
            ]
        }

    def test_logical_not_negates_compound_clause(self):
        # `!= None` parses to a compound `{$exists: true, $ne: null}` clause; its
        # negation must split into a disjunction (NOT exists OR == null).
        filters = {
            "operator": "NOT",
            "conditions": [{"field": "meta.x", "operator": "!=", "value": None}],
        }
        result = _convert_filters(filters)
        assert result == {
            "$or": [
                {
                    "$or": [
                        {"meta.x": {"$exists": False}},
                        {"meta.x": {"$eq": None}},
                    ]
                }
            ]
        }


class TestNormalizeFilters:
    def test_non_dict_raises(self):
        with pytest.raises(FilterError, match="Filters must be a dictionary"):
            _convert_filters("not_a_dict")


class TestParseLogicalCondition:
    def test_missing_operator_raises(self):
        with pytest.raises(FilterError, match="'operator' key missing"):
            _parse_logical_condition({"conditions": []})

    def test_missing_conditions_raises(self):
        with pytest.raises(FilterError, match="'conditions' key missing"):
            _parse_logical_condition({"operator": "AND"})

    def test_unknown_operator_raises(self):
        with pytest.raises(FilterError, match="Unknown operator"):
            _parse_logical_condition(
                {
                    "operator": "XOR",
                    "conditions": [{"field": "a", "operator": "==", "value": 1}],
                }
            )


class TestParseComparisonCondition:
    @pytest.mark.parametrize(
        "condition,err",
        [
            ({"operator": "==", "value": 1}, "'field' key missing"),
            ({"field": "a", "value": 1}, "'operator' key missing"),
            ({"field": "a", "operator": "=="}, "'value' key missing"),
        ],
    )
    def test_missing_keys_raise(self, condition, err):
        with pytest.raises(FilterError, match=err):
            _parse_comparison_condition(condition)
