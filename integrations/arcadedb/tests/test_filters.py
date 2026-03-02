# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for filter conversion (no ArcadeDB instance required)."""

import pytest

from haystack_integrations.document_stores.arcadedb.filters import _convert_filters


class TestFilterConversion:
    def test_none_returns_empty(self):
        assert _convert_filters(None) == ""

    def test_equality(self):
        result = _convert_filters({"field": "meta.name", "operator": "==", "value": "alice"})
        assert result == "meta.name = 'alice'"

    def test_equality_null(self):
        result = _convert_filters({"field": "meta.name", "operator": "==", "value": None})
        assert result == "meta.name IS NULL"

    def test_not_equal(self):
        result = _convert_filters({"field": "meta.name", "operator": "!=", "value": "bob"})
        assert result == "meta.name <> 'bob'"

    def test_not_equal_null(self):
        result = _convert_filters({"field": "meta.name", "operator": "!=", "value": None})
        assert result == "meta.name IS NOT NULL"

    def test_greater_than(self):
        result = _convert_filters({"field": "meta.score", "operator": ">", "value": 5})
        assert result == "meta.score > 5"

    def test_in_operator(self):
        result = _convert_filters({"field": "meta.tag", "operator": "in", "value": ["a", "b"]})
        assert result == "meta.tag IN ['a', 'b']"

    def test_not_in_operator(self):
        result = _convert_filters({"field": "meta.tag", "operator": "not in", "value": ["x"]})
        assert result == "meta.tag NOT IN ['x']"

    def test_and(self):
        result = _convert_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.a", "operator": "==", "value": 1},
                    {"field": "meta.b", "operator": ">", "value": 2},
                ],
            }
        )
        assert result == "(meta.a = 1 AND meta.b > 2)"

    def test_or(self):
        result = _convert_filters(
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.x", "operator": "==", "value": "yes"},
                    {"field": "meta.y", "operator": "==", "value": "no"},
                ],
            }
        )
        assert result == "(meta.x = 'yes' OR meta.y = 'no')"

    def test_not(self):
        result = _convert_filters(
            {
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.deleted", "operator": "==", "value": True},
                ],
            }
        )
        assert result == "NOT (meta.deleted = true)"

    def test_nested(self):
        result = _convert_filters(
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.a", "operator": "==", "value": 1},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.b", "operator": "==", "value": 2},
                            {"field": "meta.c", "operator": "==", "value": 3},
                        ],
                    },
                ],
            }
        )
        assert result == "(meta.a = 1 AND (meta.b = 2 OR meta.c = 3))"

    def test_missing_operator_raises(self):
        with pytest.raises(ValueError):
            _convert_filters({"field": "x", "value": 1})

    def test_missing_field_raises(self):
        with pytest.raises(ValueError):
            _convert_filters({"operator": "==", "value": 1})
