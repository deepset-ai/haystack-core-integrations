# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.mariadb.filters import (
    _convert_filters_to_where_clause_and_params,
    _validate_filters,
)


class TestValidateFilters:
    @pytest.mark.parametrize("filters", [None, {"field": "meta.x", "operator": "==", "value": 1}])
    def test_valid_inputs_pass(self, filters):
        _validate_filters(filters)

    @pytest.mark.parametrize(
        ("filters", "exc"),
        [("not a dict", TypeError), ({"unknown_key": "value"}, ValueError)],
    )
    def test_invalid_inputs_raise(self, filters, exc):
        with pytest.raises(exc):
            _validate_filters(filters)


class TestFilterConversion:
    @pytest.mark.parametrize(
        ("value", "clause", "params"),
        [
            ("Alice", " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) = ?", ["Alice"]),
            (30, " WHERE CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) AS SIGNED) = ?", [30]),
            (0.9, " WHERE CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) AS DECIMAL(65,30)) = ?", [0.9]),
            (True, " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) = ?", ["true"]),
            (False, " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) = ?", ["false"]),
            (None, " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) IS NULL", []),
        ],
    )
    def test_equal_meta_by_type(self, value, clause, params):
        assert _convert_filters_to_where_clause_and_params({"field": "meta.x", "operator": "==", "value": value}) == (
            clause,
            params,
        )

    @pytest.mark.parametrize(
        ("operator", "value", "clause", "params"),
        [
            (
                "!=",
                "foo",
                (
                    " WHERE (JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) IS NULL "
                    "OR JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) != ?)"
                ),
                ["foo"],
            ),
            (">", 5, " WHERE CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) AS SIGNED) > ?", [5]),
            ("<", 100, " WHERE CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) AS SIGNED) < ?", [100]),
            ("like", "%python%", " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) LIKE ?", ["%python%"]),
            ("not like", "%java%", " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) NOT LIKE ?", ["%java%"]),
        ],
    )
    def test_scalar_operators(self, operator, value, clause, params):
        assert _convert_filters_to_where_clause_and_params(
            {"field": "meta.x", "operator": operator, "value": value}
        ) == (clause, params)

    @pytest.mark.parametrize(
        ("operator", "value", "clause", "params"),
        [
            ("in", ["a", "b", "c"], " WHERE JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) IN (?, ?, ?)", ["a", "b", "c"]),
            (
                "in",
                [1.5, 2.5],
                " WHERE CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) AS DECIMAL(65,30)) IN (?, ?)",
                [1.5, 2.5],
            ),
            (
                "not in",
                ["x", "y"],
                (
                    " WHERE (JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) IS NULL "
                    "OR JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) NOT IN (?, ?))"
                ),
                ["x", "y"],
            ),
        ],
    )
    def test_in_operators(self, operator, value, clause, params):
        assert _convert_filters_to_where_clause_and_params(
            {"field": "meta.tag", "operator": operator, "value": value}
        ) == (clause, params)

    def test_top_level_field(self):
        assert _convert_filters_to_where_clause_and_params({"field": "id", "operator": "==", "value": "doc-1"}) == (
            " WHERE `id` = ?",
            ["doc-1"],
        )

    def test_nested_logical(self):
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.x", "operator": ">", "value": 0},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.tag", "operator": "==", "value": "a"},
                        {"field": "meta.tag", "operator": "==", "value": "b"},
                    ],
                },
            ],
        }
        assert _convert_filters_to_where_clause_and_params(filters) == (
            (
                " WHERE (CAST(JSON_UNQUOTE(JSON_EXTRACT(meta, '$.x')) AS SIGNED) > ? "
                "AND (JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) = ? "
                "OR JSON_UNQUOTE(JSON_EXTRACT(meta, '$.tag')) = ?))"
            ),
            [0, "a", "b"],
        )

    @pytest.mark.parametrize(("operator", "prefix"), [("WHERE", " WHERE "), ("AND", " AND ")])
    def test_operator_prefix(self, operator, prefix):
        clause, _ = _convert_filters_to_where_clause_and_params(
            {"field": "meta.x", "operator": "==", "value": 1}, operator=operator
        )
        assert clause.startswith(prefix)

    @pytest.mark.parametrize(
        ("filters", "match"),
        [
            ({"field": "meta.x", "operator": "INVALID", "value": 1}, "Unknown comparison operator"),
            (
                {"operator": "XOR", "conditions": [{"field": "meta.x", "operator": "==", "value": 1}]},
                "Unknown logical operator",
            ),
            ({"field": "meta.bad; DROP TABLE t", "operator": "==", "value": 1}, "Invalid meta field name"),
            ({"field": "meta.x", "operator": "in", "value": "not-a-list"}, None),
            ({"field": "meta.x", "operator": "like", "value": 5}, None),
            ({"field": "meta.x", "operator": "not like", "value": 5}, None),
            ({"field": "meta.x", "operator": ">", "value": "notadate"}, None),
        ],
    )
    def test_invalid_filters_raise(self, filters, match):
        with pytest.raises(FilterError, match=match):
            _convert_filters_to_where_clause_and_params(filters)
