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
    def test_none_filters_pass(self):
        _validate_filters(None)

    def test_dict_without_operator_raises(self):
        with pytest.raises(ValueError, match="Invalid filter syntax"):
            _validate_filters({"unknown_key": "value"})

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _validate_filters("not a dict")  # type: ignore[arg-type]

    def test_valid_filter_passes(self):
        _validate_filters({"field": "meta.x", "operator": "==", "value": 1})


class TestFilterConversion:
    def test_equal_meta_string(self):
        filters = {"field": "meta.name", "operator": "==", "value": "Alice"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "JSON_UNQUOTE" in clause
        assert "= ?" in clause
        assert params == ["Alice"]

    def test_equal_meta_int(self):
        filters = {"field": "meta.age", "operator": "==", "value": 30}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "CAST" in clause
        assert "SIGNED" in clause
        assert params == [30]

    def test_equal_meta_float(self):
        filters = {"field": "meta.score", "operator": "==", "value": 0.9}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "DECIMAL" in clause
        assert params == [0.9]

    def test_equal_none(self):
        filters = {"field": "meta.x", "operator": "==", "value": None}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "IS NULL" in clause
        assert params == []

    def test_not_equal(self):
        filters = {"field": "meta.x", "operator": "!=", "value": "foo"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "!= ?" in clause
        assert params == ["foo"]

    def test_greater_than(self):
        filters = {"field": "meta.count", "operator": ">", "value": 5}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "> ?" in clause
        assert params == [5]

    def test_less_than(self):
        filters = {"field": "meta.count", "operator": "<", "value": 100}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "< ?" in clause

    def test_in_operator(self):
        filters = {"field": "meta.tag", "operator": "in", "value": ["a", "b", "c"]}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "IN (?, ?, ?)" in clause
        assert params == ["a", "b", "c"]

    def test_not_in_operator(self):
        filters = {"field": "meta.tag", "operator": "not in", "value": ["x", "y"]}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "NOT IN" in clause
        assert params == ["x", "y"]

    def test_like_operator(self):
        filters = {"field": "meta.title", "operator": "like", "value": "%python%"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "LIKE ?" in clause
        assert params == ["%python%"]

    def test_not_like_operator(self):
        filters = {"field": "meta.title", "operator": "not like", "value": "%java%"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "NOT LIKE ?" in clause

    def test_top_level_field(self):
        filters = {"field": "id", "operator": "==", "value": "doc-1"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "`id`" in clause
        assert params == ["doc-1"]

    def test_and_logical(self):
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.age", "operator": ">=", "value": 18},
                {"field": "meta.active", "operator": "==", "value": "true"},
            ],
        }
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "AND" in clause
        assert len(params) == 2

    def test_or_logical(self):
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "meta.tag", "operator": "==", "value": "ai"},
                {"field": "meta.tag", "operator": "==", "value": "ml"},
            ],
        }
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "OR" in clause
        assert len(params) == 2

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
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "AND" in clause
        assert "OR" in clause
        assert len(params) == 3

    def test_where_operator(self):
        filters = {"field": "meta.x", "operator": "==", "value": 1}
        clause, _ = _convert_filters_to_where_clause_and_params(filters, operator="WHERE")
        assert clause.startswith(" WHERE ")

    def test_and_operator(self):
        filters = {"field": "meta.x", "operator": "==", "value": 1}
        clause, _ = _convert_filters_to_where_clause_and_params(filters, operator="AND")
        assert clause.startswith(" AND ")

    def test_invalid_operator_raises(self):
        filters = {"field": "meta.x", "operator": "INVALID", "value": 1}
        with pytest.raises(FilterError, match="Unknown comparison operator"):
            _convert_filters_to_where_clause_and_params(filters)

    def test_in_non_list_raises(self):
        filters = {"field": "meta.x", "operator": "in", "value": "not-a-list"}
        with pytest.raises(FilterError):
            _convert_filters_to_where_clause_and_params(filters)

    def test_gt_with_string_non_date_raises(self):
        filters = {"field": "meta.x", "operator": ">", "value": "notadate"}
        with pytest.raises(FilterError):
            _convert_filters_to_where_clause_and_params(filters)

    def test_gt_with_iso_date_passes(self):
        filters = {"field": "meta.created", "operator": ">", "value": "2024-01-01T00:00:00"}
        clause, params = _convert_filters_to_where_clause_and_params(filters)
        assert "> ?" in clause
