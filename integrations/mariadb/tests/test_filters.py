# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.mariadb.filters import (
    _convert_filters_to_where_clause_and_params,
)


class TestComparisonBuilding:
    @pytest.mark.parametrize(("value", "expected_param"), [(True, "true"), (False, "false")])
    def test_boolean_equal_binds_json_text(self, value, expected_param):
        clause, params = _convert_filters_to_where_clause_and_params(
            {"field": "meta.flag", "operator": "==", "value": value}
        )
        assert "JSON_UNQUOTE" in clause
        assert params == [expected_param]

    @pytest.mark.parametrize(("value", "cast"), [([1, 2], "SIGNED"), ([1.5, 2.5], "DECIMAL")])
    def test_in_list_numeric_cast(self, value, cast):
        clause, params = _convert_filters_to_where_clause_and_params(
            {"field": "meta.n", "operator": "in", "value": value}
        )
        assert cast in clause
        assert params == value


class TestValidation:
    @pytest.mark.parametrize(
        "filters",
        [
            {"operator": "XOR", "conditions": [{"field": "meta.x", "operator": "==", "value": 1}]},
            {"field": "meta.bad; DROP TABLE t", "operator": "==", "value": 1},
            {"field": "meta.x", "operator": "like", "value": 5},
            {"field": "meta.x", "operator": "not like", "value": 5},
        ],
    )
    def test_invalid_filters_raise(self, filters):
        with pytest.raises(FilterError):
            _convert_filters_to_where_clause_and_params(filters)
