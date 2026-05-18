# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.memory_stores.mem0.filters import normalize_filters


class TestNormalizeFilters:
    @pytest.mark.parametrize(
        "filters,expected",
        [
            (
                {"field": "score", "operator": ">=", "value": 0.8},
                {"score": {"gte": 0.8}},
            ),
            (
                {"field": "tag", "operator": "==", "value": "python"},
                {"tag": "python"},
            ),
            (
                {"field": "tag", "operator": "!=", "value": "java"},
                {"tag": {"ne": "java"}},
            ),
            (
                {"field": "cat", "operator": "in", "value": ["ml", "nlp"]},
                {"cat": {"in": ["ml", "nlp"]}},
            ),
            (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "user_id", "operator": "==", "value": "u1"},
                        {"field": "score", "operator": ">", "value": 0.5},
                    ],
                },
                {"AND": [{"user_id": "u1"}, {"score": {"gt": 0.5}}]},
            ),
        ],
    )
    def test_normalize_filters(self, filters, expected):
        assert normalize_filters(filters) == expected

    def test_unsupported_comparison_operator_raises_filter_error(self):
        with pytest.raises(FilterError, match="Unsupported filter operator"):
            normalize_filters({"field": "x", "operator": "LIKE", "value": "foo"})

    def test_unsupported_logical_operator_raises_filter_error(self):
        with pytest.raises(FilterError, match="Unsupported logical operator"):
            normalize_filters({"operator": "XOR", "conditions": []})

    def test_not_a_dict_raises_filter_error(self):
        with pytest.raises(FilterError, match="Filters must be a dictionary"):
            normalize_filters("not a dict")  # type: ignore[arg-type]
