# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.memory_stores.mem0.filters import _build_search_filters, normalize_filters


class TestNormalizeFilters:
    @pytest.mark.parametrize(
        "filters,expected",
        [
            (
                {"field": "created_at", "operator": ">=", "value": "2026-05-01T00:00:00Z"},
                {"created_at": {"gte": "2026-05-01T00:00:00Z"}},
            ),
            (
                {"field": "tag", "operator": "==", "value": "python"},
                {"metadata": {"tag": "python"}},
            ),
            (
                {"field": "tag", "operator": "!=", "value": "java"},
                {"metadata": {"tag": {"ne": "java"}}},
            ),
            (
                {"field": "categories", "operator": "in", "value": ["ml", "nlp"]},
                {"categories": {"in": ["ml", "nlp"]}},
            ),
            (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "user_id", "operator": "==", "value": "u1"},
                        {"field": "created_at", "operator": ">", "value": "2026-05-01T00:00:00Z"},
                    ],
                },
                {"AND": [{"user_id": "u1"}, {"created_at": {"gt": "2026-05-01T00:00:00Z"}}]},
            ),
        ],
    )
    def test_normalize_filters(self, filters, expected):
        assert normalize_filters(filters) == expected

    def test_unsupported_comparison_operator_raises_filter_error(self):
        with pytest.raises(FilterError, match="Unsupported filter operator"):
            normalize_filters({"field": "categories", "operator": "LIKE", "value": "foo"})

    def test_unsupported_metadata_operator_raises_filter_error(self):
        with pytest.raises(FilterError, match="Unsupported metadata filter operator"):
            normalize_filters({"field": "tag", "operator": "in", "value": ["python", "java"]})

    def test_unsupported_logical_operator_raises_filter_error(self):
        with pytest.raises(FilterError, match="Unsupported logical operator"):
            normalize_filters({"operator": "XOR", "conditions": []})

    def test_not_a_dict_raises_filter_error(self):
        with pytest.raises(FilterError, match="Filters must be a dictionary"):
            normalize_filters("not a dict")  # type: ignore[arg-type]


class TestBuildSearchFilters:
    def test_ids_only(self):
        assert _build_search_filters(user_id="u1", app_id="app1") == {"AND": [{"user_id": "u1"}, {"app_id": "app1"}]}

    def test_filters_only(self):
        filters = {"field": "user_id", "operator": "==", "value": "u1"}
        assert _build_search_filters(filters=filters) == {"user_id": "u1"}

    def test_combines_ids_and_filters_at_haystack_filter_level(self):
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "tag", "operator": "==", "value": "work"},
                {"field": "created_at", "operator": ">=", "value": "2026-05-01T00:00:00Z"},
            ],
        }
        assert _build_search_filters(filters=filters, user_id="u1", app_id="app1") == {
            "AND": [
                {"user_id": "u1"},
                {"app_id": "app1"},
                {"metadata": {"tag": "work"}},
                {"created_at": {"gte": "2026-05-01T00:00:00Z"}},
            ]
        }

    def test_raises_without_filters_or_ids(self):
        with pytest.raises(ValueError, match="Either filters or at least one"):
            _build_search_filters()
