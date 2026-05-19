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
                {"field": "updated_at", "operator": "<", "value": "2026-06-01T00:00:00Z"},
                {"updated_at": {"lt": "2026-06-01T00:00:00Z"}},
            ),
            (
                {"field": "run_id", "operator": "!=", "value": "run-1"},
                {"run_id": {"ne": "run-1"}},
            ),
            (
                {"field": "tag", "operator": "==", "value": "python"},
                {"metadata": {"tag": "python"}},
            ),
            (
                {"field": "metadata.tag", "operator": "==", "value": "python"},
                {"metadata": {"tag": "python"}},
            ),
            (
                {"field": "tag", "operator": "!=", "value": "java"},
                {"metadata": {"tag": {"ne": "java"}}},
            ),
            (
                {"field": "description", "operator": "contains", "value": "meeting"},
                {"metadata": {"description": {"contains": "meeting"}}},
            ),
            (
                {"field": "categories", "operator": "in", "value": ["ml", "nlp"]},
                {"categories": {"in": ["ml", "nlp"]}},
            ),
            (
                {"field": "categories", "operator": "contains", "value": "finance"},
                {"categories": {"contains": "finance"}},
            ),
            (
                {"field": "keywords", "operator": "icontains", "value": "invoice"},
                {"keywords": {"icontains": "invoice"}},
            ),
            (
                {"field": "memory_ids", "operator": "==", "value": ["mem-1", "mem-2"]},
                {"memory_ids": ["mem-1", "mem-2"]},
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
            (
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "category", "operator": "==", "value": "work"},
                        {"field": "category", "operator": "==", "value": "personal"},
                    ],
                },
                {"OR": [{"metadata": {"category": "work"}}, {"metadata": {"category": "personal"}}]},
            ),
            (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "user_id", "operator": "==", "value": "u1"},
                        {
                            "operator": "OR",
                            "conditions": [
                                {"field": "categories", "operator": "in", "value": ["finance"]},
                                {"field": "metadata.source", "operator": "==", "value": "email"},
                            ],
                        },
                        {
                            "operator": "NOT",
                            "conditions": [
                                {"field": "metadata.status", "operator": "==", "value": "archived"},
                            ],
                        },
                    ],
                },
                {
                    "AND": [
                        {"user_id": "u1"},
                        {"OR": [{"categories": {"in": ["finance"]}}, {"metadata": {"source": "email"}}]},
                        {"NOT": [{"metadata": {"status": "archived"}}]},
                    ]
                },
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

    @pytest.mark.parametrize(
        "filters,match",
        [
            ({"conditions": []}, "'operator' key missing"),
            ({"operator": "AND"}, "'conditions' key missing"),
            ({"field": "tag", "operator": "=="}, "'value' key missing"),
            ({"field": "tag", "value": "python"}, "'operator' key missing"),
            ({"operator": "AND", "conditions": [{"operator": "==", "value": "python"}]}, "'conditions' key missing"),
        ],
    )
    def test_malformed_filters_raise_filter_error(self, filters, match):
        with pytest.raises(FilterError, match=match):
            normalize_filters(filters)


class TestBuildSearchFilters:
    def test_ids_only(self):
        assert _build_search_filters(user_id="u1", app_id="app1") == {"AND": [{"user_id": "u1"}, {"app_id": "app1"}]}

    def test_filters_only(self):
        filters = {"field": "user_id", "operator": "==", "value": "u1"}
        assert _build_search_filters(filters=filters) == {"user_id": "u1"}

    def test_metadata_filters_only(self):
        filters = {"field": "category", "operator": "==", "value": "work"}
        assert _build_search_filters(filters=filters) == {"metadata": {"category": "work"}}

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

    def test_combines_ids_with_or_filter(self):
        filters = {
            "operator": "OR",
            "conditions": [
                {"field": "category", "operator": "==", "value": "work"},
                {"field": "category", "operator": "==", "value": "personal"},
            ],
        }

        assert _build_search_filters(filters=filters, user_id="u1") == {
            "AND": [
                {"user_id": "u1"},
                {"OR": [{"metadata": {"category": "work"}}, {"metadata": {"category": "personal"}}]},
            ]
        }

    def test_raises_without_filters_or_ids(self):
        with pytest.raises(ValueError, match="Either filters or at least one"):
            _build_search_filters()
