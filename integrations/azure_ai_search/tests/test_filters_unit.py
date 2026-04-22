# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack_integrations.document_stores.azure_ai_search.errors import AzureAISearchDocumentStoreFilterError
from haystack_integrations.document_stores.azure_ai_search.filters import _normalize_filters


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"field": "meta.name", "operator": "==", "value": "alice"}, "name eq 'alice'"),
        ({"field": "meta.active", "operator": "==", "value": True}, "active eq true"),
        ({"field": "meta.count", "operator": "==", "value": 3}, "count eq 3"),
        ({"field": "meta.name", "operator": "==", "value": None}, "name eq null"),
        ({"field": "meta.name", "operator": "!=", "value": "alice"}, "not (name eq 'alice')"),
        ({"field": "meta.active", "operator": "!=", "value": False}, "not (active eq false)"),
        ({"field": "meta.count", "operator": "!=", "value": 3}, "not (count eq 3)"),
        (
            {"field": "meta.page", "operator": "in", "value": ["1", "2"]},
            "search.in(page,'1,2',',')",
        ),
        ({"field": "meta.count", "operator": ">", "value": 5}, "count gt 5"),
        ({"field": "meta.count", "operator": ">=", "value": 5}, "count ge 5"),
        ({"field": "meta.count", "operator": "<", "value": 5}, "count lt 5"),
        ({"field": "meta.count", "operator": "<=", "value": 5}, "count le 5"),
        (
            {"field": "meta.date", "operator": ">", "value": "2020-01-01T00:00:00Z"},
            "date gt 2020-01-01T00:00:00Z",
        ),
        ({"field": "bare_field", "operator": "==", "value": "x"}, "bare_field eq 'x'"),
    ],
)
def test_normalize_filters_comparison_conditions(filters, expected):
    assert _normalize_filters(filters) == expected


@pytest.mark.parametrize(
    "filters, expected",
    [
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "alice"},
                    {"field": "meta.count", "operator": ">=", "value": 1},
                ],
            },
            "(name eq 'alice') and (count ge 1)",
        ),
        (
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "alice"},
                    {"field": "meta.name", "operator": "==", "value": "bob"},
                ],
            },
            "(name eq 'alice') or (name eq 'bob')",
        ),
        (
            {
                "operator": "NOT",
                "conditions": [{"field": "meta.name", "operator": "==", "value": "alice"}],
            },
            "not ((name eq 'alice'))",
        ),
        (
            {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "alice"},
                    {
                        "operator": "OR",
                        "conditions": [
                            {"field": "meta.count", "operator": ">", "value": 1},
                            {"field": "meta.count", "operator": "<", "value": 10},
                        ],
                    },
                ],
            },
            "(name eq 'alice') and ((count gt 1) or (count lt 10))",
        ),
    ],
)
def test_normalize_filters_logical_conditions(filters, expected):
    assert _normalize_filters(filters) == expected


@pytest.mark.parametrize(
    "filters, expected_match",
    [
        ("not a dict", "Filters must be a dictionary"),
        ({"operator": "AND"}, "Missing key"),
        ({"conditions": []}, "Missing key"),
        (
            {"operator": "XOR", "conditions": [{"field": "a", "operator": "==", "value": 1}]},
            "Unknown operator XOR",
        ),
        ({"field": "f"}, "Missing key"),
        ({"field": "f", "operator": "???", "value": 1}, "Unknown operator"),
        (
            {"field": "f", "operator": ">", "value": "not-a-date"},
            "Invalid value type",
        ),
        (
            {"field": "f", "operator": ">", "value": [1, 2]},
            "Invalid value type",
        ),
        (
            {"field": "f", "operator": "in", "value": "not-a-list"},
            "only supports a list of strings",
        ),
        (
            {"field": "f", "operator": "in", "value": [1, 2]},
            "only supports a list of strings",
        ),
    ],
)
def test_normalize_filters_raises_on_invalid_input(filters, expected_match):
    with pytest.raises(AzureAISearchDocumentStoreFilterError, match=expected_match):
        _normalize_filters(filters)
