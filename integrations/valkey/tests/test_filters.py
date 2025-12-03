# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.valkey.filters import _normalize_filters, _validate_filters

filters_data = [
    # Basic TagField equality
    (
        {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": "news"}]},
        "(@meta_category:{news})",
    ),
    # Basic NumericField comparison
    (
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": 0.8}]},
        "(@meta_score:[0.8 +inf])",
    ),
    # Complex AND condition
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.score", "operator": ">=", "value": 0.5},
                {"field": "meta.priority", "operator": "<=", "value": 10},
            ],
        },
        "(@meta_category:{news} @meta_score:[0.5 +inf] @meta_priority:[-inf 10])",
    ),
    # OR condition
    (
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {"field": "meta.category", "operator": "==", "value": "sports"},
            ],
        },
        "(@meta_category:{news} | @meta_category:{sports})",
    ),
    # IN operator for TagField
    (
        {"operator": "AND", "conditions": [{"field": "meta.status", "operator": "in", "value": ["active", "pending"]}]},
        "(@meta_status:{active | pending})",
    ),
    # IN operator for NumericField
    (
        {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": "in", "value": [1, 2, 3]}]},
        "(@meta_priority:[1 1 | 2 2 | 3 3])",
    ),
    # NOT EQUAL for TagField
    (
        {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "!=", "value": "spam"}]},
        "(-@meta_category:{spam})",
    ),
    # NOT IN for TagField
    (
        {
            "operator": "AND",
            "conditions": [{"field": "meta.status", "operator": "not in", "value": ["deleted", "archived"]}],
        },
        "(-@meta_status:{deleted | archived})",
    ),
    # Nested conditions
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.category", "operator": "==", "value": "news"},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.priority", "operator": ">", "value": 5},
                        {"field": "meta.score", "operator": ">=", "value": 0.9},
                    ],
                },
            ],
        },
        "(@meta_category:{news} (@meta_priority:[(5 +inf] | @meta_score:[0.9 +inf]))",
    ),
    # Range queries
    (
        {
            "operator": "AND",
            "conditions": [
                {"field": "meta.timestamp", "operator": ">=", "value": 1609459200},
                {"field": "meta.timestamp", "operator": "<", "value": 1640995200},
            ],
        },
        "(@meta_timestamp:[1609459200 +inf] @meta_timestamp:[-inf (1640995200])",
    ),
]


@pytest.mark.parametrize("filters, expected", filters_data)
def test_normalize_filters(filters, expected):
    result = _normalize_filters(filters)
    assert result == expected


def test_normalize_filters_invalid_operator():
    with pytest.raises(FilterError, match="Unknown logical operator 'INVALID'"):
        _normalize_filters({"operator": "INVALID", "conditions": []})


def test_normalize_filters_malformed():
    # Missing operator
    with pytest.raises(FilterError, match="'operator' key missing"):
        _normalize_filters({"conditions": []})

    # Missing conditions
    with pytest.raises(FilterError, match="'conditions' key missing"):
        _normalize_filters({"operator": "AND"})

    # Missing comparison field
    with pytest.raises(FilterError, match="'conditions' key missing"):
        _normalize_filters({"operator": "AND", "conditions": [{"operator": "==", "value": "news"}]})

    # Missing comparison operator
    with pytest.raises(FilterError, match="'operator' key missing"):
        _normalize_filters({"operator": "AND", "conditions": [{"field": "meta.category", "value": "news"}]})

    # Missing comparison value
    with pytest.raises(FilterError, match="'value' key missing"):
        _normalize_filters({"operator": "AND", "conditions": [{"field": "meta.category", "operator": "=="}]})


def test_unsupported_field():
    with pytest.raises(FilterError, match="Field 'meta_unsupported' is not supported for filtering"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.unsupported", "operator": "==", "value": "test"}]}
        )


def test_unsupported_operator_for_tag_field():
    with pytest.raises(FilterError, match="Operator '>' not supported for tag field"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": ">", "value": "news"}]}
        )


def test_invalid_value_type_for_tag_field():
    with pytest.raises(FilterError, match="TagField 'meta_category' requires string value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "==", "value": 123}]}
        )


def test_invalid_value_type_for_numeric_field():
    with pytest.raises(FilterError, match="NumericField 'meta_score' requires numeric value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.score", "operator": ">=", "value": "invalid"}]}
        )


def test_invalid_in_operator_value():
    with pytest.raises(FilterError, match="'in' operator requires a list value"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "in", "value": "not_a_list"}]}
        )


def test_invalid_list_values_for_tag_field():
    with pytest.raises(FilterError, match="TagField 'meta_category' requires string values in list"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.category", "operator": "in", "value": ["valid", 123]}]}
        )


def test_invalid_list_values_for_numeric_field():
    with pytest.raises(FilterError, match="NumericField 'meta_priority' requires numeric values in list"):
        _normalize_filters(
            {"operator": "AND", "conditions": [{"field": "meta.priority", "operator": "in", "value": [1, "invalid"]}]}
        )


def test_validate_filters_valid():
    # Should not raise any exception
    _validate_filters({"operator": "AND", "conditions": []})
    _validate_filters(None)


def test_validate_filters_invalid():
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        _validate_filters({"invalid": "filter"})


def test_special_character_escaping():
    # Test that special characters are properly escaped
    result = _normalize_filters(
        {
            "operator": "AND",
            "conditions": [{"field": "meta.category", "operator": "==", "value": "test-value.with:special@chars"}],
        }
    )
    expected = "(@meta_category:{test\\-value\\.with\\:special\\@chars})"
    assert result == expected


def test_direct_comparison_condition():
    # Test single comparison condition without logical wrapper
    result = _normalize_filters({"field": "meta.category", "operator": "==", "value": "news"})
    assert result == "@meta_category:{news}"


def test_numeric_equality():
    result = _normalize_filters(
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": "==", "value": 0.5}]}
    )
    assert result == "(@meta_score:[0.5 0.5])"


def test_numeric_not_equal():
    result = _normalize_filters(
        {"operator": "AND", "conditions": [{"field": "meta.score", "operator": "!=", "value": 0.5}]}
    )
    assert result == "(-@meta_score:[0.5 0.5])"


def test_filters_must_be_dict():
    with pytest.raises(FilterError, match="Filters must be a dictionary"):
        _normalize_filters("invalid")
