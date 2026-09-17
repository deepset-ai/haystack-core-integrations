# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import weaviate
from haystack.errors import FilterError
from weaviate.collections.classes.filters import _Operator

from haystack_integrations.document_stores.weaviate._filters import (
    _invert_condition,
    _parse_comparison_condition,
    _parse_logical_condition,
    convert_filters,
    validate_filters,
)


def test_invert_conditions():
    filters = {
        "operator": "NOT",
        "conditions": [
            {"field": "meta.number", "operator": "==", "value": 100},
            {"field": "meta.name", "operator": "==", "value": "name_0"},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.name", "operator": "==", "value": "name_1"},
                    {"field": "meta.name", "operator": "==", "value": "name_2"},
                ],
            },
        ],
    }

    inverted = _invert_condition(filters)
    assert inverted == {
        "operator": "OR",
        "conditions": [
            {"field": "meta.number", "operator": "!=", "value": 100},
            {"field": "meta.name", "operator": "!=", "value": "name_0"},
            {
                "conditions": [
                    {"field": "meta.name", "operator": "!=", "value": "name_1"},
                    {"field": "meta.name", "operator": "!=", "value": "name_2"},
                ],
                "operator": "AND",
            },
        ],
    }


def test_invert_condition_without_operator_returns_copy():
    filters = {"field": "meta.number", "value": 100}
    assert _invert_condition(filters) == filters


def test_convert_filters_raises_on_non_dict():
    with pytest.raises(FilterError, match="Filters must be a dictionary"):
        convert_filters([{"field": "meta.number", "operator": "==", "value": 1}])  # type: ignore[arg-type]


def test_parse_logical_condition_errors():
    with pytest.raises(FilterError, match="'operator' key missing"):
        _parse_logical_condition({"conditions": []})
    with pytest.raises(FilterError, match="'conditions' key missing"):
        _parse_logical_condition({"operator": "AND"})
    with pytest.raises(FilterError, match="Unknown logical operator"):
        _parse_logical_condition({"operator": "XOR", "conditions": []})


def test_parse_comparison_condition_errors():
    with pytest.raises(FilterError, match="'operator' key missing"):
        _parse_comparison_condition({"field": "meta.x", "value": 1})
    with pytest.raises(FilterError, match="'value' key missing"):
        _parse_comparison_condition({"field": "meta.x", "operator": "=="})


def test_parse_comparison_condition_unknown_operator():
    with pytest.raises(FilterError, match="Unknown comparison operator 'like'"):
        _parse_comparison_condition({"field": "meta.number", "operator": "like", "value": 100})


@pytest.mark.parametrize("filters", [None, {}, {"operator": "AND", "conditions": []}, {"conditions": []}])
def test_validate_filters_accepts_valid_input(filters):
    validate_filters(filters)


def test_validate_filters_rejects_filters_without_operator_or_conditions():
    with pytest.raises(ValueError, match="Invalid filter syntax"):
        validate_filters({"field": "meta.number", "value": 100})


def test_parse_comparison_condition_contains():
    assert _parse_comparison_condition(
        {"field": "meta.name", "operator": "contains", "value": "doc"}
    ) == weaviate.classes.query.Filter.by_property("name").like("*doc*")


@pytest.mark.parametrize("value", [100, ["a"], None])
def test_contains_rejects_non_string_values(value):
    with pytest.raises(FilterError, match="must be a string when using 'contains' comparator"):
        _parse_comparison_condition({"field": "meta.name", "operator": "contains", "value": value})


@pytest.mark.parametrize("operator", [">", ">=", "<", "<="])
def test_comparison_with_none_matches_no_document(operator):
    """An ordering operator against None must match nothing, as it does in the other Document Stores."""
    result = _parse_comparison_condition({"field": "meta.number", "operator": operator, "value": None})

    # `is_none(False) AND is_none(True)` is unsatisfiable, so no Document can match.
    assert result.operator == _Operator.AND
    assert result.filters == [
        weaviate.classes.query.Filter.by_property("number").is_none(False),
        weaviate.classes.query.Filter.by_property("number").is_none(True),
    ]


def test_equal_passes_through_values_that_are_not_dates():
    """_handle_date() must leave non-ISO strings and non-strings untouched."""
    assert _parse_comparison_condition(
        {"field": "meta.name", "operator": "==", "value": "not-a-date"}
    ) == weaviate.classes.query.Filter.by_property("name").equal("not-a-date")
    assert _parse_comparison_condition(
        {"field": "meta.number", "operator": "==", "value": 100}
    ) == weaviate.classes.query.Filter.by_property("number").equal(100)


def test_invert_nested_not_conditions():
    """NOT inverts to OR, so a NOT nested inside a NOT must round-trip back to the original meaning."""
    inner = {"operator": "NOT", "conditions": [{"field": "meta.number", "operator": "==", "value": 100}]}
    inverted = _invert_condition({"operator": "NOT", "conditions": [inner]})
    assert inverted == {
        "operator": "OR",
        "conditions": [{"operator": "OR", "conditions": [{"field": "meta.number", "operator": "!=", "value": 100}]}],
    }
