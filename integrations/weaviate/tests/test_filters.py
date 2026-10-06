# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import weaviate
from haystack.errors import FilterError
from weaviate.collections.classes.filters import _Operator

from haystack_integrations.document_stores.weaviate._filters import (
    _parse_comparison_condition,
    _parse_logical_condition,
    convert_filters,
    validate_filters,
)


def _uses_native_not(filter_) -> bool:
    """Whether the filter tree contains Weaviate's native NOT, which older servers reject."""
    if getattr(filter_, "operator", None) == _Operator.NOT:
        return True
    return any(_uses_native_not(f) for f in getattr(filter_, "filters", []))


def test_not_inverts_its_conditions_with_de_morgan():
    """A NOT node negates the conjunction of its conditions, applied client-side with De Morgan's laws."""
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

    result = convert_filters(filters)

    # NOT(a AND b AND (c OR d)) == NOT(a) OR NOT(b) OR (NOT(c) AND NOT(d))
    assert result.operator == _Operator.OR
    number_ne, name_ne, nested_and = result.filters
    # Only leaf filters compare by value, so composite filters are checked structurally.
    # `!=` also matches Documents where the field is not set.
    assert number_ne.operator == _Operator.OR
    assert number_ne.filters == [
        weaviate.classes.query.Filter.by_property("number").not_equal(100),
        weaviate.classes.query.Filter.by_property("number").is_none(True),
    ]
    assert name_ne.operator == _Operator.OR
    assert name_ne.filters[0] == weaviate.classes.query.Filter.by_property("name").not_equal("name_0")
    assert nested_and.operator == _Operator.AND
    assert [f.filters[0] for f in nested_and.filters] == [
        weaviate.classes.query.Filter.by_property("name").not_equal("name_1"),
        weaviate.classes.query.Filter.by_property("name").not_equal("name_2"),
    ]
    # Invertible operators never need the native NOT, which Weaviate < 1.33 rejects.
    assert not _uses_native_not(result)


def test_nested_not_is_not_double_negated():
    """NOT(NOT(x)) must mean x, not NOT(x)."""
    inner = {"operator": "NOT", "conditions": [{"field": "meta.number", "operator": "==", "value": 100}]}

    result = convert_filters({"operator": "NOT", "conditions": [inner]})

    # `Filter.any_of` and `Filter.all_of` collapse a single operand, leaving just the condition.
    assert result == weaviate.classes.query.Filter.by_property("number").equal(100)


def test_not_over_logical_operators():
    """NOT(a OR b) == NOT(a) AND NOT(b)."""
    filters = {
        "operator": "NOT",
        "conditions": [
            {
                "operator": "OR",
                "conditions": [
                    {"field": "meta.number", "operator": ">", "value": 10},
                    {"field": "meta.number", "operator": "in", "value": [1, 2]},
                ],
            }
        ],
    }

    result = convert_filters(filters)

    assert result.operator == _Operator.AND
    number_lte, number_not_in = result.filters
    assert number_lte == weaviate.classes.query.Filter.by_property("number").less_or_equal(10)
    assert number_not_in.operator == _Operator.AND
    assert number_not_in.filters == [
        weaviate.classes.query.Filter.by_property("number").not_equal(1),
        weaviate.classes.query.Filter.by_property("number").not_equal(2),
    ]


@pytest.mark.parametrize("operator", ["contains", "like"])
def test_not_over_operators_without_an_inverse(operator):
    """These used to raise a bare KeyError out of the inversion table."""
    filters = {"operator": "NOT", "conditions": [{"field": "meta.name", "operator": operator, "value": "x"}]}

    if operator == "contains":
        # Supported, and negated with Weaviate's native NOT since `contains` has no inverse.
        result = convert_filters(filters)
        assert result.operator == _Operator.NOT
        assert result.filters == [weaviate.classes.query.Filter.by_property("name").like("*x*")]
    else:
        with pytest.raises(FilterError, match="Unknown comparison operator 'like'"):
            convert_filters(filters)


def test_not_over_unknown_logical_operator():
    filters = {"operator": "NOT", "conditions": [{"operator": "XOR", "conditions": []}]}
    with pytest.raises(FilterError, match="Unknown logical operator 'XOR'"):
        convert_filters(filters)


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
