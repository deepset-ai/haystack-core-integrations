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
)


def test_not_wraps_its_conditions_in_a_single_negated_and():
    """A NOT node negates the conjunction of its conditions, using Weaviate's native NOT operator."""
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

    assert result.operator == _Operator.NOT
    [negated] = result.filters
    assert negated.operator == _Operator.AND
    number_eq, name_eq, nested_or = negated.filters
    assert number_eq == weaviate.classes.query.Filter.by_property("number").equal(100)
    assert name_eq == weaviate.classes.query.Filter.by_property("name").equal("name_0")
    # Only leaf filters compare by value, so the nested OR is checked structurally.
    assert nested_or.operator == _Operator.OR
    assert nested_or.filters == [
        weaviate.classes.query.Filter.by_property("name").equal("name_1"),
        weaviate.classes.query.Filter.by_property("name").equal("name_2"),
    ]


def test_nested_not_is_not_double_negated():
    """NOT(NOT(x)) must mean x, not NOT(x)."""
    inner = {"operator": "NOT", "conditions": [{"field": "meta.number", "operator": "==", "value": 100}]}

    result = convert_filters({"operator": "NOT", "conditions": [inner]})

    # `Filter.all_of` collapses a single operand, so each NOT wraps its condition directly.
    assert result.operator == _Operator.NOT
    [inner_not] = result.filters
    assert inner_not.operator == _Operator.NOT
    assert inner_not.filters == [weaviate.classes.query.Filter.by_property("number").equal(100)]


@pytest.mark.parametrize("operator", ["contains", "like"])
def test_not_over_operators_without_an_inverse(operator):
    """These used to raise a bare KeyError out of the inversion table."""
    filters = {"operator": "NOT", "conditions": [{"field": "meta.name", "operator": operator, "value": "x"}]}

    if operator == "contains":
        # Supported, and now negatable because no inversion table is consulted.
        assert convert_filters(filters).operator == _Operator.NOT
    else:
        with pytest.raises(FilterError, match="Unknown comparison operator 'like'"):
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
