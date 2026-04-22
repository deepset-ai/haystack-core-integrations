# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from haystack.errors import FilterError

from haystack_integrations.document_stores.weaviate._filters import (
    _invert_condition,
    _parse_comparison_condition,
    _parse_logical_condition,
    convert_filters,
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
