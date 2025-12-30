# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Converts Haystack filters to Astra compatible filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _convert_filters(filters: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    """
    Convert haystack filters to astra filter string capturing all boolean operators
    """
    if not filters:
        return None
    filters = _normalize_filters(filters)

    filter_statements = {}
    for key, value in filters.items():
        if key in {"$and", "$or"}:
            filter_statements[key] = value
        else:
            if key != "$in" and isinstance(value, list):
                filter_statements[key] = {"$in": value}
            elif isinstance(value, dict):
                for dkey, dvalue in value.items():
                    if dkey == "$in" and not isinstance(dvalue, list):
                        exception_message = f"$in operator must have `ARRAY`, got {dvalue} of type {type(dvalue)}"
                        raise FilterError(exception_message)
                    converted = {dkey: dvalue}
                filter_statements[key] = converted
            else:
                filter_statements[key] = value
            if key == "id":
                filter_statements["_id"] = filter_statements.pop("id")

    return filter_statements


# TODO consider other operators, or filters that are not with the same structure as field operator value
OPERATORS = {
    "==": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
    "in": "$in",
    "not in": "$nin",
    "AND": "$and",
    "OR": "$or",
}


def _parse_logical_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    conditions = [_normalize_filters(c) for c in condition["conditions"]]
    if len(conditions) > 1:
        conditions = _normalize_ranges(conditions)
    if operator not in OPERATORS:
        msg = f"Unknown operator {operator}"
        raise FilterError(msg)
    return {OPERATORS[operator]: conditions}


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "field" not in condition:
        msg = f"'field' key missing in {condition}"
        raise FilterError(msg)
    field: str = condition["field"]

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    return {field: {OPERATORS[operator]: value}}


def _normalize_ranges(conditions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merges range conditions acting on a same field.

    Example usage:

    ```python
    conditions = [
        {"range": {"date": {"lt": "2021-01-01"}}},
        {"range": {"date": {"gte": "2015-01-01"}}},
    ]
    conditions = _normalize_ranges(conditions)
    assert conditions == [
        {"range": {"date": {"lt": "2021-01-01", "gte": "2015-01-01"}}},
    ]
    ```
    """
    range_conditions = [next(iter(c["range"].items())) for c in conditions if "range" in c]
    if range_conditions:
        conditions = [c for c in conditions if "range" not in c]
        range_conditions_dict: dict[str, Any] = {}
        for field_name, comparison in range_conditions:
            if field_name not in range_conditions_dict:
                range_conditions_dict[field_name] = {}
            range_conditions_dict[field_name].update(comparison)

        for field_name, comparisons in range_conditions_dict.items():
            conditions.append({"range": {field_name: comparisons}})
    return conditions
