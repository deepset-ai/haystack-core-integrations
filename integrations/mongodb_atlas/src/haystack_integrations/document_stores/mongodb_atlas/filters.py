# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Any, Dict

from haystack.errors import FilterError
from haystack.utils.filters import convert
from pandas import DataFrame

UNSUPPORTED_TYPES_FOR_COMPARISON = (list, DataFrame)


def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters to MongoDB filters.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "operator" not in filters and "conditions" not in filters:
        filters = convert(filters)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    conditions = []
    for c in condition["conditions"]:
        if "field" in c:
            conditions.append(_parse_comparison_condition(c))
        else:
            conditions.append(_parse_logical_condition(c))

    operator = condition["operator"]
    if operator == "AND":
        return {"$and": conditions}
    elif operator == "OR":
        return {"$or": conditions}
    elif operator == "NOT":
        # MongoDB doesn't support our NOT operator (logical NAND) directly.
        # we combine $nor and $and to achieve the same effect.
        return {"$nor": [{"$and": conditions}]}

    msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if isinstance(value, DataFrame):
        value = value.to_json()

    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> Dict[str, Any]:
    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    return {field: {"$ne": value}}


def _validate_type_for_comparison(value: Any) -> None:
    msg = f"Cant compare {type(value)} using operators '>', '>=', '<', '<='."
    if isinstance(value, UNSUPPORTED_TYPES_FOR_COMPARISON):
        raise FilterError(msg)
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg += "\nStrings are only comparable if they are ISO formatted dates."
            raise FilterError(msg) from exc


def _greater_than(field: str, value: Any) -> Dict[str, Any]:
    _validate_type_for_comparison(value)
    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # we want {field: {"$gte": null}} to return an empty result
        # $gte with null values in MongoDB returns a non-empty result, while $gt aligns with our expectations
        return {field: {"$gt": value}}

    _validate_type_for_comparison(value)
    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> Dict[str, Any]:
    _validate_type_for_comparison(value)
    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> Dict[str, Any]:
    if value is None:
        # we want {field: {"$lte": null}} to return an empty result
        # $lte with null values in MongoDB returns a non-empty result, while $lt aligns with our expectations
        return {field: {"$lt": value}}
    _validate_type_for_comparison(value)

    return {field: {"$lte": value}}


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    return {field: {"$nin": value}}


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    return {field: {"$in": value}}


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}
