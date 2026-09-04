# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import datetime
from typing import Any

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """Convert Haystack filters to MongoDB-compatible Azure DocumentDB filters."""
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)
    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)
    conditions = [
        _parse_comparison_condition(item) if "field" in item else _parse_logical_condition(item)
        for item in condition["conditions"]
    ]
    operator = condition["operator"]
    if operator == "AND":
        return {"$and": conditions}
    if operator == "OR":
        return {"$or": conditions}
    if operator == "NOT":
        return {"$nor": [{"$and": conditions}]}
    msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    for key in ("field", "operator", "value"):
        if key not in condition:
            msg = f"'{key}' key missing in {condition}"
            raise FilterError(msg)
    operator = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise FilterError(msg)
    return COMPARISON_OPERATORS[operator](condition["field"], condition["value"])


def _comparison(mongo_operator: str) -> Callable[[str, Any], dict[str, Any]]:
    def convert(field: str, value: Any) -> dict[str, Any]:
        return {field: {mongo_operator: value}}

    return convert


def _ordered_comparison(mongo_operator: str) -> Callable[[str, Any], dict[str, Any]]:
    def convert(field: str, value: Any) -> dict[str, Any]:
        if isinstance(value, list):
            msg = f"Can't compare {type(value)} using ordered comparison operators."
            raise FilterError(msg)
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value)
            except (TypeError, ValueError) as error:
                msg = "Strings are only comparable if they are ISO formatted dates."
                raise FilterError(msg) from error
        if value is None and mongo_operator in {"$gte", "$lte"}:
            return {field: {"$gt" if mongo_operator == "$gte" else "$lt": None}}
        return {field: {mongo_operator: value}}

    return convert


def _membership(mongo_operator: str) -> Callable[[str, Any], dict[str, Any]]:
    def convert(field: str, value: Any) -> dict[str, Any]:
        if not isinstance(value, list):
            msg = f"{field}'s value must be a list when using a membership comparator"
            raise FilterError(msg)
        return {field: {mongo_operator: value}}

    return convert


COMPARISON_OPERATORS: dict[str, Callable[[str, Any], dict[str, Any]]] = {
    "==": _comparison("$eq"),
    "!=": _comparison("$ne"),
    ">": _ordered_comparison("$gt"),
    ">=": _ordered_comparison("$gte"),
    "<": _ordered_comparison("$lt"),
    "<=": _ordered_comparison("$lte"),
    "in": _membership("$in"),
    "not in": _membership("$nin"),
}
