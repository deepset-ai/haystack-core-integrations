# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

from haystack.errors import FilterError

# Astra Data API rejects '$gt'/'$gte'/'$lt'/'$lte' against null. In Haystack, None on these comparators should return
# no documents, so we emit a filter that matches nothing. It's a real filter to make it work also on composite filters.
ASTRA_FILTER_NO_MATCH: dict[str, Any] = {"_id": {"$in": []}}

NEGATED_COMPARATORS = {
    "$eq": "$ne",
    "$ne": "$eq",
    "$gt": "$lte",
    "$gte": "$lt",
    "$lt": "$gte",
    "$lte": "$gt",
    "$in": "$nin",
    "$nin": "$in",
}


def _convert_filters(filters: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    Converts Haystack filters to Astra compatible filters.
    """
    if not filters:
        return None
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

    operator = condition["operator"]
    conditions: list[dict[str, Any]] = [c for c in (_convert_filters(c) for c in condition["conditions"]) if c]

    if operator == "AND":
        return {"$and": conditions}
    if operator == "OR":
        return {"$or": conditions}
    if operator == "NOT":
        # NOT(c1 AND c2 AND ...) == NOT c1 OR NOT c2 OR ...
        return {"$or": [_negate(c) for c in conditions]}

    msg = f"Unknown operator {operator}"
    raise FilterError(msg)


def _negate(condition: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively negate a parsed filter using De Morgan's laws and operator inversion.
    """
    if "$and" in condition:
        return {"$or": [_negate(c) for c in condition["$and"]]}
    if "$or" in condition:
        return {"$and": [_negate(c) for c in condition["$or"]]}

    field, ops = next(iter(condition.items()))
    if not isinstance(ops, dict):
        return {field: {"$ne": ops}}
    # a multi-op clause {field: {opA: vA, opB: vB}} is opA AND opB; its negation is (NOT opA) OR (NOT opB), with each
    # disjunct as its own clause on the same field
    disjuncts = [{field: _negated_op(op, val)} for op, val in ops.items()]
    return disjuncts[0] if len(disjuncts) == 1 else {"$or": disjuncts}


def _negated_op(op: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if op == "$exists":
        return {"$exists": not value}
    if op in NEGATED_COMPARATORS:
        return {NEGATED_COMPARATORS[op]: value}
    msg = f"Cannot negate operator '{op}'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "field" not in condition:
        msg = f"'field' key missing in {condition}"
        raise FilterError(msg)
    field: str = condition["field"]
    if field == "id":
        field = "_id"

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise FilterError(msg)
    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        # Astra's '$ne: null' also matches documents missing the field; require
        # the field to exist so semantics align with `meta.get(f) is not None`.
        return {field: {"$exists": True, "$ne": None}}
    return {field: {"$ne": value}}


def _validate_type_for_comparison(value: Any) -> None:  # noqa: ANN401
    msg = f"Cannot compare {type(value).__name__} using operators '>', '>=', '<', '<='."
    if isinstance(value, list):
        raise FilterError(msg)
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg += " Strings are only comparable if they are ISO formatted dates."
            raise FilterError(msg) from exc


def _greater_than(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return ASTRA_FILTER_NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return ASTRA_FILTER_NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return ASTRA_FILTER_NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return ASTRA_FILTER_NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$lte": value}}


def _in(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if not isinstance(value, list):
        msg = f"$in operator must have `ARRAY`, got {value} of type {type(value)}"
        raise FilterError(msg)
    return {field: {"$in": value}}


def _not_in(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if not isinstance(value, list):
        msg = f"$nin operator must have `ARRAY`, got {value} of type {type(value)}"
        raise FilterError(msg)
    return {field: {"$nin": value}}


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
