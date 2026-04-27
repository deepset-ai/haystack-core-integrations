# SPDX-FileCopyrightText: 2023-present Anant Corporation <support@anant.us>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

from haystack.errors import FilterError

UNSUPPORTED_TYPES_FOR_COMPARISON = (list,)

# Astra Data API rejects '$gt'/'$gte'/'$lt'/'$lte' against null. To preserve the
# Haystack contract (None on these comparators returns no documents), we emit a
# filter that the API accepts but that matches nothing.
_NO_MATCH: dict[str, Any] = {"_id": {"$in": []}}

# Astra Data API does not implement '$not' or '$nor'; NOT is realised by
# applying De Morgan's laws and inverting the leaf comparators below.
_NEGATED_COMPARATORS = {
    "$eq": "$ne",
    "$ne": "$eq",
    "$gt": "$lte",
    "$gte": "$lt",
    "$lt": "$gte",
    "$lte": "$gt",
    "$in": "$nin",
    "$nin": "$in",
}


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


def _convert_filters(filters: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    Convert Haystack filters to the Astra Data API filter format.
    """
    if not filters:
        return None
    return _normalize_filters(filters)


# Kept for backward compatibility with anything still importing this mapping.
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
    if len(ops) != 1:
        # Compound clauses like {"$exists": True, "$ne": null} would need to
        # negate to a disjunction; not needed by current tests.
        msg = f"Cannot negate compound clause for field '{field}': {ops}"
        raise FilterError(msg)
    op, value = next(iter(ops.items()))
    if op not in _NEGATED_COMPARATORS:
        msg = f"Cannot negate operator '{op}'"
        raise FilterError(msg)
    return {field: {_NEGATED_COMPARATORS[op]: value}}


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
    if isinstance(value, UNSUPPORTED_TYPES_FOR_COMPARISON):
        raise FilterError(msg)
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg += " Strings are only comparable if they are ISO formatted dates."
            raise FilterError(msg) from exc


def _greater_than(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return _NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return _NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return _NO_MATCH
    _validate_type_for_comparison(value)
    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> dict[str, Any]:  # noqa: ANN401
    if value is None:
        return _NO_MATCH
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
