# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Converts Haystack filters into the Dakera metadata-filter DSL.

    Dakera uses a MongoDB-style operator DSL (``{"field": {"$eq": value}}``,
    ``{"$and": [...]}``), so the mapping mirrors the one used by other vector-store
    integrations.
    """
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
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    if operator in LOGICAL_OPERATORS:
        return {LOGICAL_OPERATORS[operator]: conditions}

    msg = f"Unknown logical operator '{operator}'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "field" not in condition:
        # A missing 'field' key means this is actually a logical condition.
        return _parse_logical_condition(condition)

    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    operator: str = condition["operator"]
    if field.startswith("meta."):
        # Metadata is stored flat in Dakera, so the "meta." prefix is dropped.
        field = field[len("meta.") :]

    value: Any = condition["value"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise FilterError(msg)
    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> dict[str, Any]:
    _check_scalar("equal", value)
    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> dict[str, Any]:
    _check_scalar("not equal", value)
    return {field: {"$ne": value}}


def _greater_than(field: str, value: Any) -> dict[str, Any]:
    _check_numeric("greater than", value)
    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> dict[str, Any]:
    _check_numeric("greater than equal", value)
    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> dict[str, Any]:
    _check_numeric("less than", value)
    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> dict[str, Any]:
    _check_numeric("less than equal", value)
    return {field: {"$lte": value}}


def _in(field: str, value: Any) -> dict[str, Any]:
    _check_list("in", field, value)
    return {field: {"$in": value}}


def _not_in(field: str, value: Any) -> dict[str, Any]:
    _check_list("not in", field, value)
    return {field: {"$nin": value}}


def _check_scalar(operator: str, value: Any) -> None:
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = f"Unsupported type for '{operator}' comparison: {type(value)}. Supported types are: {supported_types}"
        raise FilterError(msg)


def _check_numeric(operator: str, value: Any) -> None:
    # bool is a subclass of int but is not a meaningful ordering operand.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        supported_types = (int, float)
        msg = f"Unsupported type for '{operator}' comparison: {type(value)}. Supported types are: {supported_types}"
        raise FilterError(msg)


def _check_list(operator: str, field: str, value: Any) -> None:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using the '{operator}' comparator in Dakera"
        raise FilterError(msg)
    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = f"Unsupported type for '{operator}' comparison: {type(v)}. Supported types are: {supported_types}"
            raise FilterError(msg)


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

LOGICAL_OPERATORS = {"AND": "$and", "OR": "$or"}


def _validate_filters(filters: dict[str, Any] | None) -> None:
    """Validates the top-level shape of a Haystack filter dictionary."""
    if filters and "operator" not in filters and "conditions" not in filters and "field" not in filters:
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise ValueError(msg)
