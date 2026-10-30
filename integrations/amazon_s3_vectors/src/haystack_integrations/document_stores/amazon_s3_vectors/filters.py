# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Haystack filters to Amazon S3 Vectors compatible filters.

    Reference: https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-metadata-filtering.html
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
    conditions = [_normalize_filters(c) for c in condition["conditions"]]

    if operator in LOGICAL_OPERATORS:
        return {LOGICAL_OPERATORS[operator]: conditions}

    msg = f"Unknown logical operator '{operator}'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "field" not in condition:
        return _parse_logical_condition(condition)

    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    operator: str = condition["operator"]
    value: Any = condition["value"]

    # Strip the "meta." prefix — metadata is stored flat in S3 Vectors
    if field.startswith("meta."):
        field = field[5:]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise FilterError(msg)

    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (str, int, float, bool), "equal")
    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (str, int, float, bool), "not equal")
    return {field: {"$ne": value}}


def _greater_than(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (int, float), "greater than")
    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (int, float), "greater than equal")
    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (int, float), "less than")
    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> dict[str, Any]:
    _assert_supported_type(value, (int, float), "less than equal")
    return {field: {"$lte": value}}


def _in(field: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator"
        raise FilterError(msg)
    for v in value:
        _assert_supported_type(v, (str, int, float), "in")
    return {field: {"$in": value}}


def _not_in(field: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator"
        raise FilterError(msg)
    for v in value:
        _assert_supported_type(v, (str, int, float), "not in")
    return {field: {"$nin": value}}


def _assert_supported_type(value: Any, supported_types: tuple[type, ...], operator_name: str) -> None:
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for '{operator_name}' comparison: {type(value)}. "
            f"Types supported by S3 Vectors are: {supported_types}"
        )
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
    """Validate Haystack filter syntax."""
    if filters and "operator" not in filters and "conditions" not in filters:
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise ValueError(msg)
