# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict

from haystack.errors import FilterError
from pandas import DataFrame


def _normalize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts Haystack filters in Pinecone compatible filters.
    Reference: https://docs.pinecone.io/docs/metadata-filtering
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

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

    operator = condition["operator"]
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    if operator in LOGICAL_OPERATORS:
        return {LOGICAL_OPERATORS[operator]: conditions}

    msg = f"Unknown logical operator '{operator}'"
    raise FilterError(msg)


def _parse_comparison_condition(condition: Dict[str, Any]) -> Dict[str, Any]:
    if "field" not in condition:
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
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
        # Remove the "meta." prefix if present.
        # Documents are flattened when using the PineconeDocumentStore
        # so we don't need to specify the "meta." prefix.
        # Instead of raising an error we handle it gracefully.
        field = field[5:]

    value: Any = condition["value"]
    if isinstance(value, DataFrame):
        value = value.to_json()

    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'equal' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'not equal' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$ne": value}}


def _greater_than(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'greater than' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$gt": value}}


def _greater_than_equal(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'greater than equal' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$gte": value}}


def _less_than(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'less than' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$lt": value}}


def _less_than_equal(field: str, value: Any) -> Dict[str, Any]:
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'less than equal' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise FilterError(msg)

    return {field: {"$lte": value}}


def _not_in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = (
                f"Unsupported type for 'not in' comparison: {type(v)}. "
                f"Types supported by Pinecone are: {supported_types}"
            )
            raise FilterError(msg)

    return {field: {"$nin": value}}


def _in(field: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = (
                f"Unsupported type for 'in' comparison: {type(v)}. "
                f"Types supported by Pinecone are: {supported_types}"
            )
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

LOGICAL_OPERATORS = {"AND": "$and", "OR": "$or"}
