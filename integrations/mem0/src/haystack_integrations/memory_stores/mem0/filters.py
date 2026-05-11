# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.errors import FilterError


def normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Convert Haystack-style filters to the Mem0 filter format.

    :param filters: Haystack filter dictionary.
    :returns: Equivalent Mem0 filter dictionary.
    :raises FilterError: If the filter structure or operators are invalid.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary."
        raise FilterError(msg)
    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "operator" not in condition:
        msg = f"'operator' key missing in: {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in: {condition}"
        raise FilterError(msg)

    operator = condition["operator"].upper()
    if operator not in ("AND", "OR", "NOT"):
        msg = f"Unsupported logical operator: {operator!r}. Use AND, OR, or NOT."
        raise FilterError(msg)

    converted = [_convert(c) for c in condition["conditions"]]
    return {operator: converted}


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
    if "field" not in condition:
        msg = f"'field' key missing in: {condition}"
        raise FilterError(msg)
    if "operator" not in condition:
        msg = f"'operator' key missing in: {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in: {condition}"
        raise FilterError(msg)

    field: str = condition["field"]
    operator: str = condition["operator"]
    value: Any = condition["value"]

    op_map = {
        "==": lambda f, v: {f: v},
        "!=": lambda f, v: {f: {"ne": v}},
        ">": lambda f, v: {f: {"gt": v}},
        ">=": lambda f, v: {f: {"gte": v}},
        "<": lambda f, v: {f: {"lt": v}},
        "<=": lambda f, v: {f: {"lte": v}},
        "in": lambda f, v: {f: {"in": v if isinstance(v, list) else [v]}},
        "not in": lambda f, v: {f: {"ne": v}},
    }

    if operator not in op_map:
        msg = f"Unsupported filter operator: {operator!r}."
        raise FilterError(msg)

    return op_map[operator](field, value)


def _convert(node: dict[str, Any]) -> dict[str, Any]:
    if "field" in node:
        return _parse_comparison_condition(node)
    return _parse_logical_condition(node)
