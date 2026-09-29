# SPDX-FileCopyrightText: 2026-present EverMind AI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

_OPERATOR_MAP = {
    "==": "eq",
    "=": "eq",
    "eq": "eq",
    "!=": "ne",
    "ne": "ne",
    ">": "gt",
    "gt": "gt",
    ">=": "gte",
    "gte": "gte",
    "<": "lt",
    "lt": "lt",
    "<=": "lte",
    "lte": "lte",
    "in": "in",
}


def build_search_filters(
    *, filters: dict[str, Any] | None = None, session_id: str | None = None
) -> dict[str, Any] | None:
    """Convert Haystack metadata filters into the EverOS filter DSL and add an optional session scope."""
    converted = _convert_filter(filters) if filters else None
    session_filter = {"session_id": session_id} if session_id else None
    if converted and session_filter:
        return {"AND": [converted, session_filter]}
    return converted or session_filter


def _convert_filter(node: dict[str, Any]) -> dict[str, Any]:
    """Convert one Haystack comparison or logical filter node."""
    if "field" in node:
        field = node.get("field")
        operator = node.get("operator")
        if not isinstance(field, str) or not field:
            msg = "Haystack filters require a non-empty string 'field'."
            raise ValueError(msg)
        if not isinstance(operator, str) or operator.lower() not in _OPERATOR_MAP:
            msg = f"Unsupported EverOS filter operator: {operator!r}."
            raise ValueError(msg)
        mapped = _OPERATOR_MAP[operator.lower()]
        value = node.get("value")
        if mapped == "eq":
            return {field: value}
        return {field: {mapped: value}}

    operator = node.get("operator")
    conditions = node.get("conditions")
    if isinstance(operator, str) and isinstance(conditions, list):
        logical = operator.upper()
        if logical not in {"AND", "OR"}:
            msg = f"Unsupported EverOS logical filter operator: {operator!r}."
            raise ValueError(msg)
        if not conditions:
            msg = "Logical filters require at least one condition."
            raise ValueError(msg)
        if not all(isinstance(condition, dict) for condition in conditions):
            msg = "Every logical filter condition must be a dictionary."
            raise TypeError(msg)
        return {logical: [_convert_filter(condition) for condition in conditions]}

    msg = "Filters must use Haystack comparison or logical filter syntax."
    raise ValueError(msg)
