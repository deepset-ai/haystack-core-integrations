# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime
from typing import Any

from haystack.errors import FilterError

NO_VALUE = "no_value"


def _validate_filters(filters: dict[str, Any] | None = None) -> None:
    if filters:
        if not isinstance(filters, dict):
            msg = "Filters must be a dictionary"
            raise TypeError(msg)
        if "operator" not in filters and "conditions" not in filters:
            msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
            raise ValueError(msg)


def _convert_filters_to_where_clause_and_params(
    filters: dict[str, Any],
    operator: str = "WHERE",
) -> tuple[str, list[Any]]:
    """Convert Haystack filters to a MariaDB WHERE clause string and parameter list."""
    if "field" in filters:
        clause, values = _parse_comparison_condition(filters)
    else:
        clause, values = _parse_logical_condition(filters)

    params = [v for v in values if v != NO_VALUE]
    return f" {operator} {clause}", params


def _parse_logical_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ("AND", "OR", "NOT"):
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
        raise FilterError(msg)

    parts, all_params = [], []
    for c in condition["conditions"]:
        if "field" in c:
            clause, vals = _parse_comparison_condition(c)
        else:
            clause, vals = _parse_logical_condition(c)
        parts.append(clause)
        all_params.extend(vals)

    if operator == "NOT":
        inner = parts[0] if len(parts) == 1 else "(" + " AND ".join(parts) + ")"
        return f"COALESCE(NOT {inner}, TRUE)", all_params

    joined = f" {operator} ".join(parts)
    return f"({joined})", all_params


def _parse_comparison_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)

    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]
    sql_field = _build_field_expr(field, value)
    clause, val = COMPARISON_OPERATORS[operator](sql_field, value)
    # _in/_not_in return the value list directly; flatten it so params stay flat
    if isinstance(val, list):
        return clause, val
    return clause, [val]


_SAFE_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def _build_field_expr(field: str, value: Any) -> str:
    """Return a safe SQL expression for a document field or meta JSON path."""
    if field.startswith("meta."):
        field_name = field.split(".", 1)[1]
        if not _SAFE_FIELD_RE.match(field_name):
            msg = f"Invalid meta field name: '{field_name}'"
            raise FilterError(msg)
        # JSON_UNQUOTE(JSON_EXTRACT(meta, '$.field')) returns the value as a string.
        # Cast to numeric types when the comparison value is numeric.
        base = f"JSON_UNQUOTE(JSON_EXTRACT(meta, '$.{field_name}'))"
        if isinstance(value, bool):
            return base
        if isinstance(value, int):
            return f"CAST({base} AS SIGNED)"
        if isinstance(value, float):
            return f"CAST({base} AS DECIMAL(65,30))"
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, int) and not isinstance(first, bool):
                return f"CAST({base} AS SIGNED)"
            if isinstance(first, float):
                return f"CAST({base} AS DECIMAL(65,30))"
        return base
    # Top-level document field (id, content, etc.) — backtick-quoted
    return f"`{field}`"


def _equal(field: str, value: Any) -> tuple[str, Any]:
    if value is None:
        return f"{field} IS NULL", NO_VALUE
    return f"{field} = ?", value


def _not_equal(field: str, value: Any) -> tuple[str, Any]:
    if value is None:
        return f"{field} IS NOT NULL", NO_VALUE
    return f"({field} IS NULL OR {field} != ?)", value


def _greater_than(field: str, value: Any) -> tuple[str, Any]:
    _check_comparable(value, ">")
    return f"{field} > ?", value


def _greater_than_equal(field: str, value: Any) -> tuple[str, Any]:
    _check_comparable(value, ">=")
    return f"{field} >= ?", value


def _less_than(field: str, value: Any) -> tuple[str, Any]:
    _check_comparable(value, "<")
    return f"{field} < ?", value


def _less_than_equal(field: str, value: Any) -> tuple[str, Any]:
    _check_comparable(value, "<=")
    return f"{field} <= ?", value


def _in(field: str, value: Any) -> tuple[str, list]:
    if not isinstance(value, list):
        msg = f"'in' operator requires a list value, got {type(value)}"
        raise FilterError(msg)
    placeholders = ", ".join(["?"] * len(value))
    return f"{field} IN ({placeholders})", value


def _not_in(field: str, value: Any) -> tuple[str, list]:
    if not isinstance(value, list):
        msg = f"'not in' operator requires a list value, got {type(value)}"
        raise FilterError(msg)
    placeholders = ", ".join(["?"] * len(value))
    return f"({field} IS NULL OR {field} NOT IN ({placeholders}))", value


def _like(field: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, str):
        msg = f"'like' operator requires a string value, got {type(value)}"
        raise FilterError(msg)
    return f"{field} LIKE ?", value


def _not_like(field: str, value: Any) -> tuple[str, Any]:
    if not isinstance(value, str):
        msg = f"'not like' operator requires a string value, got {type(value)}"
        raise FilterError(msg)
    return f"{field} NOT LIKE ?", value


def _check_comparable(value: Any, op: str) -> None:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = f"Cannot compare strings with '{op}'. Strings are only comparable if they are ISO formatted dates."
            raise FilterError(msg) from exc
    if isinstance(value, list):
        msg = f"Filter value cannot be a list when using '{op}'"
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
    "like": _like,
    "not like": _not_like,
}
