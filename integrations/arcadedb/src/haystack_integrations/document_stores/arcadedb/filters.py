# SPDX-FileCopyrightText: 2025-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Convert Haystack filter dictionaries to ArcadeDB SQL WHERE clauses."""

from typing import Any


def _convert_filters(filters: dict[str, Any] | None) -> str:
    """
    Convert a Haystack filter dictionary to an ArcadeDB SQL WHERE clause.

    Supports comparison operators (==, !=, >, >=, <, <=, in, not in)
    and logical operators (AND, OR, NOT).
    """
    if not filters:
        return ""
    return _parse_condition(filters)


def _parse_condition(condition: dict[str, Any]) -> str:
    operator = condition.get("operator")
    if not operator:
        msg = f"Missing 'operator' in filter condition: {condition}"
        raise ValueError(msg)

    operator_upper = operator.upper()

    if operator_upper in ("AND", "OR"):
        if "conditions" not in condition:
            msg = f"Missing 'conditions' in filter: {condition}"
            raise ValueError(msg)
        conditions = condition.get("conditions", [])
        if not conditions:
            return ""
        parts = [_parse_condition(c) for c in conditions]
        parts = [p for p in parts if p]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        joiner = f" {operator_upper} "
        return f"({joiner.join(parts)})"

    if operator_upper == "NOT":
        conditions = condition.get("conditions", [])
        if not conditions:
            return ""
        inner = _parse_condition(conditions[0])
        return f"NOT ({inner})" if inner else ""

    field = condition.get("field")
    value = condition.get("value")

    if not field:
        msg = f"Missing 'field' in filter condition: {condition}"
        raise ValueError(msg)
    if "value" not in condition:
        msg = f"Missing 'value' in filter condition: {condition}"
        raise ValueError(msg)

    return _comparison_to_sql(field, operator, value)


def _comparison_to_sql(field: str, operator: str, value: Any) -> str:
    if operator == "==":
        if value is None:
            return f"{field} IS NULL"
        return f"{field} = {_sql_value(value)}"

    if operator == "!=":
        if value is None:
            return f"{field} IS NOT NULL"
        return f"{field} <> {_sql_value(value)}"

    if operator in (">", ">=", "<", "<="):
        if value is None:
            return "1 = 0"
        if isinstance(value, list):
            msg = "Comparison operators require numeric or datetime values, not list"
            raise ValueError(msg)
        if isinstance(value, str) and "T" not in value:
            msg = "Comparison operators require numeric or datetime (ISO) values, not plain string"
            raise ValueError(msg)
        return f"{field} {operator} {_sql_value(value)}"

    if operator == "in":
        if not isinstance(value, list):
            msg = "Operator 'in' requires value to be a list"
            raise ValueError(msg)
        values = ", ".join(_sql_value(v) for v in value)
        return f"{field} IN [{values}]"

    if operator == "not in":
        if not isinstance(value, list):
            msg = "Operator 'not in' requires value to be a list"
            raise ValueError(msg)
        values = ", ".join(_sql_value(v) for v in value)
        return f"{field} NOT IN [{values}]"

    msg = f"Unsupported filter operator: {operator}"
    raise ValueError(msg)


def _sql_value(value: Any) -> str:
    """Format a Python value as an ArcadeDB SQL literal."""
    if isinstance(value, str):
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if value is None:
        return "NULL"
    return f"'{value}'"
