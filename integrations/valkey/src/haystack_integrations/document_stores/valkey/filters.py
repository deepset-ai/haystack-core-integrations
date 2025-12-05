# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""Valkey document store filtering utilities.

This module provides filter conversion from Haystack's filter format to Valkey Search query syntax.
It supports both tag-based exact matching and numeric range filtering with logical operators.

Supported filter operations:
- TagField filters: ==, !=, in, not in (exact string matches)
- NumericField filters: ==, !=, >, >=, <, <=, in, not in (numeric comparisons)
- Logical operators: AND, OR for combining conditions

Filter syntax examples:
```python
# Simple equality filter
filters = {"field": "meta.category", "operator": "==", "value": "tech"}

# Numeric range filter
filters = {"field": "meta.priority", "operator": ">=", "value": 5}

# List membership filter
filters = {"field": "meta.status", "operator": "in", "value": ["active", "pending"]}

# Complex logical filter
filters = {
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "tech"},
        {"field": "meta.priority", "operator": ">=", "value": 3}
    ]
}
```
"""

from __future__ import annotations

from typing import Any

from haystack.errors import FilterError


def _normalize_filters(filters: dict[str, Any]) -> str:
    """
    Converts Haystack filters to Valkey Search query syntax.

    Transforms Haystack's filter format into Valkey FT.SEARCH compatible query strings.
    Supports both simple field comparisons and complex logical combinations.

    Valkey Search supports:
    - TagField: exact matches (==, !=, in, not in)
    - NumericField: all comparisons (==, !=, >, >=, <, <=, in, not in)
    - Logical operators: AND, OR

    Supported filterable fields:
    - meta_category (TagField): exact string matches
    - meta_status (TagField): status filtering
    - meta_priority (NumericField): numeric comparisons
    - meta_score (NumericField): score filtering
    - meta_timestamp (NumericField): date/time filtering

    :param filters: Haystack filter dictionary.
    :return: Valkey Search query string.
    :raises FilterError: If filter format is invalid or unsupported.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise FilterError(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    if operator == "AND":
        return f"({' '.join(conditions)})"
    elif operator == "OR":
        return f"({' | '.join(conditions)})"
    else:
        msg = f"Unknown logical operator '{operator}'. Supported: AND, OR"
        raise FilterError(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
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

    # Handle meta.field syntax
    if field.startswith("meta."):
        field = f"meta_{field[5:]}"

    # Validate field is supported
    if field not in SUPPORTED_FIELDS:
        msg = f"Field '{field}' is not supported for filtering. Supported fields: {list(SUPPORTED_FIELDS.keys())}"
        raise FilterError(msg)

    field_type = SUPPORTED_FIELDS[field]

    # Validate operator for field type
    if operator not in FIELD_TYPE_OPERATORS[field_type]:
        supported_ops = FIELD_TYPE_OPERATORS[field_type]
        msg = f"Operator '{operator}' not supported for {field_type} field '{field}'. Supported: {supported_ops}"
        raise FilterError(msg)

    return COMPARISON_OPERATORS[operator](field, value, field_type)


def _equal(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        if not isinstance(value, str):
            msg = f"TagField '{field}' requires string value, got {type(value)}"
            raise FilterError(msg)
        return f"@{field}:{{{_escape_value(value)}}}"
    else:  # numeric
        if not isinstance(value, (int, float)):
            msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
            raise FilterError(msg)
        return f"@{field}:[{value} {value}]"


def _not_equal(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        if not isinstance(value, str):
            msg = f"TagField '{field}' requires string value, got {type(value)}"
            raise FilterError(msg)
        return f"-@{field}:{{{_escape_value(value)}}}"
    else:  # numeric
        if not isinstance(value, (int, float)):
            msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
            raise FilterError(msg)
        return f"-@{field}:[{value} {value}]"


def _greater_than(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        msg = f"Operator '>' not supported for TagField '{field}'"
        raise FilterError(msg)
    if not isinstance(value, (int, float)):
        msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
        raise FilterError(msg)
    return f"@{field}:[({value} +inf]"


def _greater_than_equal(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        msg = f"Operator '>=' not supported for TagField '{field}'"
        raise FilterError(msg)
    if not isinstance(value, (int, float)):
        msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
        raise FilterError(msg)
    return f"@{field}:[{value} +inf]"


def _less_than(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        msg = f"Operator '<' not supported for TagField '{field}'"
        raise FilterError(msg)
    if not isinstance(value, (int, float)):
        msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
        raise FilterError(msg)
    return f"@{field}:[-inf ({value}]"


def _less_than_equal(field: str, value: Any, field_type: str) -> str:
    if field_type == "tag":
        msg = f"Operator '<=' not supported for TagField '{field}'"
        raise FilterError(msg)
    if not isinstance(value, (int, float)):
        msg = f"NumericField '{field}' requires numeric value, got {type(value)}"
        raise FilterError(msg)
    return f"@{field}:[-inf {value}]"


def _in(field: str, value: Any, field_type: str) -> str:
    if not isinstance(value, list):
        msg = f"'in' operator requires a list value, got {type(value)}"
        raise FilterError(msg)

    if field_type == "tag":
        for v in value:
            if not isinstance(v, str):
                msg = f"TagField '{field}' requires string values in list, got {type(v)}"
                raise FilterError(msg)
        escaped_values = [_escape_value(v) for v in value]
        # Use OR syntax for multiple tag values
        conditions = [f"@{field}:{{{val}}}" for val in escaped_values]
        return f"({' | '.join(conditions)})"
    else:  # numeric
        for v in value:
            if not isinstance(v, (int, float)):
                msg = f"NumericField '{field}' requires numeric values in list, got {type(v)}"
                raise FilterError(msg)
        # For numeric fields, use OR with separate range queries for each value
        conditions = [f"@{field}:[{v} {v}]" for v in value]
        return f"({' | '.join(conditions)})"


def _not_in(field: str, value: Any, field_type: str) -> str:
    if not isinstance(value, list):
        msg = f"'not in' operator requires a list value, got {type(value)}"
        raise FilterError(msg)

    if field_type == "tag":
        for v in value:
            if not isinstance(v, str):
                msg = f"TagField '{field}' requires string values in list, got {type(v)}"
                raise FilterError(msg)
        escaped_values = [_escape_value(v) for v in value]
        return f"-@{field}:{{{' | '.join(escaped_values)}}}"
    else:  # numeric
        for v in value:
            if not isinstance(v, (int, float)):
                msg = f"NumericField '{field}' requires numeric values in list, got {type(v)}"
                raise FilterError(msg)
        # For numeric fields, use negated OR with separate range queries
        conditions = [f"@{field}:[{v} {v}]" for v in value]
        return f"-({' | '.join(conditions)})"


def _escape_value(value: str) -> str:
    """Escape special characters in tag values for Redis Search."""
    # Escape special Redis Search characters
    special_chars = [
        ",",
        ".",
        "<",
        ">",
        "{",
        "}",
        "[",
        "]",
        '"',
        "'",
        ":",
        ";",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "-",
        "+",
        "=",
        "~",
    ]
    for char in special_chars:
        value = value.replace(char, f"\\{char}")
    return value


# Supported filterable fields and their types
SUPPORTED_FIELDS = {
    "meta_category": "tag",
    "meta_status": "tag",
    "meta_priority": "numeric",
    "meta_score": "numeric",
    "meta_timestamp": "numeric",
}

# Operators supported by field type
FIELD_TYPE_OPERATORS = {
    "tag": ["==", "!=", "in", "not in"],
    "numeric": ["==", "!=", ">", ">=", "<", "<=", "in", "not in"],
}

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


def _validate_filters(filters: dict[str, Any] | None) -> None:
    """
    Helper method to validate filter syntax.
    """
    if filters and "operator" not in filters and "conditions" not in filters:
        msg = "Invalid filter syntax. See https://docs.haystack.deepset.ai/docs/metadata-filtering for details."
        raise ValueError(msg)
