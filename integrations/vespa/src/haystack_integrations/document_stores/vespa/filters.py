# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from haystack.errors import FilterError

from .errors import VespaDocumentStoreFilterError


def _format_yql_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _is_scalar_string_value(value: Any) -> bool:
    return isinstance(value, str)


def _normalize_field_name(field: str, *, content_field: str) -> str:
    if field == "content":
        return content_field
    if field.startswith("meta."):
        return field[5:]
    return field


def _validate_ordered_filter_value(operator: str, value: Any) -> None:
    """
    Raise FilterError when a comparison operator cannot use the supplied value type.

    Aligns loosely with haystack.utils.filters behaviour for relational operators.
    """
    if operator not in {">", ">=", "<", "<="}:
        return

    if isinstance(value, bool):
        return

    if isinstance(value, (int, float)):
        return

    if isinstance(value, list):
        msg = (
            f"Filter value can't be of type {type(value)} using operators "
            "'>', '>=', '<', '<='. See Haystack metadata filtering documentation."
        )
        raise FilterError(msg)

    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = "Can't compare strings using relational operators unless they are ISO-formatted dates."
            raise FilterError(msg) from exc
        return

    msg = "Unsupported filter value type for relational operators. Use numbers or ISO/date-parseable strings."
    raise FilterError(msg)


def _convert_comparison(
    field: str,
    operator: str,
    value: Any,
    *,
    content_field: str,
) -> str:
    normalized_field = _normalize_field_name(field, content_field=content_field)

    if operator in {">", ">=", "<", "<="} and value is None:
        return "false"

    _validate_ordered_filter_value(operator, value)

    if operator == "==":
        if _is_scalar_string_value(value):
            return f"{normalized_field} contains {_format_yql_value(value)}"
        return f"{normalized_field} = {_format_yql_value(value)}"
    if operator == "!=":
        if _is_scalar_string_value(value):
            return f"!( {normalized_field} contains {_format_yql_value(value)} )"
        return f"!( {normalized_field} = {_format_yql_value(value)} )"
    if operator == ">":
        return f"{normalized_field} > {_format_yql_value(value)}"
    if operator == ">=":
        return f"{normalized_field} >= {_format_yql_value(value)}"
    if operator == "<":
        return f"{normalized_field} < {_format_yql_value(value)}"
    if operator == "<=":
        return f"{normalized_field} <= {_format_yql_value(value)}"
    if operator == "in":
        if not isinstance(value, list):
            msg = "'in' filter values must be lists"
            raise VespaDocumentStoreFilterError(msg)
        values = ", ".join(_format_yql_value(item) for item in value)
        return f"{normalized_field} in ({values})"
    if operator == "not in":
        if not isinstance(value, list):
            msg = "'not in' filter values must be lists"
            raise VespaDocumentStoreFilterError(msg)
        values = ", ".join(_format_yql_value(item) for item in value)
        return f"!( {normalized_field} in ({values}) )"
    if operator == "contains":
        return f"{normalized_field} contains {_format_yql_value(value)}"
    if operator == "not contains":
        return f"!( {normalized_field} contains {_format_yql_value(value)} )"

    msg = f"Unsupported Vespa filter operator: {operator}"
    raise VespaDocumentStoreFilterError(msg)


def _normalize_filters(filters: dict[str, Any] | None, *, content_field: str) -> str:
    """
    Convert Haystack metadata filters into a Vespa YQL expression.

    :param filters: Haystack-style filters.
    :param content_field: Vespa field name used for document content.
    :returns: Vespa YQL expression without the enclosing `where`.
    """
    if not filters:
        return "true"

    operator = filters.get("operator")
    if operator in {"AND", "OR"}:
        conditions = filters.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            msg = f"{operator} filters must contain a non-empty 'conditions' list"
            raise VespaDocumentStoreFilterError(msg)

        joiner = " and " if operator == "AND" else " or "
        nested = (_normalize_filters(condition, content_field=content_field) for condition in conditions)
        return f"( {joiner.join(nested)} )"

    if operator == "NOT":
        conditions = filters.get("conditions")
        if not isinstance(conditions, list) or not conditions:
            msg = "NOT filters must contain a non-empty 'conditions' list"
            raise VespaDocumentStoreFilterError(msg)
        normalized_parts = [_normalize_filters(condition, content_field=content_field) for condition in conditions]
        merged = " and ".join(normalized_parts)
        return f"!( ( {merged} ) )"

    field = filters.get("field")
    comparison = filters.get("operator")
    if not isinstance(field, str) or not isinstance(comparison, str):
        msg = "Leaf filters must contain 'field' and 'operator'"
        raise VespaDocumentStoreFilterError(msg)

    if "value" not in filters:
        msg = "Leaf filters must include a 'value' key."
        raise VespaDocumentStoreFilterError(msg)

    return _convert_comparison(field, comparison, filters["value"], content_field=content_field)
