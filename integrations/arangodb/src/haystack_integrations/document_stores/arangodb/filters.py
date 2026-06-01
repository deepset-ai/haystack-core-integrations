# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any

from haystack.errors import FilterError


def _is_iso_date(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False


def _field_to_aql(field: str) -> str:
    """Maps a Haystack field name to an AQL document field reference."""
    if field == "id":
        return "doc._key"
    if field == "content":
        return "doc.content"
    if field == "embedding":
        return "doc.embedding"
    if field.startswith("meta."):
        key = field[5:]
        return f"doc.meta.`{key}`"
    return f"doc.`{field}`"


def _convert_filters(filters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Converts a Haystack filter dict to an AQL FILTER expression and bind vars.

    :param filters: Haystack filter dict.
    :returns: Tuple of (aql_filter_string, bind_vars).
    """
    bind_vars: dict[str, Any] = {}
    expr = _parse_filter(filters, bind_vars, counter=[0])
    return expr, bind_vars


def _parse_filter(node: dict[str, Any], bind_vars: dict[str, Any], counter: list[int]) -> str:
    if "operator" in node and "conditions" in node:
        op = node["operator"].upper()
        parts = [_parse_filter(c, bind_vars, counter) for c in node["conditions"]]
        if op == "NOT":
            inner = " AND ".join(parts)
            return f"(NOT ({inner}))"
        joiner = " AND " if op == "AND" else " OR "
        inner = joiner.join(parts)
        return f"({inner})"

    if "field" in node and "operator" in node and "value" in node:
        return _parse_comparison(node, bind_vars, counter)

    if "NOT" in node:
        inner = _parse_filter(node["NOT"], bind_vars, counter)
        return f"(NOT {inner})"

    msg = f"Unsupported filter structure: {node}"
    raise FilterError(msg)


_COMPARISON_OPS = {">", ">=", "<", "<="}


def _parse_comparison(node: dict[str, Any], bind_vars: dict[str, Any], counter: list[int]) -> str:
    field = _field_to_aql(node["field"])
    op = node["operator"]
    value = node["value"]

    if op in _COMPARISON_OPS:
        if value is None:
            return "false"
        if isinstance(value, list):
            msg = f"Filter operator '{op}' does not support list values."
            raise FilterError(msg)
        if isinstance(value, str) and not _is_iso_date(value):
            msg = f"Filter operator '{op}' does not support non-date string values."
            raise FilterError(msg)
        idx = counter[0]
        counter[0] += 1
        var = f"fv{idx}"
        bind_vars[var] = value
        return f"({field} != null AND {field} {op} @{var})"

    if op == "==":
        if value is None:
            idx = counter[0]
            counter[0] += 1
            var = f"fv{idx}"
            bind_vars[var] = value
            return f"({field} == null OR {field} == @{var})"
        idx = counter[0]
        counter[0] += 1
        var = f"fv{idx}"
        bind_vars[var] = value
        return f"{field} == @{var}"

    if op == "!=":
        if value is None:
            idx = counter[0]
            counter[0] += 1
            var = f"fv{idx}"
            bind_vars[var] = value
            return f"({field} != null AND {field} != @{var})"
        idx = counter[0]
        counter[0] += 1
        var = f"fv{idx}"
        bind_vars[var] = value
        return f"{field} != @{var}"

    idx = counter[0]
    counter[0] += 1
    var = f"fv{idx}"
    bind_vars[var] = value

    op_map = {
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "in": "IN",
        "not in": "NOT IN",
    }

    if op in ("in", "not in"):
        if not isinstance(value, list):
            msg = f"Filter operator '{op}' requires a list value, got {type(value).__name__}."
            raise FilterError(msg)
        aql_op = "IN" if op == "in" else "NOT IN"
        return f"{field} {aql_op} @{var}"

    if op in op_map:
        return f"{field} {op_map[op]} @{var}"

    msg = f"Unsupported filter operator: {op}"
    raise FilterError(msg)
