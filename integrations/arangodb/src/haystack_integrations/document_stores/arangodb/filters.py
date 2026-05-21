# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


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
        joiner = " AND " if op == "AND" else " OR "
        inner = joiner.join(parts)
        return f"({inner})"

    if "field" in node and "operator" in node and "value" in node:
        return _parse_comparison(node, bind_vars, counter)

    if "NOT" in node:
        inner = _parse_filter(node["NOT"], bind_vars, counter)
        return f"(NOT {inner})"

    msg = f"Unsupported filter structure: {node}"
    raise ValueError(msg)


def _parse_comparison(node: dict[str, Any], bind_vars: dict[str, Any], counter: list[int]) -> str:
    field = _field_to_aql(node["field"])
    op = node["operator"]
    value = node["value"]

    idx = counter[0]
    counter[0] += 1
    var = f"fv{idx}"
    bind_vars[var] = value

    op_map = {
        "==": "==",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
        "in": "IN",
        "not in": "NOT IN",
    }

    if op == "in":
        return f"{field} IN @{var}"
    if op == "not in":
        return f"{field} NOT IN @{var}"
    if op in op_map:
        return f"{field} {op_map[op]} @{var}"

    msg = f"Unsupported filter operator: {op}"
    raise ValueError(msg)
