# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from datetime import datetime
from typing import Any, ClassVar

from haystack.errors import FilterError

_RANGE_OPS = {">", ">=", "<", "<="}
_JSON_FIELD_NAME = r"^[A-Za-z0-9_.]+$"


class FilterTranslator:
    """
    Translates Haystack 2.x filter dicts into Oracle SQL WHERE fragments.

    Example input:
        {"operator": "AND", "conditions": [
            {"field": "meta.author", "operator": "==", "value": "Alice"},
            {"field": "meta.year",   "operator": ">",  "value": 2020},
        ]}

    Example output SQL fragment:
        (JSON_VALUE(metadata, '$.author') = :p0
         AND TO_NUMBER(JSON_VALUE(metadata, '$.year')) > :p1)

    Params dict is mutated in-place; caller passes an empty dict and uses it
    for cursor.execute / cursor.executemany bindings.
    """

    _OP_MAP: ClassVar[dict[str, str]] = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }

    def translate(self, filters: dict[str, Any], params: dict[str, Any], counter: list[int]) -> str:
        """
        Translate the given filter dict into an SQL fragment, adding any necessary parameters to the params dict.
        """
        op = filters.get("operator")

        # Logical nodes
        if op in ("AND", "OR", "NOT"):
            if "conditions" not in filters:
                msg = f"'conditions' key missing in logical filter: {filters}"
                raise FilterError(msg)
            if op == "AND":
                parts = [self.translate(c, params, counter) for c in filters["conditions"]]
                return "(" + " AND ".join(parts) + ")"
            if op == "OR":
                parts = [self.translate(c, params, counter) for c in filters["conditions"]]
                return "(" + " OR ".join(parts) + ")"
            # NOT
            inner = self.translate(filters["conditions"][0], params, counter)
            return f"(NOT {inner})"

        # Comparison leaf — validate required keys first
        if "field" not in filters:
            msg = f"'field' key missing in comparison filter: {filters}"
            raise FilterError(msg)
        if "operator" not in filters:
            msg = f"'operator' key missing in comparison filter: {filters}"
            raise FilterError(msg)
        if "value" not in filters:
            msg = f"'value' key missing in comparison filter: {filters}"
            raise FilterError(msg)

        if not isinstance(op, str) or op not in {*self._OP_MAP, "in", "not in", "contains", "not contains"}:
            msg = f"Unsupported filter operator: {op!r}"
            raise FilterError(msg)

        field: str = filters["field"]
        value: Any = filters["value"]

        if op in ("contains", "not contains"):
            if isinstance(value, list):
                msg = f"{op!r} filter values must be scalar, got list"
                raise FilterError(msg)
            json_path = FilterTranslator._field_to_json_path(field)
            pname = f"p{counter[0]}"
            counter[0] += 1
            params[pname] = value
            contains_sql = f"JSON_EXISTS(metadata, '{json_path}[*]?(@ == $val)' PASSING :{pname} AS \"val\")"
            if op == "contains":
                return contains_sql
            return f"(NOT {contains_sql})"

        if op in ("in", "not in"):
            if not isinstance(value, list):
                msg = f"'in' / 'not in' filter values must be a list, got {type(value).__name__!r}"
                raise FilterError(msg)
            col = FilterTranslator._field_to_sql(field, value[0] if value else value)
            placeholders = []
            for v in value:
                pname = f"p{counter[0]}"
                counter[0] += 1
                params[pname] = v
                placeholders.append(f":{pname}")
            if op == "in":
                return f"{col} IN ({', '.join(placeholders)})"
            # NOT IN: include rows where the column is NULL, matching Python's
            # `None not in [...]` == True semantics.
            return f"({col} IS NULL OR {col} NOT IN ({', '.join(placeholders)}))"

        if op in _RANGE_OPS and (isinstance(value, (str, list)) and not _is_iso_date(value)):
            msg = f"Operator {op!r} requires a numeric or ISO-date value, got {type(value).__name__!r}: {value!r}"
            raise FilterError(msg)

        col = FilterTranslator._field_to_sql(field, value)

        # NULL-safe equality / inequality
        if op == "==" and value is None:
            return f"{col} IS NULL"
        if op == "!=" and value is None:
            return f"{col} IS NOT NULL"

        pname = f"p{counter[0]}"
        counter[0] += 1
        params[pname] = value

        if op == "!=":
            # Include rows where the column is NULL, matching Python's
            # `None != x` == True semantics.
            return f"({col} != :{pname} OR {col} IS NULL)"

        sql_op = self._OP_MAP[op]
        return f"{col} {sql_op} :{pname}"

    @staticmethod
    def _field_to_sql(field: str, value: Any) -> str:
        if field == "id":
            return "id"
        if field == "content":
            return "text"
        if field.startswith("meta."):
            key = field[len("meta.") :]
            json_path = f"JSON_VALUE(metadata, '$.{key}')"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"TO_NUMBER({json_path})"
            return json_path
        # Fallback: treat as top-level JSON key
        json_path = f"JSON_VALUE(metadata, '$.{field}')"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"TO_NUMBER({json_path})"
        return json_path

    @staticmethod
    def _field_to_json_path(field: str) -> str:
        if not field.startswith("meta."):
            msg = f"Operator 'contains' supports only metadata fields, got {field!r}"
            raise FilterError(msg)
        key = field[len("meta.") :]
        if not re.match(_JSON_FIELD_NAME, key):
            msg = f"Invalid metadata field name: {field!r}"
            raise FilterError(msg)
        return f"$.{key}"


def _infer_hybrid_filter_type(value: Any) -> str:
    if isinstance(value, bool):
        msg = "Boolean values are not supported for Oracle hybrid filters."
        raise FilterError(msg)
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    msg = "Oracle hybrid filters support only string and numeric values."
    raise FilterError(msg)


def _hybrid_filter_path(field: str) -> str:
    if not field.startswith("meta."):
        msg = "Oracle hybrid retrieval supports only filters under the 'meta.' field."
        raise FilterError(msg)
    if not re.match(_JSON_FIELD_NAME, field):
        msg = f"Invalid metadata field name: {field!r}"
        raise FilterError(msg)
    return "metadata." + field[len("meta.") :]


def to_hybrid_filter(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Converts Haystack filters into DBMS_HYBRID_VECTOR.SEARCH filter_by JSON.
    """
    op = filters.get("operator")
    if op in ("AND", "OR", "NOT"):
        if "conditions" not in filters:
            msg = f"'conditions' key missing in logical filter: {filters}"
            raise FilterError(msg)
        return {"op": op, "args": [to_hybrid_filter(condition) for condition in filters["conditions"]]}

    if "field" not in filters:
        msg = f"'field' key missing in comparison filter: {filters}"
        raise FilterError(msg)
    if "operator" not in filters:
        msg = f"'operator' key missing in comparison filter: {filters}"
        raise FilterError(msg)
    if "value" not in filters:
        msg = f"'value' key missing in comparison filter: {filters}"
        raise FilterError(msg)

    field = _hybrid_filter_path(filters["field"])
    value = filters["value"]
    if value is None:
        msg = "Oracle hybrid retrieval does not support null comparisons."
        raise FilterError(msg)
    if op in {"contains", "not contains"}:
        msg = f"Filter operation {op!r} is not supported for Oracle hybrid retrieval."
        raise FilterError(msg)

    if op in {"in", "not in"}:
        if not isinstance(value, list) or not value:
            msg = f"{op!r} filter requires a non-empty list."
            raise FilterError(msg)
        value_type = _infer_hybrid_filter_type(value[0])
        if any(_infer_hybrid_filter_type(item) != value_type for item in value):
            msg = "Oracle hybrid retrieval requires all 'in' filter values to share one type."
            raise FilterError(msg)
        hybrid_filter: dict[str, Any] = {"op": "IN", "path": field, "type": value_type, "args": value}
        if op == "not in":
            return {"op": "NOT", "args": [hybrid_filter]}
        return hybrid_filter

    hybrid_op_map = {
        "==": "=",
        "!=": "!=",
        ">": ">",
        ">=": ">=",
        "<": "<",
        "<=": "<=",
    }
    if not isinstance(op, str) or op not in hybrid_op_map:
        msg = f"Unsupported filter operator: {op!r}"
        raise FilterError(msg)

    return {
        "op": hybrid_op_map[op],
        "path": field,
        "type": _infer_hybrid_filter_type(value),
        "args": [value],
    }


def _is_iso_date(value: Any) -> bool:
    """Return True if *value* is a string that Python recognises as a valid ISO-8601 datetime."""
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
